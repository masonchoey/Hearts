import torch
import torch.nn as nn
import math

from gymnasium import spaces as gym_spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import numpy as np

# OpenSpiel Hearts observation structure (total: 5088 values)
# Based on hearts.h lines 65-72 and hearts.cc lines 241-310
OBS_STRUCTURE = {
    'pass_dir': (0, 4),           # 4 values - one-hot pass direction
    'dealt_hand': (4, 56),         # 52 values - initial cards dealt
    'passed_cards': (56, 108),     # 52 values - cards passed away
    'received_cards': (108, 160),  # 52 values - cards received from opponent
    'current_hand': (160, 212),    # 52 values - CRITICAL: cards currently in hand
    'points': (212, 356),          # 144 values - thermometer encoding of scores (36 per player * 4)
    'trick_history': (356, 5088)   # 4732 values - history of all tricks played (13 tricks * 364)
}

class PositionalEncoding(nn.Module):
    """Explicit positional encodings for sequence processing"""
    def __init__(self, embed_dim, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, embed_dim]
        
        self.register_buffer('pe', pe)  # Non-trainable parameter
    
    def forward(self, x):
        """x shape: [seq_len, batch_size, embed_dim] or [batch_size, seq_len, embed_dim]"""
        # The transformer in the main model uses batch_first=True, so we expect [batch_size, seq_len, embed_dim]
        # In this case: [32, 5, 128] means batch_size=32, seq_len=5, embed_dim=128
        
        if x.dim() == 3:
            batch_size, seq_len, embed_dim = x.shape
            # For batch_first format [batch_size, seq_len, embed_dim], directly use seq_len
            x = x + self.pe[:seq_len, :, :].squeeze(1)  # pe is [max_len, 1, embed_dim], so squeeze middle dim
        else:
            # Handle 2D case if needed
            seq_len = x.size(0)
            x = x + self.pe[:seq_len, :].squeeze(1)
        
        return self.dropout(x)

class AttentionPooling(nn.Module):
    """🔹 ADDED: Attention-based pooling instead of mean pooling"""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Learnable query vector for pooling
        self.query_vector = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Multi-head attention for pooling
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        x shape: [batch_size, seq_len, embed_dim]
        Returns: [batch_size, embed_dim]
        """
        batch_size = x.size(0)
        
        # Expand query for batch
        query = self.query_vector.expand(batch_size, -1, -1)
        
        # Apply attention pooling
        pooled, _ = self.attention(query, x, x)
        pooled = self.layer_norm(pooled)
        
        return pooled.squeeze(1)  # Remove sequence dimension

class AttentionMaskModel(TorchModelV2, nn.Module):
    """
    Structured Hearts model that properly parses the 5088-length observation tensor.
    
    Instead of treating the observation as a flat vector, this model understands
    the semantic structure:
    - Pass direction (4 values)
    - Card representations (52 values each for dealt/passed/received/current hand)
    - Points (144 values - 36 per player)
    - Trick history (4732 values)
    
    This allows the model to immediately understand which cards are in its hand,
    rather than learning this from scratch through backpropagation.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.num_outputs = num_outputs
        
        # Get sequence length from config for multi-timestep processing
        self.seq_len = model_config.get("seq_len", 13)
        self.store_history = model_config.get("store_history", True)
        
        base_space = getattr(obs_space, "original_space", obs_space)
        if isinstance(base_space, gym_spaces.Dict) and "observations" in base_space.spaces:
            obs_dim = int(np.prod(base_space["observations"].shape))
        else:
            obs_dim = int(np.prod(base_space.shape))

        # Verify observation dimension matches expected structure
        assert obs_dim == 5088, f"Expected observation dimension 5088, got {obs_dim}"

        embed_dim = model_config.get("embed_dim", 256)
        num_heads = model_config.get("num_attention_heads", 4)
        num_layers = model_config.get("num_attention_layers", 2)
        
        # ===== STRUCTURED OBSERVATION PARSING =====
        # Instead of one linear projection, we create separate embeddings
        # for different semantic components of the observation
        
        # 1. Pass direction embedding (4 -> 16)
        self.pass_dir_embed = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU()
        )
        
        # 2. Card representations (52 values each)
        # Shared embedding for all card-based features (current hand, dealt hand, etc.)
        card_embed_dim = 64
        self.card_embed = nn.Sequential(
            nn.Linear(52, card_embed_dim),
            nn.LayerNorm(card_embed_dim),
            nn.ReLU()
        )
        
        # 3. Points embedding (144 -> 64)
        # Thermometer encoding of scores for all 4 players
        self.points_embed = nn.Sequential(
            nn.Linear(144, 64),
            nn.ReLU()
        )
        
        # 4. Trick history embedding (4732 -> 128)
        # History of all 13 tricks played so far
        self.trick_history_embed = nn.Sequential(
            nn.Linear(4732, 128),
            nn.ReLU()
        )
        
        # Calculate total embedding dimension after concatenation
        # 16 (pass_dir) + 64*4 (card embeds) + 64 (points) + 128 (tricks) = 464
        structured_dim = 16 + (card_embed_dim * 4) + 64 + 128
        
        # Project structured features to model embedding dimension
        self.structured_proj = nn.Linear(structured_dim, embed_dim)
        
        # Layer normalization for better training stability
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # 🔹 ADDED: Positional encoding for sequence processing
        self.positional_encoding = PositionalEncoding(
            embed_dim, max_len=self.seq_len * 2  # Extra buffer
        )

        # 🔹 UNCHANGED: Transformer encoder (but now processes sequences properly)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 🔹 ADDED: Attention-based pooling instead of mean pooling
        self.attention_pooling = AttentionPooling(embed_dim, num_heads)
        
        # 🔹 MODIFIED: Enhanced policy head with residual connection
        self.logits_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_outputs)
        )

        # 🔹 IMPROVED: More powerful value head to increase explained variance
        # This is critical - low vf_explained_var was a major issue
        self.value_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),  # Add normalization for stability
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )
        
        self._value_out = None
        
        # 🔹 ADDED: History buffer for sequence processing
        if self.store_history:
            # Initialize as None - will be created dynamically based on actual batch size
            self.obs_history = None
            self.history_ptr = None

    def _update_history(self, obs, batch_size):
        """🔹 ADDED: Manage observation history for sequence processing"""
        if not self.store_history:
            return obs.unsqueeze(1)  # Single timestep
            
        # Initialize history buffer if not created or batch size changed
        if self.obs_history is None or self.obs_history.size(0) != batch_size:
            device = obs.device
            self.obs_history = torch.zeros(batch_size, self.seq_len, obs.size(-1), device=device)
            self.history_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Update circular buffer
        for i in range(batch_size):
            ptr = self.history_ptr[i].item()
            self.obs_history[i, ptr] = obs[i]
            self.history_ptr[i] = (ptr + 1) % self.seq_len
        
        return self.obs_history.clone()

    def _parse_observation(self, obs):
        """
        Parse the flat 5088-length observation into structured semantic components.
        
        Args:
            obs: [batch_size, 5088] tensor
            
        Returns:
            dict with parsed components
        """
        # Extract each component according to OBS_STRUCTURE
        parsed = {}
        parsed['pass_dir'] = obs[:, OBS_STRUCTURE['pass_dir'][0]:OBS_STRUCTURE['pass_dir'][1]]
        parsed['dealt_hand'] = obs[:, OBS_STRUCTURE['dealt_hand'][0]:OBS_STRUCTURE['dealt_hand'][1]]
        parsed['passed_cards'] = obs[:, OBS_STRUCTURE['passed_cards'][0]:OBS_STRUCTURE['passed_cards'][1]]
        parsed['received_cards'] = obs[:, OBS_STRUCTURE['received_cards'][0]:OBS_STRUCTURE['received_cards'][1]]
        parsed['current_hand'] = obs[:, OBS_STRUCTURE['current_hand'][0]:OBS_STRUCTURE['current_hand'][1]]
        parsed['points'] = obs[:, OBS_STRUCTURE['points'][0]:OBS_STRUCTURE['points'][1]]
        parsed['trick_history'] = obs[:, OBS_STRUCTURE['trick_history'][0]:OBS_STRUCTURE['trick_history'][1]]
        
        return parsed
    
    def _embed_structured_obs(self, parsed_obs):
        """
        Embed each semantic component and concatenate into a unified representation.
        
        Args:
            parsed_obs: dict with parsed observation components
            
        Returns:
            [batch_size, structured_dim] tensor
        """
        # Embed each component
        pass_dir_emb = self.pass_dir_embed(parsed_obs['pass_dir'])  # [B, 16]
        
        # Embed all card-based features using shared embedding
        dealt_emb = self.card_embed(parsed_obs['dealt_hand'])  # [B, 64]
        passed_emb = self.card_embed(parsed_obs['passed_cards'])  # [B, 64]
        received_emb = self.card_embed(parsed_obs['received_cards'])  # [B, 64]
        current_emb = self.card_embed(parsed_obs['current_hand'])  # [B, 64] - MOST IMPORTANT!
        
        points_emb = self.points_embed(parsed_obs['points'])  # [B, 64]
        tricks_emb = self.trick_history_embed(parsed_obs['trick_history'])  # [B, 128]
        
        # Concatenate all embeddings
        # Order: pass_dir, current_hand, dealt_hand, passed, received, points, tricks
        # We put current_hand early because it's the most important for decision making
        combined = torch.cat([
            pass_dir_emb,      # 16
            current_emb,       # 64 - CRITICAL: what cards can we play?
            dealt_emb,         # 64 - what was our original hand?
            passed_emb,        # 64 - what did we pass away?
            received_emb,      # 64 - what did we receive?
            points_emb,        # 64 - current scores
            tricks_emb         # 128 - what has been played?
        ], dim=-1)  # Total: 464
        
        return combined

    def forward(self, input_dict, state, seq_lens):
        obs_tensor = input_dict["obs"]
        if isinstance(obs_tensor, dict) and "observations" in obs_tensor:
            obs = obs_tensor["observations"].float()
            action_mask = obs_tensor.get("action_mask", None)
            if action_mask is not None:
                action_mask = action_mask.float()
            else:
                raise ValueError("action_mask is not in obs_tensor")
        else:
            raise ValueError("obs_tensor is not a dict")

        batch_size = obs.size(0)
        
        # Update observation history for sequence processing
        obs_sequence = self._update_history(obs, batch_size)  # [B, seq_len, 5088]
        
        # Parse and embed each observation in the sequence
        seq_len = obs_sequence.size(1)
        embedded_sequence = []
        
        for t in range(seq_len):
            obs_t = obs_sequence[:, t, :]  # [B, 5088]
            
            # Parse into structured components
            parsed = self._parse_observation(obs_t)
            
            # Embed structured components
            structured_emb = self._embed_structured_obs(parsed)  # [B, 464]
            
            # Project to model embedding dimension
            emb_t = self.structured_proj(structured_emb)  # [B, embed_dim]
            
            embedded_sequence.append(emb_t)
        
        # Stack into sequence: [B, seq_len, embed_dim]
        x = torch.stack(embedded_sequence, dim=1)
        x = self.input_norm(x)  # Normalize for training stability

        # Apply positional encoding for sequence processing
        x = self.positional_encoding(x)

        # Run transformer encoder to capture sequential dependencies
        features = self.transformer(x)  # [B, seq_len, embed_dim]
        
        # Use attention pooling to aggregate sequence information
        pooled = self.attention_pooling(features)  # [B, embed_dim]

        # Generate policy logits
        logits = self.logits_layer(pooled)

        # Apply action mask to prevent illegal actions
        if action_mask is not None:
            inf_mask = torch.clamp(
                torch.log(action_mask), 
                min=torch.finfo(torch.float32).min
            )
            logits = logits + inf_mask

        # Generate value estimate
        self._value_out = self.value_net(pooled).squeeze(-1)

        return logits, state

    def value_function(self):
        return self._value_out
    
    def reset_history(self):
        """🔹 ADDED: Reset history buffer (call between episodes)"""
        if self.store_history and hasattr(self, 'obs_history') and self.obs_history is not None:
            self.obs_history.fill_(0)
            self.history_ptr.fill_(0)


if __name__ == "__main__":
    ModelCatalog.register_custom_model("masked_attention_model", AttentionMaskModel)