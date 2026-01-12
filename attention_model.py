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
    - Trick history (4732 values) → NEW: processed as 52 separate per-card tokens
    
    Key architectural feature: Per-card processing (Updated 1/9/2026)
    - Each of the 52 cards in trick history is embedded separately
    - For each card, we embed: card_id, player_id, trick_idx, play_order
    - These embeddings are concatenated and projected into a single token
    - All 52 card tokens are appended to the observation sequence
    - The transformer can attend to individual card plays, learning patterns like:
      * Which cards were played by which players
      * When key cards (Queen of Spades, high hearts) were played
      * Sequential card play patterns within and across tricks
    
    Sequence structure: [obs_t1, obs_t2, ..., obs_tN, card_1, card_2, ..., card_52]
    Total tokens: seq_len + 52 (default: 13 + 52 = 65 tokens)
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

        embed_dim = model_config.get("embed_dim", 512)
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
        # Two types of card embeddings:
        # a) For binary card vectors (dealt/passed/received/current hand): use linear projection
        # b) For card indices in trick history: use embedding lookup
        
        card_embed_dim = 40
        trick_embed_dim = 8
        order_embed_dim = 4
        self.card_vector_embed = nn.Linear(52, 64)  # Binary card vector -> embedding
        self.card_embed = nn.Embedding(52, card_embed_dim)  # Card index -> embedding
        #which trick it is (0-12)
        self.trick_embed = nn.Embedding(13, trick_embed_dim )
        #NESWNES order (i think it is that i'm not sure tbh)
        self.order_embed = nn.Embedding(7, order_embed_dim)
        
        # 3. Points embedding (144 -> 64)
        # Thermometer encoding of scores for all 4 players
        self.points_embed = nn.Sequential(
            nn.Linear(144, 32),
            nn.ReLU()
        )
        
        # 4. Trick-history-per-card embedding - NEW APPROACH (1/9/2026)
        # Per-card embedding dimension: 40 (card) + 8 (trick) + 4 (order) = 52
        per_card_embed_dim = card_embed_dim + trick_embed_dim + order_embed_dim  # sum of all embedding dimensions
        
        # Calculate total embedding dimension after concatenation
        # 16 (pass_dir) + 64*4 (card embeds for 4 card features) + 64 (points) = 336
        # Note: tricks are now separate tokens, not concatenated in base_combined
        structured_dim = 16 + (64 * 4) + 64
        
        # Project structured features to model embedding dimension
        self.structured_proj = nn.Linear(structured_dim, embed_dim)
        
        # Project per-card embed tokens to model embedding dimension
        # Input: concatenated embeddings (40 + 4 + 8 + 4 = 56)
        self.per_card_embed_token_proj = nn.Linear(per_card_embed_dim, embed_dim)
        
        # Layer normalization for better training stability
        self.input_norm = nn.LayerNorm(embed_dim)
        self.per_card_token_norm = nn.LayerNorm(embed_dim)
        
        # 🔹 ADDED: Positional encoding for sequence processing
        # Max length needs to accommodate: seq_len observation timesteps + 52 card tokens
        self.positional_encoding = PositionalEncoding(
            embed_dim, max_len=self.seq_len + 52 + 10  # seq_len + 52 cards + buffer
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
        Embed each semantic component and return base features + per-trick tokens.
        
        Args:
            parsed_obs: dict with parsed observation components
            
        Returns:
            tuple: (base_embedding [batch_size, structured_dim], 
                   trick_tokens [batch_size, 13, trick_embed_dim])
        """
        batch_size = parsed_obs['pass_dir'].size(0)
        
        # Embed each component
        pass_dir_emb = self.pass_dir_embed(parsed_obs['pass_dir'])  # [B, 16]
        
        # Embed all card-based features (binary vectors) using linear projection
        dealt_emb = self.card_vector_embed(parsed_obs['dealt_hand'])  # [B, 64]
        passed_emb = self.card_vector_embed(parsed_obs['passed_cards'])  # [B, 64]
        received_emb = self.card_vector_embed(parsed_obs['received_cards'])  # [B, 64]
        current_emb = self.card_vector_embed(parsed_obs['current_hand'])  # [B, 64] - MOST IMPORTANT!
        
        points_emb = self.points_embed(parsed_obs['points'])  # [B, 64]
        

        #STILL A LOT NEEDS TO BE CHANGED
        # Process trick history as 52 separate tokens (per-card approach)
        trick_history = parsed_obs['trick_history']  # [B, 4732]
        # Each of 91 cards has 52 values: 4732 / 91 = 52 (there are 91 possible card placements with NESWNES * 13 tricks)
        # NOT in the form of (card_id, player_id, trick_idx, play_order) needs processing.
        trick_history_reshaped = trick_history.view(batch_size, 91, 52)
        
        # Embed each card placement separately. Each card placement is a 52 length one-hot encoded vector without the desired structure.
        per_card_embed_tokens = []
        #NEED TO FIX ###########################
        lead_card = True
        for card_idx in range(91):
            card_data = trick_history_reshaped[:, card_idx, :]  # [B, 52]
            #if the card_data is all zeros (no card was played in this placement), then skip
            mask = card_data.sum(dim=1) > 0  # [B] bool
            if mask.sum() == 0:
                continue
            else:
                if lead_card == True:
                    lead_card = False
                    lead_player_id = card_idx % 7
                trick_idx = card_idx // 7  # 7 positions per trick
                trick_embedding = self.trick_embed(trick_idx)  # [B, trick_embed_dim]
                # order_embedding = self.order_embed([(i - lead_player_id) % 4 for i in range(7)])
                order_id = (card_idx - lead_player_id) % 4
                order_embedding = self.order_embed(order_id)  # [B, order_embed_dim]
                #extract the card_id, player_id, trick_idx, play_order from the 52 length one-hot encoded vector
                # card_embedding = self.card_embed(card_data)
                card_id = card_data.argmax(dim=-1)  # [B]
                card_embedding = self.card_embed(card_id)  # [B, card_embed_dim]
                final_card_emb = self.per_card_embed_token_proj(torch.cat([card_embedding, player_embedding, trick_embedding, order_embedding], dim=-1))  # [B, 52]
                per_card_embed_tokens.append(final_card_emb)
        
        # Concatenate base embeddings (everything except tricks)
        # Order: pass_dir, current_hand, dealt_hand, passed, received, points
        # We put current_hand early because it's the most important for decision making
        base_combined = torch.cat([
            pass_dir_emb,      # 16
            current_emb,       # 64 - CRITICAL: what cards can we play?
            dealt_emb,         # 64 - what was our original hand?
            passed_emb,        # 64 - what did we pass away?
            received_emb,      # 64 - what did we receive?
            points_emb,        # 64 - current scores
        ], dim=-1)  # Total: 336
        
        return base_combined, per_card_embed_tokens

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
        trick_tokens_embedded = None  # Will store the trick tokens from most recent obs
        
        for t in range(seq_len):
            obs_t = obs_sequence[:, t, :]  # [B, 5088]
            
            # Parse into structured components
            parsed = self._parse_observation(obs_t)
            
            # Embed structured components - now returns base embedding + trick tokens
            base_emb, trick_tokens = self._embed_structured_obs(parsed)  # [B, 336], [B, 52, embed_dim]
            
            # Project base embedding to model embedding dimension
            emb_t = self.structured_proj(base_emb)  # [B, embed_dim]
            emb_t = self.input_norm(emb_t)  # Normalize
            
            embedded_sequence.append(emb_t)
            
            # Store trick tokens from the most recent observation (last timestep)
            if t == seq_len - 1:
                # trick_tokens are already projected to embed_dim in _embed_structured_obs
                # trick_tokens shape: [B, 52, embed_dim] (52 cards, one token per card)
                trick_tokens_embedded = self.per_card_token_norm(trick_tokens)  # Normalize
        
        # Stack observation sequence: [B, seq_len, embed_dim]
        obs_embedded = torch.stack(embedded_sequence, dim=1)
        
        # Concatenate observation sequence with trick tokens: [B, seq_len + 52, embed_dim]
        # Structure: [obs_timestep_1, ..., obs_timestep_N, card_1, ..., card_52]
        # This allows the transformer to:
        # 1. Attend across temporal observations (decision history)
        # 2. Attend to individual card placements in trick history (game history)
        # 3. Cross-attend between current state and past card plays
        x = torch.cat([obs_embedded, trick_tokens_embedded], dim=1)

        # Apply positional encoding for sequence processing
        # Encodes position information for both observation timesteps and trick tokens
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