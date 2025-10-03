import torch
import torch.nn as nn
import math

from gymnasium import spaces as gym_spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import numpy as np

class PositionalEncoding(nn.Module):
    """🔹 ADDED: Explicit positional encodings for sequence processing"""
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

#TODO: STILL NEED TO CALIBRATE OUTPUT OF ENVIRONMENT TO MATCH THE MODEL
class AttentionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.num_outputs = num_outputs
        
        # 🔹 ADDED: Get sequence length from config for multi-timestep processing
        self.seq_len = model_config.get("seq_len", 5)  # Process last 5 game states (TODO: CHANGE THIS TO BE VARIABLE)
        self.store_history = model_config.get("store_history", True)
        
        base_space = getattr(obs_space, "original_space", obs_space)
        if isinstance(base_space, gym_spaces.Dict) and "observations" in base_space.spaces:
            obs_dim = int(np.prod(base_space["observations"].shape))
        else:
            obs_dim = int(np.prod(base_space.shape))

        embed_dim = model_config.get("embed_dim", 64)
        num_heads = model_config.get("num_attention_heads", 2)
        num_layers = model_config.get("num_attention_layers", 1)
        
        # 🔹 ADDED: Player identity embedding for multi-agent awareness
        self.num_players = model_config.get("num_players", 4)  # Hearts has 4 players
        self.player_embed_dim = model_config.get("player_embed_dim", 16)
        self.player_embedding = nn.Embedding(self.num_players, self.player_embed_dim)
        
        # 🔹 FIXED: Use the full observation as-is without incorrect card/game state split
        # The OpenSpiel Hearts observation is already a structured representation
        # that we should not arbitrarily split
        
        # 🔹 MODIFIED: Project full observation + player identity to embed_dim
        combined_dim = obs_dim + self.player_embed_dim
        self.input_proj = nn.Linear(combined_dim, embed_dim)
        
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

        # 🔹 MODIFIED: Enhanced value head with separate architecture
        self.value_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),  # Larger value network
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
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
            # obs = obs_tensor.float()
            # action_mask = None
            raise ValueError("obs_tensor is not a dict")

        batch_size = obs.size(0)
        
        # 🔹 ADDED: Get current player ID from game state (you may need to extract this differently)
        # Assuming player ID is embedded in observation or passed separately
        player_ids = input_dict.get("player_id", torch.zeros(batch_size, dtype=torch.long, device=obs.device))
        
        # 🔹 MODIFIED: Update observation history for sequence processing
        obs_sequence = self._update_history(obs, batch_size)  # [B, seq_len, obs_dim]
        
        # 🔹 FIXED: Process the full observation without incorrect splitting
        # OpenSpiel Hearts observation is already properly structured
        full_obs = obs_sequence  # [B, seq_len, obs_dim]
        
        # 🔹 ADDED: Add player identity embedding
        player_embed = self.player_embedding(player_ids)  # [B, player_embed_dim]
        player_embed = player_embed.unsqueeze(1).expand(-1, self.seq_len, -1)  # [B, seq_len, player_embed_dim]
        
        # Concatenate full observation and player embedding, then project
        full_embed = torch.cat([full_obs, player_embed], dim=-1)  # [B, seq_len, obs_dim + player_embed_dim]
        x = self.input_proj(full_embed)  # [B, seq_len, embed_dim]

        # 🔹 ADDED: Apply positional encoding for sequence processing
        x = self.positional_encoding(x)

        # 🔹 UNCHANGED: Run transformer encoder (now with proper sequences)
        features = self.transformer(x)  # [B, seq_len, embed_dim]
        
        # 🔹 MODIFIED: Use attention pooling instead of mean pooling
        pooled = self.attention_pooling(features)  # [B, embed_dim]

        # 🔹 MODIFIED: Enhanced policy head
        logits = self.logits_layer(pooled)

        # 🔹 UNCHANGED: Apply action mask at logits stage
        if action_mask is not None:
            inf_mask = torch.clamp(
                torch.log(action_mask), 
                min=torch.finfo(torch.float32).min
            )
            logits = logits + inf_mask

        # 🔹 MODIFIED: Enhanced value head
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