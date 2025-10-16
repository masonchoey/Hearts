"""
Hearts AI Model Wrapper
Loads and uses RLlib-trained model for inference
"""
import numpy as np
from typing import List, Optional
import ray
from ray.rllib.algorithms.ppo import PPO
import os
from dotenv import load_dotenv

load_dotenv()

class HeartsAIModel:
    """
    Wrapper for RLlib-trained Hearts model
    Handles inference for AI players
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize the AI model
        
        Args:
            checkpoint_path: Path to RLlib checkpoint directory
        """
        self.checkpoint_path = checkpoint_path
        self.algorithm = None
        self._model_loaded = False
        
        # Don't load model immediately - will load lazily on first use
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Model checkpoint found at: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint path '{checkpoint_path}' not found. AI will use random policy.")
    
    def _load_model(self):
        """Load RLlib model from checkpoint"""
        if self._model_loaded:
            return
        
        try:
            # Initialize Ray with minimal resources and local mode for better stability
            if not ray.is_initialized():
                ray.init(
                    ignore_reinit_error=True, 
                    log_to_driver=False,
                    num_cpus=1,
                    num_gpus=0,
                    local_mode=False,
                    _temp_dir="/tmp/ray"
                )
            
            # Load the trained algorithm
            self.algorithm = PPO.from_checkpoint(self.checkpoint_path)
            self._model_loaded = True
            print(f"Successfully loaded model from {self.checkpoint_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Full error details: {repr(e)}")
            self.algorithm = None
            self._model_loaded = False
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        """
        Get action from model given observation and legal actions.
        
        The observation is a structured 5088-length tensor containing:
        - Pass direction (4 values)
        - Dealt hand (52 values)
        - Passed cards (52 values)
        - Received cards (52 values)
        - Current hand (52 values at indices 160-212) - most critical for decision making
        - Points (144 values)
        - Trick history (4732 values)
        
        Args:
            observation: info_state observation vector from OpenSpiel (5088 values)
            legal_actions: List of legal action indices
            
        Returns:
            action: Selected action index
        """
        # Lazily load model on first use
        if not self._model_loaded and self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print("Loading AI model (first use)...")
            self._load_model()
        
        if self.algorithm is None:
            # Fallback to random policy if model not loaded
            return np.random.choice(legal_actions)
        
        try:
            # Create action mask (same format as training environment)
            action_mask = np.zeros(52, dtype=np.int8)
            action_mask[legal_actions] = 1
            
            # Format observation as Dict (same as training)
            obs_dict = {
                "observations": observation.astype(np.float32),
                "action_mask": action_mask
            }
            
            # Get action from trained model
            action = self.algorithm.compute_single_action(
                obs_dict,
                explore=False,  # Use greedy policy for inference
                policy_id="default_policy"
            )
            
            # Ensure action is legal
            if action not in legal_actions:
                # If model suggests illegal action, fallback to random legal action
                action = np.random.choice(legal_actions)
            
            return int(action)
        
        except Exception as e:
            print(f"Error during AI inference: {e}")
            # Fallback to random action
            return np.random.choice(legal_actions)
    
    def __del__(self):
        """Cleanup resources"""
        if self.algorithm:
            try:
                self.algorithm.stop()
            except:
                pass


