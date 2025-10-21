"""
Hearts AI Model Wrapper
Loads and uses RLlib-trained model for inference
"""
import numpy as np
from typing import List, Optional
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path to import training environment and model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import and register the training environment (needed for checkpoint loading)
# Even though we use hearts_env_human_vs_ai for gameplay, RLlib needs the
# training environment registered to load checkpoints that were trained with it
try:
    from hearts_env_self_play import HeartsGymEnvSelfPlay
    
    def env_creator_self_play(env_config):
        """Factory for the self-play environment (used during training)"""
        return HeartsGymEnvSelfPlay(env_config)
    
    register_env("hearts_env_self_play", env_creator_self_play)
    print("✓ Registered hearts_env_self_play for checkpoint loading")
except ImportError as e:
    print(f"Warning: Could not import hearts_env_self_play: {e}")
    print("This may cause issues loading checkpoints trained with self-play environment")

# Import and register the custom model architecture (needed for checkpoint loading)
try:
    from ray.rllib.models import ModelCatalog
    from attention_model import AttentionMaskModel
    
    ModelCatalog.register_custom_model("masked_attention_model", AttentionMaskModel)
    print("✓ Registered masked_attention_model for checkpoint loading")
except ImportError as e:
    print(f"Warning: Could not import attention_model: {e}")
    print("This may cause issues loading checkpoints trained with custom architecture")

class HeartsAIModel:
    """
    Wrapper for RLlib-trained Hearts model
    Handles inference for AI players
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, eager_load: bool = False):
        """
        Initialize the AI model
        
        Args:
            checkpoint_path: Path to RLlib checkpoint directory
            eager_load: If True, load model immediately instead of lazily
        """
        self.checkpoint_path = checkpoint_path
        self.algorithm = None
        self._model_loaded = False
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Model checkpoint found at: {checkpoint_path}")
            if eager_load:
                print("Eager loading model on initialization...")
                self._load_model()
        else:
            print(f"Warning: Checkpoint path '{checkpoint_path}' not found. AI will use random policy.")
    
    def _initialize_ray(self):
        """Initialize Ray with optimized settings for inference"""
        if ray.is_initialized():
            print("Ray already initialized")
            return
        
        try:
            # Initialize Ray with optimized settings for low-latency inference
            ray.init(
                ignore_reinit_error=True,
                log_to_driver=False,
                # Allocate minimal resources for inference workload
                num_cpus=2,  # One for driver, one for worker
                num_gpus=0,
                # Use dedicated temp directory
                _temp_dir="/tmp/ray",
                # Configure for inference workload
                namespace="hearts_inference",
            )
            print(f"✓ Ray initialized successfully (version: {ray.__version__})")
            # Try to get dashboard URL, but don't fail if not available
            try:
                dashboard_url = ray.get_runtime_context().dashboard_url
                if dashboard_url:
                    print(f"✓ Ray dashboard available at: {dashboard_url}")
            except (AttributeError, Exception):
                # Dashboard URL not available in this Ray version or configuration
                print("✓ Ray dashboard: http://127.0.0.1:8265 (if enabled)")
        except Exception as e:
            print(f"Error initializing Ray: {e}")
            raise
    
    def _load_model(self):
        """Load RLlib model from checkpoint with CPU-only inference config"""
        if self._model_loaded:
            return
        
        try:
            start_time = time.time()
            
            # Initialize Ray if not already initialized
            self._initialize_ray()
            
            # Load the trained algorithm (use absolute path to avoid URI errors)
            checkpoint_path = os.path.abspath(self.checkpoint_path)
            print(f"Loading PPO model from checkpoint: {checkpoint_path}")
            
            # Load checkpoint with CPU-only settings for inference
            # The checkpoint may have been trained with GPU, but we override for inference
            from ray.rllib.algorithms.ppo import PPOConfig
            
            print("Configuring algorithm for CPU-only inference...")
            
            # Create CPU-only config for inference
            # Must match training config: old API stack + custom model
            config = (
                PPOConfig()
                .environment("hearts_env_self_play")  # Required for checkpoint compatibility
                .framework("torch")
                .api_stack(
                    enable_rl_module_and_learner=False,  # Use old API stack (checkpoint was trained with this)
                    enable_env_runner_and_connector_v2=False,
                )
                .resources(
                    num_gpus=0,  # No GPUs for inference
                )
                .env_runners(
                    num_env_runners=0,  # Single worker for inference
                    num_envs_per_env_runner=1,
                )
                .training(
                    model={
                        "custom_model": "masked_attention_model",  # Use the registered custom model
                        "custom_model_config": {},  # Model will use its default config
                    }
                )
            )
            
            # Build algorithm with CPU config
            print("Building algorithm with CPU-only configuration...")
            self.algorithm = config.build()
            
            # Restore weights from checkpoint
            print(f"Restoring weights from checkpoint: {checkpoint_path}")
            self.algorithm.restore(checkpoint_path)
            
            self._model_loaded = True
            
            load_time = time.time() - start_time
            print(f"✓ Successfully loaded model in {load_time:.2f}s")
            
            # Perform a warmup inference to ensure everything is ready
            self._warmup()
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print(f"Full error details: {repr(e)}")
            self.algorithm = None
            self._model_loaded = False
            raise
    
    def _warmup(self):
        """Perform a warmup inference to ensure model is ready"""
        try:
            print("Warming up model with dummy inference...")
            # Create dummy observation matching the expected structure
            dummy_obs = np.zeros(5088, dtype=np.float32)
            dummy_action_mask = np.zeros(52, dtype=np.int8)
            dummy_action_mask[0] = 1  # At least one legal action
            
            obs_dict = {
                "observations": dummy_obs,
                "action_mask": dummy_action_mask
            }
            
            # Perform dummy inference
            self.algorithm.compute_single_action(
                obs_dict,
                explore=False,
                policy_id="default_policy"
            )
            print("✓ Model warmup complete")
        except Exception as e:
            print(f"Warning: Model warmup failed: {e}")
    
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
        # Lazily load model on first use (fallback if not eagerly loaded)
        if not self._model_loaded and self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print("Loading AI model (lazy fallback - should not happen if eager_load=True)...")
            self._load_model()
        
        if self.algorithm is None:
            # Fallback to random policy if model not loaded
            print("Warning: Using random policy (model not loaded)")
            return np.random.choice(legal_actions)
        
        try:
            # Create action mask (same format as training environment)
            action_mask = np.zeros(52, dtype=np.int8)
            action_mask[legal_actions] = 1
            
            # Format observation as Dict (same as training)
            obs_dict = {
                "observations": np.array(observation, dtype=np.float32),
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
                print(f"Warning: Model suggested illegal action {action}, using random")
                action = np.random.choice(legal_actions)
            
            return int(action)
        
        except Exception as e:
            print(f"Error during AI inference: {e}")
            # Fallback to random action
            return np.random.choice(legal_actions)
    
    def shutdown(self):
        """Gracefully shutdown the model and clean up resources"""
        if self.algorithm:
            try:
                print("Shutting down AI model...")
                self.algorithm.stop()
                self.algorithm = None
            except Exception as e:
                print(f"Error stopping algorithm: {e}")
        
        self._model_loaded = False
    
    def __del__(self):
        """Cleanup resources"""
        self.shutdown()


