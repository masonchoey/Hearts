"""
RL vs Bots Hearts Simulation with Debug Mode

This script simulates Hearts games between a trained RL agent and various bot types,
with comprehensive debugging capabilities to analyze the RL agent's decision-making process.
Results are saved to JSON files for analysis.

USAGE:
1. Regular simulation (auto-detects most recent checkpoint):
   python rl_vs_bots.py
   
2. Specify a specific checkpoint:
   CHECKPOINT_PATH=/path/to/checkpoint_000049 python rl_vs_bots.py
   
3. Enable RL debugging via environment variables:
   DEBUG_RL=true DEBUG_FREQUENCY=5 python rl_vs_bots.py
   
4. Combined usage with specific checkpoint and debugging:
   CHECKPOINT_PATH=./PPO_2025-08-28_23-03-01/PPO_hearts_env_self_play_d3320_00000_0_2025-08-28_23-03-03/checkpoint_000049 DEBUG_RL=true python rl_vs_bots.py
   
5. Use the debug function in Python:
   from rl_vs_bots import run_debug_simulation
   run_debug_simulation("conservative", num_games=3, debug_frequency=1)

DEBUGGING FEATURES:
- Detailed observation analysis (input state, action masks, legal actions)
- Policy output inspection (selected action, probabilities, value predictions)
- Step-by-step reward tracking and game state changes
- Decision summary showing all RL agent choices
- Action legality verification and error handling
- JSON output with detailed statistics and game results

ENVIRONMENT VARIABLES:
- BOT_TYPE: "conservative", "self", or "random" (default: "conservative")
- DEBUG_RL: "true" to enable detailed RL debugging (default: "false")
- DEBUG_FREQUENCY: How often to show debug info (default: "10")
- NUM_GAMES: Number of games to play (default: "300")
- CHECKPOINT_PATH: Path to specific checkpoint for Player 0 (optional, auto-detects if not provided)
- PLAYER_1_CHECKPOINT: Path to checkpoint for Player 1 (optional, uses CHECKPOINT_PATH if not provided)
- PLAYER_2_CHECKPOINT: Path to checkpoint for Player 2 (optional, uses CHECKPOINT_PATH if not provided)
- PLAYER_3_CHECKPOINT: Path to checkpoint for Player 3 (optional, uses CHECKPOINT_PATH if not provided)
"""

# import os
# import sys
# # Sanitize sys.path to avoid mixing conda site-packages with this venv
# sys.path[:] = [p for p in sys.path if "/opt/anaconda3/" not in p]
# os.environ.pop("PYTHONPATH", None)

import pyspiel
import random
from collections import defaultdict
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
# from hearts_env_conservative import HeartsGymEnvConservative
from hearts_env_self_play import HeartsGymEnvSelfPlay
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces as gym_spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import os
import pickle
import json
from datetime import datetime
from termcolor import colored
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def show_card(card):
    suit = card % 4
    value = card // 4 + 2
    suit_to_print = ""
    if suit == 3:
        # suit_to_print = "♠️"
        suit_to_print = colored("\u2660", "white") + " "
    elif suit == 2:
        # suit_to_print = "♥️"
        suit_to_print = colored("\u2665", "red") + " "
    elif suit == 1:
        # suit_to_print = "♣️"
        suit_to_print = colored("\u2663", "green") + " "
    elif suit == 0:
        # suit_to_print = "♦️"
        suit_to_print = colored("\u2666", "light_blue") + " "
    if value == 10:
        return("T" + str(suit_to_print))
    elif value == 11:
        return("J" + str(suit_to_print))
    elif value == 12:
        return("Q" + str(suit_to_print))
    elif value == 13:
        return("K" + str(suit_to_print))
    elif value == 14:
        return("A" + str(suit_to_print))
    else:
        return(str(value) + str(suit_to_print))

def get_remaining_point_cards(played_cards):
    """Get list of remaining point cards (hearts and Queen of Spades) not yet played."""
    QUEEN_OF_SPADES = 43  # Queen of Spades in OpenSpiel Hearts representation
    # Hearts cards in OpenSpiel: 2♥=2, 3♥=6, 4♥=10, ..., A♥=50
    HEARTS_CARDS = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]
    
    # All point cards in the game
    all_point_cards = [QUEEN_OF_SPADES] + HEARTS_CARDS
    
    # Find remaining point cards
    remaining_point_cards = [card for card in all_point_cards if card not in played_cards]
    
    return remaining_point_cards

def format_card_list(cards, title):
    """Format a list of cards for terminal display."""
    if not cards:
        return f"{title}: None"
    
    card_displays = [show_card(card) for card in cards]
    return f"{title}: {', '.join(card_displays)}"

def env_creator_conservative(env_config):
    """Factory that builds a conservative-opponent Hearts environment for RLlib."""
    # return HeartsGymEnvConservative(env_config)
    return 'HI'

def env_creator_self_play(env_config):
    """Factory that builds a self-play Hearts environment for RLlib."""
    return HeartsGymEnvSelfPlay(env_config)


class ActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.num_outputs = num_outputs
        hiddens = model_config.get("fcnet_hiddens", [256, 256])

        base_space = getattr(obs_space, "original_space", obs_space)
        if isinstance(base_space, gym_spaces.Dict) and "observations" in base_space.spaces:
            obs_dim = int(np.prod(base_space["observations"].shape))
        else:
            obs_dim = int(np.prod(base_space.shape))

        layers = []
        last_dim = obs_dim
        for hidden_size in hiddens:
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(nn.ReLU())
            last_dim = hidden_size

        self.policy_net = nn.Sequential(*layers)
        self.logits_layer = nn.Linear(last_dim, num_outputs)
        self.value_net = nn.Sequential(
            nn.Linear(last_dim, max(128, last_dim)),
            nn.ReLU(),
            nn.Linear(max(128, last_dim), 1),
        )
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs_tensor = input_dict["obs"]
        if isinstance(obs_tensor, dict) and "observations" in obs_tensor:
            obs = obs_tensor["observations"].float()
            action_mask = obs_tensor.get("action_mask", None)
            if action_mask is not None:
                action_mask = action_mask.float()
        else:
            obs = obs_tensor.float()
            action_mask = None
        features = self.policy_net(obs)
        logits = self.logits_layer(features)
        if action_mask is not None:
            inf_mask = torch.clamp(torch.log(action_mask), min=torch.finfo(torch.float32).min)
            logits = logits + inf_mask
        self._value_out = self.value_net(features).squeeze(-1)
        return logits, state

    def value_function(self):
        return self._value_out


# Ensure the custom model is registered for inference time as well
ModelCatalog.register_custom_model("masked_action_model", ActionMaskModel)

class BotTypes:
    """Different types of bot strategies for Hearts."""
    
    @staticmethod
    def conservative(state, legal_actions):
        """Conservative strategy - avoid hearts and Queen of Spades when possible."""
        if not legal_actions:
            return None
            
        # Card values for decision making
        QUEEN_OF_SPADES = 43  # Queen of Spades in OpenSpiel Hearts representation
        HEARTS_CARDS = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]  # All hearts cards
        
        # 1. If we can avoid taking the Queen of Spades, do so
        non_queen_actions = [a for a in legal_actions if a != QUEEN_OF_SPADES]
        if non_queen_actions and QUEEN_OF_SPADES in legal_actions:
            legal_actions = non_queen_actions
        
        # 2. If we can avoid taking hearts, prefer that
        non_hearts_actions = [a for a in legal_actions if a not in HEARTS_CARDS]
        if non_hearts_actions:
            legal_actions = non_hearts_actions
        
        # 3. Among remaining options, prefer lower cards (more conservative)
        sorted_actions = sorted(legal_actions)
        conservative_actions = sorted_actions[:max(1, len(sorted_actions) // 2)]
        
        # Add some randomness to avoid being completely predictable
        return random.choice(conservative_actions)


class RLvsBotsSimulator:
    """Simulate Hearts games with trained RL agent vs different bot types."""
    
    def __init__(self, model_path=None, bot_type=None):
        """Initialize the simulator.
        
        Args:
            model_path: Path to the trained model checkpoint for Player 0
        """
        # Get bot type from environment variable if not specified
        if bot_type is None:
            raise ValueError("Bot type must be specified")
        self.bot_type = bot_type.lower()
        
        # Register environments
        register_env("hearts_env_conservative", env_creator_conservative)
        register_env("hearts_env_self_play", env_creator_self_play)
        
        # Build PPO agent with the same custom masked model
        ModelCatalog.register_custom_model("masked_action_model", ActionMaskModel)
        
        # Choose environment based on bot type
        if self.bot_type == "conservative":
            env_name = "hearts_env_conservative"
        elif self.bot_type == "self":
            env_name = "hearts_env_self_play"
        else:
            raise ValueError(f"Invalid bot type or bot type not specified: {self.bot_type}")
        
        # Create PPO config template
        config = (
            PPOConfig()
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .environment(env_name)
            .framework("torch")
            .training(model={"custom_model": "masked_action_model", "fcnet_hiddens": [256, 256]})
            .env_runners(num_env_runners=0)
        )
        
        # Initialize agents for each player
        self.agents = {}
        self.agent_states = {}
        
        # Player 0 - main agent being tested
        self.agents[0] = config.build()
        if model_path:
            try:
                print(f"🔄 Restoring Player 0 PPO agent from: {model_path}")
                self.agents[0].restore(model_path)
                print("✅ Successfully restored Player 0 trained PPO agent")
            except Exception as e:
                raise Exception(f"⚠️ Player 0 restore failed: {e}. Continuing with untrained weights.")
        
        # Players 1, 2, 3 - load from player-specific checkpoints if available
        player_checkpoints = {
            1: os.getenv("PLAYER_1_CHECKPOINT"),
            2: os.getenv("PLAYER_2_CHECKPOINT"), 
            3: os.getenv("PLAYER_3_CHECKPOINT")
        }
        
        for player_id in [1, 2, 3]:
            self.agents[player_id] = config.build()
            checkpoint_path = player_checkpoints[player_id]
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    print(f"🔄 Restoring Player {player_id} PPO agent from: {checkpoint_path}")
                    self.agents[player_id].restore(checkpoint_path)
                    print(f"✅ Successfully restored Player {player_id} trained PPO agent")
                except Exception as e:
                    print(f"⚠️ Player {player_id} restore failed: {e}. Using Player 0 checkpoint as fallback.")
                    if model_path:
                        try:
                            self.agents[player_id].restore(model_path)
                            print(f"✅ Player {player_id} using Player 0 checkpoint as fallback")
                        except Exception as fallback_e:
                            print(f"⚠️ Player {player_id} fallback also failed: {fallback_e}. Using untrained weights.")
            else:
                # Use Player 0 checkpoint as default if no player-specific checkpoint provided
                if model_path:
                    try:
                        print(f"🔄 Player {player_id} using Player 0 checkpoint (no specific checkpoint provided)")
                        self.agents[player_id].restore(model_path)
                        print(f"✅ Player {player_id} using Player 0 checkpoint")
                    except Exception as e:
                        print(f"⚠️ Player {player_id} fallback failed: {e}. Using untrained weights.")
        
        # Initialize agent states for each player
        for player_id in range(4):
            self.agent_states[player_id] = self.agents[player_id].get_policy().get_initial_state()
        
        # Keep reference to main agent for backward compatibility
        self.agent = self.agents[0]
        
        # Game statistics
        self.game_results = []
        self.detailed_stats = defaultdict(list)
        self.invalid_moves_count = 0
        self.total_rl_moves_count = 0
        
        # Define bot strategies based on bot type
        if self.bot_type == "conservative":
            self.bot_strategies = {
                "conservative": BotTypes.conservative,
            }
        elif self.bot_type == "self":
            # For self-play, we don't use bot strategies - all players are RL agents
            self.bot_strategies = {}
        else:  # random
            self.bot_strategies = {
                "pure_random": BotTypes.pure_random,
            }
    
    # Removed _create_fresh_action_mask method - no longer needed with single-state architecture

    def load_trained_agent(self, model_path):
        """Load the trained PPO agent."""
        try:
            print(f"🔄 Loading trained RL agent from: {model_path}")
            
            # Check if this is a valid checkpoint directory
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Checkpoint directory does not exist: {model_path}")
            
            # Look for checkpoint files in the directory
            checkpoint_files = []
            if os.path.isdir(model_path):
                # Look for checkpoint files (both old and new format)
                for item in os.listdir(model_path):
                    if item.startswith('checkpoint_') or item.startswith('checkpoint-'):
                        checkpoint_path = os.path.join(model_path, item)
                        if os.path.isfile(checkpoint_path) or os.path.isdir(checkpoint_path):
                            checkpoint_files.append(checkpoint_path)
                
                # If no checkpoint files found, try the directory itself
                if not checkpoint_files:
                    checkpoint_files = [model_path]
            else:
                checkpoint_files = [model_path]
            
            print(f"Found checkpoint files/directories: {checkpoint_files}")
            
            # Prefer restoring the already-built agent instead of from_checkpoint to avoid
            # potential ABI/version issues.
            checkpoint_to_load = checkpoint_files[0] if checkpoint_files else model_path
            try:
                self.agent.restore(checkpoint_to_load)
                print("✅ Successfully restored PPO agent via restore()")
                return
            except Exception as e:
                print(f"⚠️ restore() failed: {e}")
            raise Exception("Checkpoint restore failed")
                
        except Exception as e:
            print(f"❌ Failed to load trained agent from {model_path}: {e}")
            print("\n🔧 TROUBLESHOOTING TIPS:")
            print("1. Make sure you have trained a PPO model first")
            print("2. Check that the checkpoint directory exists and contains valid files")
            print("3. Verify the path is correct and accessible")
            print("4. For Tune checkpoints, the path should point to the algorithm directory")
            print(f"5. Current working directory: {os.getcwd()}")
            print(f"6. Contents of model_path parent: {os.listdir(os.path.dirname(model_path)) if os.path.dirname(model_path) and os.path.exists(os.path.dirname(model_path)) else 'N/A'}")
            raise
        
    
    def play_single_game(self, verbose=False, debug_rl=False):
        """Play a single Hearts game with the specified bot type.
        
        Args:
            verbose: Print general game progress
            debug_rl: Print detailed RL agent debugging information
        """
        # Pure Gym environment approach - no separate OpenSpiel state needed
        
        # Create appropriate Gym environment for RL agent based on bot type
        if self.bot_type == "conservative":
            gym_env = HeartsGymEnvConservative()
        elif self.bot_type == "self":
            gym_env = HeartsGymEnvSelfPlay()
        else:
            raise ValueError(f"Invalid bot type or bot type not specified: {self.bot_type}")
            
        gym_obs, _ = gym_env.reset()
        # Reset agent states for all players
        for player_id in range(4):
            self.agent_states[player_id] = self.agents[player_id].get_policy().get_initial_state()
        
        # Track game data
        player_scores = [0, 0, 0, 0]
        actions_taken = []
        rl_decisions = []  # Track RL agent's decision process
        played_cards = []  # Track all cards played during the game
        
        # Determine opponent strategies based on bot type
        if self.bot_type == "conservative":
            opponent_strategies = ["conservative", "conservative", "conservative"]
        elif self.bot_type == "self":
            opponent_strategies = ["self_play", "self_play", "self_play"]
        else:  # random
            opponent_strategies = ["pure_random", "pure_random", "pure_random"]
        
        if verbose:
            print("\n🃏 Starting new Hearts game:")
            print(f"  Player 0: Trained RL Agent (Main)")
            if self.bot_type == "self":
                print(f"  Player 1: RL Agent (Self-Play)")
                print(f"  Player 2: RL Agent (Self-Play)")
                print(f"  Player 3: RL Agent (Self-Play)")
                
                # Show checkpoint info for self-play
                player_checkpoints = {
                    1: os.getenv("PLAYER_1_CHECKPOINT"),
                    2: os.getenv("PLAYER_2_CHECKPOINT"), 
                    3: os.getenv("PLAYER_3_CHECKPOINT")
                }
                for pid in [1, 2, 3]:
                    if player_checkpoints[pid]:
                        checkpoint_name = os.path.basename(player_checkpoints[pid])
                        print(f"    Player {pid} Checkpoint: {checkpoint_name}")
                    else:
                        print(f"    Player {pid} Checkpoint: Same as Player 0")
            else:
                print(f"  Player 1: {opponent_strategies[0].replace('_', ' ').title()} Bot")
                print(f"  Player 2: {opponent_strategies[1].replace('_', ' ').title()} Bot") 
                print(f"  Player 3: {opponent_strategies[2].replace('_', ' ').title()} Bot")
        
        if debug_rl:
            print(f"\n🔍 RL DEBUG MODE ENABLED - Detailed RL agent analysis")
            print(f"   Bot Type: {self.bot_type}")
            print(f"   Environment: {type(gym_env).__name__}")
        
        game_turn = 0
        done = False
        total_reward = 0
        
        # Pure gym environment approach - let the gym handle all game logic
        while not done:
            # In self-play mode, ALL players are controlled by the RL agent
            # In other modes, only Player 0 is controlled by the RL agent, others are handled internally
            
            # Get legal actions from the gym environment
            legal_actions = []
            current_player = -1
            
            if hasattr(gym_env, '_last_timestep') and gym_env._last_timestep and not gym_env._last_timestep.last():
                ts = gym_env._last_timestep
                current_player = ts.observations.get("current_player", -1)
                if current_player >= 0:
                    legal_actions = ts.observations.get("legal_actions", [None])[current_player] or []
            
            # Determine if we should act based on bot type and current player
            should_act = False
            if self.bot_type == "self":
                # In self-play, we control all players
                should_act = len(legal_actions) > 0
            else:
                # In other modes, we only control player 0
                should_act = current_player == 0 and len(legal_actions) > 0
            
            if should_act:
                self.total_rl_moves_count += 1
                
                if debug_rl:
                    print(f"\n🔍 RL AGENT DECISION #{self.total_rl_moves_count}:")
                    
                    # Try to determine current trick cards from game state
                    current_trick_cards = []
                    
                    # Attempt to get trick information from the environment
                    if hasattr(gym_env, '_last_timestep') and gym_env._last_timestep:
                        ts = gym_env._last_timestep
                        
                        # Calculate current trick cards based on total moves played
                        total_moves = len(played_cards)
                        if total_moves > 0:
                            # In Hearts, every 4 moves completes a trick
                            position_in_trick = total_moves % 4
                            
                            if position_in_trick == 0:
                                # We're at the start of a new trick
                                current_trick_cards = []
                            else:
                                # We're in the middle of a trick, show the cards played so far
                                current_trick_cards = played_cards[-(position_in_trick):]
                    
                    # Show cards played in current trick
                    if current_trick_cards:
                        print(f"   🃏 {format_card_list(current_trick_cards, 'Current Trick Cards')}")
                    else:
                        print(f"   🃏 Current Trick Cards: None (start of trick)")
                    
                    # Show remaining point cards
                    remaining_points = get_remaining_point_cards(played_cards)
                    HEARTS_CARDS = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]
                    hearts_remaining = [card for card in remaining_points if card in HEARTS_CARDS]
                    queen_remaining = [card for card in remaining_points if card == 43]
                    
                    if hearts_remaining:
                        print(f"   ❤️  Remaining Hearts: {', '.join([show_card(card) for card in hearts_remaining])}")
                    else:
                        print(f"   ❤️  Remaining Hearts: None")
                        
                    if queen_remaining:
                        print(f"   ♠️  Queen of Spades: {show_card(43)} (still in play)")
                    else:
                        print(f"   ♠️  Queen of Spades: Already played")
                    
                    print(f"   🎯 Legal Actions: {len(legal_actions)} options")
                    for i, action in enumerate(legal_actions):
                        card_display = show_card(action) if action < 52 else f"Action-{action}"
                        print(f"      [{i}] Action {action}: {card_display}")
                
                try:
                    # Use the appropriate agent for the current player
                    current_agent = self.agents[current_player] if current_player in self.agents else self.agents[0]
                    current_agent_state = self.agent_states[current_player] if current_player in self.agent_states else self.agent_states[0]
                    
                    # Use gym environment's built-in action masking
                    action, new_agent_state, info_dict = current_agent.get_policy().compute_single_action(
                        gym_obs, state=current_agent_state, explore=False
                    )
                    
                    # Update the agent state for this player
                    if current_player in self.agent_states:
                        self.agent_states[current_player] = new_agent_state
                    
                    if debug_rl:
                        print(f"   🧠 RL Policy Output:")
                        print(f"      Selected Action: {action}")
                        if action < 52:
                            print(f"      Selected Card: {show_card(action)}")
                        if info_dict and 'action_logp' in info_dict:
                            print(f"      Action Log Probability: {info_dict['action_logp']:.4f}")
                        if info_dict and 'vf_preds' in info_dict:
                            print(f"      Value Function Prediction: {info_dict['vf_preds']:.4f}")
                        if info_dict and 'action_dist_inputs' in info_dict:
                            print(f"      Action Distribution Input:")
                            for legal_action in legal_actions:
                                print(f"      {show_card(legal_action)}: {info_dict['action_dist_inputs'][legal_action]:.4f}")
                    
                    # Verify action is legal (should never fail with proper action masking)
                    if action in legal_actions:
                        chosen_action = action
                        if debug_rl:
                            print(f"   ✅ Action {action} ({show_card(action) if action < 52 else f'Action-{action}'}) is LEGAL")
                    else:
                        # This should not happen with proper action masking
                        self.invalid_moves_count += 1
                        chosen_action = random.choice(legal_actions)
                        print(f"🚨 UNEXPECTED: RL agent suggested illegal action {action}, using {chosen_action}")
                        if debug_rl:
                            print(f"   ❌ Action {action} is ILLEGAL! Using random fallback: {chosen_action} ({show_card(chosen_action)})")

                except Exception as e:
                    self.invalid_moves_count += 1
                    print(f"❌ Error with RL agent: {e}, using random action")
                    chosen_action = random.choice(legal_actions) if legal_actions else 0
                    if debug_rl:
                        print(f"   💥 RL Policy Error: {e}")
                        print(f"   🎲 Using random fallback: {chosen_action} ({show_card(chosen_action)})")
                
                # Store decision details for analysis
                decision_info = {
                    'turn': game_turn,
                    'legal_actions': legal_actions.copy(),
                    'chosen_action': chosen_action,
                    'was_legal': chosen_action in legal_actions,
                    'total_rl_moves': self.total_rl_moves_count
                }
                rl_decisions.append(decision_info)
                
                if verbose and game_turn % 20 == 0:
                    print(f"  🤖 RL Agent plays action {chosen_action} ({show_card(chosen_action)})")
                elif debug_rl:
                    print(f"   🎯 FINAL DECISION: Playing {chosen_action} ({show_card(chosen_action)})")
                
                # Apply action to gym environment
                try:
                    gym_obs, reward, done, truncated, info = gym_env.step(chosen_action)
                    total_reward += reward
                    actions_taken.append((0, chosen_action))  # Player 0 is RL agent
                    
                    # Get the complete game history from the gym environment
                    # This includes moves from ALL players, not just the RL agent
                    try:
                        if hasattr(gym_env, 'get_game_history'):
                            # Use the new game history method from the gym environment
                            played_cards = gym_env.get_game_history()
                        else:
                            # Fallback: simple tracking (for environments without game history)
                            if chosen_action not in played_cards:
                                played_cards.append(chosen_action)
                                
                    except Exception as history_error:
                        # If history tracking fails, fall back to simple tracking
                        if debug_rl:
                            print(f"   ⚠️ Game history tracking failed: {history_error}")
                        # At minimum, track the RL agent's move
                        if chosen_action not in played_cards:
                            played_cards.append(chosen_action)
                    
                    # Debug: show tracking results
                    if debug_rl:
                        print(f"   📝 Tracked cards: {len(played_cards)} total")
                        if len(played_cards) <= 10:  # Show details for early game
                            recent_cards = played_cards[-min(5, len(played_cards)):]
                            print(f"      Recent cards: {[show_card(c) for c in recent_cards]}")
                    
                except Exception as e:
                    print(f"❌ Gym environment error: {e}")
                    if debug_rl:
                        print(f"   💥 Environment Step Error: {e}")
                    done = True  # End game on error
            else:
                # No legal actions means either game is over or gym is handling other players
                # In single-agent gym environments, this typically means game is done
                done = True
            
            game_turn += 1
            
            # Safety check to prevent infinite loops
            if game_turn > 200:  # Hearts games typically last ~52 turns
                print("⚠️ Game exceeded maximum turns, ending")
                done = True
        
        # Get final scores from the gym environment
        try:
            if hasattr(gym_env, '_last_timestep') and gym_env._last_timestep and gym_env._last_timestep.last():
                # Try to get rewards from final timestep 
                ts = gym_env._last_timestep
                if hasattr(ts, 'rewards') and ts.rewards:
                    # OpenSpiel rewards are typically per-player
                    rewards = ts.rewards
                    if isinstance(rewards, (list, tuple)) and len(rewards) >= 4:
                        player_scores = list(rewards)
                    else:
                        # Single reward - assume it's for RL agent
                        player_scores = [float(rewards) if not isinstance(rewards, (list, tuple)) else rewards[0], 0, 0, 0]
                else:
                    # Use accumulated reward for RL agent
                    player_scores = [total_reward, 0, 0, 0]
            else:
                # Use accumulated reward for RL agent
                player_scores = [total_reward, 0, 0, 0]
        except Exception as e:
            # Fallback: use accumulated reward
            player_scores = [total_reward, 0, 0, 0]

        for i, player_score in enumerate(player_scores):
            player_scores[i] = 26 - player_score
        
        # Determine winner (highest reward, or lowest score in Hearts)
        winner = np.argmin(player_scores) if any(score != 0 for score in player_scores[1:]) else 0
        
        # Calculate percentage scores for all players
        EXPECTED_POINTS_PER_PLAYER = 26.0 / 4.0  # 6.5 points per player
        player_percentages = [self.calculate_percentage_score(score, EXPECTED_POINTS_PER_PLAYER) for score in player_scores]
        
        game_result = {
            'timestamp': datetime.now().isoformat(),
            'player_scores': player_scores,
            'player_percentages': player_percentages,
            'expected_points_per_player': EXPECTED_POINTS_PER_PLAYER,
            'winner': int(winner),
            'rl_agent_score': player_scores[0],
            'rl_agent_percentage': player_percentages[0],
            'rl_agent_rank': sorted(player_scores).index(player_scores[0]) + 1,
            'bot_scores': player_scores[1:],
            'bot_percentages': player_percentages[1:],
            'bot_strategies': opponent_strategies,
            'total_actions': len(actions_taken),
            'game_length': game_turn,
            'rl_decisions': len(rl_decisions),
            'debug_info': {
                'total_rl_moves': self.total_rl_moves_count,
                'rl_decision_count': len(rl_decisions)
            }
        }
        
        if debug_rl and rl_decisions:
            print(f"\n🔍 RL AGENT DECISION SUMMARY:")
            print(f"   Total RL Decisions: {len(rl_decisions)}")
            legal_decisions = sum(1 for d in rl_decisions if d['was_legal'])
            print(f"   Legal Decisions: {legal_decisions}/{len(rl_decisions)} ({100*legal_decisions/len(rl_decisions):.1f}%)")
            
            # Show first few and last few decisions
            if len(rl_decisions) > 6:
                print(f"   First 3 decisions:")
                for i, decision in enumerate(rl_decisions[:3]):
                    action = decision['chosen_action']
                    card_name = show_card(action) if action < 52 else f"Action-{action}"
                    legal_status = "✅" if decision['was_legal'] else "❌"
                    print(f"      {i+1}. Turn {decision['turn']}: {action} ({card_name}) {legal_status}")
                
                print(f"   Last 3 decisions:")
                for i, decision in enumerate(rl_decisions[-3:]):
                    action = decision['chosen_action']
                    card_name = show_card(action) if action < 52 else f"Action-{action}"
                    legal_status = "✅" if decision['was_legal'] else "❌"
                    print(f"      {len(rl_decisions)-2+i}. Turn {decision['turn']}: {action} ({card_name}) {legal_status}")
            else:
                print(f"   All decisions:")
                for i, decision in enumerate(rl_decisions):
                    action = decision['chosen_action']
                    card_name = show_card(action) if action < 52 else f"Action-{action}"
                    legal_status = "✅" if decision['was_legal'] else "❌"
                    print(f"      {i+1}. Turn {decision['turn']}: {action} ({card_name}) {legal_status}")
        
        if verbose:
            print(f"\n📊 Game Results:")
            print(f"  Final Scores: {player_scores}")
            print(f"  Percentage Scores: {[f'{p:.1f}%' for p in player_percentages]}")
            winner_type = "RL Agent" if winner == 0 else (
                "RL Agent (Self-Play)" if self.bot_type == "self" else f"{opponent_strategies[winner-1]} Bot"
            )
            print(f"  Winner: Player {winner} ({winner_type})")
            print(f"  RL Agent Rank: {game_result['rl_agent_rank']}/4")
            print(f"  RL Agent Percentage: {player_percentages[0]:.1f}% (expected: 100.0%)")
            print(f"  Game Length: {game_turn} actions")
            if debug_rl:
                print(f"  RL Decisions Made: {len(rl_decisions)}")
                print(f"  Total Reward: {total_reward:.2f}")
        
        return game_result
    
    def run_simulation(self, num_games, verbose_frequency, debug_rl, debug_frequency):
        """Run multiple games with the specified bot type.
        
        Args:
            num_games: Number of games to play
            verbose_frequency: How often to print verbose game info
            debug_rl: Enable detailed RL debugging
            debug_frequency: How often to enable RL debugging (if None, uses verbose_frequency)
        """
        print(f"\n🎮 Starting simulation: {num_games} games")
        print(f"Agent Type: Trained RL Agent")
        if self.bot_type == "self":
            print(f"Opponent Type: Self-Play (All RL Agents)")
        elif self.bot_type == "conservative":
            print(f"Opponent Type: Conservative Strategy Bots")
        else:
            print(f"Opponent Type: Random Bots")
        
        if debug_rl:
            debug_freq = debug_frequency if debug_frequency is not None else verbose_frequency
            print(f"🔍 RL Debug Mode: Enabled (every {debug_freq} games)")
        
        print("=" * 50)
        
        self.game_results = []
        
        for game_num in range(num_games):
            if (game_num + 1) % verbose_frequency == 0 or game_num == 0:
                print(f"Playing game {game_num + 1}/{num_games}...")
                verbose = True
            else:
                verbose = False
            
            # Enable RL debugging based on frequency
            current_debug_rl = False
            if debug_rl:
                debug_freq = debug_frequency if debug_frequency is not None else verbose_frequency
                if (game_num + 1) % debug_freq == 0 or game_num == 0:
                    current_debug_rl = True
            
            result = self.play_single_game(verbose=verbose, debug_rl=current_debug_rl)
            
            if result:
                self.game_results.append(result)
                
                # Show quick progress update
                if (game_num + 1) % verbose_frequency == 0:
                    recent_scores = [r['rl_agent_score'] for r in self.game_results[-verbose_frequency:]]
                    avg_score = np.mean(recent_scores)
                    
                    # Calculate cumulative percentage so far
                    total_points_so_far = sum(r['rl_agent_score'] for r in self.game_results)
                    games_so_far = len(self.game_results)
                    cumulative_percentage_so_far = self.calculate_cumulative_percentage_score(total_points_so_far, games_so_far, 6.5)
                    
                    print(f"  📈 RL Agent avg score (last {len(recent_scores)} games): {avg_score:.1f}")
                    print(f"  🎯 Cumulative percentage (through {games_so_far} games): {cumulative_percentage_so_far:.1f}%")
                    
                    if debug_rl:
                        # Show debugging summary
                        recent_decisions = [r.get('rl_decisions', 0) for r in self.game_results[-verbose_frequency:]]
                        avg_decisions = np.mean(recent_decisions) if recent_decisions else 0
                        print(f"  🔍 RL Decisions per game (last {len(recent_decisions)}): {avg_decisions:.1f}")
        
        print(f"\n✅ Simulation complete: {len(self.game_results)} games played")
        return self.analyze_results()
    
    def calculate_cumulative_percentage_score(self, total_points_earned, total_games_played, expected_points_per_game=6.5):
        """Calculate cumulative percentage score across all games: (total_points / total_expected_points) * 100
        
        Args:
            total_points_earned: Total points earned across all games
            total_games_played: Number of games played
            expected_points_per_game: Expected points per game (default: 6.5)
        
        Returns:
            Cumulative percentage score as a float
        """
        total_expected_points = total_games_played * expected_points_per_game
        if total_expected_points == 0:
            return 0.0
        return (total_points_earned / total_expected_points) * 100
    
    def calculate_percentage_score(self, points_earned, expected_points_per_player=6.5):
        """Calculate percentage score for a single game: (points_earned / expected_points) * 100
        
        Args:
            points_earned: Points earned by the player in the game
            expected_points_per_player: Expected points per player (default: 26 total points / 4 players = 6.5)
        
        Returns:
            Percentage score as a float
        """
        if expected_points_per_player == 0:
            return 0.0
        return (points_earned / expected_points_per_player) * 100
    
    def analyze_results(self):
        """Analyze the simulation results."""
        if not self.game_results:
            print("❌ No game results to analyze")
            return None
        
        print("\n" + "="*60)
        if self.bot_type == "self":
            print(f"🏆 SIMULATION ANALYSIS: TRAINED RL AGENT vs 3 SELF-PLAY AGENTS")
        elif self.bot_type == "conservative":
            print(f"🏆 SIMULATION ANALYSIS: TRAINED RL AGENT vs 3 CONSERVATIVE BOTS")
        else:
            print(f"🏆 SIMULATION ANALYSIS: TRAINED RL AGENT vs 3 RANDOM BOTS")
        print("="*60)
        
        # Calculate expected points per player (26 total points / 4 players)
        EXPECTED_POINTS_PER_PLAYER = 26.0 / 4.0  # 6.5 points per player
        
        # Overall RL agent performance
        rl_scores = [r['rl_agent_score'] for r in self.game_results]
        rl_ranks = [r['rl_agent_rank'] for r in self.game_results]
        rl_wins = sum(1 for r in self.game_results if r['winner'] == 0)
        
        # Calculate cumulative percentage score for RL agent across all games
        total_rl_points = sum(rl_scores)
        total_games = len(self.game_results)
        rl_cumulative_percentage = self.calculate_cumulative_percentage_score(total_rl_points, total_games, EXPECTED_POINTS_PER_PLAYER)
        
        # Keep individual game percentages for additional analysis if needed
        rl_percentages = [self.calculate_percentage_score(score, EXPECTED_POINTS_PER_PLAYER) for score in rl_scores]
        
        analysis = {
            'total_games': len(self.game_results),
            'expected_points_per_player': EXPECTED_POINTS_PER_PLAYER,
            'rl_agent': {
                'avg_score': np.mean(rl_scores),
                'std_score': np.std(rl_scores),
                'best_score': np.min(rl_scores),
                'worst_score': np.max(rl_scores),
                'avg_rank': np.mean(rl_ranks),
                'wins': rl_wins,
                'win_rate': rl_wins / len(self.game_results),
                'invalid_moves': self.invalid_moves_count,
                'total_moves': self.total_rl_moves_count,
                'invalid_move_rate': self.invalid_moves_count / self.total_rl_moves_count if self.total_rl_moves_count > 0 else 0,
                'rank_distribution': {
                    'rank_1': sum(1 for r in rl_ranks if r == 1),
                    'rank_2': sum(1 for r in rl_ranks if r == 2),
                    'rank_3': sum(1 for r in rl_ranks if r == 3),
                    'rank_4': sum(1 for r in rl_ranks if r == 4)
                },
                # New percentage-based metrics
                'cumulative_percentage': rl_cumulative_percentage,
                'total_points_earned': total_rl_points,
                'total_expected_points': total_games * EXPECTED_POINTS_PER_PLAYER,
                'percentage_scores': {
                    'avg_percentage': np.mean(rl_percentages),
                    'std_percentage': np.std(rl_percentages),
                    'best_percentage': np.min(rl_percentages),  # Lower is better in Hearts
                    'worst_percentage': np.max(rl_percentages),
                    'all_percentages': rl_percentages
                }
            },
            'bot_performance': {}
        }
        
        print(f"\n🤖 RL AGENT PERFORMANCE:")
        print(f"  Games Played: {analysis['total_games']}")
        print(f"  Average Score: {analysis['rl_agent']['avg_score']:.2f} ± {analysis['rl_agent']['std_score']:.2f}")
        print(f"  Best Score: {analysis['rl_agent']['best_score']}")
        print(f"  Worst Score: {analysis['rl_agent']['worst_score']}")
        print(f"  Average Rank: {analysis['rl_agent']['avg_rank']:.2f}/4")
        print(f"  Wins: {analysis['rl_agent']['wins']} ({analysis['rl_agent']['win_rate']*100:.1f}%)")
        print(f"  Invalid Moves: {analysis['rl_agent']['invalid_moves']}/{analysis['rl_agent']['total_moves']} ({analysis['rl_agent']['invalid_move_rate']*100:.2f}%)")
        
        print(f"\n📊 CUMULATIVE PERCENTAGE SCORING (Across All {analysis['total_games']} Games):")
        print(f"  Total Points Earned: {analysis['rl_agent']['total_points_earned']}")
        print(f"  Total Expected Points: {analysis['rl_agent']['total_expected_points']:.1f}")
        print(f"  🎯 CUMULATIVE PERCENTAGE: {analysis['rl_agent']['cumulative_percentage']:.1f}%")
        print(f"  Expected Points per Game: {analysis['expected_points_per_player']:.1f}")
        print(f"")
        print(f"  Individual Game Statistics:")
        print(f"    Average Percentage per Game: {analysis['rl_agent']['percentage_scores']['avg_percentage']:.1f}%")
        print(f"    Standard Deviation: {analysis['rl_agent']['percentage_scores']['std_percentage']:.1f}%")
        print(f"    Best Single Game: {analysis['rl_agent']['percentage_scores']['best_percentage']:.1f}% (lowest is best)")
        print(f"    Worst Single Game: {analysis['rl_agent']['percentage_scores']['worst_percentage']:.1f}%")
        
        print(f"\n📊 RANK DISTRIBUTION:")
        rank_dist = analysis['rl_agent']['rank_distribution']
        for rank in range(1, 5):
            count = rank_dist[f'rank_{rank}']
            percentage = (count / analysis['total_games']) * 100
            print(f"  Rank {rank}: {count} games ({percentage:.1f}%)")
        
        # Analyze performance vs each bot type
        print(f"\n🎲 PERFORMANCE vs OPPONENTS:")
        
        if self.bot_type == "self":
            # For self-play, analyze how often each player position wins
            position_wins = {1: 0, 2: 0, 3: 0}
            for result in self.game_results:
                if result['winner'] != 0:  # RL agent didn't win
                    position_wins[result['winner']] += 1
            
            print(f"  Self-Play Analysis:")
            for pos in [1, 2, 3]:
                wins = position_wins[pos]
                win_rate = wins / len(self.game_results)
                print(f"    Player {pos} (RL Agent) wins: {wins} ({win_rate*100:.1f}%)")
        else:
            # Group results by which bot won when RL agent didn't win
            if self.bot_type == "conservative":
                bot_wins = {'conservative': 0}
                bot_avg_scores = {'conservative': []}
            else:
                bot_wins = {'pure_random': 0}
                bot_avg_scores = {'pure_random': []}
            
            for result in self.game_results:
                if result['winner'] != 0:  # RL agent didn't win
                    strategy_name = result['bot_strategies'][0]
                    bot_wins[strategy_name] += 1
                
                # Collect bot scores
                for i, strategy in enumerate(result['bot_strategies']):
                    bot_avg_scores[strategy].append(result['bot_scores'][i])
            
            for strategy, wins in bot_wins.items():
                avg_score = np.mean(bot_avg_scores[strategy])
                win_rate = wins / len(self.game_results)
                
                # Calculate cumulative percentage for bots across all games
                total_bot_points = sum(bot_avg_scores[strategy])
                bot_cumulative_percentage = self.calculate_cumulative_percentage_score(total_bot_points, total_games, EXPECTED_POINTS_PER_PLAYER)
                
                # Calculate individual game percentage scores for bots
                bot_percentages = [self.calculate_percentage_score(score, EXPECTED_POINTS_PER_PLAYER) for score in bot_avg_scores[strategy]]
                
                analysis['bot_performance'][strategy] = {
                    'wins': wins,
                    'win_rate': win_rate,
                    'avg_score': avg_score,
                    'cumulative_percentage': bot_cumulative_percentage,
                    'total_points_earned': total_bot_points,
                    'percentage_scores': {
                        'avg_percentage': np.mean(bot_percentages),
                        'std_percentage': np.std(bot_percentages),
                        'best_percentage': np.min(bot_percentages),
                        'worst_percentage': np.max(bot_percentages)
                    }
                }
                print(f"  {strategy.replace('_', ' ').title()}:")
                print(f"    Wins against RL: {wins} ({win_rate*100:.1f}%)")
                print(f"    Average Score: {avg_score:.2f}")
                print(f"    Cumulative Percentage: {bot_cumulative_percentage:.1f}%")
        
        return analysis
    

    
    def save_results(self, analysis):
        """Save detailed results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data for JSON serialization
        save_data = {
            'simulation_info': {
                'timestamp': timestamp,
                'total_games': len(self.game_results),
            },
            'analysis': analysis,
            'individual_games': self.game_results[:10]  # Save first 10 games as examples
        }
        
        # Convert numpy types to regular Python types for JSON
        def convert_numpy(obj):
            if isinstance(obj, np.number):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        save_data = convert_numpy(save_data)
        
        filename = f'rl_vs_{self.bot_type}_bots_results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"💾 Detailed results saved to '{filename}'")
        return filename


def find_ppo_checkpoint():
    """Automatically find the most recent PPO checkpoint directory."""
    print("🔍 Searching for trained model checkpoints...")
    
    # Look for PPO directories in OpenSpiel-Hearts project structure
    home_ray = os.path.expanduser("~/ray_results")
    search_dirs = [
        ".",  # Project root
        "hearts_phase1_basic",
        "hearts_phase2_enhanced",
        "ray_results",  # Local ray_results under project
        home_ray if os.path.exists(home_ray) else None,  # Global ray_results in home
    ]
    search_dirs = [d for d in search_dirs if d and os.path.exists(d)]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        print(f"  Searching in: {search_dir}")
        
        # Collect all candidate checkpoint paths recursively and pick newest by mtime
        candidates = []
        for root, dirs, files in os.walk(search_dir):
            # Candidate if a directory contains rllib_checkpoint.json
            if "rllib_checkpoint.json" in files:
                # Prefer the directory that contains rllib_checkpoint.json
                candidates.append(root)
            # Or if directory name looks like checkpoint_*
            if os.path.basename(root).startswith("checkpoint_"):
                if any(f.startswith("rllib_checkpoint.json") for f in files):
                    candidates.append(root)

        if candidates:
            # Sort by modification time, newest first
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            chosen = candidates[0]
            print(f"✓ Valid checkpoint found: {chosen}")
            return chosen
    
    return None


def main():
    """Main function to run the RL vs Bots simulation."""
    # Get bot type from environment variable
    bot_type = os.getenv("BOT_TYPE")
    # Check for debug mode - convert string to boolean properly
    debug_mode_str = os.getenv("DEBUG_RL", "false").lower().strip()
    debug_mode = debug_mode_str in ("true", "1", "yes", "on")
    debug_frequency = int(os.getenv("DEBUG_FREQUENCY", "1"))
    num_games = int(os.getenv("NUM_GAMES", "600"))
    
    bot_type_names = {
        "conservative": "Conservative Bots", 
        "self": "Self-Play Agents"
    }
    
    print(f"🃏 Hearts: RL Agent vs 3 {bot_type_names.get(bot_type, 'Random Bots')} Simulation")
    print(f"Bot Type: {bot_type.upper()} (set via BOT_TYPE environment variable)")
    print("=" * 65)
    
    # Check if a specific checkpoint path is provided via environment variable
    model_path = os.getenv("CHECKPOINT_PATH")
    
    if model_path:
        print(f"🎯 Using specified checkpoint: {model_path}")
        # Convert to absolute path to avoid URI scheme issues
        model_path = os.path.abspath(model_path)
        print(f"✓ Using absolute path: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Specified checkpoint path does not exist: {model_path}")
            print("\n🔧 CHECKPOINT PATH TROUBLESHOOTING:")
            print("1. Verify the path is correct and accessible")
            print("2. Use absolute paths for best results")
            print("3. Ensure the checkpoint directory contains valid PPO model files")
            print("4. Check that rllib_checkpoint.json exists in the directory")
            return
    else:
        # Automatically find the trained model
        print("🔍 No CHECKPOINT_PATH specified, auto-detecting most recent checkpoint...")
        model_path = find_ppo_checkpoint()
        
        if not model_path or not os.path.exists(model_path):
            print("❌ No trained model found!")
            print("\n🔧 TO TRAIN A MODEL FIRST:")
            print("1. Run the main training script (e.g., main.py) to train a PPO agent")
            print("2. Or use Ray Tune to train multiple experiments")
            print("3. Make sure the checkpoint is saved to a directory")
            print("4. Specify checkpoint path via CHECKPOINT_PATH environment variable")
            print("\n📁 EXPECTED DIRECTORY STRUCTURE:")
            print("  - For Tune: ray_results/PPO_*/PPO_hearts_env_*/")
            print("  - For direct training: checkpoint_*/")
            print("\n🔍 CURRENT DIRECTORY CONTENTS:")
            current_files = os.listdir(".")
            print(f"  {current_files}")
            return
    
    print(f"✓ Found trained model directory: {model_path}")
    
    # Convert to absolute path to avoid URI scheme issues (if not already done)
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
        print(f"✓ Using absolute path: {model_path}")
    
    # Create simulator - this will error if model can't be loaded
    try:
        simulator = RLvsBotsSimulator(model_path, bot_type)
    except Exception as e:
        print(f"❌ Failed to initialize simulator: {e}")
        print("\n🔧 POSSIBLE SOLUTIONS:")
        print("1. Verify the checkpoint directory contains valid PPO model files")
        print("2. Check if the checkpoint was created with a compatible version of Ray/RLlib")
        print("3. Ensure the environment registration matches the training setup")
        print("4. Try using an absolute path instead of relative path")
        return
    
    if debug_mode == True:
        print(f"🔍 DEBUG MODE ENABLED - Will show detailed RL analysis every {debug_frequency} games")
    
    # Run simulation
    print("Starting simulation...")
    print(f"   Bot Type: {bot_type}")
    print(f"   Num Games: {num_games}")
    print(f"   Debug Mode: {debug_mode}")
    print(f"   Debug Frequency: {debug_frequency}") 

    analysis = simulator.run_simulation(
        num_games, 
        10, 
        debug_mode,
        debug_frequency
    )
    
    if analysis:
        # Save results
        simulator.save_results(analysis)
        
        print("\n🎉 Simulation completed successfully!")
    else:
        print("❌ Simulation failed.")


# def run_debug_simulation(bot_type, num_games, debug_frequency):
#     """
#     Convenience function to run a small simulation with full RL debugging enabled.
    
#     Args:
#         bot_type: Type of bots to play against ("conservative", "self", or "random")
#         num_games: Number of games to play (default: 10 for debugging)
#         debug_frequency: How often to show debug info (default: every game)
    
#     Example usage:
#         # Debug every game against conservative bots
#         run_debug_simulation("conservative", 5, 1)
        
#         # Debug every 3rd game in self-play
#         run_debug_simulation("self", 10, 3)
#     """
#     print(f"🔍 RUNNING DEBUG SIMULATION")
#     print(f"   Bot Type: {bot_type}")
#     print(f"   Games: {num_games}")
#     print(f"   Debug Frequency: every {debug_frequency} game(s)")
#     print("=" * 50)
    
#     # Set environment variables for this run
#     os.environ["BOT_TYPE"] = bot_type
#     os.environ["DEBUG_RL"] = "true"
#     os.environ["DEBUG_FREQUENCY"] = str(debug_frequency)
#     os.environ["NUM_GAMES"] = str(num_games)
    
#     # Find model and run
#     model_path = find_ppo_checkpoint()
#     if not model_path:
#         print("❌ No trained model found! Please train a model first.")
#         return
    
#     try:
#         simulator = RLvsBotsSimulator(model_path, bot_type)
#         analysis = simulator.run_simulation(
#             num_games=num_games,
#             verbose_frequency=max(1, debug_frequency),
#             debug_rl=True,
#             debug_frequency=debug_frequency
#         )
        
#         if analysis:
#             print(f"\n🎯 QUICK SUMMARY:")
#             print(f"   RL Agent Win Rate: {analysis['rl_agent']['win_rate']*100:.1f}%")
#             print(f"   Average Score: {analysis['rl_agent']['avg_score']:.2f}")
#             print(f"   🎯 Cumulative Percentage: {analysis['rl_agent']['cumulative_percentage']:.1f}%")
#             print(f"   Average Rank: {analysis['rl_agent']['avg_rank']:.2f}/4")
#             print(f"   Invalid Moves: {analysis['rl_agent']['invalid_moves']}/{analysis['rl_agent']['total_moves']}")
            
#     except Exception as e:
#         print(f"❌ Debug simulation failed: {e}")


if __name__ == "__main__":
    main() 