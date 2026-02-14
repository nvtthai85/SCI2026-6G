"""
PPO-Based Risk-Adaptive Security Controller for CryptoTrust-6G
Implements Proximal Policy Optimization for dynamic security level selection
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config import PPO_CONFIG, REWARD_WEIGHTS, SECURITY_LEVELS
from utils import ExperimentLogger, set_random_seed

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using NumPy-based PPO.")


@dataclass
class Experience:
    """Single experience tuple"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


@dataclass
class PPOStats:
    """PPO training statistics"""
    episode_rewards: List[float]
    episode_lengths: List[int]
    policy_losses: List[float]
    value_losses: List[float]
    entropy_losses: List[float]
    security_level_distribution: Dict[str, int]


if TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """
        Actor-Critic network for PPO
        
        Architecture:
        - Shared feature extractor (2 hidden layers)
        - Actor head (policy)
        - Critic head (value function)
        """
        
        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_layers: List[int] = None
        ):
            super().__init__()
            
            hidden_layers = hidden_layers or PPO_CONFIG["hidden_layers"]
            
            # Shared feature extractor
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_layers[0]),
                nn.ReLU(),
                nn.Linear(hidden_layers[0], hidden_layers[1]),
                nn.ReLU()
            )
            
            # Actor (policy) head
            self.actor = nn.Sequential(
                nn.Linear(hidden_layers[1], action_dim),
                nn.Softmax(dim=-1)
            )
            
            # Critic (value) head
            self.critic = nn.Linear(hidden_layers[1], 1)
        
        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass
            
            Args:
                state: State tensor
            
            Returns:
                Tuple of (action probabilities, state value)
            """
            features = self.shared(state)
            action_probs = self.actor(features)
            state_value = self.critic(features)
            return action_probs, state_value
        
        def act(self, state: torch.Tensor) -> Tuple[int, float, float]:
            """
            Select action using current policy
            
            Args:
                state: Current state
            
            Returns:
                Tuple of (action, log_prob, value)
            """
            action_probs, value = self.forward(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
        
        def evaluate(
            self,
            states: torch.Tensor,
            actions: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Evaluate actions for PPO update
            
            Args:
                states: Batch of states
                actions: Batch of actions
            
            Returns:
                Tuple of (log_probs, values, entropy)
            """
            action_probs, values = self.forward(states)
            dist = Categorical(action_probs)
            
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            return log_probs, values.squeeze(), entropy


class PPOController:
    """
    PPO-based Risk-Adaptive Security Controller
    
    Dynamically selects security levels based on current threat assessment
    """
    
    def __init__(
        self,
        state_dim: int = None,
        action_dim: int = None,
        device: str = "cpu",
        logger: Optional[ExperimentLogger] = None
    ):
        self.state_dim = state_dim or PPO_CONFIG["state_dim"]
        self.action_dim = action_dim or PPO_CONFIG["action_dim"]
        self.device = device
        self.logger = logger
        
        # Initialize actor-critic network
        self.policy = ActorCritic(self.state_dim, self.action_dim).to(device)
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=PPO_CONFIG["learning_rate"]
        )
        
        # PPO hyperparameters
        self.gamma = PPO_CONFIG["gamma"]
        self.gae_lambda = PPO_CONFIG["gae_lambda"]
        self.clip_ratio = PPO_CONFIG["clip_ratio"]
        self.value_coef = PPO_CONFIG["value_coef"]
        self.entropy_coef = PPO_CONFIG["entropy_coef"]
        self.update_epochs = PPO_CONFIG["update_epochs"]
        self.batch_size = PPO_CONFIG["batch_size"]
        
        # Experience buffer
        self.buffer = []
        
        # Training statistics
        self.stats = PPOStats(
            episode_rewards=[],
            episode_lengths=[],
            policy_losses=[],
            value_losses=[],
            entropy_losses=[],
            security_level_distribution={f"L{i}": 0 for i in range(5)}
        )
        
        # Action to security level mapping
        self.action_to_level = {i: f"L{i}" for i in range(self.action_dim)}
        
        # Previous action for stability penalty
        self.prev_action = 0
    
    def log(self, message: str):
        if self.logger:
            self.logger.info(message)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, str]:
        """
        Select security level based on current state
        
        Args:
            state: Current state vector
            deterministic: If True, select action with highest probability
        
        Returns:
            Tuple of (action_index, security_level_name)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.policy(state_tensor)
        
        if deterministic:
            action = action_probs.argmax().item()
        else:
            action, log_prob, value = self.policy.act(state_tensor)
        
        security_level = self.action_to_level[action]
        self.stats.security_level_distribution[security_level] += 1
        
        return action, security_level
    
    def compute_reward(
        self,
        action: int,
        tdr: float,
        latency_ms: float,
        attack_present: bool,
        attack_detected: bool
    ) -> float:
        """
        Compute reward based on the reward function from the paper:
        R(s,a) = α·TDR(a) - β·L_crypto(a)/L_max - γ·|a - a_{t-1}| - δ·𝟙[attack ∧ level < L3]
        
        Args:
            action: Selected action (security level index)
            tdr: Tampering detection rate achieved
            latency_ms: Cryptographic latency in milliseconds
            attack_present: Whether an attack was present
            attack_detected: Whether the attack was detected
        
        Returns:
            Computed reward value
        """
        alpha = REWARD_WEIGHTS["alpha"]
        beta = REWARD_WEIGHTS["beta"]
        gamma = REWARD_WEIGHTS["gamma"]
        delta = REWARD_WEIGHTS["delta"]
        L_max = REWARD_WEIGHTS["L_max"]
        
        # TDR component (normalized to 0-1)
        tdr_reward = alpha * tdr
        
        # Latency penalty (normalized)
        latency_penalty = beta * (latency_ms / L_max)
        
        # Level change penalty (stability)
        level_change_penalty = gamma * abs(action - self.prev_action)
        
        # Attack under low security penalty
        attack_penalty = 0.0
        if attack_present and not attack_detected and action < 3:  # L0, L1, L2
            attack_penalty = delta
        
        reward = tdr_reward - latency_penalty - level_change_penalty - attack_penalty
        
        # Update previous action
        self.prev_action = action
        
        return reward
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Store experience in buffer"""
        self.buffer.append(Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value
        ))
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
        
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                next_val = values[t + 1] if t + 1 < len(values) else next_value
                delta = rewards[t] + self.gamma * next_val - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected experiences
        
        Returns:
            Dictionary with update statistics
        """
        if len(self.buffer) == 0:
            return {}
        
        # Extract data from buffer
        states = torch.FloatTensor([e.state for e in self.buffer]).to(self.device)
        actions = torch.LongTensor([e.action for e in self.buffer]).to(self.device)
        old_log_probs = torch.FloatTensor([e.log_prob for e in self.buffer]).to(self.device)
        rewards = [e.reward for e in self.buffer]
        values = [e.value for e in self.buffer]
        dones = [e.done for e in self.buffer]
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(self.update_epochs):
            # Get current policy outputs
            log_probs, values_pred, entropy = self.policy.evaluate(states, actions)
            
            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = self.value_coef * nn.functional.mse_loss(values_pred, returns)
            
            # Entropy loss
            entropy_loss = -self.entropy_coef * entropy.mean()
            
            # Total loss
            loss = policy_loss + value_loss + entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        # Store statistics
        avg_policy_loss = total_policy_loss / self.update_epochs
        avg_value_loss = total_value_loss / self.update_epochs
        avg_entropy_loss = total_entropy_loss / self.update_epochs
        
        self.stats.policy_losses.append(avg_policy_loss)
        self.stats.value_losses.append(avg_value_loss)
        self.stats.entropy_losses.append(avg_entropy_loss)
        
        # Clear buffer
        self.buffer = []
        
        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss
        }
    
    def get_security_level_distribution(self) -> Dict[str, float]:
        """Get normalized distribution of security level selections"""
        total = sum(self.stats.security_level_distribution.values())
        if total == 0:
            return {k: 0.0 for k in self.stats.security_level_distribution}
        return {k: v / total * 100 for k, v in self.stats.security_level_distribution.items()}
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats
        }, path)
        self.log(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.stats = checkpoint['stats']
        self.log(f"Model loaded from {path}")


# NumPy-based fallback controller
class NumpyPPOController:
    """
    Simple rule-based controller when PyTorch is not available
    Uses threat level thresholds to select security levels
    """
    
    def __init__(self, logger: Optional[ExperimentLogger] = None):
        self.logger = logger
        self.stats = {
            "security_level_distribution": {f"L{i}": 0 for i in range(5)}
        }
        self.prev_action = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, str]:
        """Select security level based on threat level in state"""
        # Assume first element of state is threat level (0-1)
        threat_level = state[0] if len(state) > 0 else 0.5
        
        # Map threat level to security level
        if threat_level < 0.1:
            action = 0  # L0
        elif threat_level < 0.3:
            action = 1  # L1
        elif threat_level < 0.6:
            action = 2  # L2
        elif threat_level < 0.8:
            action = 3  # L3
        else:
            action = 4  # L4
        
        security_level = f"L{action}"
        self.stats["security_level_distribution"][security_level] += 1
        
        return action, security_level
    
    def compute_reward(
        self,
        action: int,
        tdr: float,
        latency_ms: float,
        attack_present: bool,
        attack_detected: bool
    ) -> float:
        """Compute reward"""
        alpha = REWARD_WEIGHTS["alpha"]
        beta = REWARD_WEIGHTS["beta"]
        gamma = REWARD_WEIGHTS["gamma"]
        delta = REWARD_WEIGHTS["delta"]
        L_max = REWARD_WEIGHTS["L_max"]
        
        reward = alpha * tdr - beta * (latency_ms / L_max) - gamma * abs(action - self.prev_action)
        
        if attack_present and not attack_detected and action < 3:
            reward -= delta
        
        self.prev_action = action
        return reward
    
    def get_security_level_distribution(self) -> Dict[str, float]:
        total = sum(self.stats["security_level_distribution"].values())
        if total == 0:
            return {k: 0.0 for k in self.stats["security_level_distribution"]}
        return {k: v / total * 100 for k, v in self.stats["security_level_distribution"].items()}


def create_ppo_controller(device: str = "cpu", logger: Optional[ExperimentLogger] = None):
    """Create appropriate PPO controller based on available libraries"""
    if TORCH_AVAILABLE:
        return PPOController(device=device, logger=logger)
    else:
        return NumpyPPOController(logger=logger)


class SecurityEnvironment:
    """
    Simulated environment for training PPO controller
    Generates states with varying threat levels and evaluates security decisions
    """
    
    def __init__(self, threat_distribution: Dict[str, float] = None):
        """
        Args:
            threat_distribution: Distribution of time at each threat level
                                 Default matches paper: L0:42.5%, L1:35.8%, etc.
        """
        # Default distribution from paper
        self.threat_distribution = threat_distribution or {
            "L0": 0.425,  # 42.5% - Very low risk
            "L1": 0.358,  # 35.8% - Low risk  
            "L2": 0.134,  # 13.4% - Medium risk
            "L3": 0.058,  # 5.8% - High risk
            "L4": 0.025   # 2.5% - Very high risk (attack)
        }
        
        self.state_dim = PPO_CONFIG["state_dim"]
        self.current_threat_level = 0.0
        self.steps = 0
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state"""
        self.steps = 0
        self.current_threat_level = self._sample_threat_level()
        return self._get_state()
    
    def _sample_threat_level(self) -> float:
        """Sample threat level based on distribution"""
        rand = np.random.random()
        cumsum = 0
        for level, prob in self.threat_distribution.items():
            cumsum += prob
            if rand < cumsum:
                level_idx = int(level[1])  # Extract number from "L0", "L1", etc.
                # Map to threat level range
                threat_ranges = {
                    0: (0.0, 0.1),
                    1: (0.1, 0.3),
                    2: (0.3, 0.6),
                    3: (0.6, 0.8),
                    4: (0.8, 1.0)
                }
                low, high = threat_ranges[level_idx]
                return np.random.uniform(low, high)
        return 0.5
    
    def _get_state(self) -> np.ndarray:
        """Generate state vector"""
        state = np.zeros(self.state_dim)
        
        # State components:
        # [0] Threat level estimate (0-1)
        # [1] Packet loss rate
        # [2] Latency jitter variance
        # [3] Failed auth attempts
        # [4] Current security level (normalized)
        # [5] Time since last level change
        # [6] Bandwidth utilization
        # [7] Queue length
        # [8-11] Entropy-based traffic features
        
        state[0] = self.current_threat_level
        state[1] = np.random.exponential(0.01)  # Packet loss
        state[2] = np.random.exponential(0.1)   # Jitter
        state[3] = np.random.poisson(0.5) if self.current_threat_level > 0.5 else 0
        state[4] = 0  # Will be updated based on action
        state[5] = self.steps / 100  # Normalized time
        state[6] = np.random.uniform(0.2, 0.8)  # Bandwidth
        state[7] = np.random.exponential(5)     # Queue
        state[8:12] = np.random.normal(0, 1, 4)  # Entropy features
        
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return results
        
        Args:
            action: Security level to apply (0-4)
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.steps += 1
        
        # Determine if attack is present based on threat level
        attack_present = self.current_threat_level > 0.5
        
        # Get expected latency for this security level
        level_key = f"L{action}"
        latency_ms = SECURITY_LEVELS[level_key]["latency_ms"]
        
        # Add some noise to latency
        latency_ms *= np.random.uniform(0.9, 1.1)
        
        # Determine TDR based on security level
        base_tdr = {0: 0.20, 1: 0.785, 2: 0.962, 3: 0.975, 4: 0.985}
        tdr = base_tdr[action]
        
        # Determine if attack was detected
        attack_detected = attack_present and (np.random.random() < tdr)
        
        # Sample new threat level
        self.current_threat_level = self._sample_threat_level()
        
        # Get next state
        next_state = self._get_state()
        next_state[4] = action / 4  # Normalized current level
        
        # Episode done after certain steps
        done = self.steps >= 1000
        
        info = {
            "attack_present": attack_present,
            "attack_detected": attack_detected,
            "latency_ms": latency_ms,
            "tdr": tdr,
            "threat_level": self.current_threat_level
        }
        
        return next_state, tdr, latency_ms, attack_present, attack_detected, done, info


if __name__ == "__main__":
    # Test PPO controller
    from utils import ExperimentLogger
    
    logger = ExperimentLogger("ppo_test", "ppo_controller")
    
    logger.phase_start("PPO Controller Test")
    
    # Create controller and environment
    controller = create_ppo_controller(device="cpu", logger=logger)
    env = SecurityEnvironment()
    
    # Run test episodes
    num_episodes = 100
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action, level = controller.select_action(state)
            next_state, tdr, latency, attack, detected, done, info = env.step(action)
            
            reward = controller.compute_reward(action, tdr, latency, attack, detected)
            episode_reward += reward
            
            if TORCH_AVAILABLE:
                # Get log_prob and value for PPO
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                _, log_prob, value = controller.policy.act(state_tensor)
                controller.store_experience(state, action, reward, next_state, done, log_prob, value)
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Update policy every few episodes
        if TORCH_AVAILABLE and (episode + 1) % 10 == 0:
            update_stats = controller.update()
            if update_stats:
                logger.info(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
                           f"Policy Loss={update_stats['policy_loss']:.4f}")
    
    # Print final statistics
    logger.info(f"\nAverage Episode Reward: {np.mean(episode_rewards):.2f}")
    logger.info(f"Security Level Distribution:")
    for level, pct in controller.get_security_level_distribution().items():
        logger.info(f"  {level}: {pct:.1f}%")
    
    logger.phase_end("PPO Controller Test")
    logger.finalize()
    
    print("\nPPO Controller test completed!")
