import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser(description="Energy Boids Simulation")
parser.add_argument('--n_episodes', type=int, default=100, help='Number of training episodes')
parser.add_argument('--episode_length', type=int, default=300, help='Length of each episode')
parser.add_argument('--method',type=int,default=0)
args = parser.parse_args()
seed = 18
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
class Categorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p):
        if p.dim() == 1:
            p = p.unsqueeze(0)        
        action_idx = torch.multinomial(p, num_samples=1)        
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(1, action_idx, 1.0)
        ctx.save_for_backward(one_hot, p)

        return one_hot

    @staticmethod
    def backward(ctx, grad_output):
        one_hot, p = ctx.saved_tensors
        p_safe = p.clamp(min=1e-6, max=1.0 - 1e-6)
        weights = (one_hot - p) / (p_safe * (1 - p_safe))
        
        return grad_output.expand_as(p) * weights
    
class StochasticCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, uncertainty_in=None):
        if p.dim() == 1:
            p = p.unsqueeze(0)
        
        result = torch.multinomial(p, num_samples=1)
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        
        var_chosen = (1.0 - p) / p.clamp(min=1e-6)
        var_not_chosen = p / (1.0 - p).clamp(min=1e-6)
        op_variance = one_hot * var_chosen + (1 - one_hot) * var_not_chosen
        uncertainty_out = uncertainty_in + op_variance if uncertainty_in is not None else op_variance
        
        ctx.save_for_backward(result, p, uncertainty_out)
        return one_hot, uncertainty_out
    
    @staticmethod
    def backward(ctx, grad_output, grad_uncertainty=None):
        result, p, uncertainty = ctx.saved_tensors
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        
        w_chosen = (1.0 / p) / 2
        w_non_chosen = (1.0 / (1.0 - p)) / 2
        confidence = 1.0 / (1.0 + uncertainty.clamp(min=1e-6))
        adjusted_ws = (one_hot * w_chosen + (1 - one_hot) * w_non_chosen) * confidence
        
        grad_p = grad_output.sum(-1, keepdim=True).expand_as(p) * adjusted_ws
        return grad_p, None
    

class EnergyBoids(nn.Module):
    """
    Differentiable Energy Boids simulation with discrete movement via Gumbel Softmax
    """
    
    def __init__(self, 
                 grid_size: int = 50,
                 n_agents: int = 50,
                 initial_energy: float = 100.0,
                 move_cost: float = 1.0,
                 death_threshold: float = 10.0,
                 sharing_radius: float = 2.0,
                 sharing_threshold: float = 10.0,
                 device: str =  'cpu'):
        super().__init__()
        
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.initial_energy = initial_energy
        self.move_cost = move_cost
        self.death_threshold = death_threshold
        self.sharing_radius = sharing_radius
        self.sharing_threshold = sharing_threshold
        self.device = device
        
        self.energy_transfer_rate = nn.Parameter(torch.tensor(0.1)) 
        self.movement_bias = nn.Parameter(torch.randn(5)) 
        self.energy_movement_scale = nn.Parameter(torch.tensor(0.01))  
        
        self.actions = torch.tensor([
            [0, 0],   # stay
            [0, -1],  # up
            [0, 1],   # down
            [-1, 0],  # left
            [1, 0]    # right
        ], dtype=torch.float32, device=device)
        
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset environment and return initial state"""
        self.positions = torch.randint(0, self.grid_size, (self.n_agents, 2), 
                                     dtype=torch.float32, device=self.device)
        
        self.energies = torch.full((self.n_agents,), self.initial_energy, 
                                 dtype=torch.float32, device=self.device)
        
        self.alive = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)
        
        self.visited_positions = set()
        self._update_coverage()
        
        return self.get_state()
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get current state"""
        return {
            'positions': self.positions.clone(),
            'energies': self.energies.clone(),
            'alive': self.alive.clone(),
            'coverage': torch.tensor(len(self.visited_positions), dtype=torch.float32, device=self.device)
        }
    
    def _get_movement_logits(self) -> torch.Tensor:
        """Compute movement logits based on agent energy levels"""
        energy_factor = self.energies.unsqueeze(1) * self.energy_movement_scale
        
        logits = self.movement_bias.unsqueeze(0).repeat(self.n_agents, 1)
        
        logits[:, 0] -= energy_factor.squeeze()
        logits[:, 1:] += energy_factor
        
        logits[~self.alive] = torch.tensor([-1000., -1000., -1000., -1000., -1000.], device=self.device)
        logits[~self.alive, 0] = 0.  # Force dead agents to stay
        
        return logits
    
    def _sample_actions(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample actions using Gumbel Softmax for differentiability"""
        method = args.method
        probs = F.softmax(logits, dim=-1)
        if(method == 0): action_probs = F.gumbel_softmax(probs, tau=temperature, hard=True)
        elif(method == 1): action_probs = Categorical.apply(probs)
        elif(method == 2): action_probs, _ = StochasticCategorical.apply(probs)
        # print(action_probs)
        return action_probs
    
    def _apply_movement(self, action_probs: torch.Tensor) -> torch.Tensor:
        """Apply movement based on action probabilities"""
        movement_delta = torch.sum(action_probs.unsqueeze(2) * self.actions.unsqueeze(0), dim=1)
        
        new_positions = self.positions + movement_delta
        
        new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)
        
        move_prob = 1.0 - action_probs[:, 0]  # probability of any move action
        energy_cost = move_prob * self.move_cost
        
        return new_positions, energy_cost
    
    def _compute_distances(self) -> torch.Tensor:
        """Compute pairwise distances between agents"""
        pos_diff = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)
        distances = torch.norm(pos_diff, dim=2)
        return distances
    
    def _energy_sharing(self) -> torch.Tensor:
        """Perform energy sharing between nearby agents"""
        distances = self._compute_distances()
        
        neighbor_mask = (distances <= self.sharing_radius) & (distances > 0)
        
        alive_mask = self.alive.unsqueeze(0) & self.alive.unsqueeze(1)
        neighbor_mask = neighbor_mask & alive_mask
        
        energy_diff = self.energies.unsqueeze(1) - self.energies.unsqueeze(0)
        
        sharing_mask = neighbor_mask & (energy_diff > self.sharing_threshold)
        
        transfer_rate = torch.sigmoid(self.energy_transfer_rate)  # Ensure [0,1]
        energy_to_transfer = self.energies.unsqueeze(1) * transfer_rate
        
        num_receivers = sharing_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        energy_per_receiver = energy_to_transfer / num_receivers
        
        energy_given = (sharing_mask * energy_per_receiver).sum(dim=1)
        energy_received = (sharing_mask * energy_per_receiver).sum(dim=0)
        
        net_energy_change = energy_received - energy_given
        
        return net_energy_change
    
    def _update_coverage(self):
        """Update visited positions for coverage calculation"""
        for i in range(self.n_agents):
            if self.alive[i]:
                pos_tuple = (int(self.positions[i, 0].item()), int(self.positions[i, 1].item()))
                self.visited_positions.add(pos_tuple)
    
    def step(self, temperature: float = 1.0) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Execute one simulation step"""
        logits = self._get_movement_logits()
        action_probs = self._sample_actions(logits, temperature)
        
        new_positions, energy_cost = self._apply_movement(action_probs)
        self.positions = new_positions
        
        self.energies = self.energies - energy_cost
        
        energy_change = self._energy_sharing()
        self.energies = self.energies + energy_change
        
        self.alive = self.alive & (self.energies > self.death_threshold)
        
        self._update_coverage()
        
        reward = self._compute_reward()
        
        return self.get_state(), reward
    
    def _compute_reward(self) -> torch.Tensor:
        """Compute reward based on coverage, survival, and energy"""
        coverage_reward = len(self.visited_positions) / (self.grid_size ** 2)
        
        survival_reward = self.alive.sum().float() / self.n_agents
        
        if self.alive.sum() > 0:
            energy_reward = self.energies[self.alive].mean() / self.initial_energy
        else:
            energy_reward = torch.tensor(0.0, device=self.device)
        
        total_reward = (0.4 * coverage_reward + 
                       0.4 * survival_reward + 
                       0.2 * energy_reward)
        
        return total_reward
    
    def simulate_episode(self, episode_length: int, temperature: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """Simulate a complete episode"""
        self.reset()
        total_reward = 0.0
        
        states_history = []
        rewards_history = []
        
        for step in range(episode_length):
            state, reward = self.step(temperature)
            total_reward = total_reward + reward
            
            states_history.append(state)
            rewards_history.append(reward)
            
            if not self.alive.any():
                print(f"All agents dead at step {step}. Ending episode early.")
                break
        
        return total_reward, {
            'states': states_history,
            'rewards': rewards_history,
            'final_coverage': len(self.visited_positions),
            'final_alive': self.alive.sum().item(),
            'final_avg_energy': self.energies[self.alive].mean().item() if self.alive.any() else 0.0
        }


def train_energy_boids(n_episodes: int = 100, episode_length: int = 300):
    """Training loop for Energy Boids"""
    
    # Initialize environment
    env = EnergyBoids(grid_size=50, n_agents=50)
    
    # Print initial configuration
    print("=== INITIAL CONFIGURATION ===")
    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"Number of agents: {env.n_agents}")
    print(f"Initial energy: {env.initial_energy}")
    print(f"Move cost: {env.move_cost}")
    print(f"Death threshold: {env.death_threshold}")
    print(f"Sharing radius: {env.sharing_radius}")
    print(f"Sharing threshold: {env.sharing_threshold}")
    print(f"Device: {env.device}")
    
    print("\n=== INITIAL PARAMETERS ===")
    print(f"Energy transfer rate (raw): {env.energy_transfer_rate.item():.6f}")
    print(f"Energy transfer rate (sigmoid): {torch.sigmoid(env.energy_transfer_rate).item():.6f}")
    print(f"Movement bias: {env.movement_bias.detach().cpu().numpy()}")
    print(f"Energy movement scale: {env.energy_movement_scale.item():.6f}")
    print("=" * 50)
    
    # Optimizer
    optimizer = torch.optim.Adam(env.parameters(), lr=0.01)
    
    # Training loop
    losses = []
    rewards = []
    
    for episode in range(n_episodes):
        print(f"\n--- Episode {episode} ---")
        optimizer.zero_grad()
        
        # Simulate episode
        total_reward, info = env.simulate_episode(episode_length, temperature=1.0)
        
        loss = -total_reward
        
        loss.backward()
        
        print(f"Before optimizer step:")
        print(f"  Transfer rate grad: {env.energy_transfer_rate.grad.item() if env.energy_transfer_rate.grad is not None else 'None':.6f}")
        print(f"  Movement bias grad norm: {env.movement_bias.grad.norm().item() if env.movement_bias.grad is not None else 'None':.6f}")
        print(f"  Energy scale grad: {env.energy_movement_scale.grad.item() if env.energy_movement_scale.grad is not None else 'None':.6f}")
        
        optimizer.step()
        
        # Log progress
        losses.append(loss.item())
        rewards.append(total_reward.item())
        
        print(f"Episode {episode} Results:")
        print(f"  Total Reward: {total_reward.item():.4f}")
        print(f"  Final Coverage: {info['final_coverage']}")
        print(f"  Final Alive: {info['final_alive']}")
        print(f"  Final Avg Energy: {info['final_avg_energy']:.2f}")
        print(f"Updated transfer rate (raw): {env.energy_transfer_rate.item():.6f}")
        print(f"  Updated transfer rate (sigmoid): {torch.sigmoid(env.energy_transfer_rate).item():.6f}")

    return env, losses, rewards


if __name__ == "__main__":
    # Train the model
    trained_env, losses, rewards = train_energy_boids(n_episodes=10)  # Reduced for debugging
    
    print(f"\n=== FINAL LEARNED PARAMETERS ===")
    print(f"Energy transfer rate (raw): {trained_env.energy_transfer_rate.item():.6f}")
    print(f"Energy transfer rate: {torch.sigmoid(trained_env.energy_transfer_rate).item():.6f}")
    print(f"Movement bias: {trained_env.movement_bias.detach().cpu().numpy()}")
    print(f"Energy-movement scale: {trained_env.energy_movement_scale.item():.6f}")
    
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Initial reward: {rewards[0]:.4f}")
    print(f"Final reward: {rewards[-1]:.4f}")
    print(f"Maximum reward: {max(rewards):.4f}")
    print(f"Reward improvement: {rewards[-1] - rewards[0]:.4f}")