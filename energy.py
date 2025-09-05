import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np

class EnergyBoids(nn.Module):
    """
    Differentiable Energy Boids simulation with discrete movement via Gumbel Softmax
    """
    
    def __init__(self, 
                 grid_size: int = 20,
                 n_agents: int = 50,
                 initial_energy: float = 100.0,
                 move_cost: float = 1.0,
                 death_threshold: float = 10.0,
                 sharing_radius: float = 2.0,
                 sharing_threshold: float = 10.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.initial_energy = initial_energy
        self.move_cost = move_cost
        self.death_threshold = death_threshold
        self.sharing_radius = sharing_radius
        self.sharing_threshold = sharing_threshold
        self.device = device
        
        # Learnable parameters
        self.energy_transfer_rate = nn.Parameter(torch.tensor(0.1))  # p in [0, 1]
        self.movement_bias = nn.Parameter(torch.randn(5))  # bias for each action
        self.energy_movement_scale = nn.Parameter(torch.tensor(0.01))  # how energy affects movement
        
        # Action encoding: [stay, up, down, left, right]
        self.actions = torch.tensor([
            [0, 0],   # stay
            [0, -1],  # up
            [0, 1],   # down
            [-1, 0],  # left
            [1, 0]    # right
        ], dtype=torch.float32, device=device)
        
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset environment and return initial state"""
        # Initialize agent positions randomly
        self.positions = torch.randint(0, self.grid_size, (self.n_agents, 2), 
                                     dtype=torch.float32, device=self.device)
        
        # Initialize energies
        self.energies = torch.full((self.n_agents,), self.initial_energy, 
                                 dtype=torch.float32, device=self.device)
        
        # Track alive status
        self.alive = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)
        
        # Coverage tracking
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
        # Higher energy agents are more likely to move (explore)
        # Lower energy agents prefer to stay put (conserve)
        energy_factor = self.energies.unsqueeze(1) * self.energy_movement_scale
        
        # Base logits with learnable bias
        logits = self.movement_bias.unsqueeze(0).repeat(self.n_agents, 1)
        
        # Stay action gets negative energy factor (low energy -> prefer stay)
        logits[:, 0] -= energy_factor.squeeze()
        # Movement actions get positive energy factor (high energy -> prefer movement)
        logits[:, 1:] += energy_factor
        
        # Mask dead agents to always stay
        logits[~self.alive] = torch.tensor([-1000., -1000., -1000., -1000., -1000.], device=self.device)
        logits[~self.alive, 0] = 0.  # Force dead agents to stay
        
        return logits
    
    def _sample_actions(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample actions using Gumbel Softmax for differentiability"""
        # Gumbel Softmax sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        logits_with_noise = (logits + gumbel_noise) / temperature
        action_probs = F.softmax(logits_with_noise, dim=1)
        
        return action_probs
    
    def _apply_movement(self, action_probs: torch.Tensor) -> torch.Tensor:
        """Apply movement based on action probabilities"""
        # Compute expected movement delta
        movement_delta = torch.sum(action_probs.unsqueeze(2) * self.actions.unsqueeze(0), dim=1)
        
        # Apply movement
        new_positions = self.positions + movement_delta
        
        # Handle boundaries (clamp to grid)
        new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)
        
        # Compute energy cost (expected cost based on action probabilities)
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
        
        # Find neighbors within sharing radius
        neighbor_mask = (distances <= self.sharing_radius) & (distances > 0)
        
        # Only alive agents can participate
        alive_mask = self.alive.unsqueeze(0) & self.alive.unsqueeze(1)
        neighbor_mask = neighbor_mask & alive_mask
        
        # Energy differences (donors have higher energy)
        energy_diff = self.energies.unsqueeze(1) - self.energies.unsqueeze(0)
        
        # Sharing happens when energy difference exceeds threshold
        sharing_mask = neighbor_mask & (energy_diff > self.sharing_threshold)
        
        # Calculate energy transfers
        transfer_rate = torch.sigmoid(self.energy_transfer_rate)  # Ensure [0,1]
        energy_to_transfer = self.energies.unsqueeze(1) * transfer_rate
        
        # Count receivers for each donor
        num_receivers = sharing_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Distribute energy equally among receivers
        energy_per_receiver = energy_to_transfer / num_receivers
        
        # Calculate net energy change for each agent
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
        # Get movement logits and sample actions
        logits = self._get_movement_logits()
        action_probs = self._sample_actions(logits, temperature)
        
        # Apply movement
        new_positions, energy_cost = self._apply_movement(action_probs)
        self.positions = new_positions
        
        # Apply energy cost
        self.energies = self.energies - energy_cost
        
        # Energy sharing
        energy_change = self._energy_sharing()
        self.energies = self.energies + energy_change
        
        # Update alive status
        self.alive = self.alive & (self.energies > self.death_threshold)
        
        # Update coverage
        self._update_coverage()
        
        # Compute reward
        reward = self._compute_reward()
        
        return self.get_state(), reward
    
    def _compute_reward(self) -> torch.Tensor:
        """Compute reward based on coverage, survival, and energy"""
        # Coverage reward (normalized by grid size)
        coverage_reward = len(self.visited_positions) / (self.grid_size ** 2)
        
        # Survival reward (fraction of agents alive)
        survival_reward = self.alive.sum().float() / self.n_agents
        
        # Energy reward (average energy of alive agents)
        if self.alive.sum() > 0:
            energy_reward = self.energies[self.alive].mean() / self.initial_energy
        else:
            energy_reward = torch.tensor(0.0, device=self.device)
        
        # Weighted combination
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
            
            # Early termination if all agents dead
            if not self.alive.any():
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
    env = EnergyBoids(grid_size=20, n_agents=50)
    
    # Optimizer
    optimizer = torch.optim.Adam(env.parameters(), lr=0.01)
    
    # Training loop
    losses = []
    rewards = []
    
    for episode in range(n_episodes):
        optimizer.zero_grad()
        
        # Simulate episode
        total_reward, info = env.simulate_episode(episode_length, temperature=1.0)
        
        # Loss is negative reward
        loss = -total_reward
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log progress
        losses.append(loss.item())
        rewards.append(total_reward.item())
        
        if episode % 20 == 0:
            print(f"Episode {episode}: Reward = {total_reward.item():.4f}, "
                  f"Coverage = {info['final_coverage']}, "
                  f"Alive = {info['final_alive']}, "
                  f"Avg Energy = {info['final_avg_energy']:.2f}")
            print(f"Transfer rate: {torch.sigmoid(env.energy_transfer_rate).item():.4f}")
    
    return env, losses, rewards


# Example usage and visualization
if __name__ == "__main__":
    # Train the model
    trained_env, losses, rewards = train_energy_boids(n_episodes=100)
    
    # Plot training progress
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss (-Reward)')
    
    ax2.plot(rewards)
    ax2.set_title('Training Reward')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    
    plt.tight_layout()
    plt.show()
    
    # Final learned parameters
    print(f"\nFinal learned parameters:")
    print(f"Energy transfer rate: {torch.sigmoid(trained_env.energy_transfer_rate).item():.4f}")
    print(f"Movement bias: {trained_env.movement_bias.detach().cpu().numpy()}")
    print(f"Energy-movement scale: {trained_env.energy_movement_scale.item():.4f}")