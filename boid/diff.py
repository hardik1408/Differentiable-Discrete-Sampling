import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict

class Categorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p):
        if p.dim() == 1:
            p = p.unsqueeze(0)
        result = torch.multinomial(p, num_samples=1)
        ctx.save_for_backward(result, p)
        return result.float()
    
    @staticmethod
    def backward(ctx, grad_output):
        result, p = ctx.saved_tensors
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(1, result.long(), 1.0)
        p_safe = p.clamp(min=1e-6, max=1.0-1e-6)
        weights = (one_hot - p) / (p_safe * (1-p_safe))
        
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
        return result, uncertainty_out
    
    @staticmethod
    def backward(ctx, grad_output, grad_uncertainty=None):
        result, p, uncertainty = ctx.saved_tensors
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        
        w_chosen = (1.0 / p) / 2
        w_non_chosen = (1.0 / (1.0 - p)) / 2
        confidence = 1.0 / (1.0 + uncertainty.clamp(min=1e-6))
        adjusted_ws = (one_hot * w_chosen + (1 - one_hot) * w_non_chosen) * confidence
        adjusted_ws = adjusted_ws / adjusted_ws.mean(dim=-1, keepdim=True).clamp(min=1e-6)
        
        grad_p = grad_output.expand_as(p) * adjusted_ws
        return grad_p, None

class LearnableStochasticCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, alpha, beta, uncertainty_in=None):
        if p.dim() == 1:
            p = p.unsqueeze(0)
            
        result = torch.multinomial(p, num_samples=1)
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        
        var_chosen = (1.0 - p) / p.clamp(min=1e-6)
        var_not_chosen = p / (1.0 - p).clamp(min=1e-6)
        op_variance = one_hot * var_chosen + (1 - one_hot) * var_not_chosen
        uncertainty_out = uncertainty_in + op_variance if uncertainty_in is not None else op_variance
        
        ctx.save_for_backward(result, p, alpha, beta, uncertainty_out)
        return result, uncertainty_out
    
    @staticmethod
    def backward(ctx, grad_output, grad_uncertainty=None):
        result, p, alpha, beta, uncertainty = ctx.saved_tensors
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        
        w_chosen = (1.0 / p) / 2
        w_non_chosen = (1.0 / (1.0 - p)) / 2
        confidence = 1.0 / (1.0 + uncertainty.clamp(min=1e-6))
        
        scaling = torch.sigmoid(alpha) * (1 + torch.tanh(beta))
        adjusted_ws = (one_hot * w_chosen + (1 - one_hot) * w_non_chosen) * confidence * scaling
        adjusted_ws = adjusted_ws / adjusted_ws.mean(dim=-1, keepdim=True).clamp(min=1e-6)
        
        grad_p = grad_output.expand_as(p) * adjusted_ws
        
        variance_term = (one_hot * w_chosen + (1 - one_hot) * w_non_chosen) * confidence
        grad_alpha = (grad_output.expand_as(p) * variance_term * 
                     torch.sigmoid(alpha) * (1 - torch.sigmoid(alpha)) * 
                     (1 + torch.tanh(beta))).sum()
        
        grad_beta = (grad_output.expand_as(p) * variance_term * 
                    torch.sigmoid(alpha) * (1 - torch.tanh(beta)**2)).sum()
        
        return grad_p, grad_alpha, grad_beta, None

class DifferentiableBoidsEnv:
    def __init__(self, num_agents, world_size, device='cuda', batch_size=32):
        self.num_agents = num_agents
        self.world_size = world_size
        self.device = device
        self.batch_size = batch_size
        
        self.action_vectors = torch.tensor([
            [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [-1, 1], [1, -1], [-1, -1]
        ], dtype=torch.float32, device=device)
        self.num_actions = 9
        
        self.max_speed = 5.0
        self.energy_decay = 0.5
        self.initial_energy = 100.0
        self.death_threshold = 10.0
        
        self.step_count = 0
        self.reset()
    
    def reset(self):
        self.positions = torch.rand(self.batch_size, self.num_agents, 2, 
                                  device=self.device, requires_grad=True) * self.world_size
        self.velocities = torch.zeros(self.batch_size, self.num_agents, 2, 
                                    device=self.device, requires_grad=True)
        self.energies = torch.full((self.batch_size, self.num_agents), self.initial_energy, 
                                 device=self.device, requires_grad=True)
        
        self.alive = torch.ones(self.batch_size, self.num_agents, dtype=torch.bool, device=self.device)
        self.step_count = 0
        return self.get_state()
    
    def step(self, action_probs):
        action_probs = action_probs.reshape(self.batch_size, self.num_agents, -1)
        
        # Convert action probabilities to action vectors
        action_vectors = torch.einsum('bna,ad->bnd', action_probs, self.action_vectors)
        
        # Apply alive mask - FIXED: Use differentiable masking
        alive_mask = self.alive.unsqueeze(-1).float()
        masked_actions = action_vectors * alive_mask
        
        # Update velocities with momentum
        self.velocities = 0.8 * self.velocities + 0.2 * masked_actions * self.max_speed
        
        # Clamp velocity magnitude
        vel_magnitude = torch.norm(self.velocities, dim=-1, keepdim=True)
        vel_magnitude_clamped = torch.clamp(vel_magnitude, max=self.max_speed)
        self.velocities = self.velocities * (vel_magnitude_clamped / (vel_magnitude + 1e-8))
        
        # Update positions
        self.positions = self.positions + self.velocities * alive_mask
        self.positions = self.positions % self.world_size
        
        # Update energy - FIXED: Use differentiable energy updates
        movement_cost = torch.norm(self.velocities, dim=-1) * self.energy_decay
        self.energies = self.energies - movement_cost * self.alive.float()
        
        # Update alive status - FIXED: Use differentiable alive updates
        # Instead of boolean operations, use smooth transitions
        death_prob = torch.sigmoid((self.death_threshold - self.energies) * 10.0)  # Steep sigmoid
        alive_float = self.alive.float() * (1.0 - death_prob)
        
        # Update alive mask (for next step)
        self.alive = (self.energies >= self.death_threshold) & self.alive
        
        rewards = self.compute_rewards(alive_float)
        
        self.step_count += 1
        next_state = self.get_state()
        
        info = {
            'agents_alive': self.alive.sum(dim=1).detach().cpu().numpy(),
            'step': self.step_count,
            'avg_energy': self.energies[self.alive].mean().item() if self.alive.any() else 0.0
        }
        
        return next_state, rewards, info
    
    def compute_rewards(self, alive_float=None):
        if alive_float is None:
            alive_float = self.alive.float()
        
        # Energy reward
        energy_reward = torch.clamp(self.energies / self.initial_energy, 0, 1) * 0.1
        
        # Cooperation reward - distance to nearest neighbor
        pos_expanded = self.positions.unsqueeze(2)  # [batch, agents, 1, 2]
        pos_expanded_t = self.positions.unsqueeze(1)  # [batch, 1, agents, 2]
        distances = torch.norm(pos_expanded - pos_expanded_t, dim=-1)  # [batch, agents, agents]
        
        # Mask self-distances and dead agents
        eye_mask = torch.eye(self.num_agents, device=self.device).unsqueeze(0).bool()
        distances = distances.masked_fill(eye_mask, 1000.0)
        
        alive_mask = alive_float.unsqueeze(1) * alive_float.unsqueeze(2)
        distances = distances + (1.0 - alive_mask) * 1000.0  # Add large distance for dead agents
        
        min_distances, _ = torch.min(distances, dim=-1)
        cooperation_reward = torch.exp(-min_distances / 20.0) * 0.05
        
        # Social reward - number of nearby agents
        nearby_mask = (distances < 50.0).float() * alive_mask
        neighbor_count = nearby_mask.sum(dim=-1)
        social_reward = torch.tanh(neighbor_count * 0.1) * 0.02
        
        total_reward = energy_reward + cooperation_reward + social_reward
        
        return total_reward * alive_float
    
    def get_state(self):
        # Normalize state components
        pos_norm = self.positions / self.world_size
        vel_norm = self.velocities / self.max_speed
        energy_norm = self.energies.unsqueeze(-1) / self.initial_energy
        alive_state = self.alive.unsqueeze(-1).float()
        
        batch_size, num_agents = pos_norm.shape[:2]
        
        # Get neighbor information
        pos_expanded = self.positions.unsqueeze(2)
        pos_expanded_t = self.positions.unsqueeze(1)
        all_distances = torch.norm(pos_expanded - pos_expanded_t, dim=-1)
        
        eye_mask = torch.eye(num_agents, device=self.device).unsqueeze(0).bool()
        all_distances = all_distances.masked_fill(eye_mask, 1e6)
        
        alive_mask = self.alive.unsqueeze(1) & self.alive.unsqueeze(2)
        all_distances = all_distances.masked_fill(~alive_mask, 1e6)
        
        min_distances, closest_indices = torch.min(all_distances, dim=-1)
        valid_neighbors = min_distances < 1e5
        
        neighbor_info = torch.zeros(batch_size, num_agents, 4, device=self.device)
        
        # FIXED: More efficient neighbor info computation
        for b in range(batch_size):
            valid_mask = valid_neighbors[b] & self.alive[b]
            if valid_mask.any():
                valid_agents = torch.where(valid_mask)[0]
                closest_idx = closest_indices[b][valid_mask]
                
                # Relative position and velocity
                neighbor_info[b, valid_agents, :2] = (
                    self.positions[b, closest_idx] - self.positions[b, valid_agents]
                ) / self.world_size
                neighbor_info[b, valid_agents, 2:4] = (
                    self.velocities[b, closest_idx] - self.velocities[b, valid_agents]
                ) / self.max_speed
        
        state = torch.cat([pos_norm, vel_norm, energy_norm, alive_state, neighbor_info], dim=-1)
        return state.reshape(batch_size * num_agents, -1)
    
    def get_coverage_stats(self):
        grid_size = 20
        cell_size = self.world_size / grid_size
        
        # Only consider alive agents
        alive_positions = self.positions[self.alive]
        if len(alive_positions) == 0:
            return 0, 0.0
        
        grid_x = torch.floor(alive_positions[:, 0] / cell_size).long()
        grid_y = torch.floor(alive_positions[:, 1] / cell_size).long()
        
        grid_x = torch.clamp(grid_x, 0, grid_size - 1)
        grid_y = torch.clamp(grid_y, 0, grid_size - 1)
        
        # Count unique cells
        unique_cells = torch.unique(grid_x * grid_size + grid_y)
        
        return len(unique_cells), len(unique_cells) / (grid_size * grid_size)
    
    def get_avg_energy(self):
        alive_energies = self.energies[self.alive]
        if len(alive_energies) > 0:
            return alive_energies.mean().item()
        return 0.0
    
    def get_agents_alive_count(self):
        return self.alive.sum().item()

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DifferentiableAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, device='cuda'):
        self.device = device
        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.alpha = nn.Parameter(torch.tensor(1.0, device=device))
        self.beta = nn.Parameter(torch.tensor(1.0, device=device))
        
        self.uncertainty_tracker = None
    
    def compute_probs(self, state):
        logits = self.policy_net(state)
        return F.softmax(logits, dim=-1)
    
    def compute_logits(self, state):
        return self.policy_net(state)
    
    def sample_stochastic_ad(self, state):
        probs = self.compute_probs(state)
        sample_indices = Categorical.apply(probs)
        action_probs = F.one_hot(sample_indices.squeeze(-1).long(), self.action_dim).float()
        return action_probs
    
    def sample_stochastic_categorical(self, state):
        probs = self.compute_probs(state)
        sample_indices, uncertainty = StochasticCategorical.apply(probs, self.uncertainty_tracker)
        self.uncertainty_tracker = uncertainty
        action_probs = F.one_hot(sample_indices.squeeze(-1).long(), self.action_dim).float()
        return action_probs
    
    def sample_learnable_aug(self, state):
        probs = self.compute_probs(state)
        sample_indices, uncertainty = LearnableStochasticCategorical.apply(
            probs, self.alpha, self.beta, self.uncertainty_tracker
        )
        self.uncertainty_tracker = uncertainty
        action_probs = F.one_hot(sample_indices.squeeze(-1).long(), self.action_dim).float()
        return action_probs
    
    def sample_gumbel(self, state, tau=1.0):
        logits = self.compute_logits(state)
        action_probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        return action_probs
    
    def sample_action(self, state, method, tau=1.0):
        if self.uncertainty_tracker is None:
            batch_size = state.shape[0]
            self.uncertainty_tracker = torch.zeros(batch_size, self.action_dim, device=self.device)
        
        if method == "Learnable AUG":
            return self.sample_learnable_aug(state)
        elif method == "Gumbel":
            return self.sample_gumbel(state, tau)
        elif method == "Stochastic AD":
            return self.sample_stochastic_ad(state)
        elif method == "Fixed AUG":
            return self.sample_stochastic_categorical(state)
        else:
            return self.sample_stochastic_ad(state)
    
    def reset_uncertainty(self):
        self.uncertainty_tracker = None

class ExperimentRunner:
    def __init__(self, device='cuda'):
        self.device = device
        
        self.num_agents = 10
        self.world_size = 100.0
        self.batch_size = 10
        
        self.learning_rate = 3e-4
        self.num_episodes = 51 # Increased for better training
        self.episode_length = 100
        
        self.env = DifferentiableBoidsEnv(
            num_agents=self.num_agents,
            world_size=self.world_size,
            device=self.device,
            batch_size=self.batch_size
        )
        
        self.state_dim = 10
        self.action_dim = 9
        
        self.agent = DifferentiableAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        # Separate optimizers for better control
        self.policy_optimizer = torch.optim.Adam(self.agent.policy_net.parameters(), lr=self.learning_rate)
        self.param_optimizer = torch.optim.Adam([self.agent.alpha, self.agent.beta], lr=self.learning_rate * 0.1)
        
        self.training_rewards = []
        self.max_rewards = []
        self.final_agents_alive = []
        self.coverage_ratios = []
        self.avg_energies = []
        self.gradient_norms = []
    
    def train_episode(self, method="Stochastic AD"):
        self.agent.reset_uncertainty()
        
        state = self.env.reset()
        total_reward = 0.0
        max_reward = 0.0
        
        for step in range(self.episode_length):
            action_probs = self.agent.sample_action(state, method, tau=1.0)
            
            next_state, rewards, info = self.env.step(action_probs)
            
            step_reward = rewards.sum()
            total_reward += step_reward
            max_reward = max(max_reward, step_reward.item())
            
            state = next_state
            
            if info['agents_alive'].sum() == 0:
                break
        
        final_agents_alive = self.env.get_agents_alive_count()
        coverage_cells, coverage_ratio = self.env.get_coverage_stats()
        avg_energy = self.env.get_avg_energy()
        
        loss = -total_reward
        
        # Separate optimization
        self.policy_optimizer.zero_grad()
        if method in ["Learnable AUG"]:
            self.param_optimizer.zero_grad()
        
        loss.backward()

        # Gradient clipping
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), 1.0)
        
        if method in ["Learnable AUG"]:
            param_grad_norm = torch.nn.utils.clip_grad_norm_([self.agent.alpha, self.agent.beta], 1.0)
            total_grad_norm = policy_grad_norm + param_grad_norm
        else:
            total_grad_norm = policy_grad_norm
        
        self.gradient_norms.append(total_grad_norm.item())
        
        self.policy_optimizer.step()
        if method in ["Learnable AUG"]:
            self.param_optimizer.step()
        
        return total_reward.item(), loss.item(), max_reward, final_agents_alive, coverage_ratio, avg_energy
    
    def train(self, method="Stochastic AD"):
        print(f"Training with method: {method}")
        print(f"Device: {self.device}")
        
        for episode in range(self.num_episodes):
            total_reward, loss, max_reward, final_agents_alive, coverage_ratio, avg_energy = self.train_episode(method)
            
            self.training_rewards.append(total_reward)
            self.max_rewards.append(max_reward)
            self.final_agents_alive.append(final_agents_alive)
            self.coverage_ratios.append(coverage_ratio)
            self.avg_energies.append(avg_energy)
            
            if episode % 10 == 0:
                recent_indices = max(0, len(self.max_rewards) - 20)
                recent_max_rewards = self.max_rewards[recent_indices:]
                recent_agents_alive = self.final_agents_alive[recent_indices:]
                recent_coverage = self.coverage_ratios[recent_indices:]
                recent_energy = self.avg_energies[recent_indices:]
                
                avg_max_reward = np.mean(recent_max_rewards)
                avg_agents_alive = np.mean(recent_agents_alive)
                avg_coverage = np.mean(recent_coverage)
                avg_avg_energy = np.mean(recent_energy)
                
                print(f"Episode {episode}: Max Reward: {avg_max_reward:.2f}, "
                      f"Agents Alive: {avg_agents_alive:.1f}, "
                      f"Coverage: {avg_coverage:.3f}, "
                      f"Avg Energy: {avg_avg_energy:.1f}")
                
                # if method in ["Learnable AUG"]:
                #     print(f"  Alpha: {self.agent.alpha.item():.3f}, Beta: {self.agent.beta.item():.3f}")
        
        print("Training completed!")
    
    def evaluate(self, num_eval_episodes=10, method="Stochastic AD"):
        print(f"Evaluating with method: {method}")
        
        eval_rewards = []
        eval_max_rewards = []
        eval_final_agents_alive = []
        eval_coverage_ratios = []
        eval_avg_energies = []
        
        with torch.no_grad():
            for episode in range(num_eval_episodes):
                self.agent.reset_uncertainty()
                state = self.env.reset()
                episode_reward = 0.0
                max_reward = 0.0
                
                for step in range(self.episode_length):
                    action_probs = self.agent.sample_action(state, method, tau=0.1)
                    next_state, rewards, info = self.env.step(action_probs)
                    
                    step_reward = rewards.sum().item()
                    episode_reward += step_reward
                    max_reward = max(max_reward, step_reward)
                    state = next_state
                    
                    if info['agents_alive'].sum() == 0:
                        break
                
                final_agents_alive = self.env.get_agents_alive_count()
                coverage_cells, coverage_ratio = self.env.get_coverage_stats()
                avg_energy = self.env.get_avg_energy()
                
                eval_rewards.append(episode_reward)
                eval_max_rewards.append(max_reward)
                eval_final_agents_alive.append(final_agents_alive)
                eval_coverage_ratios.append(coverage_ratio)
                eval_avg_energies.append(avg_energy)
        
        results = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_max_reward': np.mean(eval_max_rewards),
            'std_max_reward': np.std(eval_max_rewards),
            'avg_final_agents_alive': np.mean(eval_final_agents_alive),
            'std_final_agents_alive': np.std(eval_final_agents_alive),
            'avg_coverage_ratio': np.mean(eval_coverage_ratios),
            'std_coverage_ratio': np.std(eval_coverage_ratios),
            'avg_avg_energy': np.mean(eval_avg_energies),
            'std_avg_energy': np.std(eval_avg_energies),
            'all_rewards': eval_rewards,
            'training_rewards': self.training_rewards,
            'max_rewards': self.max_rewards,
            'final_agents_alive': self.final_agents_alive,
            'coverage_ratios': self.coverage_ratios,
            'avg_energies': self.avg_energies,
            'gradient_norms': self.gradient_norms
        }
        
        print(f"Results - Max Reward: {results['avg_max_reward']:.2f}, "
              f"Agents Alive: {results['avg_final_agents_alive']:.1f}, "
              f"Coverage: {results['avg_coverage_ratio']:.3f}, "
              f"Avg Energy: {results['avg_avg_energy']:.1f}")
        
        return results

def run_experiment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test gradient flow first
    print("Testing gradient flow...")
    runner = ExperimentRunner(device=device)
    
    # Check if gradients are flowing
    state = runner.env.reset()
    action_probs = runner.agent.sample_action(state, "Fixed AUG")
    next_state, rewards, info = runner.env.step(action_probs)
    loss = -rewards.sum()
    
    runner.policy_optimizer.zero_grad()
    loss.backward()
    
    total_grad_norm = 0
    for param in runner.agent.policy_net.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item()
    
    print(f"Total gradient norm: {total_grad_norm:.6f}")
    if total_grad_norm >= 1e-6:
        print("Gradients are flowing correctly")
    else:
        print("Gradients are not flowing")
        print(total_grad_norm)
    
    methods = ["Stochastic AD", "Fixed AUG", "Learnable AUG", "Gumbel"]
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running experiment with {method}")
        print(f"{'='*50}")
        
        runner = ExperimentRunner(device=device)
        
        start_time = time.time()
        runner.train(method=method)
        training_time = time.time() - start_time
        
        eval_results = runner.evaluate(method=method)
        eval_results['training_time'] = training_time
        
        results[method] = eval_results
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for method, result in results.items():
        print(f"{method}:")
        print(f"  Max Reward: {result['avg_max_reward']:.2f}")
        print(f"  Final Agents Alive: {result['avg_final_agents_alive']:.1f}")
        print(f"  Coverage Ratio: {result['avg_coverage_ratio']:.3f}")
        print(f"  Avg Energy: {result['avg_avg_energy']:.1f}")
        print(f"  Training Time: {result['training_time']:.1f}s")
        print()
    
    return results

if __name__ == "__main__":
    results = run_experiment()