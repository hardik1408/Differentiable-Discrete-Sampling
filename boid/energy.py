import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

# Placeholder imports for your custom estimators
# from your_estimators import StochasticCategorical, LearnableStochasticCategorical
class StochasticCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, uncertainty_in=None):
        result = torch.multinomial(p, num_samples=1)
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        var_chosen = (1.0 - p) / p.clamp(min=1e-10)
        var_not_chosen = p / (1.0 - p).clamp(min=1e-10)
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
        adjusted_ws = adjusted_ws / adjusted_ws.mean().clamp(min=1e-10)
        grad_p = grad_output.expand_as(p) * adjusted_ws
        return grad_p, None

class LearnableStochasticCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, uncertainty_in=None, alpha=None, beta=None):
        result = torch.multinomial(p, num_samples=1)
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        var_chosen = (1.0 - p) / p.clamp(min=1e-10)
        var_not_chosen = p / (1.0 - p).clamp(min=1e-10)
        op_variance = one_hot * var_chosen + (1 - one_hot) * var_not_chosen
        uncertainty_out = uncertainty_in + op_variance if uncertainty_in is not None else op_variance
        ctx.save_for_backward(result, p, uncertainty_out, alpha, beta)
        return result, uncertainty_out

    @staticmethod
    def backward(ctx, grad_output, grad_uncertainty=None):
        result, p, uncertainty, alpha, beta = ctx.saved_tensors
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        w_chosen = (1.0 / p) / 2
        w_non_chosen = (1.0 / (1.0 - p)) / 2
        confidence = 1.0 / (1.0 + alpha * uncertainty.clamp(min=1e-6))
        adjusted_ws = (one_hot * w_chosen * beta + (1 - one_hot) * w_non_chosen * (1 - beta)) * confidence
        adjusted_ws = adjusted_ws / adjusted_ws.mean().clamp(min=1e-10)
        grad_p = grad_output.expand_as(p) * adjusted_ws
        confidence_derivative = -uncertainty.clamp(min=1e-6) * confidence * confidence
        grad_alpha = torch.sum(
            grad_output.expand_as(p)
            * ((one_hot * w_chosen * beta + (1 - one_hot) * w_non_chosen * (1 - beta)) * confidence_derivative)
        )
        balance_derivative = one_hot * w_chosen - (1 - one_hot) * w_non_chosen
        grad_beta = torch.sum(grad_output.expand_as(p) * (balance_derivative * confidence))
        return grad_p, None, grad_alpha, grad_beta
    
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
        one_hot = torch.zeros_like(p).scatter_(1, result, 1.0)
        one_hot.scatter_(1, result.long(), 1.0)
        p_safe = p.clamp(min=1e-6, max=1.0-1e-6)
        weights = (one_hot - p) / (p * (1-p)) 
        return grad_output.expand_as(p) * weights
    
class DifferentiableBoidsEnvironment:
    def __init__(self, n_robots=100, world_size=100.0, dt=0.1):
        self.n_robots = n_robots
        self.world_size = world_size
        self.dt = dt
        self.max_energy = 100.0
        self.death_threshold = 0.1
        
        # Environment parameters
        self.harvesting_rate = 0.5  # Energy gained per timestep
        self.base_consumption = 0.8  # Base energy consumption
        self.movement_cost = 0.1    # Additional cost per unit velocity
        
        # Area exploration parameters
        self.cell_size = 0.5  # Grid cell size for area calculation
        self.exploration_bonus = 2.0  # Reward multiplier for new area
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.positions = torch.randn(self.n_robots, 2) * self.world_size * 0.3
        self.velocities = torch.zeros(self.n_robots, 2)
        self.energies = torch.ones(self.n_robots) * self.max_energy * 0.8
        self.alive_mask = torch.ones(self.n_robots, dtype=torch.bool)
        self.timestep = 0
        
        # Area exploration tracking
        self.visited_cells = set()
        self.total_area_explored = 0.0
        self.area_history = []  # Track area over time
        
        # Initialize with starting positions
        self._update_visited_area()
        
        return self.get_state()
    
    def get_state(self):
        """Get current state for policy network"""
        # Compute relative positions and energies to neighbors
        rel_positions = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)  # [n_robots, n_robots, 2]
        distances = torch.norm(rel_positions, dim=2)  # [n_robots, n_robots]
        
        # Create state representation for each robot
        states = []
        for i in range(self.n_robots):
            if not self.alive_mask[i]:
                states.append(torch.zeros(8))  # Dead robot state
                continue
                
            # Local state: position, velocity, energy
            local_state = torch.cat([
                self.positions[i],          # [2]
                self.velocities[i],         # [2] 
                self.energies[i:i+1],       # [1]
            ])
            
            # Neighbor information (closest 3 neighbors)
            neighbor_dists = distances[i].clone()
            neighbor_dists[i] = float('inf')  # Ignore self
            _, neighbor_indices = torch.topk(neighbor_dists, k=min(3, self.n_robots-1), largest=False)
            
            neighbor_info = []
            for j in neighbor_indices:
                if self.alive_mask[j]:
                    neighbor_info.extend([
                        rel_positions[i, j, 0].item(),  # Relative x
                        rel_positions[i, j, 1].item(),  # Relative y
                        self.energies[j].item(),        # Neighbor energy
                    ])
                else:
                    neighbor_info.extend([0.0, 0.0, 0.0])
            
            # Pad to fixed size
            while len(neighbor_info) < 9:  # 3 neighbors * 3 features
                neighbor_info.append(0.0)
            
            robot_state = torch.cat([local_state, torch.tensor(neighbor_info[:9])])
            states.append(robot_state)
        
        return torch.stack(states)  # [n_robots, state_dim]
    
    def compute_boids_forces(self, positions, velocities, energies, alive_mask, params):
        """Compute boids forces with energy considerations"""
        n = positions.shape[0]
        forces = torch.zeros_like(positions)
        
        for i in range(n):
            if not alive_mask[i]:
                continue
                
            neighbors = []
            neighbor_forces = torch.zeros(2)
            
            for j in range(n):
                if i == j or not alive_mask[j]:
                    continue
                    
                rel_pos = positions[j] - positions[i]
                distance = torch.norm(rel_pos) + 1e-8
                
                # Only consider neighbors within interaction radius
                if distance < params['interaction_radius']:
                    neighbors.append(j)
                    
                    # Separation force (avoid collision)
                    if distance < params['separation_radius']:
                        separation = -rel_pos / (distance ** 2)
                        neighbor_forces += params['w_separation'] * separation
                    
                    # Alignment force (match velocity)
                    alignment = velocities[j] - velocities[i]
                    neighbor_forces += params['w_alignment'] * alignment
                    
                    # Cohesion force (move toward group center) - modified for exploration
                    cohesion = rel_pos / distance
                    # Reduce cohesion when energy is high to encourage exploration
                    exploration_factor = 1.0 + (energies[i] / self.max_energy) * 0.5
                    neighbor_forces += params['w_cohesion'] * cohesion / exploration_factor
                    
                    # Energy-based forces
                    energy_diff = energies[j] - energies[i]
                    if energies[i] < 0.3 * self.max_energy:  # Low energy
                        if energy_diff > 0:  # Neighbor has more energy
                            energy_seek = rel_pos / distance
                            neighbor_forces += params['w_energy_seek'] * energy_seek
                    
                    if energies[i] > 0.7 * self.max_energy and energy_diff < -0.2 * self.max_energy:
                        # High energy robot avoiding low energy clusters - explore instead
                        energy_avoid = -rel_pos / distance
                        neighbor_forces += params['w_energy_avoid'] * energy_avoid
                    
                    # Exploration force: high-energy robots seek unexplored areas
                    if energies[i] > 0.6 * self.max_energy:
                        # Check if this direction leads to less explored areas
                        future_pos = positions[i] + rel_pos * 0.1
                        future_cell_x = int(future_pos[0].item() / self.cell_size) if hasattr(future_pos[0], 'item') else int(future_pos[0] / self.cell_size)
                        future_cell_y = int(future_pos[1].item() / self.cell_size) if hasattr(future_pos[1], 'item') else int(future_pos[1] / self.cell_size)
                        
                        if (future_cell_x, future_cell_y) not in self.visited_cells:
                            exploration_force = rel_pos / distance
                            neighbor_forces += params.get('w_exploration', 0.3) * exploration_force
            
            forces[i] = neighbor_forces
        
        return forces
    
    def compute_energy_transfers(self, positions, energies, alive_mask, params):
        """Compute energy transfers between nearby robots"""
        n = positions.shape[0]
        energy_deltas = torch.zeros(n)
        
        for i in range(n):
            if not alive_mask[i]:
                continue
                
            for j in range(n):
                if i == j or not alive_mask[j]:
                    continue
                    
                distance = torch.norm(positions[j] - positions[i])
                
                # Energy transfer only at close range
                if distance < params['transfer_radius']:
                    energy_diff = energies[i] - energies[j]
                    
                    # Transfer from high to low energy robot
                    if energy_diff > params['transfer_threshold']:
                        # Soft transfer decision
                        transfer_prob = torch.sigmoid(energy_diff - params['transfer_threshold'])
                        proximity_weight = torch.exp(-distance / params['transfer_radius'])
                        
                        transfer_amount = params['transfer_rate'] * transfer_prob * proximity_weight
                        transfer_amount = torch.min(transfer_amount, energies[i] * 0.1)  # Max 10% of energy
                        
                        energy_deltas[i] -= transfer_amount
                        energy_deltas[j] += transfer_amount * params['transfer_efficiency']
        
        return energy_deltas
    
    def _update_visited_area(self):
        """Update the set of visited grid cells and calculate area"""
        previous_area = len(self.visited_cells)
        
        # Add current positions of alive robots to visited cells
        for i in range(self.n_robots):
            if self.alive_mask[i]:
                # Convert continuous position to discrete grid cell
                cell_x = int(self.positions[i, 0].item() / self.cell_size)
                cell_y = int(self.positions[i, 1].item() / self.cell_size)
                self.visited_cells.add((cell_x, cell_y))
        
        # Calculate current total area
        self.total_area_explored = len(self.visited_cells) * (self.cell_size ** 2)
        self.area_history.append(self.total_area_explored)
        
        # Return number of newly explored cells this step
        return len(self.visited_cells) - previous_area
    
    def get_exploration_metrics(self):
        """Get detailed exploration metrics"""
        return {
            'total_area': self.total_area_explored,
            'cells_visited': len(self.visited_cells),
            'area_efficiency': self.total_area_explored / max(self.timestep, 1),
            'coverage_ratio': self.total_area_explored / (self.world_size ** 2)
        }
    
    def step(self, actions, params):
        """Execute one environment step"""
        # Convert discrete actions to continuous force multipliers
        action_multipliers = torch.tensor([0.5, 0.8, 1.0, 1.2, 1.5])[actions]  # 5 discrete actions
        
        # Compute boids forces
        forces = self.compute_boids_forces(
            self.positions, self.velocities, self.energies, self.alive_mask, params
        )
        
        # Apply action multipliers to forces
        forces = forces * action_multipliers.unsqueeze(1)
        
        # Update velocities and positions
        self.velocities = self.velocities + forces * self.dt
        
        # Velocity limits
        max_velocity = 2.0
        velocity_norms = torch.norm(self.velocities, dim=1, keepdim=True)
        velocity_norms = torch.clamp(velocity_norms, min=1e-8)
        self.velocities = self.velocities * torch.clamp(velocity_norms, max=max_velocity) / velocity_norms
        
        # Update positions
        self.positions = self.positions + self.velocities * self.dt
        
        # Boundary conditions (toroidal world)
        self.positions = self.positions % self.world_size
        
        # Compute energy transfers
        energy_transfers = self.compute_energy_transfers(
            self.positions, self.energies, self.alive_mask, params
        )
        
        # Update energies
        movement_costs = self.movement_cost * torch.norm(self.velocities, dim=1)
        self.energies = (self.energies 
                        + self.harvesting_rate * self.dt 
                        - self.base_consumption * self.dt 
                        - movement_costs * self.dt
                        + energy_transfers * self.dt)
        
        # Update alive mask
        self.alive_mask = self.energies > self.death_threshold
        
        # Zero out dead robots
        self.energies = self.energies * self.alive_mask.float()
        self.velocities = self.velocities * self.alive_mask.float().unsqueeze(1)
        
        # Update area exploration
        new_cells_explored = self._update_visited_area()
        
        self.timestep += 1
        
        return self.get_state(), self.compute_reward(new_cells_explored), self.is_done()
    
    def compute_reward(self, new_cells_explored=0):
        """Compute collective reward based on area exploration and survival"""
        n_alive = torch.sum(self.alive_mask.float())
        
        # Option 1: Continuous reward (for faster training)
        if self.timestep < 500:  # During episode
            # Reward for exploring new areas
            exploration_reward = new_cells_explored * self.exploration_bonus
            
            # Reward for maintaining swarm cohesion while exploring
            survival_multiplier = (n_alive / self.n_robots) ** 2  # Quadratic penalty for deaths
            
            # Energy efficiency bonus (optional)
            avg_energy = torch.mean(self.energies[self.alive_mask]) if n_alive > 0 else 0
            energy_bonus = 0.1 * avg_energy / self.max_energy
            
            return exploration_reward * survival_multiplier + energy_bonus
        
        # Option 2: Sparse episode-end reward (higher variance)
        else:  # Episode end
            # Primary reward: total area explored
            base_reward = self.total_area_explored
            
            # Bonus for survival time (longer episodes allow more exploration)
            survival_bonus = self.timestep * 0.1
            
            # Penalty for poor energy management (high variance = inefficient sharing)
            if n_alive > 1:
                energy_penalty = torch.var(self.energies[self.alive_mask]) * 0.05
            else:
                energy_penalty = 0
            
            return base_reward + survival_bonus - energy_penalty
    
    def is_done(self):
        """Check if episode is done"""
        return torch.sum(self.alive_mask.float()) < 0.1 * self.n_robots or self.timestep > 500


class BoidsPolicy(nn.Module):
    def __init__(self, state_dim=14, hidden_dim=64, n_actions=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
    def forward(self, state):
        return self.network(state)
    
    def compute_control_probs(self, state):
        """Compute action probabilities"""
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def compute_control_logits(self, state):
        """Compute action logits"""
        return self.forward(state)
    
    def sample_control_stoch(self, state):
        """Uses the custom differentiable categorical estimator."""
        probs = self.compute_control_probs(state)
        sample = Categorical.apply(probs.unsqueeze(0))
        control_idx = int(sample.item())
        log_prob = torch.log(probs[control_idx] + 1e-8)
        return control_idx, log_prob
    
    def sample_control_custom(self, state):
        logits = self.compute_control_probs(state)
        sample, _ = StochasticCategorical.apply(logits.unsqueeze(0), None)
        # Placeholder - replace with your custom estimator
        # sample = torch.multinomial(logits, 1)
        control_idx = sample.item()
        log_prob = torch.log(logits[control_idx] + 1e-8)
        return control_idx, log_prob
    
    def sample_control_cat(self, state, policy_params, tau=1.0):
        probs = self.compute_control_probs(state)
        sample, _ = LearnableStochasticCategorical.apply(
            probs.unsqueeze(0),
            None,
            policy_params['alpha'],
            policy_params['beta']
        )
        # Placeholder - replace with your learnable estimator
        # sample = torch.multinomial(probs, 1)
        control_idx = sample.item()
        log_prob = torch.log(probs[control_idx] + 1e-8)
        return control_idx, log_prob
    
    def sample_control_gumbel(self, state, tau=1.0):
        """Gumbel softmax baseline"""
        logits = self.compute_control_logits(state)
        gumbel_samples = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        control_idx = torch.argmax(gumbel_samples, dim=-1).item()
        probs = F.softmax(logits, dim=-1)
        log_prob = torch.log(probs[control_idx] + 1e-8)
        return control_idx, log_prob
    
    def sample_control(self, state, method, policy_params=None, tau=1.0):
        """Main sampling function using your interface"""
        if method == "Learnable AUG":
            return self.sample_control_cat(state, policy_params, tau)
        elif method == "Gumbel":
            return self.sample_control_gumbel(state, tau)
        elif method == "Stochastic AD":
            return self.sample_control_stoch(state)
        else:
            return self.sample_control_custom(state)


class BoidsExperiment:
    def __init__(self, n_robots=10, n_episodes=1000):
        self.n_robots = n_robots
        self.n_episodes = n_episodes
        self.env = DifferentiableBoidsEnvironment(n_robots)
        self.policy = BoidsPolicy()
        
        # Default boids parameters
        self.default_params = {
            'w_separation': 2.0,
            'w_alignment': 1.0,
            'w_cohesion': 1.0,
            'w_energy_seek': 1.5,
            'w_energy_avoid': 0.5,
            'w_exploration': 0.8,  
            'interaction_radius': 2.0,
            'separation_radius': 0.5,
            'transfer_radius': 1.0,
            'transfer_threshold': 20.0,
            'transfer_rate': 2.0,
            'transfer_efficiency': 0.8,
        }
        
        # Policy parameters for learnable estimators
        self.policy_params = {
            'alpha': nn.Parameter(torch.tensor(1.0)),
            'beta': nn.Parameter(torch.tensor(1.0))
        }
        
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.policy_params.values()), 
            lr=0.001
        )
        
    def run_episode(self, method, params=None):
        """Run single episode with given sampling method"""
        if params is None:
            params = self.default_params
            
        state = self.env.reset()
        episode_reward = 0
        log_probs = []
        rewards = []
        
        while not self.env.is_done():
            actions = []
            episode_log_probs = []
            
            # Sample actions for all alive robots
            for i in range(self.n_robots):
                if self.env.alive_mask[i]:
                    action, log_prob = self.policy.sample_control(
                        state[i], method, self.policy_params
                    )
                    actions.append(action)
                    episode_log_probs.append(log_prob)
                else:
                    actions.append(0)  # Dead robot action
                    episode_log_probs.append(torch.tensor(0.0))
            
            # Execute step
            actions_tensor = torch.tensor(actions)
            next_state, reward, done = self.env.step(actions_tensor, params)
            
            episode_reward += reward
            log_probs.extend(episode_log_probs)
            rewards.append(reward)
            
            state = next_state
            
            if done:
                break
        
        # Get final exploration metrics
        exploration_metrics = self.env.get_exploration_metrics()
        
        return episode_reward, log_probs, rewards, exploration_metrics
    
    def compute_gradient_variance(self, method, n_trials=10):
        """Compute gradient variance for given method"""
        gradients = []
        
        for trial in range(n_trials):
            episode_reward, log_probs, rewards, _ = self.run_episode(method)
            
            # Compute policy gradient
            policy_loss = 0
            for log_prob, reward in zip(log_probs, rewards):
                if log_prob.requires_grad:
                    policy_loss -= log_prob * reward
            
            # Compute gradients
            self.optimizer.zero_grad()
            if policy_loss != 0:
                policy_loss.backward(retain_graph=True)
                
                # Collect gradients
                grad_vector = []
                for param in self.policy.parameters():
                    if param.grad is not None:
                        grad_vector.append(param.grad.flatten())
                
                if grad_vector:
                    gradients.append(torch.cat(grad_vector))
        
        if gradients:
            grad_tensor = torch.stack(gradients)
            return torch.var(grad_tensor, dim=0).mean().item()
        return 0.0
    
    def benchmark_methods(self, methods, n_episodes_per_method=100):
        """Benchmark different gradient estimation methods"""
        results = {}
        
        for method in methods:
            print(f"Benchmarking {method}...")
            
            rewards = []
            variances = []
            times = []
            exploration_areas = []
            coverage_ratios = []
            
            for episode in range(n_episodes_per_method):
                start_time = time.time()
                
                episode_reward, _, _, _ = self.run_episode(method)
                rewards.append(episode_reward)
                
                # Compute gradient variance every 10 episodes
                if episode % 10 == 0:
                    variance = self.compute_gradient_variance(method, n_trials=5)
                    variances.append(variance)
                
                times.append(time.time() - start_time)
                
                if episode % 10 == 0:
                    print(f"  Episode {episode}, Avg Reward: {np.mean(rewards[-20:]):.3f}")
            
            results[method] = {
                'rewards': rewards,
                'variances': variances,
                'times': times,
                'avg_reward': np.mean(rewards),
                'avg_variance': np.mean(variances),
                'avg_time': np.mean(times)
            }
        
        return results
    
    def plot_results(self, results):
        """Plot benchmark results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot rewards over time
        ax = axes[0, 0]
        for method, data in results.items():
            ax.plot(data['rewards'], label=method, alpha=0.7)
        ax.set_title('Episode Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        
        # Plot gradient variance
        ax = axes[0, 1]
        methods = list(results.keys())
        variances = [results[m]['avg_variance'] for m in methods]
        ax.bar(methods, variances)
        ax.set_title('Average Gradient Variance')
        ax.set_ylabel('Variance')
        plt.xticks(rotation=45)
        
        # Plot average rewards
        ax = axes[1, 0]
        avg_rewards = [results[m]['avg_reward'] for m in methods]
        ax.bar(methods, avg_rewards)
        ax.set_title('Average Episode Reward')
        ax.set_ylabel('Reward')
        plt.xticks(rotation=45)
        
        # Plot computation time
        ax = axes[1, 1]
        avg_times = [results[m]['avg_time'] for m in methods]
        ax.bar(methods, avg_times)
        ax.set_title('Average Computation Time')
        ax.set_ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main experiment runner"""
    print("Starting Differentiable Boids Energy Experiment...")
    
    # Initialize experiment
    experiment = BoidsExperiment(n_robots=100, n_episodes=100)
    
    # Methods to benchmark
    methods = [
        "Learnable AUG",
        "Gumbel", 
        "Stochastic AD",
        "Fixed AUG"
    ]
    
    # Run benchmark
    results = experiment.benchmark_methods(methods, n_episodes_per_method=50)
    
    # Print results summary
    print("\n" + "="*50)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*50)
    
    for method, data in results.items():
        print(f"\n{method}:")
        print(f"  Average Reward: {data['avg_reward']:.4f}")
        # print(f"  Gradient Variance: {data['avg_variance']:.6f}")
        # print(f"  Computation Time: {data['avg_time']:.4f}s")
    
    # Plot results
    experiment.plot_results(results)
    
    return results


if __name__ == "__main__":
    results = main()