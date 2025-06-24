import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import random

seed = 42  
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
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

class DifferentiableBoids:
    def __init__(self, n_boids=10, dim=2, max_speed=2.0, perception_radius=3.0):
        self.n_boids = n_boids
        self.dim = dim
        self.max_speed = max_speed
        self.perception_radius = perception_radius
        
        # Control actions: [forward, left, right, speed_up, slow_down]
        self.n_actions = 5
        
        # Policy network parameters
        self.policy_net = nn.Sequential(
            nn.Linear(dim * 2, 64),  # position and velocity
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_actions)
        )
        
        # Initialize learnable parameters for LearnableStochasticCategorical
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
    def compute_control_probs(self, state, policy_params):
        """Compute control probabilities from state"""
        # state: [position, velocity] concatenated
        logits = self.policy_net(state)
        return F.softmax(logits, dim=-1)
    
    def compute_control_logits(self, state, policy_params):
        """Compute control logits from state"""
        return self.policy_net(state)
    
    def sample_control_stoch(self, state, policy_params):
        """
        Uses the custom differentiable categorical estimator.
        """
        probs = self.compute_control_probs(state, policy_params)
        sample = Categorical.apply(probs.unsqueeze(0))
        control_idx = int(sample.item())
        log_prob = torch.log(probs[control_idx] + 1e-8)
        return control_idx, log_prob
    
    def sample_control_custom(self, state, policy_params):
            logits = self.compute_control_probs(state, policy_params)
            sample, _ = StochasticCategorical.apply(logits.unsqueeze(0), None)
            control_idx = sample.item()
            log_prob = torch.log(logits[control_idx] + 1e-8)
            return control_idx, log_prob
    
    def sample_control_cat(self, state, policy_params, tau=1.0):
        # Uses the LearnableStochasticCategorical estimator
        probs = self.compute_control_probs(state, policy_params)
        # Forward pass returns sample and updated uncertainty (ignored here)
        sample, _ = LearnableStochasticCategorical.apply(
            probs.unsqueeze(0),
            None,
            policy_params['alpha'],
            policy_params['beta']
        )
        control_idx = sample.item()
        log_prob = torch.log(probs[control_idx] + 1e-8)
        return control_idx, log_prob

    
    def sample_control_gumbel(self, state, policy_params, tau=1.0):
        """Gumbel softmax baseline"""
        logits = self.compute_control_logits(state, policy_params)
        gumbel_samples = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        control_idx = torch.argmax(gumbel_samples, dim=-1).item()
        probs = F.softmax(logits, dim=-1)
        log_prob = torch.log(probs[control_idx] + 1e-8)
        return control_idx, log_prob
    
    def sample_control(self, state, policy_params, method="custom", tau=1.0):
        """Main sampling function using your interface"""
        if method == "Learnable AUG":
            return self.sample_control_cat(state, policy_params, tau)
        elif method == "Gumbel":
            return self.sample_control_gumbel(state, policy_params, tau)
        elif method == "Stochastic AD":
            return self.sample_control_stoch(state,policy_params)
        else:
            return self.sample_control_custom(state, policy_params)
    
    def apply_control(self, positions, velocities, actions):
        """Apply control actions to boids - returns new velocities"""
        # actions: [n_boids] list of action indices
        new_velocities = velocities.clone()
        
        for i, action in enumerate(actions):
            if action == 0:  # forward
                new_velocities[i] = velocities[i]
            elif action == 1:  # left
                if self.dim == 2:
                    rotation = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32)
                    new_velocities[i] = torch.matmul(rotation, velocities[i])
            elif action == 2:  # right
                if self.dim == 2:
                    rotation = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)
                    new_velocities[i] = torch.matmul(rotation, velocities[i])
            elif action == 3:  # speed up
                new_velocities[i] = velocities[i] * 1.1
            elif action == 4:  # slow down
                new_velocities[i] = velocities[i] * 0.9
        
        # Clip velocities to max speed
        speeds = torch.norm(new_velocities, dim=1, keepdim=True)
        max_speed_tensor = torch.tensor(self.max_speed)
        speed_ratios = torch.minimum(max_speed_tensor / (speeds + 1e-8), torch.ones_like(speeds))
        new_velocities = new_velocities * speed_ratios
        
        return new_velocities
    
    def boids_update(self, positions, velocities):
        """Standard boids rules: separation, alignment, cohesion"""
        n_boids = positions.shape[0]
        
        # Compute pairwise distances
        pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [n_boids, n_boids, dim]
        distances = torch.norm(pos_diff, dim=2)  # [n_boids, n_boids]
        
        # Create neighbor mask
        neighbor_mask = (distances < self.perception_radius) & (distances > 0)
        
        # Separation
        separation = torch.zeros_like(positions)
        for i in range(n_boids):
            neighbors = neighbor_mask[i]
            if neighbors.any():
                close_neighbors = distances[i] < (self.perception_radius / 2)
                close_neighbors = close_neighbors & (distances[i] > 0)
                if close_neighbors.any():
                    separation[i] = -torch.mean(pos_diff[i][close_neighbors], dim=0)
        
        # Alignment
        alignment = torch.zeros_like(velocities)
        for i in range(n_boids):
            neighbors = neighbor_mask[i]
            if neighbors.any():
                alignment[i] = torch.mean(velocities[neighbors], dim=0) - velocities[i]
        
        # Cohesion
        cohesion = torch.zeros_like(positions)
        for i in range(n_boids):
            neighbors = neighbor_mask[i]
            if neighbors.any():
                center_of_mass = torch.mean(positions[neighbors], dim=0)
                cohesion[i] = center_of_mass - positions[i]
        
        # Combine forces
        total_force = 0.5 * separation + 0.3 * alignment + 0.2 * cohesion
        
        # Update velocities
        new_velocities = velocities + 0.1 * total_force
        
        # Clip to max speed
        speeds = torch.norm(new_velocities, dim=1, keepdim=True)
        max_speed_tensor = torch.tensor(self.max_speed)
        speed_ratios = torch.minimum(max_speed_tensor / (speeds + 1e-8), torch.ones_like(speeds))
        new_velocities = new_velocities * speed_ratios
        
        # Update positions
        new_positions = positions + new_velocities * 0.1
        
        return new_positions, new_velocities
    
    def simulate_chain(self, chain_length, method="custom", tau=1.0):
        """Simulate a chain of boids updates with differentiable control using REINFORCE"""
        # Initialize positions and velocities - these don't need gradients directly
        positions = torch.randn(self.n_boids, self.dim) * 5.0
        velocities = torch.randn(self.n_boids, self.dim) * 0.5
        
        # Policy parameters
        policy_params = {
            'alpha': self.alpha,
            'beta': self.beta
        }
        
        log_probs = []  # Store log probabilities for REINFORCE
        trajectory = []
        
        for step in range(chain_length):
            # Sample actions for each boid
            actions = []
            step_log_probs = []
            
            for boid_idx in range(self.n_boids):
                # Create state vector (position and velocity) - detach to avoid in-place issues
                state = torch.cat([positions[boid_idx].detach(), velocities[boid_idx].detach()])
                
                # Sample control action
                action, log_prob = self.sample_control(state, policy_params, method, tau)
                actions.append(action)
                step_log_probs.append(log_prob)
            
            # Store log probabilities for gradient computation
            log_probs.extend(step_log_probs)
            
            # Apply controls (this doesn't need gradients, just transforms the state)
            velocities = self.apply_control(positions, velocities, actions)
            
            # Apply boids dynamics (this also doesn't need gradients through positions)
            positions, velocities = self.boids_update(positions, velocities)
            
            trajectory.append((positions.clone(), velocities.clone()))
        
        return trajectory, torch.stack(log_probs)

class GradientBenchmark:
    def __init__(self):
        self.boids = DifferentiableBoids(n_boids=10, dim=2)
        self.methods = ["Learnable AUG", "Gumbel", "Stochastic AD", "Fixed AUG"]
        self.chain_lengths = [5,10,50,100]
        self.n_trials = 10  # Reduced for faster testing
        
    def compute_loss(self, trajectory):
        """Compute loss function for the trajectory"""
        # Target: maintain formation while moving towards center
        target_center = torch.tensor([0.0, 0.0])
        
        total_loss = 0.0
        for positions, velocities in trajectory:
            # Distance to target center
            center_of_mass = torch.mean(positions, dim=0)
            center_loss = torch.norm(center_of_mass - target_center) ** 2
            
            # Formation maintenance (variance in pairwise distances)
            pairwise_dists = torch.cdist(positions, positions)
            # Only consider upper triangle to avoid double counting
            upper_triangle = torch.triu(pairwise_dists, diagonal=1)
            non_zero_dists = upper_triangle[upper_triangle > 0]
            if len(non_zero_dists) > 0:
                formation_loss = torch.var(non_zero_dists)
            else:
                formation_loss = torch.tensor(0.0)
            
            total_loss += center_loss + 0.1 * formation_loss
        
        return total_loss
    
    def measure_gradients(self, method, chain_length, n_trials):
        """Measure gradient statistics using REINFORCE"""
        gradients = []
        losses = []
        for trial in range(n_trials):
            try:
                # Zero gradients
                self.boids.policy_net.zero_grad()
                if self.boids.alpha.grad is not None:
                    self.boids.alpha.grad.zero_()
                if self.boids.beta.grad is not None:
                    self.boids.beta.grad.zero_()
                
                # Run simulation
                trajectory, log_probs = self.boids.simulate_chain(chain_length, method)
                
                # Compute loss (reward)
                loss = self.compute_loss(trajectory)
                losses.append(loss.detach().item())
                # REINFORCE: multiply log probabilities by loss (negative reward)
                # This creates the policy gradient: ∇J = E[∇log π(a|s) * R]
                policy_loss = torch.sum(log_probs) * loss.detach()  # Detach loss to avoid second-order gradients
                
                # Backward pass
                policy_loss.backward()
                # Collect gradients
                trial_gradients = []
                for param in self.boids.policy_net.parameters():
                    if param.grad is not None:
                        trial_gradients.append(param.grad.flatten().detach().clone())
                
                # Also collect alpha/beta gradients if they exist
                if self.boids.alpha.grad is not None:
                    trial_gradients.append(self.boids.alpha.grad.flatten().detach().clone())
                if self.boids.beta.grad is not None:
                    trial_gradients.append(self.boids.beta.grad.flatten().detach().clone())
                
                if trial_gradients:
                    gradients.append(torch.cat(trial_gradients))
                
            except Exception as e:
                print(f"Error in trial {trial} for {method}, chain {chain_length}: {e}")
                continue
        
        if gradients and len(gradients) > 0:
            gradients = torch.stack(gradients)
            grad_mean = torch.mean(gradients, dim=0)
            grad_var = torch.var(gradients, dim=0)
            
            return {
                'mean_norm': torch.norm(grad_mean).item(),
                'mean_var': torch.mean(grad_var).item(),
                'total_var': torch.var(gradients.flatten()).item(),
                'n_successful_trials': len(gradients),
                'loss_mean': np.mean(losses) if losses else 0.0,
            }
        else:
            return {
                'mean_norm': 0.0, 
                'mean_var': 0.0, 
                'total_var': 0.0,
                'n_successful_trials': 0,
            }
    
    def run_benchmark(self):
        """Run the complete benchmark"""
        results = defaultdict(lambda: defaultdict(dict))
        
        print("Running gradient estimator benchmark...")
        print("Methods:", self.methods)
        print("Chain lengths:", self.chain_lengths)
        print("Trials per configuration:", self.n_trials)
        print("-" * 50)
        
        for method in self.methods:
            print(f"\nBenchmarking method: {method}")
            for chain_length in self.chain_lengths:
                print(f"  Chain length: {chain_length}")
                
                start_time = time.time()
                stats = self.measure_gradients(method, chain_length, self.n_trials)
                end_time = time.time()
                
                results[method][chain_length] = stats
                results[method][chain_length]['time'] = end_time - start_time
                
                print(f"    Mean norm: {stats['mean_norm']:.4f}")
                print(f"    Mean var: {stats['mean_var']:.4f}")
                print(f"    Total var: {stats['total_var']:.4f}")
                print(f"    Successful trials: {stats['n_successful_trials']}/{self.n_trials}")
                print(f"    Loss mean: {stats['loss_mean']:.4f}")
                print(f"    Time: {end_time - start_time:.2f}s")
        
        return results
    
    def plot_results(self, results):
        """Plot benchmark results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['mean_norm', 'mean_var', 'total_var','loss_mean']
        titles = ['Gradient Mean Norm', 'Mean Gradient Variance', 'Total Gradient Variance','Loss Mean']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            for method in self.methods:
                x_vals = self.chain_lengths
                y_vals = [results[method][length][metric] for length in self.chain_lengths]
                ax.plot(x_vals, y_vals, marker='o', label=method,linewidth=2)
            
            ax.set_xlabel('Chain Length')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def summary_table(self, results):
        """Print summary table"""
        print("\n" + "="*96)
        print("GRADIENT ESTIMATOR BENCHMARK SUMMARY")
        print("="*96)
        
        print(f"{'Method':<15} {'Chain':<6} {'Mean Norm':<12} {'Mean Var':<12} {'Total Var':<12} {'Loss Mean':<8}")
        print("-" * 96)
        
        for method in self.methods:
            for chain_length in self.chain_lengths:
                stats = results[method][chain_length]
                print(f"{method:<15} {chain_length:<6} {stats['mean_norm']:<12.4f} "
                      f"{stats['mean_var']:<12.4f} {stats['total_var']:<12.4f} "
                      f"{stats['loss_mean']:<8.2f}")
            print("-" * 96)

def main():
    """Main benchmarking function"""
    benchmark = GradientBenchmark()
    results = benchmark.run_benchmark()
    
    # Print summary
    benchmark.summary_table(results)
    
    # Plot results
    fig = benchmark.plot_results(results)
    
    # Analysis
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Find best method for each chain length
    for chain_length in benchmark.chain_lengths:
        print(f"\nChain Length {chain_length}:")
        
        # Compare total variance (lower is better)
        method_vars = {method: results[method][chain_length]['total_var'] 
                      for method in benchmark.methods}
        best_method = min(method_vars, key=method_vars.get)
        
        print(f"  Best method (lowest variance): {best_method}")
        print(f"  Variance: {method_vars[best_method]:.4f}")
        
        # Show relative performance
        custom_var = method_vars.get('custom', float('inf'))
        print(f"  Custom method variance: {custom_var:.4f}")
        
        if custom_var != float('inf'):
            for method, var in method_vars.items():
                if method != 'custom' and var > 0:
                    improvement = ((var - custom_var) / var) * 100
                    print(f"  vs {method}: {improvement:+.1f}% improvement")

if __name__ == "__main__":
    main()
