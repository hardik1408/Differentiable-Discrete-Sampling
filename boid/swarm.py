import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
import time

# Placeholder implementations for your custom gradient estimators
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

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Learnable parameters for LearnableStochasticCategorical
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        return self.network(x)
    
    def get_policy_params(self):
        return {'alpha': self.alpha, 'beta': self.beta}

class Config:
    def __init__(self):
        self.tau = 0.5  # Gumbel softmax temperature

class BoidAgent:
    def __init__(self, agent_id: int, policy: PolicyNetwork, config: Config):
        self.agent_id = agent_id
        self.policy = policy
        self.config = config
    
    def sample_neighbor_selection(self, state: torch.Tensor, num_neighbors: int, method: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample which neighbors to select using different methods"""
        if num_neighbors == 0:
            return torch.empty(0, dtype=torch.long), torch.tensor(0.0)
        
        # Get policy logits for neighbor selection
        logits = self.policy(state)
        
        # Create a mask for available neighbors
        if num_neighbors < logits.shape[-1]:
            # Zero out logits for non-existent neighbors
            mask = torch.zeros_like(logits)
            mask[:num_neighbors] = 1.0
            logits = logits + (1 - mask) * (-1e9)
        
        # Normalize logits to probabilities
        probs = F.softmax(logits, dim=-1)
        params = self.policy.get_policy_params()
        
        if method == 'Gumbel':
            # Use Gumbel-Softmax with different temperature based on agent_id
            temp = self.config.tau * (1.0 + 0.1 * self.agent_id)  # Vary temperature per agent
            gumbel_samples = F.gumbel_softmax(logits, tau=temp, hard=True)
            selected_indices = torch.where(gumbel_samples > 0.5)[0]
            if len(selected_indices) == 0:
                selected_indices = torch.tensor([torch.argmax(probs).item()])
            # Limit selection
            selected_indices = selected_indices[:min(3, num_neighbors)]
            log_prob = torch.sum(torch.log(probs[selected_indices] + 1e-8))
            
        elif method == 'Learnable AUG':
            # Use learnable parameters with agent-specific modulation
            alpha_mod = params['alpha'] * (1.0 + 0.05 * self.agent_id)
            beta_mod = torch.sigmoid(params['beta'] + 0.1 * self.agent_id)  # Agent-specific beta
            
            selected_indices = []
            log_prob = 0
            remaining_probs = probs.clone()
            
            for _ in range(min(3, num_neighbors)):
                if len(selected_indices) >= num_neighbors:
                    break
                # Sample using learnable parameters
                action_tensor, _ = LearnableStochasticCategorical.apply(
                    remaining_probs.unsqueeze(0), None, alpha_mod, beta_mod
                )
                idx = action_tensor.item()
                if idx < num_neighbors and idx not in selected_indices:
                    selected_indices.append(idx)
                    log_prob += torch.log(remaining_probs[idx] + 1e-8)
                    # Reduce probability of selected neighbor for next iteration
                    remaining_probs[idx] *= 0.1
                    remaining_probs = F.softmax(remaining_probs, dim=-1)
                else:
                    break
            
            selected_indices = torch.tensor(selected_indices if selected_indices else [0])
            
        elif method == 'StochasticAD':
            # Use pure stochastic sampling with agent-specific noise
            noisy_probs = probs + torch.randn_like(probs) * 0.01 * (1 + self.agent_id * 0.1)
            noisy_probs = F.softmax(noisy_probs, dim=-1)
            
            selected_indices = []
            log_prob = 0
            remaining_probs = noisy_probs.clone()
            
            for _ in range(min(3, num_neighbors)):
                if len(selected_indices) >= num_neighbors:
                    break
                action_tensor = Categorical.apply(remaining_probs.unsqueeze(0))
                idx = int(action_tensor.item())
                if idx < num_neighbors and idx not in selected_indices:
                    selected_indices.append(idx)
                    log_prob += torch.log(remaining_probs[idx] + 1e-8)
                    # Remove selected neighbor
                    remaining_probs[idx] = 0
                    if remaining_probs.sum() > 0:
                        remaining_probs = remaining_probs / remaining_probs.sum()
                    else:
                        break
                else:
                    break
            
            selected_indices = torch.tensor(selected_indices if selected_indices else [0])
            
        else:  # StochasticCategorical
            # Use basic stochastic categorical with agent-specific variance
            variance_factor = 1.0 + 0.1 * self.agent_id
            adjusted_probs = probs * variance_factor
            adjusted_probs = F.softmax(adjusted_probs, dim=-1)
            
            selected_indices = []
            log_prob = 0
            remaining_probs = adjusted_probs.clone()
            
            for _ in range(min(3, num_neighbors)):
                if len(selected_indices) >= num_neighbors:
                    break
                action_tensor, _ = StochasticCategorical.apply(remaining_probs.unsqueeze(0), None)
                idx = action_tensor.item()
                if idx < num_neighbors and idx not in selected_indices:
                    selected_indices.append(idx)
                    log_prob += torch.log(remaining_probs[idx] + 1e-8)
                    # Remove selected neighbor
                    remaining_probs[idx] = 0
                    if remaining_probs.sum() > 0:
                        remaining_probs = remaining_probs / remaining_probs.sum()
                    else:
                        break
                else:
                    break
            
            selected_indices = torch.tensor(selected_indices if selected_indices else [0])
        
        return selected_indices, log_prob

class DifferentiableBoids:
    def __init__(self, num_agents: int, max_neighbors: int = 5, world_size: float = 10.0):
        self.num_agents = num_agents
        self.max_neighbors = max_neighbors
        self.world_size = world_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Boids parameters
        self.separation_weight = 2.0
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0
        self.max_speed = 2.0
        self.perception_radius = 3.0
        
        # Enhanced state dimension to make agents more distinguishable
        self.state_dim = 8  # [pos_x, pos_y, vel_x, vel_y, num_neighbors, min_distance, agent_id, avg_neighbor_distance]
        
    def get_neighbors_vectorized(self, positions: torch.Tensor) -> torch.Tensor:
        """Get all pairwise distances and neighbor masks"""
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.norm(diff, dim=2)
        neighbor_mask = (distances < self.perception_radius) & (distances > 0)
        return distances, neighbor_mask, diff
    
    def get_agent_state(self, agent_idx: int, positions: torch.Tensor, velocities: torch.Tensor, 
                       distances: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        """Get enhanced state representation for a single agent"""
        neighbors = neighbor_mask[agent_idx]
        neighbor_distances = distances[agent_idx][neighbors]
        
        own_pos = positions[agent_idx]
        own_vel = velocities[agent_idx]
        
        # Neighbor statistics
        if neighbor_distances.shape[0] > 0:
            num_neighbors = torch.tensor(float(neighbor_distances.shape[0])).to(self.device)
            min_distance = neighbor_distances.min()
            avg_neighbor_distance = neighbor_distances.mean()
        else:
            num_neighbors = torch.tensor(0.0).to(self.device)
            min_distance = torch.tensor(self.perception_radius).to(self.device)
            avg_neighbor_distance = torch.tensor(self.perception_radius).to(self.device)
        
        # Agent ID as a feature (normalized)
        agent_id_normalized = torch.tensor(float(agent_idx) / self.num_agents).to(self.device)
        
        # Enhanced state with more distinguishing features
        state = torch.cat([
            own_pos,  # [2]
            own_vel,  # [2] 
            num_neighbors.unsqueeze(0),  # [1]
            min_distance.unsqueeze(0),   # [1]
            agent_id_normalized.unsqueeze(0),  # [1] - helps distinguish agents
            avg_neighbor_distance.unsqueeze(0)  # [1]
        ])  # Total: [8]
        
        return state
    
    def select_neighbors_with_policy(self, agent_idx: int, positions: torch.Tensor, 
                                   velocities: torch.Tensor, distances: torch.Tensor,
                                   neighbor_mask: torch.Tensor, agents: List[BoidAgent], 
                                   method: str) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Select neighbors for an agent using the policy"""
        neighbors = neighbor_mask[agent_idx]
        neighbor_indices = torch.where(neighbors)[0]
        
        if len(neighbor_indices) == 0:
            return torch.zeros(2).to(self.device), []
        
        # Sort neighbors by distance (closest first)
        neighbor_distances = distances[agent_idx][neighbors]
        sorted_distances, sort_idx = neighbor_distances.sort()
        neighbor_indices = neighbor_indices[sort_idx]
        
        # Limit to max_neighbors
        if len(neighbor_indices) > self.max_neighbors:
            neighbor_indices = neighbor_indices[:self.max_neighbors]
        
        # Get agent state
        state = self.get_agent_state(agent_idx, positions, velocities, distances, neighbor_mask)
        
        # Sample neighbor selection using policy
        selected_neighbor_idx, log_prob = agents[agent_idx].sample_neighbor_selection(
            state, len(neighbor_indices), method
        )
        
        # Map selected indices back to actual neighbor indices
        if len(selected_neighbor_idx) > 0:
            # Ensure indices are within bounds
            valid_idx = selected_neighbor_idx[selected_neighbor_idx < len(neighbor_indices)]
            if len(valid_idx) > 0:
                selected_neighbors = neighbor_indices[valid_idx]
            else:
                selected_neighbors = neighbor_indices[:1]  # Select at least one neighbor
        else:
            selected_neighbors = neighbor_indices[:1]  # Select at least one neighbor
        
        # Compute boids forces with selected neighbors  
        force = self.compute_boids_force(agent_idx, positions, velocities, selected_neighbors)
        
        return force, [log_prob]
    
    def compute_boids_force(self, agent_idx: int, positions: torch.Tensor, 
                          velocities: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
        """Compute boids forces for selected neighbors"""
        if len(neighbor_indices) == 0:
            return torch.zeros(2).to(self.device)
        
        agent_pos = positions[agent_idx]
        agent_vel = velocities[agent_idx]
        neighbor_pos = positions[neighbor_indices]
        neighbor_vel = velocities[neighbor_indices]
        
        # Separation force
        diff = agent_pos - neighbor_pos
        distances = torch.norm(diff, dim=1, keepdim=True)
        separation = (diff / (distances + 1e-8)).sum(dim=0)
        
        # Alignment force
        alignment = neighbor_vel.mean(dim=0) - agent_vel
        
        # Cohesion force
        cohesion = neighbor_pos.mean(dim=0) - agent_pos
        
        total_force = (self.separation_weight * separation + 
                      self.alignment_weight * alignment + 
                      self.cohesion_weight * cohesion)
        
        return total_force
    
    def simulate_step(self, positions: torch.Tensor, velocities: torch.Tensor, 
                     agents: List[BoidAgent], method: str) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Single simulation step"""
        distances, neighbor_mask, _ = self.get_neighbors_vectorized(positions)
        
        new_positions = positions.clone()
        new_velocities = velocities.clone()
        all_log_probs = []
        
        for i in range(self.num_agents):
            force, log_probs = self.select_neighbors_with_policy(
                i, positions, velocities, distances, neighbor_mask, agents, method
            )
            
            # Update velocity with force
            new_velocities[i] += force * 0.1  # dt = 0.1
            
            # Limit speed
            speed = torch.norm(new_velocities[i])
            if speed > self.max_speed:
                new_velocities[i] = new_velocities[i] / speed * self.max_speed
            
            # Update position
            new_positions[i] += new_velocities[i] * 0.1  # dt = 0.1
            
            # Keep in bounds (toroidal world)
            new_positions[i] = new_positions[i] % self.world_size
            
            all_log_probs.extend(log_probs)
        
        return new_positions, new_velocities, all_log_probs
    
    def compute_flocking_coherence(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute flocking coherence as negative variance from centroid"""
        centroid = positions.mean(dim=0)
        distances_to_centroid = torch.norm(positions - centroid, dim=1)
        coherence = -distances_to_centroid.var()
        return coherence

def run_gradient_variance_experiment():
    """Main experiment function"""
    # Hyperparameters
    NUM_AGENTS = 10
    MAX_NEIGHBORS = 5
    GRADIENT_CHAIN_LENGTHS = [3, 5, 7, 9, 11]
    METHODS = ['Gumbel', 'Learnable AUG', 'StochasticAD', 'StochasticCategorical']
    NUM_RUNS = 20
    HIDDEN_DIM = 64
    LEARNING_RATE = 0.001
    STATE_DIM = 8  # Updated state dimension
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {}
    
    for method in METHODS:
        print(f"\n=== Testing Method: {method} ===")
        results[method] = {}
        
        for chain_length in GRADIENT_CHAIN_LENGTHS:
            print(f"Chain length: {chain_length}")
            
            gradient_variances = []
            expected_param_count = None
            
            for run in range(NUM_RUNS):
                # Different seed for each method and run combination
                seed = run + 42 + hash(method) % 1000
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Initialize environment
                boids = DifferentiableBoids(NUM_AGENTS, MAX_NEIGHBORS)
                
                # Use a single shared policy to ensure consistent parameter count
                shared_policy = PolicyNetwork(
                    input_dim=STATE_DIM, 
                    hidden_dim=HIDDEN_DIM, 
                    output_dim=MAX_NEIGHBORS
                ).to(device)
                
                config = Config()
                agents = []
                for agent_id in range(NUM_AGENTS):
                    agent = BoidAgent(agent_id, shared_policy, config)
                    agents.append(agent)
                
                # Random initial positions and velocities with more variation
                positions = torch.rand(NUM_AGENTS, 2).to(device) * boids.world_size
                velocities = (torch.randn(NUM_AGENTS, 2).to(device) * 0.5) + \
                           (torch.rand(NUM_AGENTS, 2).to(device) - 0.5) * 0.3
                
                # Create optimizer
                optimizer = torch.optim.Adam(shared_policy.parameters(), lr=LEARNING_RATE)
                optimizer.zero_grad()
                
                # Simulate for chain_length steps
                all_log_probs = []
                total_loss = 0
                
                for step in range(chain_length):
                    positions, velocities, log_probs = boids.simulate_step(
                        positions, velocities, agents, method
                    )
                    all_log_probs.extend(log_probs)
                    
                    # Compute loss (negative flocking coherence)
                    coherence = boids.compute_flocking_coherence(positions)
                    step_loss = -coherence
                    total_loss += step_loss
                
                # Compute gradients
                if all_log_probs:
                    # REINFORCE-style loss
                    policy_loss = -sum(all_log_probs) * total_loss.detach()
                    policy_loss.backward()
                    
                    # Collect gradients from shared policy
                    gradients = []
                    for param in shared_policy.parameters():
                        if param.grad is not None:
                            gradients.append(param.grad.clone().flatten())
                    
                    if gradients:
                        all_grads = torch.cat(gradients)
                        grad_numpy = all_grads.detach().cpu().numpy()
                        
                        # Check parameter count consistency
                        if expected_param_count is None:
                            expected_param_count = len(grad_numpy)
                        elif len(grad_numpy) != expected_param_count:
                            print(f"Warning: Parameter count mismatch. Expected {expected_param_count}, got {len(grad_numpy)}")
                            continue
                        
                        gradient_variances.append(grad_numpy)
                
                # Clean up
                optimizer.zero_grad()
            
            if gradient_variances and len(gradient_variances) > 1:
                try:
                    # Convert to numpy array - should work now with consistent shapes
                    gradients_array = np.array(gradient_variances)
                    print(f"  Gradient array shape: {gradients_array.shape}")
                    
                    # Calculate statistics
                    mean_gradient = np.mean(gradients_array, axis=0)
                    gradient_variance_per_param = np.var(gradients_array, axis=0)
                    
                    avg_mean_gradient = np.mean(np.abs(mean_gradient))
                    avg_gradient_variance = np.mean(gradient_variance_per_param)
                    std_gradient_variance = np.std(gradient_variance_per_param)
                    
                    results[method][chain_length] = {
                        'avg_mean_gradient': avg_mean_gradient,
                        'avg_gradient_variance': avg_gradient_variance,
                        'std_gradient_variance': std_gradient_variance,
                        'num_parameters': len(mean_gradient),
                        'successful_runs': len(gradient_variances)
                    }
                    print(f"  Successful runs: {len(gradient_variances)}/{NUM_RUNS}")
                    print(f"  Avg |mean gradient|: {avg_mean_gradient:.6f}")
                    print(f"  Avg gradient variance: {avg_gradient_variance:.6f} ± {std_gradient_variance:.6f}")
                    print(f"  Num parameters: {len(mean_gradient)}")
                    
                except Exception as e:
                    print(f"  Error processing gradients: {e}")
                    print(f"  Gradient shapes: {[g.shape for g in gradient_variances[:5]]}")
                    results[method][chain_length] = {
                        'avg_mean_gradient': 0.0,
                        'avg_gradient_variance': 0.0,
                        'std_gradient_variance': 0.0,
                        'num_parameters': 0,
                        'successful_runs': 0
                    }
            else:
                results[method][chain_length] = {
                    'avg_mean_gradient': 0.0,
                    'avg_gradient_variance': 0.0,
                    'std_gradient_variance': 0.0,
                    'num_parameters': 0,
                    'successful_runs': len(gradient_variances) if gradient_variances else 0
                }
                print(f"  Insufficient gradient data: {len(gradient_variances) if gradient_variances else 0} successful runs")
    
    # Print summary results
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print(f"{'Method':<20} {'Chain Length':<12} {'Avg |Mean Grad|':<15} {'Avg Grad Var':<15} {'Std Grad Var':<15} {'Success Rate':<12}")
    print("-" * 90)
    
    for method in METHODS:
        for chain_length in GRADIENT_CHAIN_LENGTHS:
            result = results[method][chain_length]
            success_rate = result.get('successful_runs', 0) / NUM_RUNS
            print(f"{method:<20} {chain_length:<12} {result['avg_mean_gradient']:<15.6f} {result['avg_gradient_variance']:<15.6f} {result['std_gradient_variance']:<15.6f} {success_rate:<12.2f}")
    
    return results

if __name__ == "__main__":
    print("Starting Differentiable Boids Gradient Variance Experiment")
    print(f"PyTorch version: {torch.__version__}")
    
    start_time = time.time()
    results = run_gradient_variance_experiment()
    end_time = time.time()
    
    print(f"\nExperiment completed in {end_time - start_time:.2f} seconds")