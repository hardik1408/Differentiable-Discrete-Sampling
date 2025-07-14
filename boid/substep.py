import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict
from typing import Dict, Any, List, Optional

# Import AgentTorch components following the tutorial
from agent_torch.config import (
    ConfigBuilder,
    StateBuilder,
    AgentBuilder,
    PropertyBuilder,
    EnvironmentBuilder,
    SubstepBuilder,
    PolicyBuilder,
    TransitionBuilder
)
from agent_torch.core import Registry
from agent_torch.core.helpers import get_var
from agent_torch.core.substep import SubstepObservation, SubstepAction, SubstepTransition
from agent_torch.core.environment import envs

# Preserve your exact gradient estimators
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

# Policy Network (same as your original)
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Global state for neural network and parameters
class GlobalPolicyState:
    def __init__(self, state_dim, action_dim, hidden_dim=128, device='cuda'):
        self.device = device
        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        self.alpha = nn.Parameter(torch.tensor(1.0, device=device))
        self.beta = nn.Parameter(torch.tensor(1.0, device=device))
        self.uncertainty_tracker = None
        self.action_vectors = torch.tensor([
            [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [-1, 1], [1, -1], [-1, -1]
        ], dtype=torch.float32, device=device)
    
    def reset_uncertainty(self, batch_size, action_dim):
        self.uncertainty_tracker = torch.zeros(batch_size, action_dim, device=self.device)
    
    def compute_probs(self, state):
        logits = self.policy_net(state)
        return F.softmax(logits, dim=-1)
    
    def compute_logits(self, state):
        return self.policy_net(state)

# Global instance - following AgentTorch module pattern
global_policy = None
registry = Registry()

def init_global_policy(state_dim=10, action_dim=9, hidden_dim=128, device='cuda'):
    global global_policy
    global_policy = GlobalPolicyState(state_dim, action_dim, hidden_dim, device)
    return global_policy

# Substep implementations following AgentTorch tutorial patterns

@registry.register_substep("observe_neighbors", "policy")
class ObserveNeighbors(SubstepAction):
    def forward(self, state: Dict[str, Any], observations) -> Dict[str, Any]:
        positions = get_var(state, self.input_variables["positions"])
        velocities = get_var(state, self.input_variables["velocities"])
        energies = get_var(state, self.input_variables["energies"])
        alive = get_var(state, self.input_variables["alive"])
        
        # Get environment parameters from fixed arguments
        world_size = self.fixed_args["world_size"]
        max_speed = self.fixed_args["max_speed"]
        initial_energy = self.fixed_args["initial_energy"]
        
        batch_size, num_agents = positions.shape[:2]
        
        # Normalize state components
        pos_norm = positions / world_size
        vel_norm = velocities / max_speed
        energy_norm = energies.unsqueeze(-1) / initial_energy
        alive_state = alive.unsqueeze(-1).float()
        
        # Compute neighbor information
        pos_expanded = positions.unsqueeze(2)
        pos_expanded_t = positions.unsqueeze(1)
        all_distances = torch.norm(pos_expanded - pos_expanded_t, dim=-1)
        
        eye_mask = torch.eye(num_agents, device=positions.device).unsqueeze(0).bool()
        all_distances = all_distances.masked_fill(eye_mask, 1e6)
        
        alive_mask = alive.unsqueeze(1) & alive.unsqueeze(2)
        all_distances = all_distances.masked_fill(~alive_mask, 1e6)
        
        min_distances, closest_indices = torch.min(all_distances, dim=-1)
        valid_neighbors = min_distances < 1e5
        
        neighbor_info = torch.zeros(batch_size, num_agents, 4, device=positions.device)
        
        for b in range(batch_size):
            valid_mask = valid_neighbors[b] & alive[b]
            if valid_mask.any():
                valid_agents = torch.where(valid_mask)[0]
                closest_idx = closest_indices[b][valid_mask]
                
                neighbor_info[b, valid_agents, :2] = (
                    positions[b, closest_idx] - positions[b, valid_agents]
                ) / world_size
                neighbor_info[b, valid_agents, 2:4] = (
                    velocities[b, closest_idx] - velocities[b, valid_agents]
                ) / max_speed
        
        agent_state = torch.cat([pos_norm, vel_norm, energy_norm, alive_state, neighbor_info], dim=-1)
        flattened_state = agent_state.reshape(batch_size * num_agents, -1)
        
        return {
            self.output_variables[0]: flattened_state
        }

@registry.register_substep("decide_actions", "policy")
class DecideActions(SubstepAction):
    def forward(self, state: Dict[str, Any], observations) -> Dict[str, Any]:
        global global_policy
        
        # CRITICAL FIX: Get agent_state directly from observations with proper gradient flow
        if "agent_state" in observations:
            agent_state = observations["agent_state"]
        else:
            # Fallback: if agent_state not in observations, we have a problem
            print("WARNING: agent_state not found in observations!")
            positions = get_var(state, self.input_variables["positions"])
            batch_size, num_agents = positions.shape[:2]
            agent_state = torch.randn(batch_size * num_agents, 10, device=positions.device, requires_grad=True)
        
        method = self.fixed_args.get("method", "Stochastic AD")
        tau = self.fixed_args.get("tau", 1.0)
        
        # Get original shape info from positions to reconstruct batch dimensions
        positions = get_var(state, self.input_variables["positions"])
        batch_size, num_agents = positions.shape[:2]
        
        # CRITICAL: Ensure agent_state requires gradients
        if not agent_state.requires_grad:
            print(f"WARNING: agent_state does not require gradients! Method: {method}")
            agent_state = agent_state.detach().requires_grad_(True)
        
        # Initialize uncertainty tracker if needed
        if global_policy.uncertainty_tracker is None:
            global_policy.reset_uncertainty(batch_size * num_agents, 9)
        
        # CRITICAL: Check that global_policy.policy_net parameters require gradients
        policy_requires_grad = any(p.requires_grad for p in global_policy.policy_net.parameters())
        if not policy_requires_grad:
            print("WARNING: Policy network parameters do not require gradients!")
        
        # Compute action probabilities using your estimators
        if method == "Learnable AUG":
            action_probs = self._sample_learnable_aug(agent_state)
        elif method == "Gumbel":
            action_probs = self._sample_gumbel(agent_state, tau)
        elif method == "Fixed AUG":
            action_probs = self._sample_stochastic_categorical(agent_state)
        else:  # "Stochastic AD"
            action_probs = self._sample_stochastic_ad(agent_state)
        
        # CRITICAL: Verify action_probs has gradients
        if not action_probs.requires_grad:
            print(f"WARNING: action_probs does not require gradients! Method: {method}")
        
        # Convert to action vectors
        action_vectors = torch.einsum('ba,ad->bd', action_probs, global_policy.action_vectors)
        action_vectors = action_vectors.reshape(batch_size, num_agents, 2)
        
        # CRITICAL: Verify final action_vectors has gradients
        if not action_vectors.requires_grad:
            print(f"WARNING: action_vectors does not require gradients! Method: {method}")
        
        return {
            self.output_variables[0]: action_vectors
        }
    
    def _sample_stochastic_ad(self, state):
        # CRITICAL: Ensure gradient flow through policy network
        probs = global_policy.compute_probs(state)
        
        # Check if probs has gradients
        if not probs.requires_grad:
            print("WARNING: probs from policy network do not require gradients in Stochastic AD!")
        
        # CRITICAL FIX: Instead of converting to hard one-hot, use the original probs
        # Your Categorical.apply already implements the proper backward pass
        # So we can just return the original probabilities with your custom gradients
        sample_indices = Categorical.apply(probs)
        
        # Use straight-through estimator: forward pass uses hard selection, backward uses soft probs
        # This preserves your custom gradient computation while maintaining differentiability
        action_probs = probs + (torch.zeros_like(probs).scatter_(1, sample_indices.long(), 1.0) - probs).detach()
        
        # Verify gradients are preserved
        if not action_probs.requires_grad:
            print("WARNING: action_probs lost gradients in Stochastic AD!")
        
        return action_probs
    
    def _sample_stochastic_categorical(self, state):
        probs = global_policy.compute_probs(state)
        
        if not probs.requires_grad:
            print("WARNING: probs from policy network do not require gradients in Fixed AUG!")
        
        sample_indices, uncertainty = StochasticCategorical.apply(probs, global_policy.uncertainty_tracker)
        global_policy.uncertainty_tracker = uncertainty
        
        # CRITICAL FIX: Use straight-through estimator to preserve gradients
        # Forward: hard one-hot, Backward: your custom gradients through probs
        hard_onehot = torch.zeros_like(probs).scatter_(1, sample_indices.long(), 1.0)
        action_probs = probs + (hard_onehot - probs).detach()
        
        if not action_probs.requires_grad:
            print("WARNING: action_probs lost gradients in Fixed AUG!")
        
        return action_probs
    
    def _sample_learnable_aug(self, state):
        probs = global_policy.compute_probs(state)
        
        if not probs.requires_grad:
            print("WARNING: probs from policy network do not require gradients in Learnable AUG!")
        
        sample_indices, uncertainty = LearnableStochasticCategorical.apply(
            probs, global_policy.alpha, global_policy.beta, global_policy.uncertainty_tracker
        )
        global_policy.uncertainty_tracker = uncertainty
        
        # CRITICAL FIX: Use straight-through estimator to preserve gradients
        # This ensures your custom backward pass in LearnableStochasticCategorical is used
        hard_onehot = torch.zeros_like(probs).scatter_(1, sample_indices.long(), 1.0)
        action_probs = probs + (hard_onehot - probs).detach()
        
        if not action_probs.requires_grad:
            print("WARNING: action_probs lost gradients in Learnable AUG!")
        
        return action_probs
    
    def _sample_gumbel(self, state, tau=1.0):
        logits = global_policy.compute_logits(state)
        
        if not logits.requires_grad:
            print("WARNING: logits from policy network do not require gradients in Gumbel!")
        
        # CRITICAL: Gumbel-Softmax naturally preserves gradients!
        action_probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        
        if not action_probs.requires_grad:
            print("WARNING: action_probs lost gradients in Gumbel!")
        
        return action_probs

@registry.register_substep("update_boid_state", "transition")
class UpdateBoidState(SubstepTransition):
    def forward(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        positions = get_var(state, self.input_variables["positions"])
        velocities = get_var(state, self.input_variables["velocities"])
        energies = get_var(state, self.input_variables["energies"])
        alive = get_var(state, self.input_variables["alive"])
        
        # Get parameters from fixed and learnable arguments
        world_size = self.fixed_args["world_size"]
        max_speed = self.fixed_args["max_speed"]
        initial_energy = self.fixed_args["initial_energy"]
        energy_decay = self.learnable_args["energy_decay"]
        death_threshold = self.learnable_args["death_threshold"]
        
        action_vectors = action["action_vectors"]
        
        # Apply alive mask to actions
        alive_mask = alive.unsqueeze(-1).float()
        masked_actions = action_vectors * alive_mask
        
        # Update velocities
        new_velocities = 0.8 * velocities + 0.2 * masked_actions * max_speed
        
        # Clamp velocity magnitude
        vel_magnitude = torch.norm(new_velocities, dim=-1, keepdim=True)
        vel_magnitude_clamped = torch.clamp(vel_magnitude, max=max_speed)
        new_velocities = new_velocities * (vel_magnitude_clamped / (vel_magnitude + 1e-8))
        
        # Update positions with wrapping
        new_positions = positions + new_velocities * alive_mask
        new_positions = new_positions % world_size
        
        # Update energies
        movement_cost = torch.norm(new_velocities, dim=-1) * energy_decay
        new_energies = energies - movement_cost * alive.float()
        
        # Update alive status
        death_prob = torch.sigmoid((death_threshold - new_energies) * 10.0)
        alive_float = alive.float() * (1.0 - death_prob)
        new_alive = (new_energies >= death_threshold) & alive
        
        # Compute rewards
        rewards = self._compute_rewards(new_positions, new_energies, alive_float, world_size, initial_energy)
        
        return {
            self.output_variables[0]: new_positions,
            self.output_variables[1]: new_velocities,
            self.output_variables[2]: new_energies,
            self.output_variables[3]: new_alive,
            self.output_variables[4]: rewards
        }
    
    def _compute_rewards(self, positions, energies, alive_float, world_size, initial_energy):
        # Energy reward
        energy_reward = torch.clamp(energies / initial_energy, 0, 1) * 0.1
        
        # Distance-based cooperation reward
        pos_expanded = positions.unsqueeze(2)
        pos_expanded_t = positions.unsqueeze(1)
        distances = torch.norm(pos_expanded - pos_expanded_t, dim=-1)
        
        # Mask self-distances and dead agents
        num_agents = positions.shape[1]
        eye_mask = torch.eye(num_agents, device=positions.device).unsqueeze(0).bool()
        distances = distances.masked_fill(eye_mask, 1000.0)
        
        alive_mask = alive_float.unsqueeze(1) * alive_float.unsqueeze(2)
        distances = distances + (1.0 - alive_mask) * 1000.0
        
        min_distances, _ = torch.min(distances, dim=-1)
        cooperation_reward = torch.exp(-min_distances / 20.0) * 0.05
        
        # Social reward
        nearby_mask = (distances < 50.0).float() * alive_mask
        neighbor_count = nearby_mask.sum(dim=-1)
        social_reward = torch.tanh(neighbor_count * 0.1) * 0.02
        
        total_reward = energy_reward + cooperation_reward + social_reward
        return total_reward * alive_float

# Configuration Builder following AgentTorch tutorial
def create_boids_config(num_agents=10, num_episodes=51, num_steps_per_episode=100, 
                       batch_size=10, world_size=100.0, device='cuda', method="Stochastic AD"):
    
    # Initialize global policy
    init_global_policy(state_dim=10, action_dim=9, device=device)
    
    config = ConfigBuilder()
    
    # 1. Simulation Metadata (REQUIRED PARAMETERS)
    metadata = {
        "num_agents": num_agents,
        "num_episodes": num_episodes,
        "num_steps_per_episode": num_steps_per_episode,
        "num_substeps_per_step": 1,
        "device": device,
        "calibration": True  # Enable calibration for learnable parameters
    }
    config.set_metadata(metadata)
    
    # 2. State Configuration
    state_builder = StateBuilder()
    
    # Add boids agents
    agent_builder = AgentBuilder("boids", num_agents)
    
    # Agent properties
    positions = PropertyBuilder("positions")\
        .set_dtype("float")\
        .set_shape([batch_size, num_agents, 2])\
        .set_value(torch.rand(batch_size, num_agents, 2, device=device) * world_size)
    
    velocities = PropertyBuilder("velocities")\
        .set_dtype("float")\
        .set_shape([batch_size, num_agents, 2])\
        .set_value(torch.zeros(batch_size, num_agents, 2, device=device))
    
    energies = PropertyBuilder("energies")\
        .set_dtype("float")\
        .set_shape([batch_size, num_agents])\
        .set_value(torch.full((batch_size, num_agents), 100.0, device=device))
    
    alive = PropertyBuilder("alive")\
        .set_dtype("bool")\
        .set_shape([batch_size, num_agents])\
        .set_value(torch.ones(batch_size, num_agents, dtype=torch.bool, device=device))
    
    agent_builder.add_property(positions)
    agent_builder.add_property(velocities)
    agent_builder.add_property(energies)
    agent_builder.add_property(alive)
    
    state_builder.add_agent("boids", agent_builder)
    
    # Add environment variables
    env_builder = EnvironmentBuilder()
    
    world_size_env = PropertyBuilder("world_size")\
        .set_dtype("float")\
        .set_shape([])\
        .set_value(world_size)
    
    max_speed_env = PropertyBuilder("max_speed")\
        .set_dtype("float")\
        .set_shape([])\
        .set_value(5.0)
    
    initial_energy_env = PropertyBuilder("initial_energy")\
        .set_dtype("float")\
        .set_shape([])\
        .set_value(100.0)
    
    env_builder.add_variable(world_size_env)
    env_builder.add_variable(max_speed_env)
    env_builder.add_variable(initial_energy_env)
    
    state_builder.set_environment(env_builder)
    config.set_state(state_builder.to_dict())
    
    # 3. Substep Configuration
    boids_substep = SubstepBuilder("BoidsStep", "Boids simulation with gradient estimators")
    boids_substep.add_active_agent("boids")
    boids_substep.config["observation"] = {"boids": None}  # Following tutorial pattern
    
    # Add observation/policy for neighbor detection
    observe_policy = PolicyBuilder()
    
    # Learnable parameters for observation
    world_size_param = PropertyBuilder.create_argument(
        name="World size parameter",
        value=world_size,
        learnable=False
    ).config
    
    max_speed_param = PropertyBuilder.create_argument(
        name="Max speed parameter",
        value=5.0,
        learnable=False
    ).config
    
    initial_energy_param = PropertyBuilder.create_argument(
        name="Initial energy parameter",
        value=100.0,
        learnable=False
    ).config
    
    observe_policy.add_policy(
        "observe_neighbors",
        "ObserveNeighbors",
        {
            "positions": "agents/boids/positions",
            "velocities": "agents/boids/velocities",
            "energies": "agents/boids/energies",
            "alive": "agents/boids/alive"
        },
        ["agent_state"],
        {
            "world_size": world_size_param,
            "max_speed": max_speed_param,
            "initial_energy": initial_energy_param
        }
    )
    
    # Add action decision policy
    method_param = PropertyBuilder.create_argument(
        name="Gradient estimation method",
        value=method,
        learnable=False
    ).config
    
    tau_param = PropertyBuilder.create_argument(
        name="Gumbel temperature",
        value=1.0,
        learnable=False
    ).config
    
    observe_policy.add_policy(
        "decide_actions",
        "DecideActions",
        {"positions": "agents/boids/positions"},
        ["action_vectors"],
        {"method": method_param, "tau": tau_param}
    )
    
    boids_substep.set_policy("boids", observe_policy)
    
    # Add transition
    transition = TransitionBuilder()
    
    # Learnable environment parameters for transition
    energy_decay_param = PropertyBuilder.create_argument(
        name="Energy decay rate",
        value=0.5,
        learnable=True
    ).config
    
    death_threshold_param = PropertyBuilder.create_argument(
        name="Death threshold",
        value=10.0,
        learnable=True
    ).config
    
    transition.add_transition(
        "update_boid_state",
        "UpdateBoidState",
        {
            "positions": "agents/boids/positions",
            "velocities": "agents/boids/velocities",
            "energies": "agents/boids/energies",
            "alive": "agents/boids/alive"
        },
        ["positions", "velocities", "energies", "alive", "rewards"],
        {
            "world_size": world_size_param,
            "max_speed": max_speed_param,
            "energy_decay": energy_decay_param,
            "death_threshold": death_threshold_param,
            "initial_energy": initial_energy_param
        }
    )
    
    boids_substep.set_transition(transition)
    config.add_substep("0", boids_substep)
    
    return config

# Simple population class following tutorial patterns
class BoidsPopulation:
    def __init__(self, num_agents=10, batch_size=10, world_size=100.0, device='cuda'):
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.world_size = world_size
        self.device = device

# Experiment Runner following AgentTorch patterns
class AgentTorchBoidsExperiment:
    def __init__(self, device='cuda'):
        self.device = device
        self.num_agents = 10
        self.world_size = 100.0
        self.batch_size = 10
        self.learning_rate = 3e-4
        self.num_episodes = 51
        self.episode_length = 100
        
        self.training_rewards = []
        self.max_rewards = []
        self.final_agents_alive = []
        self.coverage_ratios = []
        self.avg_energies = []
        self.gradient_norms = []
        
        # Following tutorial pattern - create module-level registry
        self.registry = registry
        
    def setup_simulation(self, method="Stochastic AD"):
        # Create config following tutorial
        config = create_boids_config(
            num_agents=self.num_agents,
            num_episodes=self.num_episodes,
            num_steps_per_episode=self.episode_length,
            batch_size=self.batch_size,
            world_size=self.world_size,
            device=self.device,
            method=method
        )
        
        # Create population
        population = BoidsPopulation(
            num_agents=self.num_agents,
            batch_size=self.batch_size,
            world_size=self.world_size,
            device=self.device
        )
        
        # AgentTorch expects a module-like object with __path__ attribute for models
        # Let's skip envs.create and go directly to manual runner since it's more reliable
        print("Using manual runner for better control and reliability")
        self.runner = self._create_manual_runner(config, population, method)
        
        # Setup optimizers
        global global_policy
        self.policy_optimizer = torch.optim.Adam(global_policy.policy_net.parameters(), lr=self.learning_rate)
        self.param_optimizer = torch.optim.Adam([global_policy.alpha, global_policy.beta], lr=self.learning_rate * 0.1)
        
        return self.runner
    
    def _create_manual_runner(self, config, population, method):
        """Manual runner when AgentTorch's envs.create fails"""
        class ManualRunner:
            def __init__(self, config, population, method_name):
                self.config_dict = config.to_dict()
                self.population = population
                self.method = method_name
                self.state = None
                self.step_count = 0
                
            def reset(self):
                # Initialize state from config
                config_state = self.config_dict["state"]
                self.state = {
                    "agents": {
                        "boids": {
                            "positions": config_state["agents"]["boids"]["properties"]["positions"]["value"].clone().detach().requires_grad_(True),
                            "velocities": config_state["agents"]["boids"]["properties"]["velocities"]["value"].clone().detach().requires_grad_(True),
                            "energies": config_state["agents"]["boids"]["properties"]["energies"]["value"].clone().detach().requires_grad_(True),
                            "alive": config_state["agents"]["boids"]["properties"]["alive"]["value"].clone().detach()
                        }
                    },
                    "environment": {
                        "world_size": config_state["environment"]["world_size"]["value"],
                        "max_speed": config_state["environment"]["max_speed"]["value"],
                        "initial_energy": config_state["environment"]["initial_energy"]["value"]
                    }
                }
                self.step_count = 0
                
            def step(self, num_steps):
                global global_policy
                
                for _ in range(num_steps):
                    # Execute substeps following the configuration
                    
                    # 1. Observation phase - ObserveNeighbors
                    observe_input_vars = {
                        "positions": "positions",
                        "velocities": "velocities",
                        "energies": "energies",
                        "alive": "alive"
                    }
                    observe_output_vars = ["agent_state"]
                    observe_args = {
                        "learnable": {},  # No learnable parameters for observation
                        "fixed": {
                            "world_size": self.state["environment"]["world_size"],
                            "max_speed": self.state["environment"]["max_speed"],
                            "initial_energy": self.state["environment"]["initial_energy"]
                        }
                    }
                    
                    observe_substep = ObserveNeighbors(
                        config=self.config_dict,
                        input_variables=observe_input_vars,
                        output_variables=observe_output_vars,
                        arguments=observe_args
                    )
                    
                    obs_state = {
                        "positions": self.state["agents"]["boids"]["positions"],
                        "velocities": self.state["agents"]["boids"]["velocities"],
                        "energies": self.state["agents"]["boids"]["energies"],
                        "alive": self.state["agents"]["boids"]["alive"]
                    }
                    
                    observations = observe_substep.forward(obs_state, None)
                    
                    # 2. Policy phase - DecideActions
                    policy_input_vars = {"positions": "positions"}
                    policy_output_vars = ["action_vectors"]
                    policy_args = {
                        "learnable": {},  # No learnable parameters for policy decision
                        "fixed": {
                            "method": self.method, 
                            "tau": 1.0
                        }
                    }
                    
                    policy_substep = DecideActions(
                        config=self.config_dict,
                        input_variables=policy_input_vars,
                        output_variables=policy_output_vars,
                        arguments=policy_args
                    )
                    
                    actions = policy_substep.forward(obs_state, observations)
                    
                    # 3. Transition phase - UpdateBoidState
                    transition_input_vars = {
                        "positions": "positions",
                        "velocities": "velocities",
                        "energies": "energies",
                        "alive": "alive"
                    }
                    transition_output_vars = ["positions", "velocities", "energies", "alive", "rewards"]
                    transition_args = {
                        "learnable": {
                            "energy_decay": 0.5,
                            "death_threshold": 10.0
                        },
                        "fixed": {
                            "world_size": self.state["environment"]["world_size"],
                            "max_speed": self.state["environment"]["max_speed"],
                            "initial_energy": self.state["environment"]["initial_energy"]
                        }
                    }
                    
                    transition_substep = UpdateBoidState(
                        config=self.config_dict,
                        input_variables=transition_input_vars,
                        output_variables=transition_output_vars,
                        arguments=transition_args
                    )
                    
                    new_state = transition_substep.forward(obs_state, actions)
                    
                    # Update state
                    self.state["agents"]["boids"]["positions"] = new_state["positions"]
                    self.state["agents"]["boids"]["velocities"] = new_state["velocities"]
                    self.state["agents"]["boids"]["energies"] = new_state["energies"]
                    self.state["agents"]["boids"]["alive"] = new_state["alive"]
                    self.state["agents"]["boids"]["rewards"] = new_state["rewards"]
                    
                    self.step_count += 1
                    
                    actions = policy_substep.forward(obs_state, observations)
                    
                    # # 3. Transition phase - UpdateBoidState
                    # transition_substep = UpdateBoidState()
                    # transition_substep.input_variables = {
                    #     "positions": "positions",
                    #     "velocities": "velocities",
                    #     "energies": "energies",
                    #     "alive": "alive"
                    # }
                    # transition_substep.output_variables = ["positions", "velocities", "energies", "alive", "rewards"]
                    # transition_substep.learnable_args = {
                    #     "world_size": self.state["environment"]["world_size"],
                    #     "max_speed": self.state["environment"]["max_speed"],
                    #     "energy_decay": 0.5,
                    #     "death_threshold": 10.0,
                    #     "initial_energy": self.state["environment"]["initial_energy"]
                    # }
                    
                    # new_state = transition_substep.forward(obs_state, actions)
                    
                    # # Update state
                    # self.state["agents"]["boids"]["positions"] = new_state["positions"]
                    # self.state["agents"]["boids"]["velocities"] = new_state["velocities"]
                    # self.state["agents"]["boids"]["energies"] = new_state["energies"]
                    # self.state["agents"]["boids"]["alive"] = new_state["alive"]
                    # self.state["agents"]["boids"]["rewards"] = new_state["rewards"]
                    
                    # self.step_count += 1
        
        return ManualRunner(config, population, method)
    
    def train_episode(self, method="Stochastic AD"):
        global global_policy
        global_policy.reset_uncertainty(self.batch_size * self.num_agents, 9)
        
        # Reset simulation
        self.runner.reset()
        
        total_reward = 0.0
        max_reward = 0.0
        
        # Track cumulative coverage throughout the episode
        grid_size = 20
        cell_size = self.world_size / grid_size
        visited_cells = set()
        
        for step in range(self.episode_length):
            # Execute one simulation step
            self.runner.step(1)
            
            # Get rewards from state
            state = self.runner.state
            if "rewards" in state.get("agents", {}).get("boids", {}):
                rewards = state["agents"]["boids"]["rewards"]
                step_reward = rewards.sum()
                total_reward += step_reward
                max_reward = max(max_reward, step_reward.item())
                
                # Track coverage during this step
                positions = state["agents"]["boids"]["positions"]
                alive = state["agents"]["boids"]["alive"]
                
                # Add current positions to visited cells
                alive_expanded = alive.unsqueeze(-1).expand_as(positions)
                alive_positions = positions[alive_expanded].view(-1, 2)
                
                if alive_positions.shape[0] > 0:
                    grid_x = torch.floor(alive_positions[:, 0] / cell_size).long()
                    grid_y = torch.floor(alive_positions[:, 1] / cell_size).long()
                    
                    grid_x = torch.clamp(grid_x, 0, grid_size - 1)
                    grid_y = torch.clamp(grid_y, 0, grid_size - 1)
                    
                    cell_ids = (grid_x * grid_size + grid_y).tolist()
                    visited_cells.update(cell_ids)
                
                # Check if any agents are alive
                if alive.sum() == 0:
                    print(f"All agents died at step {step}")
                    break
        
        # Get final statistics
        final_state = self.runner.state
        alive_final = final_state["agents"]["boids"]["alive"]
        positions_final = final_state["agents"]["boids"]["positions"]
        energies_final = final_state["agents"]["boids"]["energies"]
        
        final_agents_alive = alive_final.sum().item()
        
        # Use cumulative coverage instead of just final position coverage
        cumulative_coverage_ratio = len(visited_cells) / (grid_size * grid_size)
        
        avg_energy = self._compute_avg_energy(energies_final, alive_final)
        
        # Compute loss and backpropagate
        loss = -total_reward
        
        self.policy_optimizer.zero_grad()
        if method == "Learnable AUG":
            self.param_optimizer.zero_grad()
        
        loss.backward()
        
        # Gradient clipping
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy.policy_net.parameters(), 1.0)
        
        if method == "Learnable AUG":
            param_grad_norm = torch.nn.utils.clip_grad_norm_([global_policy.alpha, global_policy.beta], 1.0)
            total_grad_norm = policy_grad_norm + param_grad_norm
        else:
            total_grad_norm = policy_grad_norm
        
        self.gradient_norms.append(total_grad_norm.item())
        
        # Optimizer steps
        self.policy_optimizer.step()
        if method == "Learnable AUG":
            self.param_optimizer.step()
        
        return total_reward.item(), loss.item(), max_reward, final_agents_alive, cumulative_coverage_ratio, avg_energy
    
    def _compute_coverage(self, positions, alive):
        """
        Compute the percentage of grid cells that have been visited by alive agents.
        Returns a value between 0.0 and 1.0 representing the fraction of area covered.
        """
        grid_size = 20
        cell_size = self.world_size / grid_size
        
        # Get positions of alive agents across all batches
        # positions shape: [batch_size, num_agents, 2]
        # alive shape: [batch_size, num_agents]
        
        # Flatten batch and agent dimensions for alive agents only
        alive_expanded = alive.unsqueeze(-1).expand_as(positions)  # [batch_size, num_agents, 2]
        alive_positions = positions[alive_expanded].view(-1, 2)  # [num_alive_agents_total, 2]
        
        if alive_positions.shape[0] == 0:
            return 0.0
        
        # Convert positions to grid coordinates
        grid_x = torch.floor(alive_positions[:, 0] / cell_size).long()
        grid_y = torch.floor(alive_positions[:, 1] / cell_size).long()
        
        # Clamp to valid grid range
        grid_x = torch.clamp(grid_x, 0, grid_size - 1)
        grid_y = torch.clamp(grid_y, 0, grid_size - 1)
        
        # Convert 2D coordinates to unique cell IDs
        cell_ids = grid_x * grid_size + grid_y
        
        # Count unique cells visited
        unique_cells = torch.unique(cell_ids)
        coverage_ratio = len(unique_cells) / (grid_size * grid_size)
        
        return coverage_ratio
    
    def _compute_avg_energy(self, energies, alive):
        """
        Compute average energy of alive agents across all batches.
        """
        # energies shape: [batch_size, num_agents]
        # alive shape: [batch_size, num_agents]
        
        alive_energies = energies[alive]  # Get energies of alive agents only
        if alive_energies.numel() > 0:
            return alive_energies.mean().item()
        return 0.0
    
    def train(self, method="Stochastic AD"):
        print(f"Training with method: {method}")
        print(f"Device: {self.device}")
        
        # Setup simulation
        self.setup_simulation(method)
        
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
                global global_policy
                global_policy.reset_uncertainty(self.batch_size * self.num_agents, 9)
                
                # Reset simulation
                self.runner.reset()
                
                episode_reward = 0.0
                max_reward = 0.0
                
                # Track cumulative coverage for evaluation
                grid_size = 20
                cell_size = self.world_size / grid_size
                visited_cells = set()
                
                for step in range(self.episode_length):
                    # Execute one simulation step
                    self.runner.step(1)
                    
                    # Get rewards from current state
                    state = self.runner.state
                    if "rewards" in state.get("agents", {}).get("boids", {}):
                        rewards = state["agents"]["boids"]["rewards"]
                        step_reward = rewards.sum().item()
                        episode_reward += step_reward
                        max_reward = max(max_reward, step_reward)
                        
                        # Track coverage during this step
                        positions = state["agents"]["boids"]["positions"]
                        alive = state["agents"]["boids"]["alive"]
                        
                        # Add current positions to visited cells
                        alive_expanded = alive.unsqueeze(-1).expand_as(positions)
                        alive_positions = positions[alive_expanded].view(-1, 2)
                        
                        if alive_positions.shape[0] > 0:
                            grid_x = torch.floor(alive_positions[:, 0] / cell_size).long()
                            grid_y = torch.floor(alive_positions[:, 1] / cell_size).long()
                            
                            grid_x = torch.clamp(grid_x, 0, grid_size - 1)
                            grid_y = torch.clamp(grid_y, 0, grid_size - 1)
                            
                            cell_ids = (grid_x * grid_size + grid_y).tolist()
                            visited_cells.update(cell_ids)
                        
                        # Check if any agents are alive
                        if alive.sum() == 0:
                            break
                
                # Get final statistics
                final_state = self.runner.state
                alive_final = final_state["agents"]["boids"]["alive"]
                energies_final = final_state["agents"]["boids"]["energies"]
                
                final_agents_alive = alive_final.sum().item()
                cumulative_coverage_ratio = len(visited_cells) / (grid_size * grid_size)
                avg_energy = self._compute_avg_energy(energies_final, alive_final)
                
                eval_rewards.append(episode_reward)
                eval_max_rewards.append(max_reward)
                eval_final_agents_alive.append(final_agents_alive)
                eval_coverage_ratios.append(cumulative_coverage_ratio)
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

# Main experiment function following AgentTorch patterns
def run_agenttorch_experiment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test gradient flow first
    print("Testing gradient flow...")
    experiment = AgentTorchBoidsExperiment(device=device)
    experiment.setup_simulation("Fixed AUG")
    
    # Execute one step to test gradient flow
    experiment.runner.reset()
    experiment.runner.step(1)
    
    state = experiment.runner.state
    if "rewards" in state.get("agents", {}).get("boids", {}):
        rewards = state["agents"]["boids"]["rewards"]
        loss = -rewards.sum()
        
        experiment.policy_optimizer.zero_grad()
        loss.backward()
        
        total_grad_norm = 0
        global global_policy
        for param in global_policy.policy_net.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
        
        print(f"Total gradient norm: {total_grad_norm:.6f}")
        if total_grad_norm >= 1e-6:
            print("Gradients are flowing correctly")
        else:
            print("Gradients are not flowing")
    
    methods = ["Stochastic AD", "Fixed AUG", "Learnable AUG", "Gumbel"]
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running experiment with {method}")
        print(f"{'='*50}")
        
        experiment = AgentTorchBoidsExperiment(device=device)
        
        start_time = time.time()
        experiment.train(method=method)
        training_time = time.time() - start_time
        
        eval_results = experiment.evaluate(method=method)
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

# Alternative simplified interface that matches your original API
class AgentTorchBoidsRunner:
    """Simplified interface that closely matches your original ExperimentRunner API"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.experiment = AgentTorchBoidsExperiment(device=device)
        
    def train(self, method="Stochastic AD"):
        """Train using the specified gradient estimation method"""
        return self.experiment.train(method=method)
    
    def evaluate(self, num_eval_episodes=10, method="Stochastic AD"):
        """Evaluate the trained model"""
        return self.experiment.evaluate(num_eval_episodes, method)
    
    @property
    def training_rewards(self):
        return self.experiment.training_rewards
    
    @property
    def max_rewards(self):
        return self.experiment.max_rewards
    
    @property
    def final_agents_alive(self):
        return self.experiment.final_agents_alive
    
    @property
    def coverage_ratios(self):
        return self.experiment.coverage_ratios
    
    @property
    def avg_energies(self):
        return self.experiment.avg_energies
    
    @property
    def gradient_norms(self):
        return self.experiment.gradient_norms

# Usage example following AgentTorch tutorial patterns
def run_boids_simulation():
    """Run boids simulation following AgentTorch tutorial patterns."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create and run simulation for each method
    methods = ["Stochastic AD", "Fixed AUG", "Learnable AUG", "Gumbel"]
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running Boids simulation with {method}")
        print(f"{'='*50}")
        
        # Create simulation runner following tutorial pattern
        runner = AgentTorchBoidsRunner(device=device)
        
        # Train the model
        start_time = time.time()
        runner.train(method=method)
        training_time = time.time() - start_time
        
        # Evaluate the model
        eval_results = runner.evaluate(method=method)
        eval_results['training_time'] = training_time
        
        results[method] = eval_results
        
        # Print episode statistics following tutorial pattern
        print(f"Method {method} completed:")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Final performance: Max Reward {eval_results['avg_max_reward']:.2f}")
    
    return results

if __name__ == "__main__":
    # Run the experiments following AgentTorch patterns
    results = run_agenttorch_experiment()
    
    # Alternative: Run with tutorial-style interface
    # results = run_boids_simulation()