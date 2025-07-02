import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import time
from collections import defaultdict

# Placeholder classes for gradient estimators
class StochasticCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, uncertainty_in=None):
        # p can be [batch_size, n_categories] now
        if p.dim() == 1:
            p = p.unsqueeze(0)
        
        result = torch.multinomial(p, num_samples=1)  # [batch_size, 1]
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
        adjusted_ws = adjusted_ws / adjusted_ws.mean(dim=-1, keepdim=True).clamp(min=1e-10)
        
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
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class BoidsEnvironment:
    def __init__(self, num_agents, world_size, device='cuda', batch_size=32):
        self.num_agents = num_agents
        self.world_size = world_size
        self.device = device
        self.batch_size = batch_size
        
        self.max_speed = 5.0
        self.energy_decay = 0.7
        self.energy_transfer_rate = 0.05
        self.energy_transfer_distance = 3.0
        self.initial_energy = 100.0
        self.min_energy = 0.1
        self.death_energy = 10
        
        self.grid_resolution = 2.0
        self.exploration_bonus = 5.0
        self.coverage_decay = 0.99
        
        self.num_actions = 9
        self.action_vectors = torch.tensor([
            [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [-1, 1], [1, -1], [-1, -1]
        ], dtype=torch.float32, device=device)
        
        self.reset()
    
    def reset(self):
        self.positions = torch.rand(self.batch_size, self.num_agents, 2, device=self.device) * self.world_size
        self.velocities = torch.zeros(self.batch_size, self.num_agents, 2, device=self.device)
        self.energies = torch.full((self.batch_size, self.num_agents), self.initial_energy, device=self.device)
        self.alive = torch.ones(self.batch_size, self.num_agents, dtype=torch.bool, device=self.device)
        
        self.visited_cells = set()
        self.coverage_map = defaultdict(float)
        self.total_coverage = 0.0
        
        self.episode_rewards = torch.zeros(self.batch_size, device=self.device)
        self.episode_length = 0
        self.max_episode_length = 500
        self.agents_alive_history = []
        
        return self.get_state()
    
    def get_state(self):
        batch_size, num_agents = self.positions.shape[:2]
        
        basic_state = torch.cat([
            self.positions / self.world_size,
            self.velocities / self.max_speed,
            self.energies.unsqueeze(-1) / self.initial_energy,
            self.alive.unsqueeze(-1).float()
        ], dim=-1)
        
        distances = torch.cdist(self.positions, self.positions)
        distances = distances + torch.eye(num_agents, device=self.device) * 1000
        
        k = min(3, num_agents - 1)
        nearest_distances, _ = torch.topk(distances, k, dim=-1, largest=False)
        neighbor_features = nearest_distances / self.world_size
        neighbor_features = neighbor_features.reshape(batch_size, num_agents, k)
        
        state = torch.cat([basic_state, neighbor_features], dim=-1)
        return state.reshape(batch_size * num_agents, -1)
    
    def step(self, actions):
        actions = actions.reshape(self.batch_size, self.num_agents)
        
        action_vectors = self.action_vectors[actions]
        
        alive_mask = self.alive.unsqueeze(-1).float()
        self.velocities = 0.7 * self.velocities + 0.3 * action_vectors * self.max_speed * alive_mask
        
        vel_magnitudes = torch.norm(self.velocities, dim=-1, keepdim=True)
        vel_magnitudes = torch.clamp(vel_magnitudes, max=self.max_speed)
        safe_norm = torch.norm(self.velocities, dim=-1, keepdim=True) + 1e-8
        self.velocities = self.velocities / safe_norm * vel_magnitudes
        
        old_positions = self.positions.clone()
        self.positions += self.velocities * alive_mask
        self.positions = torch.remainder(self.positions, self.world_size)
        
        movement_cost = torch.norm(self.velocities, dim=-1) * self.energy_decay
        self.energies -= movement_cost * self.alive.float()
        # print(self.energies)
        self.energy_exchange()
        
        self.alive = (self.energies > self.death_energy) & self.alive
        self.energies = torch.clamp(self.energies, min=0.0)
        
        dead_positions = ~self.alive
        self.positions[dead_positions] = self.positions[dead_positions].detach()
        self.velocities[dead_positions] = 0.0
        
        rewards = self.compute_rewards(old_positions)
        rewards[~self.alive] = -1.0
        
        self.episode_length += 1
        agents_alive_count = self.alive.sum(dim=1).float()
        self.agents_alive_history.append(agents_alive_count.cpu().numpy().copy())
        
        done = (self.episode_length >= self.max_episode_length) or (agents_alive_count == 0).any()
        
        self.episode_rewards += rewards.sum(dim=1)
        next_state = self.get_state()
        
        return next_state, rewards.flatten(), done, {'agents_alive': agents_alive_count}
    
    def energy_exchange(self):
        batch_size, num_agents = self.positions.shape[:2]
        distances = torch.cdist(self.positions, self.positions)
        within_range = distances < self.energy_transfer_distance
        within_range = within_range & ~torch.eye(num_agents, dtype=torch.bool, device=self.device)
        
        alive_mask = self.alive.unsqueeze(-1) & self.alive.unsqueeze(-2)
        within_range = within_range & alive_mask
        
        for b in range(batch_size):
# In energy_exchange() method:
            # if b == 0:  # Log first batch only
            #     print(f"Energy stats - Min: {self.energies[b].min():.2f}, "
            #         f"Max: {self.energies[b].max():.2f}, "
            #         f"Mean: {self.energies[b].mean():.2f}")
            alive_agents = torch.where(self.alive[b])[0]
            if len(alive_agents) <= 1:
                continue
                
            for i in alive_agents:
                i_idx = i.item()
                neighbors = torch.where(within_range[b, i_idx])[0]
                if len(neighbors) > 0:
                    neighbor_energies = self.energies[b, neighbors]
                    agent_energy = self.energies[b, i_idx]
                    
                    energy_diff = agent_energy - neighbor_energies
                    transfer_to = neighbors[energy_diff > 10.0]
                    
                    if len(transfer_to) > 0 and agent_energy > 20.0:
                        max_transfer = min(agent_energy * 0.1, self.energy_transfer_rate * len(transfer_to))
                        transfer_per_agent = max_transfer / len(transfer_to)
                        
                        self.energies[b, i_idx] -= max_transfer
                        self.energies[b, transfer_to] += transfer_per_agent
        
        self.energies = torch.clamp(self.energies, min=0.0)
    
    def compute_rewards(self, old_positions):
        batch_size, num_agents = self.positions.shape[:2]
        rewards = torch.zeros(batch_size, num_agents, device=self.device)
        
        for b in range(batch_size):
            for a in range(num_agents):
                if not self.alive[b, a]:
                    continue
                    
                cell_x = int(self.positions[b, a, 0].item() // self.grid_resolution)
                cell_y = int(self.positions[b, a, 1].item() // self.grid_resolution)
                cell = (cell_x, cell_y)
                
                visit_count = self.coverage_map.get(cell, 0)
                exploration_reward = self.exploration_bonus * np.exp(-0.1 * visit_count)
                rewards[b, a] += exploration_reward
                
                self.coverage_map[cell] = visit_count + 1
        
        alive_mask = self.alive.float()
        
        energy_bonus = torch.clamp(self.energies / self.initial_energy, 0, 1) * 0.1 * alive_mask
        rewards += energy_bonus
        
        low_energy_penalty = torch.where(self.energies < 20.0, -0.2, 0.0) * alive_mask
        rewards += low_energy_penalty
        
        distances = torch.cdist(self.positions, self.positions)
        distances = distances + torch.eye(num_agents, device=self.device) * 1000
        
        alive_distances = distances.clone()
        alive_mask_2d = self.alive.unsqueeze(-1) & self.alive.unsqueeze(-2)
        alive_distances[~alive_mask_2d] = 1000.0
        
        min_distances, _ = torch.min(alive_distances, dim=-1)
        cooperation_bonus = torch.where(min_distances < 10.0, 0.1, 0.0) * alive_mask
        rewards += cooperation_bonus
        
        return rewards
    
    def get_coverage_stats(self):
        total_cells = len(self.coverage_map)
        avg_visits = np.mean(list(self.coverage_map.values())) if self.coverage_map else 0
        return total_cells, avg_visits
    
    def get_agents_alive_history(self):
        return np.array(self.agents_alive_history) if self.agents_alive_history else np.array([])

class BoidsAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, device='cuda'):
        self.device = device
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy_params = {
            'alpha': nn.Parameter(torch.tensor(1.0, device=device)),
            'beta': nn.Parameter(torch.tensor(1.0, device=device))
        }
    
    def compute_control_probs(self, state, policy_params):
        logits = self.policy_net(state)
        return F.softmax(logits, dim=-1)
    
    def compute_control_logits(self, state, policy_params):
        return self.policy_net(state)
    
    def sample_control_stoch(self, state, policy_params):
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
        probs = self.compute_control_probs(state, policy_params)
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
        logits = self.compute_control_logits(state, policy_params)
        gumbel_samples = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        control_idx = torch.argmax(gumbel_samples, dim=-1).item()
        probs = F.softmax(logits, dim=-1)
        log_prob = torch.log(probs[control_idx] + 1e-8)
        return control_idx, log_prob
    
    def sample_control(self, state, policy_params, method, tau=1.0):
        if method == "Learnable AUG":
            return self.sample_control_cat(state, policy_params, tau)
        elif method == "Gumbel":
            return self.sample_control_gumbel(state, policy_params, tau)
        elif method == "Stochastic AD":
            return self.sample_control_stoch(state, policy_params)
        else:
            return self.sample_control_custom(state, policy_params)

class BoidsTrainer:
    def __init__(self, device='cuda'):
        self.device = device
        
        self.num_agents = 100
        self.world_size = 1000.0
        self.batch_size = 16
        
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.num_episodes = 50
        self.episode_length = 300
        
        self.env = BoidsEnvironment(
            num_agents=self.num_agents,
            world_size=self.world_size,
            device=self.device,
            batch_size=self.batch_size
        )
        
        self.state_dim = 9
        self.action_dim = 9
        
        self.agent = BoidsAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        params = list(self.agent.policy_net.parameters()) + list(self.agent.policy_params.values())
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        
        self.training_rewards = []
        self.training_coverage = []
        self.training_agents_alive = []
    
    def train_episode(self, method="Stochastic AD"):
        state = self.env.reset()
        episode_log_probs = []
        episode_rewards = []
        episode_agents_alive = []
        
        for step in range(self.episode_length):
            actions = []
            log_probs = []
            
            for i in range(self.batch_size * self.num_agents):
                agent_state = state[i]
                action, log_prob = self.agent.sample_control(
                    agent_state, 
                    self.agent.policy_params, 
                    method
                )
                actions.append(action)
                log_probs.append(log_prob)
            
            actions = torch.tensor(actions, device=self.device)
            log_probs = torch.stack(log_probs)
            
            next_state, rewards, done, info = self.env.step(actions)
            
            episode_log_probs.append(log_probs)
            episode_rewards.append(rewards)
            episode_agents_alive.append(info['agents_alive'])
            
            state = next_state
            
            if done:
                break
        
        returns = []
        R = 0
        for r in reversed(episode_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.cat(returns)
        log_probs = torch.cat(episode_log_probs)
        
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = -(log_probs * returns).mean()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        total_reward = sum([r.sum().item() for r in episode_rewards])
        coverage_stats = self.env.get_coverage_stats()
        agents_alive_final = episode_agents_alive[-1] if episode_agents_alive else np.zeros(self.batch_size)
# After the training loop, before returning:
        agents_alive_history = self.env.get_agents_alive_history()
        survival_rate = np.mean(agents_alive_history, axis=1) if len(agents_alive_history) > 0 else []

        return total_reward, coverage_stats, policy_loss.item(), agents_alive_final, {
            'episode_length': step + 1,
            'survival_curve': agents_alive_history,
            'final_survival_rate': survival_rate[-1] if len(survival_rate) > 0 else 0
        }
        # return total_reward, coverage_stats, policy_loss.item(), agents_alive_final
    
    def train(self, method="Stochastic AD"):
        print(f"Training with method: {method}")
        print(f"Device: {self.device}")
        
        for episode in range(self.num_episodes):
            total_reward, (coverage_cells, avg_visits), loss, agents_alive,episode_stats = self.train_episode(method)
            
            self.training_rewards.append(total_reward)
            self.training_coverage.append(coverage_cells)
            self.training_agents_alive.append(agents_alive.mean())
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.training_rewards[-100:]) if len(self.training_rewards) >= 100 else np.mean(self.training_rewards)
                avg_coverage = np.mean(self.training_coverage[-100:]) if len(self.training_coverage) >= 100 else np.mean(self.training_coverage)
                avg_alive = np.mean(self.training_agents_alive[-100:]) if len(self.training_agents_alive) >= 100 else np.mean(self.training_agents_alive)
                
                print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, "
                      f"Coverage: {avg_coverage:.1f} cells, Agents Alive: {avg_alive:.1f}, Loss: {loss:.4f},"
      f"Final Survival Rate: {episode_stats['final_survival_rate']:.2f}")
        
        print("Training completed!")
    
    def evaluate(self, num_eval_episodes=10, method="Stochastic AD"):
        print(f"\nEvaluating with method: {method}")
        
        eval_rewards = []
        eval_coverage = []
        eval_agents_alive = []
        eval_survival_curves = []
        
        for episode in range(num_eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_survival = []
            
            for step in range(self.episode_length):
                actions = []
                
                for i in range(self.batch_size * self.num_agents):
                    agent_state = state[i]
                    action, _ = self.agent.sample_control(
                        agent_state, 
                        self.agent.policy_params, 
                        method,
                        tau=0.1  # Lower temperature for more deterministic behavior
                    )
                    actions.append(action)
                
                actions = torch.tensor(actions, device=self.device)
                next_state, rewards, done, info = self.env.step(actions)
                
                episode_reward += rewards.sum().item()
                episode_survival.append(info['agents_alive'].cpu().numpy())
                state = next_state
                
                if done:
                    break
            
            coverage_stats = self.env.get_coverage_stats()
            eval_rewards.append(episode_reward)
            eval_coverage.append(coverage_stats[0])
            final_alive = episode_survival[-1] if episode_survival else np.zeros(self.batch_size)
            eval_agents_alive.append(final_alive.mean())
            eval_survival_curves.append(np.array(episode_survival))
        
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        avg_coverage = np.mean(eval_coverage)
        std_coverage = np.std(eval_coverage)
        avg_alive = np.mean(eval_agents_alive)
        std_alive = np.std(eval_agents_alive)
        
        print(f"Evaluation Results:")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Average Coverage: {avg_coverage:.1f} ± {std_coverage:.1f} cells")
        print(f"Average Agents Alive: {avg_alive:.1f} ± {std_alive:.1f}")
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_coverage': avg_coverage,
            'std_coverage': std_coverage,
            'avg_agents_alive': avg_alive,
            'std_agents_alive': std_alive,
            'all_rewards': eval_rewards,
            'all_coverage': eval_coverage,
            'all_agents_alive': eval_agents_alive,
            'survival_curves': eval_survival_curves
        }

def run_experiment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    methods = ["Stochastic AD", "Learnable AUG", "Gumbel", "Fixed AUG"]
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running experiment with {method}")
        print(f"{'='*50}")
        
        trainer = BoidsTrainer(device=device)
        
        start_time = time.time()
        trainer.train(method=method)
        training_time = time.time() - start_time
        
        eval_results = trainer.evaluate(method=method)
        eval_results['training_time'] = training_time
        eval_results['training_rewards'] = trainer.training_rewards
        eval_results['training_coverage'] = trainer.training_coverage
        eval_results['training_agents_alive'] = trainer.training_agents_alive
        
        results[method] = eval_results
    
    # Placeholder for plotting function
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for method, result in results.items():
        print(f"\n{method}:")
        print(f"  Avg Reward: {result['avg_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Avg Coverage: {result['avg_coverage']:.1f} ± {result['std_coverage']:.1f}")
        print(f"  Avg Agents Alive: {result['avg_agents_alive']:.1f} ± {result['std_agents_alive']:.1f}")
        print(f"  Training Time: {result['training_time']:.1f}s")
    
    return results

if __name__ == "__main__":
    results = run_experiment()