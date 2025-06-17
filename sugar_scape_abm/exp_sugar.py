import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
# Custom estimators
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
    
@dataclass
class ExperimentConfig:
    grid_size: int = 50
    num_agents: int = 100
    gradient_chain_length: int = 3
    sugar_growth_rate: float = 0.1
    agent_vision: int = 3
    learning_rate: float = 0.01
    num_episodes: int = 1000
    max_steps_per_episode: int = 200
    tau: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, chain_length: int, hidden_dim: int = 64):
        super().__init__()
        layers = []
        cur_dim = input_dim
        for i in range(chain_length):
            if i == chain_length - 1:
                layers.append(nn.Linear(cur_dim, output_dim))
            else:
                layers.extend([nn.Linear(cur_dim, hidden_dim), nn.ReLU()])
                cur_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.alpha = nn.Parameter(torch.ones(output_dim) * 0.5)
        self.beta = nn.Parameter(torch.ones(output_dim) * 0.5)

    def forward(self, x):
        return self.net(x)

    def get_policy_params(self):
        return {'alpha': self.alpha, 'beta': self.beta}

class SugarscapeEnvironment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.sugar = np.random.uniform(0, 4, (self.config.grid_size,)*2)
        self.max_sugar = np.random.uniform(2, 4, (self.config.grid_size,)*2)
        self.pollution = np.zeros_like(self.sugar)
        return self.get_state()

    def get_state(self):
        return {'sugar': self.sugar.copy(), 'pollution': self.pollution.copy()}

    def step(self):
        self.sugar = np.minimum(self.max_sugar, self.sugar + self.config.sugar_growth_rate)
        self.pollution *= 0.99
        return self.get_state()

    def harvest(self, x: int, y: int) -> float:
        x %= self.config.grid_size; y %= self.config.grid_size
        val = self.sugar[x, y]
        self.sugar[x, y] = 0
        self.pollution[x, y] += 0.1
        return val

class SugarscapeAgent:
    def __init__(self, agent_id: int, config: ExperimentConfig):
        self.id, self.config = agent_id, config
        self.device = torch.device(config.device)
        self.x = np.random.randint(0, config.grid_size)
        self.y = np.random.randint(0, config.grid_size)
        self.sugar = np.random.uniform(5, 15)
        self.vision = np.random.randint(1, config.agent_vision + 1)
        self.metabolism = np.random.uniform(1, 3)
        self.age = 0
        self.max_age = np.random.randint(60, 160)
        self.policy = PolicyNetwork(4, 4, config.gradient_chain_length).to(self.device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.reset_history()

    def reset_history(self):
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def get_obs(self, env: SugarscapeEnvironment) -> torch.Tensor:
        obs = []
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for i, (dx, dy) in enumerate(dirs):
            best = (0.0, 0.0, self.vision + 1)
            for d in range(1, self.vision + 1):
                nx = (self.x + dx * d) % env.config.grid_size
                ny = (self.y + dy * d) % env.config.grid_size
                sugar = env.sugar[nx, ny]
                pol = env.pollution[nx, ny]
                if sugar > best[0]:
                    best = (sugar, pol, d)
            obs.append([best[0], best[1], best[2] / self.vision, float(i)])
        return torch.tensor(obs, dtype=torch.float32, device=self.device).mean(dim=0)

    def sample(self, state: torch.Tensor, method: str) -> Tuple[int, torch.Tensor]:
        params = self.policy.get_policy_params()
        if method == 'Gumbel':
            logits = self.policy(state)
            sm = F.gumbel_softmax(logits, tau=self.config.tau, hard=True)
            action = sm.argmax().item()
            lp = torch.log(F.softmax(logits, dim=-1)[action] + 1e-8)
        elif method == 'Learnable AUG':
            logits = self.policy(state)
            probs = F.softmax(logits, dim=-1)
            action_tensor, _ = LearnableStochasticCategorical.apply(probs.unsqueeze(0), None, params['alpha'], params['beta'])
            action = action_tensor.item()
            lp = torch.log(probs[action] + 1e-8)
        elif method == 'StochasticAD':
            logits = self.policy(state)
            probs = F.softmax(logits, dim=-1)
            action_tensor = Categorical.apply(probs.unsqueeze(0))
            action = int(action_tensor.item())
            # print(action)
            lp = torch.log(probs[action] + 1e-8)

        else:
            logits = self.policy(state)
            probs = F.softmax(logits, dim=-1)
            action_tensor, _ = StochasticCategorical.apply(probs.unsqueeze(0), None)
            action = action_tensor.item()
            lp = torch.log(probs[action] + 1e-8)
        return action, lp

    def step(self, env: SugarscapeEnvironment, method: str) -> float:
        state = self.get_obs(env)
        action, lp = self.sample(state, method)
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        self.x = (self.x + dx) % env.config.grid_size
        self.y = (self.y + dy) % env.config.grid_size
        reward = env.harvest(self.x, self.y) - self.metabolism
        self.age += 1
        self.log_probs.append(lp)
        self.rewards.append(reward)
        return reward

    def update(self) -> float:
        if not self.rewards:
            return 0.0
        returns = sum(self.rewards)
        loss = -(torch.stack(self.log_probs) * returns).mean()
        self.optim.zero_grad()
        loss.backward()  # backprop through gradient chain
        grad_vars = [p.grad.var().item() for p in self.policy.parameters() if p.grad is not None]
        avg_var = float(np.mean(grad_vars)) if grad_vars else 0.0
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optim.step()
        self.reset_history()
        return avg_var

    def alive(self) -> bool:
        return self.sugar > 0 and self.age < self.max_age

class SugarscapeExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self, methods: List[str], chain_lengths: List[int], runs: int = 3) -> Dict:
        all_results: Dict[str, Dict[int, Dict]] = {}
        for method in methods:
            all_results[method] = {}
            print(f"Starting experiments for method: {method}")
            for L in chain_lengths:
                print(f"  Chain length: {L}")
                self.config.gradient_chain_length = L
                run_results = [self._single_run(method) for _ in range(runs)]
                all_results[method][L] = self._aggregate(run_results)

        print("Final results:")
        for chain in chain_lengths:
            print()
            print(f"for chain length {chain}:")
            
            for method in methods:
                print(f"=== {method} ===")
                # print(type(all_results[method][chain]['rewards_mean']))
                print(f"Mean rewards: {np.mean(all_results[method][chain]['rewards_mean'])}")
                print(f"Mean gradient variance: {np.mean(all_results[method][chain]['gvs_mean'])}")


        # print("\nFinal aggregated results:")
        # for method, res_dict in all_results.items():
        #     for L, res in res_dict.items():
        #         mean_reward = res['rewards_mean'][-1] if len(res['rewards_mean']) > 0 else None
        #         mean_variance = res['variance_mean'][-1] if len(res['variance_mean']) > 0 else None
        #         print(f"Method: {method}, Chain Length: {L}, Final Mean Reward: {mean_reward:.4f}, Final Variance: {mean_variance:.4f}")

        return all_results

    def _single_run(self, method: str) -> Dict:
        env = SugarscapeEnvironment(self.config)
        agents = [SugarscapeAgent(i, self.config) for i in range(self.config.num_agents)]
        episode_rewards: List[float] = []
        gradient_vars: List[float] = []
        for ep in range(self.config.num_episodes):
            env.reset()
            for agent in agents:
                agent.reset_history()
            for _ in range(self.config.max_steps_per_episode):
                alive_agents = [a for a in agents if a.alive()]
                if not alive_agents:
                    break
                rewards_step = [a.step(env, method) for a in alive_agents]
                env.step()
                avg_rew = np.mean(rewards_step) if rewards_step else 0.0
                episode_rewards.append(avg_rew)
                for a in alive_agents:
                    gv = a.update()
                    gradient_vars.append(gv)
        return {'rewards': episode_rewards, 'gvs': gradient_vars}

    def _aggregate(self, runs_res: List[Dict]) -> Dict:
        rewards_lists = [r['rewards'] for r in runs_res]
        gvs_lists = [r['gvs'] for r in runs_res]
        max_r = max(len(lst) for lst in rewards_lists)
        max_g = max(len(lst) for lst in gvs_lists)
        padded_r = [lst + [lst[-1]]*(max_r-len(lst)) for lst in rewards_lists]
        padded_g = [lst + [lst[-1]]*(max_g-len(lst)) for lst in gvs_lists]
        return {
            'rewards_mean': np.mean(padded_r, axis=0),
            'gvs_mean': np.mean(padded_g, axis=0)
        }

    def plot(self, results: Dict[str, Dict[int, Dict]]):
        plt.figure(figsize=(8, 5))
        for method, res in results.items():
            lengths = sorted(res.keys())
            avg_vars = [np.mean(res[L]['gvs_mean']) for L in lengths]
            # Use marker and line for better visibility
            plt.plot(lengths, avg_vars, marker='o', linewidth=2, label=method)
            # Annotate last value for each method
            # plt.annotate(f"{avg_vars[-1]:.3f}", (lengths[-1], avg_vars[-1]), 
            #      textcoords="offset points", xytext=(5,5), ha='left', fontsize=9)
        plt.xlabel('Gradient Chain Length')
        plt.ylabel('Average Gradient Variance')
        plt.yscale('log')  # Use log scale for better separation if values are close
        plt.legend()
        plt.title('Variance vs Chain Length')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig("variance_vs_chain_length.png", dpi=150)
        plt.show()
        
# def mean(lst):
#     return sum(lst) / len(lst) if lst else 0.0

if __name__ == '__main__':
    print('Initializing experiment config...')
    config = ExperimentConfig(num_agents=50, num_episodes=100)
    methods = [ 'StochasticAD','Fixed AUG', 'Learnable AUG','Gumbel']
    chain_lengths = [5,10,50,100]
    print(f'Testing methods: {methods}')
    print(f'Testing chain lengths: {chain_lengths}')
    exp = SugarscapeExperiment(config)
    results = exp.run(methods, chain_lengths, runs=3)
    print('Experiment complete. Plotting results...')
    exp.plot(results)
    with open('results.json', 'w') as f:
        json.dump({m: {str(L): {'rewards_mean': res[L]['rewards_mean'].tolist(),
                                'gvs_mean': res[L]['gvs_mean'].tolist()} for L in res} for m, res in results.items()}, f, indent=2)
    print('Results saved to results.json')
