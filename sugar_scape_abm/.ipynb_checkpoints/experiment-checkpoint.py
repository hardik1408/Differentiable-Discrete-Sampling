import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# Autograd functions for the different gradient estimators
class Categorical(torch.autograd.Function):
    """Standard categorical sampling with a custom gradient"""
    generate_vmap_rule = True
    
    @staticmethod
    def forward(ctx, p):
        result = torch.multinomial(p, num_samples=1)
        ctx.save_for_backward(result, p)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        result, p = ctx.saved_tensors
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        eps = 1e-8
        w_chosen = (1.0 / (p + eps)) / 2  
        w_non_chosen = (1.0 / (1.0 - p + eps)) / 2  
        ws = one_hot * w_chosen + (1 - one_hot) * w_non_chosen
        grad_output_expanded = grad_output.expand_as(p)
        return grad_output_expanded * ws

class StochasticCategorical(torch.autograd.Function):
    """StochasticAD-based categorical sampling with uncertainty tracking"""
    @staticmethod
    def forward(ctx, p, uncertainty_in=None):
        result = torch.multinomial(p, num_samples=1)
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        var_chosen = (1.0 - p) / p.clamp(min=1e-10)
        var_not_chosen = p / (1.0 - p).clamp(min=1e-10)
        op_variance = one_hot * var_chosen + (1 - one_hot) * var_not_chosen
        if uncertainty_in is not None:
            uncertainty_out = uncertainty_in + op_variance
        else:
            uncertainty_out = op_variance
        ctx.save_for_backward(result, p, uncertainty_out)
        return result, uncertainty_out
    
    @staticmethod
    def backward(ctx, grad_output, grad_uncertainty=None):
        result, p, uncertainty = ctx.saved_tensors
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        w_chosen = (1.0 / p.clamp(min=1e-10)) / 2
        w_non_chosen = (1.0 / (1.0 - p).clamp(min=1e-10)) / 2
        confidence = 1.0 / (1.0 + uncertainty.clamp(min=1e-6))
        adjusted_ws = (one_hot * w_chosen + (1 - one_hot) * w_non_chosen) * confidence
        adjusted_ws = adjusted_ws / adjusted_ws.mean().clamp(min=1e-10)
        grad_p = grad_output.expand_as(p) * adjusted_ws
        return grad_p, None

# SugarScape Environment
class SugarScape:
    def __init__(self, size=25, max_sugar=4, growth_rate=1, device="cpu"):
        self.size = size
        self.max_sugar = max_sugar
        self.growth_rate = growth_rate
        self.device = device
        
        # Initialize environment
        self.reset()
    
    def reset(self):
        # Create sugar distribution - higher values near the corners
        self.sugar_capacity = torch.zeros((self.size, self.size), device=self.device)
        for i in range(self.size):
            for j in range(self.size):
                dist_to_corner = min(
                    (i**2 + j**2),
                    (i**2 + (self.size-1-j)**2),
                    ((self.size-1-i)**2 + j**2),
                    ((self.size-1-i)**2 + (self.size-1-j)**2)
                )
                self.sugar_capacity[i, j] = self.max_sugar * max(0, 1 - np.sqrt(dist_to_corner) / (self.size//2))
        
        # Initialize with random sugar levels
        self.sugar = torch.rand((self.size, self.size), device=self.device) * self.sugar_capacity
        return self.sugar.clone()
    
    def step(self):
        # Sugar growth
        self.sugar = torch.min(
            self.sugar + self.growth_rate * torch.rand((self.size, self.size), device=self.device),
            self.sugar_capacity
        )
        return self.sugar.clone()
    
    def harvest(self, position):
        """Harvest sugar at position and return amount harvested"""
        i, j = position
        if 0 <= i < self.size and 0 <= j < self.size:
            amount = self.sugar[i, j].clone()
            self.sugar[i, j] = 0
            return amount
        return torch.tensor(0.0, device=self.device)
    
    def get_state(self):
        return self.sugar.clone()

# Agent definition
class Agent:
    def __init__(self, id, env_size=25, vision_range=4, hidden_size=64, device="cpu"):
        self.id = id
        self.device = device
        self.env_size = env_size
        self.vision_range = vision_range
        self.action_size = 5  # Stay, Up, Down, Left, Right
        
        # Agent state
        self.position = None
        self.sugar_level = None
        self.metabolism = None
        self.age = None
        self.max_age = None
        
        # Policy network - takes local sugar distribution as input
        input_size = (2*vision_range+1)**2 + 2  # Vision window + sugar level and age
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.action_size)
        ).to(device)
    
    def reset(self, sugar_init=5, metabolism_range=(1, 3), max_age_range=(50, 100)):
        # Position randomly on the grid
        self.position = (
            torch.randint(0, self.env_size, (1,)).item(),
            torch.randint(0, self.env_size, (1,)).item()
        )
        
        # Initial sugar level
        self.sugar_level = sugar_init
        
        # Random metabolism (sugar consumption per step)
        self.metabolism = torch.FloatTensor(1).uniform_(*metabolism_range).item()
        
        # Age properties
        self.age = 0
        self.max_age = torch.randint(*max_age_range, (1,)).item()
        
        return self.position, self.sugar_level, self.metabolism, self.age, self.max_age
    
    def get_observation(self, env_state):
        """Extract observation window centered on agent's position"""
        i, j = self.position
        vision = torch.zeros((2*self.vision_range+1, 2*self.vision_range+1), device=self.device)
        
        # Fill observation window
        for di in range(-self.vision_range, self.vision_range+1):
            for dj in range(-self.vision_range, self.vision_range+1):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.env_size and 0 <= nj < self.env_size:
                    vision[di+self.vision_range, dj+self.vision_range] = env_state[ni, nj]
        
        # Append sugar level and normalized age
        observation = torch.cat([
            vision.flatten(),
            torch.tensor([self.sugar_level / 10, self.age / self.max_age], device=self.device)
        ])
        
        return observation
    
    def compute_action_probs(self, env_state, params=None):
        """Compute action probabilities from policy network"""
        observation = self.get_observation(env_state)
        
        if params is not None:
            # Use the provided external parameters
            logits = self._forward_with_params(observation, params)
        else:
            # Use the internal policy network
            logits = self.policy_net(observation)
        
        probs = torch.softmax(logits, dim=0)
        return probs
    
    def _forward_with_params(self, observation, params):
        """Forward pass using external parameters"""
        # First layer
        hidden = torch.matmul(params['W1'], observation) + params['b1']
        hidden = torch.relu(hidden)
        
        # Second layer
        logits = torch.matmul(params['W2'], hidden) + params['b2']
        return logits
    
    def sample_action(self, env_state, params=None, method="categorical"):
        """Sample action based on the specified gradient estimation method"""
        probs = self.compute_action_probs(env_state, params)
        
        if method == "categorical":
            action = Categorical.apply(probs.unsqueeze(0))
            action_idx = action.item()
            log_prob = None
            
        elif method == "cat++":
            action, _ = StochasticCategorical.apply(probs.unsqueeze(0), None)
            action_idx = action.item()
            log_prob = None
            
        elif method == "gumbel":
            # Gumbel-Softmax with straight-through estimator
            gumbel_sample = torch.nn.functional.gumbel_softmax(probs.unsqueeze(0), tau=1.0, hard=True)
            action_idx = torch.argmax(gumbel_sample).item()
            log_prob = None
            
        elif method == "reinforce":
            # REINFORCE estimator
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            action_idx = action.item()
            log_prob = dist.log_prob(action)
        
        return action_idx, log_prob
    
    def get_new_position(self, action_idx):
        """Get new position based on action"""
        i, j = self.position
        
        if action_idx == 1:  # Up
            i = max(0, i - 1)
        elif action_idx == 2:  # Down
            i = min(self.env_size - 1, i + 1)
        elif action_idx == 3:  # Left
            j = max(0, j - 1)
        elif action_idx == 4:  # Right
            j = min(self.env_size - 1, j + 1)
        # action_idx == 0 is "Stay"
        
        return (i, j)
    
    def step(self, action_idx, env):
        """Execute action and update agent state"""
        # Move to new position
        self.position = self.get_new_position(action_idx)
        
        # Harvest sugar
        harvested = env.harvest(self.position)
        self.sugar_level += harvested
        
        # Consume sugar for metabolism
        self.sugar_level -= self.metabolism
        
        # Increment age
        self.age += 1
        
        # Check if agent is alive
        alive = (self.sugar_level > 0) and (self.age < self.max_age)
        
        return harvested, alive

# Multi-agent SugarScape Simulation
class SugarScapeSimulation:
    def __init__(self, n_agents=25, env_size=25, max_sugar=4, growth_rate=1, vision_range=4, hidden_size=64, device="cpu"):
        self.device = device
        self.n_agents = n_agents
        self.env_size = env_size
        
        # Create environment
        self.env = SugarScape(size=env_size, max_sugar=max_sugar, growth_rate=growth_rate, device=device)
        
        # Create agents
        self.agents = [
            Agent(i, env_size=env_size, vision_range=vision_range, hidden_size=hidden_size, device=device) 
            for i in range(n_agents)
        ]
        
        # Track statistics
        self.alive_agents = n_agents
        self.total_sugar = 0
        
    def reset(self):
        """Reset the environment and all agents"""
        # Reset environment
        self.env.reset()
        
        # Reset agents
        for agent in self.agents:
            agent.reset()
        
        self.alive_agents = self.n_agents
        self.total_sugar = sum(agent.sugar_level for agent in self.agents)
        
        return self.env.get_state()
    
    def run_episode(self, policy_params=None, max_steps=100, gradient_method="categorical"):
        """Run a full episode with the given policy parameters"""
        # Reset environment and agents
        self.reset()
        
        # Initialize tracking variables
        episode_reward = torch.tensor(0.0, device=self.device, requires_grad=True)
        steps_survived = 0
        agents_alive = []
        
        # For REINFORCE
        log_probs = []
        rewards = []
        
        # Run episode
        for step in range(max_steps):
            # Environment grows resources
            self.env.step()
            
            # Track which agents are still alive
            agents_alive = [a for a in self.agents if a.sugar_level > 0 and a.age < a.max_age]
            if not agents_alive:
                break
                
            # Each agent takes an action
            step_reward = torch.tensor(0.0, device=self.device, requires_grad=(gradient_method != "reinforce"))
            for agent in agents_alive:
                # Get action from policy
                action_idx, log_prob = agent.sample_action(
                    self.env.get_state(), 
                    params=policy_params, 
                    method=gradient_method
                )
                
                # Execute action
                harvested, alive = agent.step(action_idx, self.env)
                
                # Update step reward
                step_reward = step_reward + harvested
                
                # Store log probability for REINFORCE
                if gradient_method == "reinforce" and log_prob is not None:
                    log_probs.append(log_prob)
            
            # Add step reward to episode reward
            if gradient_method == "reinforce":
                rewards.append(step_reward.item())  # Store as scalar for REINFORCE
            else:
                episode_reward = episode_reward + step_reward  # Keep differentiable for other methods
            
            steps_survived = step + 1
            self.alive_agents = len(agents_alive)
            self.total_sugar = sum(agent.sugar_level for agent in agents_alive)
        
        # For REINFORCE, construct the differentiable loss
        if gradient_method == "reinforce" and log_probs:
            # Calculate discounted rewards
            discounted_rewards = []
            R = 0
            gamma = 0.99  # Discount factor
            for r in reversed(rewards):
                R = r + gamma * R
                discounted_rewards.insert(0, R)
                
            # Convert to tensor and normalize
            discounted_rewards = torch.tensor(discounted_rewards, device=self.device)
            if len(discounted_rewards) > 1:
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
                
            # Calculate REINFORCE loss
            policy_loss = []
            for log_prob, R in zip(log_probs, discounted_rewards):
                policy_loss.append(-log_prob * R)
                
            if policy_loss:
                policy_loss = torch.stack(policy_loss).sum()
                # We'll return the negative policy loss as our "reward" - it's actually a loss to be minimized
                episode_reward = -policy_loss
        
        # Return total reward and additional statistics
        return episode_reward, steps_survived, self.alive_agents, self.total_sugar
    
    def visualize(self, title=None):
        """Visualize the current state of the environment"""
        plt.figure(figsize=(10, 8))
        
        # Plot sugar distribution
        sugar_map = self.env.get_state().cpu().numpy()
        plt.imshow(sugar_map, cmap='YlOrBr', vmin=0, vmax=self.env.max_sugar)
        
        # Plot agents
        for agent in self.agents:
            if agent.sugar_level > 0 and agent.age < agent.max_age:
                i, j = agent.position
                size = 50 + 50 * (agent.sugar_level / 10)  # Size based on sugar level
                plt.scatter(j, i, s=size, c='blue', alpha=0.7)
        
        if title:
            plt.title(title)
        plt.colorbar(label='Sugar Level')
        plt.tight_layout()
        return plt

# Optimization function for learning policy parameters
def optimize_policy(simulation, n_episodes=100, lr=0.01, gradient_method="categorical"):
    """Optimize policy parameters using the specified gradient estimation method"""
    device = simulation.device
    
    # Get dimensions from an agent's policy network
    agent = simulation.agents[0]
    modules = list(agent.policy_net.modules())
    
    # Extract dimensions
    input_size = modules[1].in_features
    hidden_size = modules[1].out_features
    output_size = modules[3].out_features
    
    # Initialize policy parameters
    policy_params = {
        'W1': torch.nn.Parameter(torch.randn(hidden_size, input_size, device=device) * 0.1),
        'b1': torch.nn.Parameter(torch.zeros(hidden_size, device=device)),
        'W2': torch.nn.Parameter(torch.randn(output_size, hidden_size, device=device) * 0.1),
        'b2': torch.nn.Parameter(torch.zeros(output_size, device=device))
    }
    
    # Create optimizer
    optimizer = torch.optim.Adam(policy_params.values(), lr=lr)
    
    # Track statistics
    rewards_history = []
    survival_history = []
    alive_agents_history = []
    sugar_history = []
    
    # Run optimization
    for episode in range(n_episodes):
        # Reset gradients
        optimizer.zero_grad()
        
        # Run episode
        episode_result, steps_survived, alive_agents, total_sugar = simulation.run_episode(
            policy_params=policy_params,
            gradient_method=gradient_method
        )
        
        # For REINFORCE, episode_result is already the loss
        # For other methods, it's the reward that needs to be negated to get the loss
        if gradient_method == "reinforce":
            loss = episode_result  # Already negative policy loss
            reward_value = -loss.item()  # For tracking
        else:
            loss = -episode_result  # Negative reward as loss to minimize
            reward_value = episode_result.item()  # For tracking
        
        # Store statistics
        rewards_history.append(reward_value)
        survival_history.append(steps_survived)
        alive_agents_history.append(alive_agents)
        sugar_history.append(total_sugar)
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        # Print progress
        if episode % 10 == 0 or episode == n_episodes - 1:
            print(f"Episode {episode}, Reward: {reward_value:.2f}, "
                  f"Steps: {steps_survived}, Agents Alive: {alive_agents}, "
                  f"Total Sugar: {total_sugar:.2f}")
    
    return policy_params, {
        'rewards': rewards_history,
        'survival': survival_history,
        'alive_agents': alive_agents_history,
        'total_sugar': sugar_history
    }

# Benchmark different gradient estimators
def benchmark_gradient_estimators(n_runs=5, n_episodes=50, env_size=25, n_agents=25, lr=0.01, device="cpu"):
    """Compare the performance of different gradient estimators"""
    methods = ["categorical", "cat++", "gumbel", "reinforce"]
    results = {method: {
        'rewards': [],
        'survival': [],
        'alive_agents': [],
        'total_sugar': [],
        'time': []
    } for method in methods}
    
    for method in methods:
        print(f"\nBenchmarking {method} gradient estimator:")
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}:")
            
            # Create simulation
            simulation = SugarScapeSimulation(
                n_agents=n_agents,
                env_size=env_size,
                device=device
            )
            
            # Measure training time
            start_time = time.time()
            
            # Train policy
            _, stats = optimize_policy(
                simulation,
                n_episodes=n_episodes,
                lr=lr,
                gradient_method=method
            )
            
            # Store time
            elapsed = time.time() - start_time
            results[method]['time'].append(elapsed)
            
            # Store statistics from the final episode
            results[method]['rewards'].append(stats['rewards'][-1])
            results[method]['survival'].append(stats['survival'][-1])
            results[method]['alive_agents'].append(stats['alive_agents'][-1])
            results[method]['total_sugar'].append(stats['total_sugar'][-1])
            
            print(f"    Final Reward: {stats['rewards'][-1]:.2f}, "
                  f"Steps: {stats['survival'][-1]}, "
                  f"Agents Alive: {stats['alive_agents'][-1]}, "
                  f"Total Sugar: {stats['total_sugar'][-1]:.2f}, "
                  f"Time: {elapsed:.2f} sec")
            
            # Plot learning curves for this run
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(stats['rewards'])
            plt.title(f'{method} - Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            
            plt.subplot(2, 2, 2)
            plt.plot(stats['survival'])
            plt.title(f'{method} - Steps Survived')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            
            plt.subplot(2, 2, 3)
            plt.plot(stats['alive_agents'])
            plt.title(f'{method} - Agents Alive at End')
            plt.xlabel('Episode')
            plt.ylabel('Agents')
            
            plt.subplot(2, 2, 4)
            plt.plot(stats['total_sugar'])
            plt.title(f'{method} - Total Sugar at End')
            plt.xlabel('Episode')
            plt.ylabel('Sugar')
            
            plt.tight_layout()
            plt.savefig(f"sugarscape_{method}_run_{run}.png")
            plt.close()
    
    # Compute averages
    for method in methods:
        for key in ['rewards', 'survival', 'alive_agents', 'total_sugar', 'time']:
            results[method][f'avg_{key}'] = np.mean(results[method][key])
            results[method][f'std_{key}'] = np.std(results[method][key])
    
    # Plot comparative results
    plot_comparative_results(results, methods)
    
    return results

def plot_comparative_results(results, methods):
    """Create plots comparing the performance of different gradient estimators"""
    # Extract data for plotting
    avg_rewards = [results[m]['avg_rewards'] for m in methods]
    std_rewards = [results[m]['std_rewards'] for m in methods]
    
    avg_survival = [results[m]['avg_survival'] for m in methods]
    std_survival = [results[m]['std_survival'] for m in methods]
    
    avg_alive = [results[m]['avg_alive_agents'] for m in methods]
    std_alive = [results[m]['std_alive_agents'] for m in methods]
    
    avg_sugar = [results[m]['avg_total_sugar'] for m in methods]
    std_sugar = [results[m]['std_total_sugar'] for m in methods]
    
    avg_time = [results[m]['avg_time'] for m in methods]
    std_time = [results[m]['std_time'] for m in methods]
    
    # Create plot
    plt.figure(figsize=(15, 15))
    
    # Plot rewards
    plt.subplot(3, 2, 1)
    plt.bar(methods, avg_rewards, yerr=std_rewards)
    plt.title('Average Final Reward')
    plt.ylabel('Reward')
    
    # Plot survival steps
    plt.subplot(3, 2, 2)
    plt.bar(methods, avg_survival, yerr=std_survival)
    plt.title('Average Steps Survived')
    plt.ylabel('Steps')
    
    # Plot alive agents
    plt.subplot(3, 2, 3)
    plt.bar(methods, avg_alive, yerr=std_alive)
    plt.title('Average Agents Alive at End')
    plt.ylabel('Agents')
    
    # Plot total sugar
    plt.subplot(3, 2, 4)
    plt.bar(methods, avg_sugar, yerr=std_sugar)
    plt.title('Average Total Sugar at End')
    plt.ylabel('Sugar')
    
    # Plot training time
    plt.subplot(3, 2, 5)
    plt.bar(methods, avg_time, yerr=std_time)
    plt.title('Average Training Time')
    plt.ylabel('Seconds')
    
    plt.tight_layout()
    plt.savefig("sugarscape_comparative_results.png")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run SugarScape benchmark
    results = benchmark_gradient_estimators(
        n_runs=5,
        n_episodes=50,
        env_size=25,
        n_agents=25,
        lr=0.01,
        device=device
    )
    
    # Print final results
    print("\nBenchmark Results:")
    methods = ["categorical", "cat++", "gumbel", "reinforce"]
    for method in methods:
        print(f"\nGradient Estimator: {method}")
        print(f"  Average Final Reward: {results[method]['avg_rewards']:.2f} ± {results[method]['std_rewards']:.2f}")
        print(f"  Average Steps Survived: {results[method]['avg_survival']:.2f} ± {results[method]['std_survival']:.2f}")
        print(f"  Average Agents Alive: {results[method]['avg_alive_agents']:.2f} ± {results[method]['std_alive_agents']:.2f}")
        print(f"  Average Total Sugar: {results[method]['avg_total_sugar']:.2f} ± {results[method]['std_total_sugar']:.2f}")
        print(f"  Average Training Time: {results[method]['avg_time']:.2f} ± {results[method]['std_time']:.2f} sec")