import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# Autograd functions for the different gradient estimators
class Categorical(torch.autograd.Function):
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

class LearnableStochasticCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, uncertainty_in=None, alpha=None, beta=None):
        # Sample from categorical distribution
        result = torch.multinomial(p, num_samples=1)
        # Create one-hot encoding of the result
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        var_chosen = (1.0 - p) / p.clamp(min=1e-10)
        var_not_chosen = p / (1.0 - p).clamp(min=1e-10)
        # Calculate operation variance
        op_variance = one_hot * var_chosen + (1 - one_hot) * var_not_chosen
        # Propagate uncertainty
        if uncertainty_in is not None:
            uncertainty_out = uncertainty_in + op_variance
        else:
            uncertainty_out = op_variance
        # Save context for backward pass
        ctx.save_for_backward(result, p, uncertainty_out, alpha, beta)
        return result, uncertainty_out
    
    @staticmethod
    def backward(ctx, grad_output, grad_uncertainty=None):
        result, p, uncertainty, alpha, beta = ctx.saved_tensors
        # Create one-hot encoding
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        # Base weights for chosen and non-chosen outcomes
        w_chosen = (1.0 / p.clamp(min=1e-10)) / 2
        w_non_chosen = (1.0 / (1.0 - p).clamp(min=1e-10)) / 2
        # Apply the learnable alpha parameter to control confidence scaling
        confidence = 1.0 / (1.0 + alpha * uncertainty.clamp(min=1e-6))
        # Apply the learnable beta parameter to control balance between chosen/non-chosen
        adjusted_ws = (one_hot * w_chosen * beta + (1 - one_hot) * w_non_chosen * (1 - beta)) * confidence
        # Normalize to maintain gradient scale
        adjusted_ws = adjusted_ws / adjusted_ws.mean().clamp(min=1e-10)
        # Apply the adjusted weights to gradients
        grad_p = grad_output.expand_as(p) * adjusted_ws

        # Compute gradients for alpha and beta
        # Gradient for alpha: how changing alpha affects the adjusted weights and thus the output
        confidence_derivative = -uncertainty.clamp(min=1e-6) * confidence * confidence
        grad_alpha = torch.sum(grad_output.expand_as(p) * ((one_hot * w_chosen * beta + 
                                                          (1 - one_hot) * w_non_chosen * (1 - beta)) * 
                                                          confidence_derivative))
        
        # Gradient for beta: how changing beta affects the balance and thus the output
        balance_derivative = one_hot * w_chosen - (1 - one_hot) * w_non_chosen
        grad_beta = torch.sum(grad_output.expand_as(p) * (balance_derivative * confidence))
        
        # Return gradients for inputs and learnable parameters
        # The gradient for p, uncertainty_in, alpha, beta
        return grad_p, None, grad_alpha, grad_beta

class Cat(nn.Module):
    def __init__(self, learnable=True, init_alpha=1.0, init_beta=0.5):
        """
        Learnable Categorical sampling module
        
        Args:
            learnable: If True, alpha and beta are learnable parameters
            init_alpha: Initial value for alpha (confidence scaling)
            init_beta: Initial value for beta (chosen/non-chosen balance)
        """
        super(Cat, self).__init__()
        
        # Create learnable parameters
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(init_alpha))
            self.beta = nn.Parameter(torch.tensor(init_beta))
        else:
            self.register_buffer('alpha', torch.tensor(init_alpha))
            self.register_buffer('beta', torch.tensor(init_beta))
        
        self.learnable = learnable
    
    def forward(self, p, uncertainty_in=None):
        """
        Forward pass
        
        Args:
            p: Probability distribution (batch_size, num_categories)
            uncertainty_in: Optional incoming uncertainty
            
        Returns:
            result: Sampled indices
            uncertainty_out: Output uncertainty
        """
        return LearnableStochasticCategorical.apply(p, uncertainty_in, self.alpha, self.beta)

# Environment definition
class ResourceGrid:
    def __init__(self, size=5, n_resources=3, device="cpu"):
        self.size = size
        self.n_resources = n_resources
        self.device = device
        self.reset()
        
    def reset(self):
        # Initialize grid with random resource locations
        self.grid = torch.zeros((self.size, self.size), device=self.device)
        resource_positions = torch.randperm(self.size * self.size)[:self.n_resources]
        for pos in resource_positions:
            row, col = pos.item() // self.size, pos.item() % self.size
            self.grid[row, col] = 1.0
        return self.grid.clone()
    
    def collect_resource(self, position):
        row, col = position
        if 0 <= row < self.size and 0 <= col < self.size and self.grid[row, col] > 0:
            reward = self.grid[row, col].item()
            self.grid[row, col] = 0
            return reward
        return 0.0
    
    def collect_resource_tensor(self, position, grid_state):
        row, col = position
        if 0 <= row < self.size and 0 <= col < self.size and grid_state[row, col] > 0:
            reward = grid_state[row, col].clone()
            grid_state[row, col] = 0
            return reward
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def get_state(self):
        return self.grid.clone()

# Agent definition
class Agent:
    def __init__(self, id, grid_size=5, comm_size=3, hidden_size=16, device="cpu"):
        self.id = id
        self.device = device
        self.comm_size = comm_size  # Size of discrete message
        self.action_size = 4  # Up, Down, Left, Right
        self.grid_size = grid_size
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(grid_size * grid_size + comm_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.action_size + comm_size)
        ).to(device)
        
        # Add Cat++ modules with learnable parameters
        self.action_cat = Cat(learnable=True, init_alpha=1.0, init_beta=0.5).to(device)
        self.message_cat = Cat(learnable=True, init_alpha=1.0, init_beta=0.5).to(device)
        
        # Current position and message
        self.position = None
        self.message = None
        
    def reset(self, position):
        self.position = position
        self.message = torch.zeros(self.comm_size, device=self.device)
        return self.position, self.message
    
    def compute_action_probs(self, grid_state, messages):
        # Concatenate grid state and messages from other agents
        state = torch.cat([grid_state.flatten(), messages.flatten()])
        logits = self.policy_net(state)
        
        # Split logits into action and message components
        action_logits = logits[:self.action_size]
        message_logits = logits[self.action_size:]
        
        # Compute probabilities
        action_probs = torch.softmax(action_logits, dim=0)
        message_probs = torch.softmax(message_logits, dim=0)
        
        return action_probs, message_probs
    
    def sample_action_message(self, grid_state, messages, method="cat++"):
        action_probs, message_probs = self.compute_action_probs(grid_state, messages)
        
        # Sample action and message based on the specified method
        if method == "cat++":
            action, action_uncertainty = self.action_cat(action_probs.unsqueeze(0), None)
            message, message_uncertainty = self.message_cat(message_probs.unsqueeze(0), None)
            action_idx = action.item()
            message_idx = message.item()
        elif method == "categorical":
            action = Categorical.apply(action_probs.unsqueeze(0))
            message = Categorical.apply(message_probs.unsqueeze(0))
            action_idx = action.item()
            message_idx = message.item()
        elif method == "gumbel":
            action_sample = torch.nn.functional.gumbel_softmax(action_probs, tau=1.0, hard=True)
            message_sample = torch.nn.functional.gumbel_softmax(message_probs, tau=1.0, hard=True)
            action_idx = torch.argmax(action_sample).item()
            message_idx = torch.argmax(message_sample).item()
        elif method == "reinforce":
            # Use torch.distributions for reinforce
            action_dist = torch.distributions.Categorical(action_probs)
            message_dist = torch.distributions.Categorical(message_probs)
            action_idx = action_dist.sample().item()
            message_idx = message_dist.sample().item()
            
            # Calculate log probabilities
            log_prob_action = action_dist.log_prob(torch.tensor(action_idx, device=self.device))
            log_prob_message = message_dist.log_prob(torch.tensor(message_idx, device=self.device))
            log_prob = log_prob_action + log_prob_message
            
            # Create one-hot message
            new_message = torch.zeros_like(self.message)
            new_message[message_idx] = 1.0
            
            # Calculate new position
            new_position = self.compute_new_position(action_idx, grid_state.shape)
            
            return new_position, new_message, log_prob
        
        # Calculate new position
        new_position = self.compute_new_position(action_idx, grid_state.shape)
        
        # Create one-hot message
        new_message = torch.zeros_like(self.message)
        new_message[message_idx] = 1.0
        
        # For direct gradient methods, we don't need log probs
        return new_position, new_message, None
    
    def compute_new_position(self, action_idx, grid_shape):
        row, col = self.position
        if action_idx == 0:  # Up
            row = max(0, row - 1)
        elif action_idx == 1:  # Down
            row = min(grid_shape[0] - 1, row + 1)
        elif action_idx == 2:  # Left
            col = max(0, col - 1)
        elif action_idx == 3:  # Right
            col = min(grid_shape[1] - 1, col + 1)
        return (row, col)

# Multi-agent system
class MultiAgentSystem:
    def __init__(self, n_agents=3, grid_size=5, n_resources=3, comm_size=3, hidden_size=16, device="cpu"):
        self.device = device
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.comm_size = comm_size
        
        # Create environment
        self.env = ResourceGrid(size=grid_size, n_resources=n_resources, device=device)
        
        # Create agents
        self.agents = [Agent(i, grid_size=grid_size, comm_size=comm_size, hidden_size=hidden_size, device=device) 
                        for i in range(n_agents)]
        
    def reset(self):
        # Reset environment
        grid_state = self.env.reset()
        
        # Reset agents to random positions
        positions = torch.randperm(self.grid_size * self.grid_size)[:self.n_agents]
        agent_positions = []
        
        for i, agent in enumerate(self.agents):
            row, col = positions[i].item() // self.grid_size, positions[i].item() % self.grid_size
            agent.reset((row, col))
            agent_positions.append((row, col))
            
        return grid_state
    
    def run_episode_direct(self, max_steps=20, method="cat++", verbose=False):
        """Run episode for direct gradient methods (Cat++, StochasticAD, Gumbel-Softmax)"""
        grid_state = self.reset()
        total_reward = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for step in range(max_steps):
            if verbose and step % 5 == 0:
                print(f"Step {step}, Current reward: {total_reward.item():.2f}")
            
            # Each agent takes an action
            for agent in self.agents:
                # Get messages from other agents
                if self.n_agents > 1:
                    other_messages = torch.cat([a.message.unsqueeze(0) for a in self.agents if a.id != agent.id])
                else:
                    other_messages = torch.zeros((1, self.comm_size), device=self.device)
                
                # Compute action probabilities
                action_probs, message_probs = agent.compute_action_probs(grid_state, other_messages)
                
                # Sample actions and messages based on method
                if method == "cat++":
                    action, _ = agent.action_cat(action_probs.unsqueeze(0), None)
                    message, _ = agent.message_cat(message_probs.unsqueeze(0), None)
                    action_idx = action.item()
                    message_idx = message.item()
                elif method == "categorical":
                    action = Categorical.apply(action_probs.unsqueeze(0))
                    message = Categorical.apply(message_probs.unsqueeze(0))
                    action_idx = action.item()
                    message_idx = message.item()
                elif method == "gumbel":
                    # Get hard samples while keeping gradients
                    action_gumbel = torch.nn.functional.gumbel_softmax(action_probs.unsqueeze(0), tau=1.0, hard=True)
                    message_gumbel = torch.nn.functional.gumbel_softmax(message_probs.unsqueeze(0), tau=1.0, hard=True)
                    action_idx = torch.argmax(action_gumbel, dim=1).item()
                    message_idx = torch.argmax(message_gumbel, dim=1).item()
                
                # Update agent position
                new_position = agent.compute_new_position(action_idx, grid_state.shape)
                
                # Handle resource collection in a differentiable way
                row, col = new_position
                if 0 <= row < self.grid_size and 0 <= col < self.grid_size and grid_state[row, col] > 0:
                    # Connect the reward to the action probabilities to make it differentiable
                    reward_value = grid_state[row, col].clone()
                    
                    # Create a path between the action and the reward
                    reward_connection = action_probs[action_idx] * reward_value
                    
                    # Add to total reward
                    total_reward = total_reward + reward_connection
                    
                    # Remove resource from grid
                    grid_state[row, col] = 0
                
                # Create one-hot message - keep it differentiable
                if method == "gumbel":
                    # Gumbel-Softmax already gives differentiable one-hot
                    new_message = message_gumbel.squeeze(0)
                else:
                    # Create one-hot for other methods
                    new_message = torch.zeros_like(agent.message)
                    new_message[message_idx] = 1.0
                
                # Update agent state
                agent.position = new_position
                agent.message = new_message
            
            # End episode early if all resources collected
            if torch.sum(grid_state) == 0:
                if verbose:
                    print("All resources collected!")
                break
        
        return total_reward

    def run_episode_reinforce(self, max_steps=20, verbose=False):
        """Run episode for REINFORCE method"""
        grid_state = self.reset()
        total_reward = 0.0
        log_probs = []
        rewards = []
        
        for step in range(max_steps):
            step_reward = 0.0
            
            if verbose and step % 5 == 0:
                print(f"Step {step}, Current reward: {total_reward:.2f}")
            
            # Each agent takes an action
            for agent in self.agents:
                # Get messages from other agents
                if self.n_agents > 1:
                    other_messages = torch.cat([a.message.unsqueeze(0) for a in self.agents if a.id != agent.id])
                else:
                    other_messages = torch.zeros((1, self.comm_size), device=self.device)
                
                # Use standard reinforce method
                action_probs, message_probs = agent.compute_action_probs(grid_state, other_messages)
                
                # Sample using torch.distributions
                action_dist = torch.distributions.Categorical(action_probs)
                message_dist = torch.distributions.Categorical(message_probs)
                
                action_idx = action_dist.sample().item()
                message_idx = message_dist.sample().item()
                
                # Calculate log probabilities
                log_prob_action = action_dist.log_prob(torch.tensor(action_idx, device=self.device))
                log_prob_message = message_dist.log_prob(torch.tensor(message_idx, device=self.device))
                log_prob = log_prob_action + log_prob_message
                log_probs.append(log_prob)
                
                # Update position
                new_position = agent.compute_new_position(action_idx, grid_state.shape)
                
                # Collect resource
                reward = self.env.collect_resource(new_position)
                step_reward += reward
                
                # Create one-hot message
                new_message = torch.zeros_like(agent.message)
                new_message[message_idx] = 1.0
                
                # Update agent state
                agent.position = new_position
                agent.message = new_message
            
            # Store reward for this step
            total_reward += step_reward
            rewards.append(step_reward)
            
            # End episode early if all resources collected
            if torch.sum(grid_state) == 0:
                if verbose:
                    print("All resources collected!")
                break
        
        return total_reward, log_probs, rewards

# Optimization functions for direct gradient methods (Cat++, StochasticAD, Gumbel-Softmax)
def optimize_multi_agent_direct(system, max_steps,n_episodes=1000, lr=0.01, method="cat++"):
    # Collect all policy parameters
    policy_params = []
    for agent in system.agents:
        policy_params.extend(agent.policy_net.parameters())
        
        # Add Cat++ parameters if using that method
        if method == "cat++":
            if hasattr(agent, 'action_cat'):
                policy_params.extend(agent.action_cat.parameters())
            if hasattr(agent, 'message_cat'):
                policy_params.extend(agent.message_cat.parameters())
    
    optimizer = torch.optim.Adam(policy_params, lr=lr)
    rewards_history = []
    
    # Monitor the learnable parameters
    alpha_history = []
    beta_history = []
    
    for episode in range(n_episodes):
        optimizer.zero_grad()
        
        # Verbose output every 100 episodes
        verbose = (episode % 100 == 0)
        
        # Run episode and get reward tensor
        total_reward = system.run_episode_direct(max_steps=max_steps,method=method, verbose=verbose)
        rewards_history.append(total_reward.item())
        
        # Track learnable parameters if using cat++
        if method == "cat++":
            avg_action_alpha = sum(agent.action_cat.alpha.item() for agent in system.agents) / len(system.agents)
            avg_action_beta = sum(agent.action_cat.beta.item() for agent in system.agents) / len(system.agents)
            alpha_history.append(avg_action_alpha)
            beta_history.append(avg_action_beta)
        
        # Direct optimization - maximize reward
        loss = -total_reward  # Negative reward as loss to minimize
        loss.backward()
        optimizer.step()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward.item():.2f}")
            if method == "cat++" and episode % 100 == 0:
                print(f"  Alpha: {avg_action_alpha:.4f}, Beta: {avg_action_beta:.4f}")
    
    result = {"rewards": rewards_history}
    if method == "cat++":
        result["alpha_history"] = alpha_history
        result["beta_history"] = beta_history
    
    return result

# Optimization function for REINFORCE
def optimize_multi_agent_reinforce(system, max_steps,n_episodes=1000, lr=0.01):
    # Collect all policy parameters
    policy_params = []
    for agent in system.agents:
        policy_params.extend(agent.policy_net.parameters())
    
    optimizer = torch.optim.Adam(policy_params, lr=lr)
    rewards_history = []
    
    for episode in range(n_episodes):
        optimizer.zero_grad()
        
        # Verbose output every 100 episodes
        verbose = (episode % 100 == 0)
        
        # Run episode and collect trajectories
        total_reward, log_probs, step_rewards = system.run_episode_reinforce(max_steps,verbose=verbose)
        rewards_history.append(total_reward)
        
        # Calculate returns with discount factor
        returns = []
        R = 0
        gamma = 0.9
        for r in reversed(step_rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=system.device)
        
        # Normalize returns for stable training
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # REINFORCE loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        if policy_loss:  # Only compute loss if there are policy decisions
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
    
    return {"rewards": rewards_history}

# Benchmark function
def benchmark_multi_agent(max_steps,n_agents=3, grid_size=5, n_resources=4, n_episodes=500, n_runs=5, lr=0.01, device="cpu"):
    all_results = {
        "cat++": {}, 
        "categorical": {}, 
        "gumbel": {},
        "reinforce": {}
    }
    
    for method in all_results.keys():
        print(f"\nBenchmarking using the {method} estimator:")
        
        run_rewards = []
        run_times = []
        learnable_params_history = []
        
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}: ", end="", flush=True)
            
            # Create system
            system = MultiAgentSystem(
                n_agents=n_agents,
                grid_size=grid_size,
                n_resources=n_resources,
                device=device
            )
            
            # Train and time
            start_time = time.time()
            
            if method == "reinforce":
                # Use REINFORCE optimization
                results = optimize_multi_agent_reinforce(
                    system,
                    max_steps=max_steps,
                    n_episodes=n_episodes,
                    lr=lr
                )
                rewards_history = results["rewards"]
            else:
                # Use direct gradient optimization for other methods
                results = optimize_multi_agent_direct(
                    system,
                    max_steps=max_steps,
                    n_episodes=n_episodes,
                    lr=lr,
                    method=method
                )
                rewards_history = results["rewards"]
                
                # Save learnable parameters history if using cat++
                if method == "cat++" and "alpha_history" in results:
                    learnable_params_history.append({
                        "alpha": results["alpha_history"],
                        "beta": results["beta_history"]
                    })
                
            elapsed = time.time() - start_time
            
            # Evaluate final performance
            final_reward = sum(rewards_history[-10:]) / 10  # Average of last 10 episodes
            run_rewards.append(final_reward)
            run_times.append(elapsed)
            
            print(f"Final Reward = {final_reward:.2f}, Time = {elapsed:.2f} sec")
            
        avg_reward = np.mean(run_rewards)
        avg_time = np.mean(run_times)
        all_results[method] = {
            "avg_final_reward": avg_reward, 
            "avg_time_sec": avg_time
        }
        
        if method == "cat++" and learnable_params_history:
            all_results[method]["learnable_params"] = learnable_params_history
            
        print(f"  Average final reward: {avg_reward:.2f}, Average time: {avg_time:.2f} sec")
    
    return all_results

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Starting Multi-Agent Coordination Benchmark")
    
    # Run a single episode to debug
    print("\nRunning debug episode with cat++ estimator...")
    system = MultiAgentSystem(
        n_agents=3,
        grid_size=5,
        n_resources=4,
        device=device
    )
    total_reward = system.run_episode_direct(method="cat++", verbose=True)
    print(f"Debug episode total reward: {total_reward.item()}")
    
    max_steps = 75
    # Only proceed to full benchmark if debug episode works
    if total_reward.item() > 0:
        results = benchmark_multi_agent(
            max_steps,
            n_agents=3,
            grid_size=5,
            n_resources=4,
            n_episodes=300,
            n_runs=5,
            lr=0.01,
            device=device
        )
        
        print("\nBenchmark Results:")
        for method, metrics in results.items():
            print(f"\nEstimator: {method}")
            print(f"Step length: {max_steps}, ")
            print(f"  Average Final Reward={metrics['avg_final_reward']:.2f}, "
                 f"Average Time={metrics['avg_time_sec']:.2f} sec")
                  
        # Plot results
        methods = list(results.keys())
        rewards = [results[m]['avg_final_reward'] for m in methods]
        times = [results[m]['avg_time_sec'] for m in methods]
        
        # Create side-by-side bar charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Performance comparison
        ax1.bar(methods, rewards)
        ax1.set_title('Average Final Reward')
        ax1.set_ylabel('Reward')
        ax1.set_ylim(bottom=0, top=max(rewards) * 1.1)  # Set y-axis to start at 0
        
        # Time comparison
        ax2.bar(methods, times)
        ax2.set_title('Average Training Time')
        ax2.set_ylabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(f"multi_agent_results_{max_steps}.png")
        plt.show()
        
        # If Cat++ was run, plot the learnable parameter evolution
        if "cat++" in results and "learnable_params" in results["cat++"]:
            # Plot alpha and beta evolution for the first run
            params = results["cat++"]["learnable_params"][0]
            episodes = range(len(params["alpha"]))
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(episodes, params["alpha"])
            plt.title('Alpha Parameter Evolution')
            plt.xlabel('Episode')
            plt.ylabel('Alpha Value')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(episodes, params["beta"])
            plt.title('Beta Parameter Evolution')
            plt.xlabel('Episode')
            plt.ylabel('Beta Value')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"cat_plus_plus_parameters_{max_steps}.png")
            plt.show()
        
        # Plot learning curves if available
        print("\nRunning final comprehensive comparison for learning curves...")
        learning_curves = {}
        
        for method in methods:
            print(f"Generating learning curve for {method}...")
            
            # Create a fresh system
            system = MultiAgentSystem(
                n_agents=3,
                grid_size=5,
                n_resources=4,
                device=device
            )
            
            # Run a single training run to capture learning curve
            if method == "reinforce":
                results = optimize_multi_agent_reinforce(
                    system,
                    max_steps=max_steps,
                    n_episodes=100,
                    lr=0.01
                )
            else:
                results = optimize_multi_agent_direct(
                    system,
                    max_steps=max_steps,
                    n_episodes=100,
                    lr=0.01,
                    method=method
                )
            
            # Store learning curve
            learning_curves[method] = results["rewards"]
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        for method, rewards in learning_curves.items():
            # Apply smoothing for better visualization
            window_size = 20
            smoothed_rewards = []
            for i in range(len(rewards)):
                start_idx = max(0, i - window_size + 1)
                smoothed_rewards.append(np.mean(rewards[start_idx:i+1]))
            
            plt.plot(smoothed_rewards, label=method)
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Learning Curves for Different Gradient Estimators')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"learning_curves_{max_steps}.png")
        plt.show()
    else:
        print("Debug episode failed - resources not being collected. Please fix before running full benchmark.")
