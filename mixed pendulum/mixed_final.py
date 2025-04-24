import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import time
class Categorical(torch.autograd.Function):
    generate_vmap_rule = True  # for functorch if needed

    @staticmethod
    def forward(ctx, p):
        # p: shape (..., k)
        result = torch.multinomial(p, num_samples=1)  # shape (..., 1)
        ctx.save_for_backward(result, p)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, p = ctx.saved_tensors
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        eps = 1e-8  # avoid division by zero
        # Compute approximate gradients:
        w_chosen = (1.0 / (p + eps)) / 2  
        w_non_chosen = (1.0 / (1.0 - p + eps)) / 2  
        ws = one_hot * w_chosen + (1 - one_hot) * w_non_chosen
        grad_output_expanded = grad_output.expand_as(p)
        return grad_output_expanded * ws
class StochasticCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, uncertainty_in=None):
        result = torch.multinomial(p, num_samples=1)
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        var_chosen = (1.0 - p) / p
        var_not_chosen = p / (1.0 - p)
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
        w_chosen = (1.0 / p) / 2
        w_non_chosen = (1.0 / (1.0 - p)) / 2
        confidence = 1.0 / (1.0 + uncertainty.clamp(min=1e-6))
        adjusted_ws = (one_hot * w_chosen + (1 - one_hot) * w_non_chosen) * confidence
        adjusted_ws = adjusted_ws / adjusted_ws.mean().clamp(min=1e-10)
        grad_p = grad_output.expand_as(p) * adjusted_ws
        return grad_p, None
# Pendulum system with discrete control actions.
class DiscreteControlPendulum:
    def __init__(self, dt=0.1, g=9.8, m=1.0, l=1.0, max_torque=5.0, n_controls=3, device="cpu"):
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l
        self.max_torque = max_torque
        self.n_controls = n_controls
        self.device = device

        # Create discrete control values:
        if n_controls % 2 == 1:
            self.control_values = torch.linspace(-max_torque, max_torque, n_controls, device=device)
        else:
            step = 2 * max_torque / (n_controls - 1)
            self.control_values = torch.arange(-max_torque, max_torque + step/2, step, device=device)

    def step(self, state, control_idx):
        # state: tensor [theta, omega]
        theta, omega = state[0], state[1]
        u = self.control_values[control_idx]  # discrete torque control
        # Dynamics: angular acceleration computed using Euler integration.
        alpha = (u - self.m * self.g * self.l * torch.sin(theta)) / (self.m * self.l**2)
        new_omega = omega + alpha * self.dt
        new_theta = theta + new_omega * self.dt
        # Normalize theta to [-pi, pi]
        new_theta = (new_theta + math.pi) % (2 * math.pi) - math.pi
        return torch.stack([new_theta, new_omega])

    def compute_control_logits(self, state, policy_params):
        # Compute features from state:
        theta, omega = state[0], state[1]
        state_tensor = torch.stack([torch.sin(theta), torch.cos(theta), omega])
        hidden = torch.matmul(policy_params['W1'], state_tensor) + policy_params['b1']
        hidden = torch.tanh(hidden)
        logits = torch.matmul(policy_params['W2'], hidden) + policy_params['b2']
        return logits

    def compute_control_probs(self, state, policy_params):
        logits = self.compute_control_logits(state, policy_params)
        probs = torch.softmax(logits, dim=0)
        return probs

    def sample_control_custom(self, state, policy_params):
        """
        Uses the custom differentiable categorical estimator.
        """
        probs = self.compute_control_probs(state, policy_params)
        sample = Categorical.apply(probs.unsqueeze(0))
        control_idx = sample.item()
        log_prob = torch.log(probs[control_idx] + 1e-8)
        return control_idx, log_prob

    def sample_control_cat(self, state, policy_params, tau=1.0):
        """
        Uses the Gumbel-Softmax estimator (with straight-through sampling).
        """
        logits = self.compute_control_probs(state, policy_params)
        # PyTorch has a built-in implementation for gumbel softmax.
        # gumbel_samples = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True, dim=0)
        # control_idx = torch.argmax(gumbel_samples).item()
        # # We compute log probability from the softmax probabilities
        # probs = torch.softmax(logits, dim=0)
        sample, _ = StochasticCategorical.apply(logits.unsqueeze(0), None)
        control_idx = sample.item()
        log_prob = torch.log(logits[control_idx] + 1e-8)
        return control_idx, log_prob
    
    def sample_control_gumbel(self, state, policy_params, tau=1.0):
        """
        Uses the Gumbel-Softmax estimator (with straight-through sampling).
        """
        logits = self.compute_control_logits(state, policy_params)
        # PyTorch has a built-in implementation for gumbel softmax.
        gumbel_samples = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True, dim=0)
        control_idx = torch.argmax(gumbel_samples).item()
        # We compute log probability from the softmax probabilities
        probs = torch.softmax(logits, dim=0)
        log_prob = torch.log(probs[control_idx] + 1e-8)
        return control_idx, log_prob

    def sample_control(self, state, policy_params, method="custom", tau=1.0):
        """
        Samples a discrete control action and returns both the control index and the differentiable log probability.
        method: "custom" (default) or "gumbel"
        """
        if method == "cat++":
            return self.sample_control_cat(state, policy_params, tau)
        elif method == "gumbel":
            return self.sample_control_gumbel(state, policy_params, tau)
        else:
            return self.sample_control_custom(state, policy_params)

    def run_episode(self, policy_params, max_steps=200, estimator_method="custom", tau=1.0, render=False):
        """
        Runs an episode, collecting log probabilities and rewards at each step.
        """
        state = torch.tensor([math.pi, 0.0], dtype=torch.float32, device=self.device)
        states = [state]
        controls = []
        log_probs = []
        rewards = []

        for _ in range(max_steps):
            control_idx, log_prob = self.sample_control(state, policy_params, method=estimator_method, tau=tau)
            controls.append(control_idx)
            log_probs.append(log_prob)
            state = self.step(state, control_idx)
            states.append(state)
            # Reward: best when pendulum is upright (theta near zero).
            reward = torch.cos(state[0])
            rewards.append(reward)

        if render:
            self.render_episode(states, controls)
        return states, controls, log_probs, rewards

    def render_episode(self, states, controls):
        thetas = [s[0].detach().cpu().numpy() for s in states]
        torques = [self.control_values[idx].item() for idx in controls] + [0]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.plot(thetas)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Angle (rad)')
        ax1.set_title('Pendulum Angle Over Time')
        ax1.grid(True)

        ax2.step(range(len(torques)), torques)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Control Torque')
        ax2.set_title('Control Actions')
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig("pendulum_episode.png")
        plt.show()

# Optimization routine using the REINFORCE surrogate gradient.
def optimize_pendulum_policy(pendulum, n_episodes=100, lr=0.01, estimator_method="custom", tau=1.0):
    device = pendulum.device
    input_size = 3    # features: sin(theta), cos(theta), omega
    hidden_size = 8
    output_size = pendulum.n_controls

    policy_params = {
        'W1': torch.nn.Parameter(torch.randn(hidden_size, input_size, device=device) * 0.1),
        'b1': torch.nn.Parameter(torch.zeros(hidden_size, device=device)),
        'W2': torch.nn.Parameter(torch.randn(output_size, hidden_size, device=device) * 0.1),
        'b2': torch.nn.Parameter(torch.zeros(output_size, device=device))
    }

    optimizer = torch.optim.Adam(policy_params.values(), lr=lr)
    rewards_history = []

    for episode in range(n_episodes):
        optimizer.zero_grad()
        _, _, log_probs, rewards = pendulum.run_episode(policy_params, max_steps=200, estimator_method=estimator_method, tau=tau)
        # Using REINFORCE loss.
        loss = -sum([lp * r for lp, r in zip(log_probs, rewards)])
        loss.backward()
        optimizer.step()

        total_reward = sum([r.item() for r in rewards])
        rewards_history.append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

    return policy_params, rewards_history

# Benchmark routine that compares both estimators.
def benchmark(n_controls_list, n_episodes=100, n_runs=4, lr=0.01, tau=1.0, device="cpu"):
    all_results = {"categorical": {}, "cat++": {}, "gumbel": {}}
    for method in [ "cat++","categorical","gumbel"]:
        print(f"\nBenchmarking using the {method} estimator:")
        for n_controls in n_controls_list:
            print(f"  n_controls = {n_controls}:")
            run_rewards = []
            run_times = []
            for run in range(n_runs):
                print(f"    Run {run+1}/{n_runs}: ", end="", flush=True)
                pendulum = DiscreteControlPendulum(n_controls=n_controls, device=device)
                start_time = time.time()
                policy_params, rewards_history = optimize_pendulum_policy(
                    pendulum,
                    n_episodes=n_episodes,
                    lr=lr,
                    estimator_method=method,
                    tau=tau
                )
                elapsed = time.time() - start_time
                _, _, _, rewards = pendulum.run_episode(policy_params, max_steps=200, estimator_method=method, tau=tau)
                final_reward = sum([r.item() for r in rewards])
                run_rewards.append(final_reward)
                run_times.append(elapsed)
                print(f"Final Reward = {final_reward:.2f}, Time = {elapsed:.2f} sec")
                # plt.figure(figsize=(8, 5))
                # plt.plot(rewards_history)
                # plt.xlabel("Episode")
                # plt.ylabel("Total Reward")
                # plt.title(f"Estimator: {method}, n_controls={n_controls}, Run {run+1}")
                # plt.grid(True)
                # plt.show()
                # plt.close()

            avg_reward = np.mean(run_rewards)
            avg_time = np.mean(run_times)
            all_results[method][n_controls] = {"avg_final_reward": avg_reward, "avg_time_sec": avg_time}
            print(f"    Average final reward: {avg_reward:.2f}, Average time: {avg_time:.2f} sec")
    return all_results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_controls_list = [3, 5, 7, 11, 21]
    results = benchmark(n_controls_list=n_controls_list, n_episodes=100, n_runs=4, lr=0.1, tau=1.0, device=device)
    print("\nBenchmark Results:")
    for method in results:
        print(f"\nEstimator: {method}")
        for n_controls, metrics in results[method].items():
            print(f"  n_controls={n_controls}: Average Final Reward={metrics['avg_final_reward']:.2f}, "
                  f"Average Time={metrics['avg_time_sec']:.2f} sec")
