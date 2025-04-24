import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from torch.distributions import Categorical

class myCategorical(torch.autograd.Function):
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
        one_hot.scatter_(1, result, 1.0)
        p_safe = p.clamp(min=1e-6, max=1.0-1e-6)
        weights = one_hot * (1/(2*p_safe)) + (1-one_hot)*(1/(2*(1-p_safe)))
        return grad_output.expand_as(p) * weights
class StochasticCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, uncertainty_in=None):
        # Sample using multinomial
        result = torch.multinomial(p, num_samples=1)
        
        # Create one-hot encoding
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        
        # Calculate variance for uncertainty propagation
        var_chosen = (1.0 - p) / p.clamp(min=1e-6)
        var_not_chosen = p / (1.0 - p).clamp(min=1e-6)
        op_variance = one_hot * var_chosen + (1 - one_hot) * var_not_chosen
        
        if uncertainty_in is not None:
            uncertainty_out = uncertainty_in + op_variance
        else:
            uncertainty_out = op_variance
            
        # Save tensors for backward
        ctx.save_for_backward(one_hot, p, uncertainty_out)
        
        # Return the one-hot vector instead of indices to maintain gradients
        return one_hot, uncertainty_out

    @staticmethod
    def backward(ctx, grad_output, grad_uncertainty=None):
        one_hot, p, uncertainty = ctx.saved_tensors
        
        w_chosen = (1.0 / p.clamp(min=1e-6)) / 2
        w_non_chosen = (1.0 / (1.0 - p).clamp(min=1e-6)) / 2
        confidence = 1.0 / (1.0 + uncertainty.clamp(min=1e-6))
        
        adjusted_ws = (one_hot * w_chosen + (1 - one_hot) * w_non_chosen) * confidence
        adjusted_ws = adjusted_ws / adjusted_ws.mean().clamp(min=1e-10)
        
        grad_p = grad_output * adjusted_ws
        
        return grad_p, None
# -------------------------------
# Baseline Random Walk 
# -------------------------------
def generate_long_baseline(total_steps=1000, p_param=20.0, num_simulations=1000, device='cuda'):
    """Generate baseline trajectories with full position tracking"""
    full_trajectories = torch.zeros((num_simulations, total_steps), device=device)
    final_positions = torch.zeros(num_simulations, device=device)
    
    torch.manual_seed(42)
    theta = 20.0
    for sim in range(num_simulations):
        x = 0.0
        for step in range(total_steps):
            q = math.exp(-(x + theta) / p_param)
            q = min(max(q, 1e-6), 1.0)
            sample = torch.bernoulli(torch.tensor([q], device=device)).item()
            action = 1 if sample == 1.0 else -1
            # Ensure boundary condition at the start
            if x + theta == 0 and action == -1:
                action = 1
            x += action
            full_trajectories[sim, step] = x
        final_positions[sim] = x
    
    split_idx = int(total_steps * 0.7)
    return {
        'train_mean': full_trajectories[:, :split_idx].mean(0),
        'test_mean': full_trajectories[:, split_idx:].mean(0),
        'baseline_mean': full_trajectories.mean(0),
        'final_positions': final_positions,
        'full_trajectories': full_trajectories
    }

# -------------------------------
# Trajectory Simulation
# -------------------------------
def simulate_full_trajectory(theta, num_steps=1000, p_param=20.0, 
                            device='cpu', estimator='categorical', tau=0.1):
    """Simulate full trajectory with specified estimator"""
    trajectory = torch.zeros(num_steps, device=device)
    x = torch.tensor([0.0], device=device)
    
    for step in range(num_steps):
        q = torch.exp(-(x + theta) / p_param).clamp(1e-6, 1.0-1e-6)
        prob = torch.cat([q, 1.0 - q]).view(1, -1)
        if estimator == 'categorical':
            sample = myCategorical.apply(prob)  
            move = 1 - 2 * sample  # If sample==0, move=+1; if sample==1, move=-1.
            move = move.squeeze()
            # print(prob)
        elif estimator == 'gumbel':
            sample = F.gumbel_softmax(torch.log(prob), tau=tau, hard=True)
            move = 1 - 2 * sample[:, 0]
            move = move.squeeze()
        else:  # StochasticCategorical
            one_hot, _ = StochasticCategorical.apply(prob, None)
            # Use one-hot directly since it's now a differentiable tensor
            move = 2 * one_hot[:, 0] - 1  # If first prob is chosen, move=+1; else move=-1
            move = move.squeeze()
        
        # Boundary condition: when at starting boundary, force a +1 step.
        if x + theta == 0 and move == -1:
            move = 1
        x += move
        # print(x)
        trajectory[step] = x
        
    return trajectory

# -------------------------------
# Training and Evaluation Loop
# -------------------------------
def train_evaluate_estimator(estimator_type, baseline_stats, num_epochs=2, 
                            lr=0.01, num_simulations=10, device='cpu'):
    """Full training and evaluation workflow"""
    # Unpack baseline statistics (ignoring variances for now)
    (train_mean, _, test_mean, _) = baseline_stats
    
    theta = torch.tensor([20.0], device=device, requires_grad=True)
    optimizer = optim.Adam([theta], lr=lr)
    
    train_losses = []
    test_losses = []
    theta_history = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0
        
        for _ in range(num_simulations):
            traj = simulate_full_trajectory(theta, estimator=estimator_type, device=device)
            # print(traj)
            train_loss = F.mse_loss(traj[:len(train_mean)], train_mean)
            test_loss = F.mse_loss(traj[len(train_mean):], test_mean)
            # print(train_loss)
            train_loss.backward()
            
            epoch_train_loss += train_loss.item()
            epoch_test_loss += test_loss.item()
        
        optimizer.step()
        
        avg_train_loss = epoch_train_loss / num_simulations
        avg_test_loss = epoch_test_loss / num_simulations
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        theta_history.append(theta.item())
        
        print(f"Epoch {epoch+1}/{num_epochs} | {estimator_type}")
        print(f"Train MSE: {avg_train_loss:.4f} | Test MSE: {avg_test_loss:.4f}")
        print(f"Theta: {theta.item():.4f}\n")
    
    return train_losses, test_losses, theta_history

def wasserstein_distance(sim_final, base_final):
    sorted_sim = torch.sort(sim_final)[0]
    sorted_base = torch.sort(base_final)[0]
    min_length = min(len(sorted_sim), len(sorted_base))
    return torch.abs(sorted_sim[:min_length] - sorted_base[:min_length]).mean()

def evaluate_model(theta, estimator_type, baseline_data, num_samples=1000, device='cpu'):
    with torch.no_grad():
        sim_final = torch.stack([
            simulate_full_trajectory(theta, device=device, estimator=estimator_type)[-1]
        ])
    
    base_final = baseline_data['final_positions']
    
    return {
        'wasserstein': wasserstein_distance(sim_final, base_final),
        'mean_absolute_error': F.mse_loss(sim_final.mean(), base_final.mean()),
        'variance_ratio': sim_final.var() / base_final.var()
    }

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate baseline data for 1000 steps.
    baseline_data = generate_long_baseline(total_steps=1000, device=device)
    num_epochs = 10
    results = {}
    final_thetas = {}
    for estimator in ['cat++','categorical', 'gumbel']:
        train_loss, test_loss, theta_hist = train_evaluate_estimator(
            estimator, 
            (baseline_data['train_mean'], None, baseline_data['test_mean'], None),
            num_epochs=num_epochs,device=device
        )
        final_theta = torch.tensor([theta_hist[-1]], device=device)
        final_thetas[estimator] = final_theta
        print(f"------Estimator: {estimator}------")
        print(f"average train loss: {sum(train_loss)/len(train_loss)}")
        print(f"average test loss: {sum(test_loss)/len(test_loss)}")
        print(f" Minimum Test Loss: {min(test_loss)}")
        print(f"Final theta: {final_theta.item()}")
        print("-------------------------")
    #     metrics = evaluate_model(final_theta, estimator, baseline_data, device=device)
        
    #     results[estimator] = {
    #         'train_loss': train_loss,
    #         'test_loss': test_loss,
    #         'theta': theta_hist,
    #         **metrics
    #     }
    
    # # Display final benchmark results.
    # print("\nFinal Benchmark Results:")
    # for est in results:
    #     print(f"{est.upper()} Estimator:")
    #     print(f"  - Wasserstein Distance: {results[est]['wasserstein']:.4f}")
    #     print(f"  - Mean Absolute Error: {results[est]['mean_absolute_error']:.4f}")
    #     print(f"  - Variance Ratio: {results[est]['variance_ratio']:.4f}")
    #     print(f"  - Final Theta: {results[est]['theta'][-1]:.4f}\n")
    
    # # -------------------------------
    # # Plot trajectories for comparison
    # # -------------------------------
    # # Use the baseline average trajectory
    # baseline_mean = baseline_data['baseline_mean'].cpu().numpy()
    # steps = range(len(baseline_mean))
    
    # Simulate one trajectory using the final theta for each estimator.
    # categorical_traj = simulate_full_trajectory(final_thetas['categorical'], device=device, estimator='categorical').cpu().numpy()
    # gumbel_traj = simulate_full_trajectory(final_thetas['gumbel'], device=device, estimator='gumbel').cpu().numpy()
    # scat_traj = simulate_full_trajectory(final_thetas['cat++'], device=device, estimator='cat++').cpu().numpy()
    # baseline_traj = baseline_data['full_trajectories'][0].cpu().numpy()
    # plt.figure(figsize=(10, 6))
    # plt.plot(steps, baseline_traj, label="Baseline Trajectory", linewidth=2)
    # plt.plot(steps, categorical_traj, label="Categorical Trajectory", linestyle="--")
    # plt.plot(steps, gumbel_traj, label="Gumbel Trajectory", linestyle=":")
    # plt.plot(steps, scat_traj, label="Categorical++ Trajectory", linestyle="-.")
    # plt.xlabel("Step")
    # plt.ylabel("Position")
    # plt.title("Comparison of 1000-Step Random Walk Trajectories")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
