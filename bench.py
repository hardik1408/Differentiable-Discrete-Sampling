import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt

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
        one_hot.scatter_(1, result, 1.0)
        p_safe = p.clamp(min=1e-6, max=1.0-1e-6)
        weights = one_hot * (1/(2*p_safe)) + (1-one_hot)*(1/(2*(1-p_safe)))
        return grad_output.expand_as(p) * weights

# -------------------------------
# Baseline Random Walk 
# -------------------------------
def generate_long_baseline(total_steps=100, p_param=20.0, num_simulations=1000, device='cuda'):
    """Generate baseline trajectories with full position tracking"""
    full_trajectories = torch.zeros((num_simulations, total_steps), device=device)
    final_positions = torch.zeros(num_simulations, device=device)
    
    torch.manual_seed(42)
    for sim in range(num_simulations):
        x = 0.0
        for step in range(total_steps):
            q = math.exp(-x / p_param)
            q = min(max(q, 1e-6), 1.0)
            sample = torch.bernoulli(torch.tensor([q], device=device)).item()
            action = 1 if sample == 1.0 else -1
            if x == 0 and action == -1:
                action = 1
            x += action
            full_trajectories[sim, step] = x
        final_positions[sim] = x
    
    split_idx = int(total_steps * 0.7)
    return {
        'train_mean': full_trajectories[:, :split_idx].mean(0),
        'test_mean': full_trajectories[:, split_idx:].mean(0),
        'final_positions': final_positions,
        'full_trajectories': full_trajectories
    }

# -------------------------------
# trajectory simulation
# -------------------------------
def simulate_full_trajectory(theta, num_steps=100, p_param=20.0, 
                            device='cpu', estimator='categorical', tau=0.1):
    """Simulate full trajectory with specified estimator"""
    trajectory = torch.zeros(num_steps, device=device)
    x = torch.tensor([0.0], device=device)
    
    for step in range(num_steps):
        # print(estimator)
        q = torch.exp(-(x.detach() + theta) / p_param).clamp(1e-6, 1.0-1e-6)
        prob = torch.cat([q, 1.0 - q]).view(1, -1)
        
        if estimator == 'categorical':
            sample = Categorical.apply(prob)  
            move = 1 - 2 * sample  # If sample==0, move=+1; if sample==1, move=-1.
            move = move.squeeze()  #
        elif estimator == 'gumbel':
            sample = F.gumbel_softmax(torch.log(prob), tau=tau, hard=True)
            move = 1 - 2 * sample[:, 0]
            move = move.squeeze()
        else:
            raise ValueError("Unknown estimator type.")
        
        # print(move.shape)
        # move = torch.where(x < 1e-6, torch.tensor(1.0, device=device), move)
        # print(x.shape, move.shape)
        if x == 0 and move == -1:
            move = 1
        x += move
        trajectory[step] = x
        
    return trajectory

# -------------------------------
# Training and Evaluation Loop
# -------------------------------
def train_evaluate_estimator(estimator_type, baseline_stats, num_epochs=100, 
                            lr=0.001, num_simulations=10, device='cpu'):
    """Full training and evaluation workflow"""
    # Unpack baseline statistics
    (train_mean, train_var, 
     test_mean, test_var) = baseline_stats
    
    theta = torch.tensor([0.0], device=device, requires_grad=True)
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
            
            train_loss = F.mse_loss(traj[:len(train_mean)], train_mean)
            # print(train_loss)
            test_loss = F.mse_loss(traj[len(train_mean):], test_mean)
            
            train_loss.backward(retain_graph=True)
            
            epoch_train_loss += train_loss.item()
            epoch_test_loss += test_loss.item()
        
        optimizer.step()
        
        # Store metrics
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
            for _ in range(num_samples)
        ])
    
    base_final = baseline_data['final_positions']
    
    return {
        'wasserstein': wasserstein_distance(sim_final, base_final),
        'mean_absolute_error': F.l1_loss(sim_final.mean(), base_final.mean()),
        'variance_ratio': sim_final.var() / base_final.var()
    }
# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    baseline_data = generate_long_baseline(device=device)   # baseline data
    
    results = {}
    for estimator in ['categorical', 'gumbel']:
        train_loss, test_loss, theta_hist = train_evaluate_estimator(
            estimator, 
            (baseline_data['train_mean'], None, baseline_data['test_mean'], None),
            device=device
        )
        
        final_theta = torch.tensor([theta_hist[-1]], device=device)
        metrics = evaluate_model(final_theta, estimator, baseline_data, device=device)
        
        results[estimator] = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'theta': theta_hist,
            **metrics
        }
    
    # Plot results
    print("\nFinal Benchmark Results:")
    for est in results:
        print(f"{est.upper()} Estimator:")
        print(f"  - Wasserstein Distance: {results[est]['wasserstein']:.4f}")
        print(f"  - Mean Absolute Error: {results[est]['mean_absolute_error']:.4f}")
        print(f"  - Variance Ratio: {results[est]['variance_ratio']:.4f}")
        print(f"  - Final Theta: {results[est]['theta'][-1]:.4f}\n")    
        print("\nFinal Evaluation:")

    plt.figure(figsize=(12, 6))
    for estimator in results:
        plt.plot(results[estimator]['train_loss'], '--', label=f'{estimator} (Train)')
        plt.plot(results[estimator]['test_loss'], '-', label=f'{estimator} (Test)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training and Test MSE Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    for estimator in results:
        plt.plot(results[estimator]['theta'], label=f'{estimator}')
    plt.xlabel('Epoch')
    plt.ylabel('Theta Value')
    plt.title('Parameter Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()
