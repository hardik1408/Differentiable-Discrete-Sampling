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
        # ensure p is [batch, n_choices]
        if p.dim() == 1:
            p = p.unsqueeze(0)

        # sample an index (long tensor)
        idx = torch.multinomial(p, num_samples=1)         # dtype=torch.int64
        # build one-hot from idx
        one_hot = torch.zeros_like(p).scatter_(1, idx, 1.0)

        # save the index and the original probabilities
        ctx.save_for_backward(idx, p)
        return one_hot

    @staticmethod
    def backward(ctx, grad_output):
        idx, p = ctx.saved_tensors   # idx is int64, p is float
        # rebuild one-hot
        one_hot = torch.zeros_like(p).scatter_(1, idx, 1.0)

        # clamp to avoid division by zero
        p_safe = p.clamp(min=1e-6, max=1.0-1e-6)

        # REINFORCE‑style weights: ∇ₚ log p_i = 1/p_i  if i was chosen, else 0
        # but you had a symmetric form; here’s one common choice:
        weights = one_hot * (1.0 / p_safe)

        # multiply by incoming gradient
        grad_p = grad_output * weights
        return grad_p

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
def generate_long_baseline(total_steps=1000, p_param=20.0, n_choices=5, num_simulations=1000, device='cuda'):
    """Generate baseline trajectories with full position tracking (n-ary random walk with multinomial sampling)"""
    full_trajectories = torch.zeros((num_simulations, total_steps), device=device)
    final_positions = torch.zeros(num_simulations, device=device)
    
    torch.manual_seed(42)
    theta = 20.0

    # Define possible moves
    if n_choices % 2 == 1:
        moves = torch.arange(-(n_choices//2), n_choices//2 + 1, device=device, dtype=torch.float32)
    else:
        moves = torch.arange(-(n_choices-1)/2, n_choices/2 + 0.1, 1.0, device=device)
    
    for sim in range(num_simulations):
        x = torch.tensor(0.0, device=device)
        for step in range(total_steps):
            # Compute move probabilities
            probs = torch.exp(-(torch.abs(x + moves) - torch.abs(x) + theta) / p_param)
            probs = probs / probs.sum()

            # Sample move using multinomial
            one_hot = torch.multinomial(probs, num_samples=1)
            move = moves[one_hot.squeeze()]
            
            x = x + move
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
                            device='cpu', estimator='categorical', tau=0.1, n_choices=5):
    """Simulate full trajectory with specified estimator"""
    trajectory = torch.zeros(num_steps, device=device)
    x = torch.tensor([0.0], device=device)

    if n_choices % 2 == 1:
        moves = torch.arange(-(n_choices//2), n_choices//2 + 1, device=device, dtype=torch.float32,requires_grad=True)
    else:
        moves = torch.arange(-(n_choices-1)/2, n_choices/2 + 0.1, 1.0, device=device)

    for step in range(num_steps):
        probs = torch.zeros(n_choices, dtype=torch.float32, device=device)
        for i in range(n_choices):
            direction = moves[i]
            prob_factor = torch.exp(-(torch.abs(x + direction) - torch.abs(x) + theta) / p_param)
            probs[i] = prob_factor
        
        probs = probs / probs.sum()
        prob = probs.view(1, -1)  # Shape [1, n_choices]
        
        if estimator=='categorical':
            sample = myCategorical.apply(prob)      # one‐hot [1, n_choices]
            move   = (sample @ moves.view(-1,1)).squeeze()
        elif estimator == 'gumbel':
            sample = F.gumbel_softmax(torch.log(prob), tau=tau, hard=True)
            move = (sample @ moves.view(-1, 1)).squeeze()
        else:
            sample,_ = StochasticCategorical.apply(prob)
            move = (sample @ moves.view(-1, 1)).squeeze()
        
        x = x + move
        # print(x)
        trajectory[step] = x
    # print(trajectory[0])
    return trajectory



# -------------------------------
# Training and Evaluation Loop
# -------------------------------
def train_evaluate_estimator(estimator_type, baseline_stats, num_epochs=10, 
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
            # for _ in range(num_samples)
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
    print("generating baseline data")
    baseline_data = generate_long_baseline(total_steps=1000, device=device)
    
    results = {}
    final_thetas = {}
    for estimator in ['cat++','categorical', 'gumbel']:
        print(f"training {estimator}")
        train_loss, test_loss, theta_hist = train_evaluate_estimator(
            estimator, 
            (baseline_data['train_mean'], None, baseline_data['test_mean'], None),
            device=device
        )
        print("train loss",sum(train_loss)/len(train_loss))
        print("test loss",sum(test_loss)/len(test_loss))
        print(min(test_loss))
        final_theta = torch.tensor([theta_hist[-1]], device=device)
        final_thetas[estimator] = final_theta
        print(f"------Estimator: {estimator}------")
        print(f"average train loss: {sum(train_loss)/len(train_loss)}")
        print(f"average test loss: {sum(test_loss)/len(test_loss)}")
        print(f" Minimum Test Loss: {min(test_loss)}")
        print("-------------------------")

