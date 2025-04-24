import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

# -------------------------------
# Correlated random walks
# -------------------------------
def correlated_random_walk_categorical(n_steps, correlation_strength, p, device, n_choices=3, memory_length=5):
    """
    Implements a random walk with correlated noise where each step depends on previous steps.
    
    Args:
        n_steps: Number of time steps to simulate
        correlation_strength: How strongly past moves influence current ones (0 to 1)
        p: Scaling parameter for transition probabilities
        device: Torch device
        n_choices: Number of possible moves at each step
        memory_length: How many previous steps influence the current one
    
    Returns:
        path: List of positions
        move_history: Tensor of move history
    """
    x = 0.0  # initial state
    path = [0.0]
    
    # Define possible moves
    if n_choices % 2 == 1:
        moves = torch.arange(-(n_choices//2), n_choices//2 + 1, device=device)
    else:
        moves = torch.arange(-(n_choices-1)/2, n_choices/2 + 0.1, 1.0, device=device)
    
    # Initialize memory buffer with zeros (no preferred direction at the start)
    move_history = torch.zeros(memory_length, device=device)
    
    for step in range(n_steps):
        # Base probabilities depend on current position - moves that take the walker further from origin are less likely
        base_probs = torch.zeros(n_choices, dtype=torch.float32, device=device)
        for i in range(n_choices):
            direction = moves[i]
            # More likely to move toward the origin
            prob_factor = math.exp(-(abs(x + direction) - abs(x)) / p)
            base_probs[i] = prob_factor
        base_probs = base_probs / base_probs.sum()
        
        # Calculate correlated probabilities based on history if available
        if step > 0:
            # Weighted average of previous moves (recent moves have stronger influence)
            weights = torch.tensor([math.pow(0.8, memory_length - i - 1) 
                                   for i in range(memory_length)], device=device)
            weighted_history = (move_history * weights).sum() / weights.sum()
            
            # Bias toward continuing in the same direction as recent history
            correlated_probs = torch.zeros(n_choices, dtype=torch.float32, device=device)
            for i in range(n_choices):
                move_val = moves[i]
                similarity = -abs(move_val - weighted_history)
                correlated_probs[i] = math.exp(similarity)
            correlated_probs = correlated_probs / correlated_probs.sum()
            
            # Combine base and correlated probabilities
            final_probs = (1 - correlation_strength) * base_probs + correlation_strength * correlated_probs
        else:
            final_probs = base_probs
        
        # Sample a move from the categorical distribution
        sample = Categorical.apply(final_probs.unsqueeze(0))
        move_idx = sample.item()
        move_val = moves[move_idx].item()
        
        # Update position and record the path
        x += move_val
        path.append(x)
        
        # Update move history (shift and add new move)
        if memory_length > 0:
            move_history = torch.cat([move_history[1:], torch.tensor([move_val], device=device)])
    
    return path, move_history

def correlated_random_walk_stochcat(n_steps, corr, p, device, n_choices=3, mem_len=5):
    """
    Same as correlated_random_walk_categorical but uses StochasticCategorical.
    """
    x = 0.0
    path = [0.0]
    if n_choices % 2 == 1:
        moves = torch.arange(-(n_choices//2), n_choices//2+1, device=device)
    else:
        moves = torch.arange(-(n_choices-1)/2, n_choices/2+0.1, 1.0, device=device)
    move_history = torch.zeros(mem_len, device=device)

    for step in range(n_steps):
        # Base probabilities depend on current position - moves that take the walker further from origin are less likely
        base_probs = torch.zeros(n_choices, dtype=torch.float32, device=device)
        for i in range(n_choices):
            direction = moves[i]
            # More likely to move toward the origin
            prob_factor = math.exp(-(abs(x + direction) - abs(x)) / p)
            base_probs[i] = prob_factor
        base_probs = base_probs / base_probs.sum()
        
        # Calculate correlated probabilities based on history if available
        if step > 0:
            # Weighted average of previous moves (recent moves have stronger influence)
            weights = torch.tensor([math.pow(0.8, mem_len - i - 1) 
                                   for i in range(mem_len)], device=device)
            weighted_history = (move_history * weights).sum() / weights.sum()
            
            # Bias toward continuing in the same direction as recent history
            correlated_probs = torch.zeros(n_choices, dtype=torch.float32, device=device)
            for i in range(n_choices):
                move_val = moves[i]
                similarity = -abs(move_val - weighted_history)
                correlated_probs[i] = math.exp(similarity)
            correlated_probs = correlated_probs / correlated_probs.sum()
            
            # Combine base and correlated probabilities
            final_probs = (1 - corr) * base_probs + corr * correlated_probs
        else:
            final_probs = base_probs
        sample, _ = StochasticCategorical.apply(final_probs.unsqueeze(0), None)
        move_idx = sample.item()

        move_val = moves[move_idx].item()
        x += move_val
        path.append(x)

        if mem_len > 0:
            move_history = torch.cat([move_history[1:], torch.tensor([move_val], device=device)])

    return path, move_history

def analyze_correlated_walks(correlation_range=[0.0, 0.3, 0.6, 0.9], n_steps=100, tau=10):
    """
    Run and analyze correlated random walks using both categorical and Gumbel sampling methods.
    Plots the trajectories and autocorrelations side by side for comparison.
    
    Args:
        correlation_range: List of correlation strength values.
        n_steps: Number of time steps for the walk.
        tau: Temperature parameter for the Gumbel softmax sampling.
    
    Returns:
        A dictionary with results for both sampling methods.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = n_steps  # scaling parameter
    
    # Dictionaries to store paths for each sampling method.
    categorical_results = {}
    gumbel_results = {}
    
    # Run simulations for each correlation strength.
    for correlation in correlation_range:
        cat_path, _ = correlated_random_walk_categorical(
            n_steps=n_steps,
            correlation_strength=correlation,
            p=p,
            device=device
        )
        gum_path, _ = correlated_random_walk_stochcat(
            n_steps=n_steps,
            corr=correlation,
            p=p,
            device=device
        )
        categorical_results[correlation] = cat_path
        gumbel_results[correlation] = gum_path
    
    # Plot trajectory comparisons.
    n_plots = len(correlation_range)
    plt.figure(figsize=(15, 10))
    
    for i, correlation in enumerate(correlation_range):
        plt.subplot(2, 2, i+1)
        # Plot categorical walk (solid line)
        plt.plot(categorical_results[correlation], label="Categorical", color='blue')
        # Plot Gumbel walk (dashed line)
        plt.plot(gumbel_results[correlation], label="categorical++", color='red')
        plt.title(f"Correlation Strength: {correlation}")
        plt.xlabel("Time Step")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("stoch/correlated_walks_comparison.png")
    plt.show()
    
    # Plot autocorrelation comparisons for step sizes.
    plt.figure(figsize=(15, 10))
    
    for i, correlation in enumerate(correlation_range):
        plt.subplot(2, 2, i+1)
        # Compute autocorrelation for categorical
        path_cat = np.array(categorical_results[correlation])
        steps_cat = np.diff(path_cat)
        autocorr_cat = np.correlate(steps_cat, steps_cat, mode='full')
        autocorr_cat = autocorr_cat[len(autocorr_cat)//2:]
        autocorr_cat = autocorr_cat / (autocorr_cat[0] if autocorr_cat[0] != 0 else 1)
        
        # Compute autocorrelation for Gumbel
        path_gum = np.array(gumbel_results[correlation])
        steps_gum = np.diff(path_gum)
        autocorr_gum = np.correlate(steps_gum, steps_gum, mode='full')
        autocorr_gum = autocorr_gum[len(autocorr_gum)//2:]
        autocorr_gum = autocorr_gum / (autocorr_gum[0] if autocorr_gum[0] != 0 else 1)
        
        # Plot both autocorrelations
        lags = np.arange(len(autocorr_cat))
        plt.plot(lags[:30], autocorr_cat[:30], label="Categorical", color='blue')
        plt.plot(lags[:30], autocorr_gum[:30], label="categorical++", color='red')
        plt.title(f"Autocorrelation (Correlation={correlation})")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("stoch/step_autocorrelation_comparison.png")
    plt.show()
    
    return {"categorical": categorical_results, "gumbel": gumbel_results}
# -------------------------------
# Gradient‐flow benchmark
# -------------------------------
def test_gradient_flow(n_trials=5, n_steps=100, tau=10.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correlation_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    p = n_steps
    target = 20.0
    mem_len = 10
    n_choices = 3

    results = {}
    methods = ['categorical', 'stochcat' , 'gumbel']

    for corr in correlation_range:
        print(f"\n--- Correlation = {corr} ---")
        results[corr] = {}

        for method in methods:
            # initialize a learnable starting value
            init_val = torch.tensor([10.0], requires_grad=True, device=device)
            optimizer = torch.optim.Adam([init_val], lr=0.01)
            loss_history = []

            for trial in range(n_trials):
                optimizer.zero_grad()
                x = init_val.clone()
                move_history = torch.zeros(mem_len, device=device)

                # define moves
                if n_choices % 2 == 1:
                    moves = torch.arange(-(n_choices//2), n_choices//2+1, device=device)
                else:
                    moves = torch.arange(-(n_choices-1)/2, n_choices/2+0.1, 1.0, device=device)

                for step in range(n_steps):
                    # build final_probs identical to your existing code
                    base_probs = torch.zeros(n_choices, device=device)
                    for i in range(n_choices):
                        d = moves[i]
                        delta = torch.abs(x + d) - torch.abs(x)
                        base_probs[i] = torch.exp(-delta / p)
                    base_probs = base_probs / base_probs.sum()

                    if step > 0:
                        weights = torch.tensor([0.8**(mem_len-i-1) for i in range(mem_len)], device=device)
                        w_hist = (move_history * weights).sum() / weights.sum()
                        corr_probs = torch.zeros_like(base_probs)
                        for i in range(n_choices):
                            corr_probs[i] = math.exp(-abs(moves[i] - w_hist.item()))
                        corr_probs = corr_probs / corr_probs.sum()
                        final_probs = (1-corr)*base_probs + corr*corr_probs
                    else:
                        final_probs = base_probs

                    # ---- sampling dispatch ----
                    if method == 'categorical':
                        sample = Categorical.apply(final_probs.unsqueeze(0))
                        idx = sample.item()
                        move_val = moves[idx].item()

                    elif method == 'stochcat':
                        sample, _ = StochasticCategorical.apply(final_probs.unsqueeze(0), None)
                        idx = sample.item()
                        move_val = moves[idx].item()

                    else:
                        gumbel_sample = F.gumbel_softmax(final_probs.unsqueeze(0), tau=tau, hard=True)
                        move_idx = gumbel_sample.argmax(dim=1).item()
                        move_val = moves[move_idx].item()

                    x = x + move_val
                    move_history = torch.cat([move_history[1:], torch.tensor([move_val], device=device)])

                loss = torch.abs(x - target)
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())

                print(f"{method:10s} | trial {trial:3d} | final x={x.item():.3f} | loss={loss.item():.3f}")

            results[corr][method] = {
                'loss_history': loss_history,
                'final_initial_value': init_val.item()
            }

    return results

if __name__ == "__main__":
    print("Running correlated random walk analysis...")
    # analyze_correlated_walks(correlation_range=[0.0, 0.3, 0.6, 0.9], n_steps=200)
    results = test_gradient_flow(n_trials=10, n_steps=100)
    for corr, res in results.items():
        print(f"\nCorrelation {corr}:")
        for m in res:
            print(f"  {m:10s} → final init value = {res[m]['final_initial_value']:.4f}")
