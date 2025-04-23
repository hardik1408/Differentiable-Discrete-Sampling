import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
# Custom differentiable categorical sampling function
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

def correlated_random_walk_gumbel(n_steps, correlation_strength, p, device, n_choices=3, memory_length=5, tau=1.0):
    """
    Implements a correlated random walk but uses Gumbel sampling via the gumbel_softmax function.
    
    Args:
        n_steps: Number of time steps to simulate.
        correlation_strength: How strongly past moves influence current ones (0 to 1).
        p: Scaling parameter for transition probabilities.
        device: Torch device.
        n_choices: Number of possible moves at each step.
        memory_length: Number of previous steps that influence the current one.
        tau: Temperature parameter for the gumbel softmax.
    
    Returns:
        path: List of positions.
        move_history: Tensor of move history.
    """
    x = 0.0  # initial state
    path = [0.0]
    
    if n_choices % 2 == 1:
        moves = torch.arange(-(n_choices // 2), n_choices // 2 + 1, device=device)
    else:
        moves = torch.arange(-(n_choices - 1) / 2, n_choices / 2 + 0.1, 1.0, device=device)
    
    move_history = torch.zeros(memory_length, device=device)
    
    for step in range(n_steps):
        # Base probabilities favoring moves that bring the walker toward the origin.
        base_probs = torch.zeros(n_choices, dtype=torch.float32, device=device)
        for i in range(n_choices):
            direction = moves[i]
            prob_factor = math.exp(-(abs(x + direction) - abs(x)) / p)
            base_probs[i] = prob_factor
        base_probs = base_probs / base_probs.sum()
        
        if step > 0:
            weights = torch.tensor([math.pow(0.8, memory_length - i - 1) for i in range(memory_length)], device=device)
            weighted_history = (move_history * weights).sum() / weights.sum()
            
            correlated_probs = torch.zeros(n_choices, dtype=torch.float32, device=device)
            for i in range(n_choices):
                move_val = moves[i]
                similarity = -abs(move_val - weighted_history)
                correlated_probs[i] = math.exp(similarity)
            correlated_probs = correlated_probs / correlated_probs.sum()
            
            final_probs = (1 - correlation_strength) * base_probs + correlation_strength * correlated_probs
        else:
            final_probs = base_probs
        
        # Sample a move using gumbel_softmax: output is a one-hot vector.
        gumbel_sample = F.gumbel_softmax(final_probs.unsqueeze(0), tau=tau, hard=True)
        move_idx = gumbel_sample.argmax(dim=1).item()
        move_val = moves[move_idx].item()
        
        x += move_val
        path.append(x)
        
        if memory_length > 0:
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
        gum_path, _ = correlated_random_walk_gumbel(
            n_steps=n_steps,
            correlation_strength=correlation,
            p=p,
            device=device,
            tau=tau
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
        plt.plot(gumbel_results[correlation], label="Gumbel", color='red')
        plt.title(f"Correlation Strength: {correlation}")
        plt.xlabel("Time Step")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("correlated_walks_comparison.png")
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
        plt.plot(lags[:30], autocorr_gum[:30], label="Gumbel", color='red', linestyle='--')
        plt.title(f"Autocorrelation (Correlation={correlation})")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("step_autocorrelation_comparison.png")
    plt.show()
    
    return {"categorical": categorical_results, "gumbel": gumbel_results}

def test_gradient_flow_benchmark(n_trials=30, n_steps=100, target=50.0, loss_threshold=5.0, smooth_weight=0.9):
    """
    Benchmarks gradient flow for both categorical and Gumbel methods.
    Logs and plots average performance across multiple runs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correlation_range = [0.0, 0.2, 0.4, 0.6, 0.8]
    p = n_steps
    memory_length = 5
    n_choices = 3

    def smooth(values, weight):
        smoothed = []
        last = values[0]
        for v in values:
            last = last * weight + (1 - weight) * v
            smoothed.append(last)
        return smoothed

    methods = ['categorical', 'gumbel']
    summary = {}

    for method in methods:
        summary[method] = {}
        for corr in correlation_range:
            print(f"\nMethod: {method}, Correlation: {corr}")
            final_losses = []
            all_loss_histories = []

            for trial in range(n_trials):
                initial_value = torch.tensor([0.0], requires_grad=True, device=device)
                optimizer = torch.optim.Adam([initial_value], lr=0.01)
                move_history = torch.zeros(memory_length, device=device)
                moves = torch.arange(-(n_choices // 2), n_choices // 2 + 1, device=device)
                loss_history = []

                for step in range(n_steps):
                    optimizer.zero_grad()
                    x = initial_value.clone()

                    probs = torch.zeros(n_choices, device=device)
                    for i in range(n_choices):
                        direction = moves[i]
                        delta = torch.abs(x + direction) - torch.abs(x)
                        probs[i] = torch.exp(-delta / p)
                    probs = probs / probs.sum()

                    if step > 0:
                        weights = torch.tensor([math.pow(0.8, memory_length - i - 1) for i in range(memory_length)], device=device)
                        weighted_history = (move_history * weights).sum() / weights.sum()

                        corr_probs = torch.zeros(n_choices, device=device)
                        for i in range(n_choices):
                            sim = -torch.abs(moves[i] - weighted_history)
                            corr_probs[i] = torch.exp(sim)
                        corr_probs = corr_probs / corr_probs.sum()

                        probs = (1 - corr) * probs + corr * corr_probs

                    if method == 'categorical':
                        sample = Categorical.apply(probs.unsqueeze(0)).item()
                        move_val = moves[sample].item()
                    elif method == 'gumbel':
                        tau = 0.5  # Could anneal
                        logits = torch.log(probs + 1e-10)
                        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
                        soft_sample = torch.nn.functional.softmax((logits + gumbel_noise) / tau, dim=0)
                        move_val = (soft_sample * moves).sum()

                    x = x + move_val
                    loss = torch.abs(x - target)
                    loss.backward()
                    optimizer.step()
                    loss_history.append(loss.item())

                    if memory_length > 0:
                        move_history = torch.cat([move_history[1:], torch.tensor([move_val], device=device)])

                final_losses.append(loss_history[-1])
                all_loss_histories.append(loss_history)

            # Aggregate stats
            final_losses = np.array(final_losses)
            mean_loss = final_losses.mean()
            std_loss = final_losses.std()
            success_rate = np.mean(final_losses < loss_threshold)

            summary[method][corr] = {
                'mean_final_loss': mean_loss,
                'std_final_loss': std_loss,
                'success_rate': success_rate,
                'loss_histories': all_loss_histories
            }

            print(f"  Final Loss Mean: {mean_loss:.3f} | Std: {std_loss:.3f} | Success (<{loss_threshold}): {success_rate:.2%}")

            # Plot
            plt.figure(figsize=(10, 4))
            all_losses = np.array(all_loss_histories)
            smoothed = np.array([smooth(l, smooth_weight) for l in all_losses])
            mean_curve = smoothed.mean(axis=0)
            std_curve = smoothed.std(axis=0)

            plt.plot(mean_curve, label=f"{method} (mean)")
            plt.fill_between(range(n_steps), mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)
            plt.title(f"Loss Curves (Corr={corr}, Method={method})")
            plt.xlabel("Optimization Step")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"gradient_benchmark_{method}_corr{corr}.png")
            plt.show()

    return summary

def test_gradient_flow(n_trials=5, n_steps=100, tau=10.0):
    """
    Test and benchmark gradient flow through the correlated random walk using both sampling methods.
    For each correlation strength, the function runs an optimization experiment for both the categorical
    and gumbel methods, and creates a separate plot comparing their loss histories over trials.

    Args:
        n_trials: Number of optimization steps (trials) per correlation strength.
        n_steps: Number of steps in the random walk for each trial.
        tau: Temperature parameter for the Gumbel softmax.
    
    Returns:
        results: Dictionary keyed by correlation strength, with loss history and final initial values
                 for both sampling methods.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correlation_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    p = n_steps  # Scaling parameter for probability computation
    target = 20.0  # Target end position for loss computation
    memory_length = 10
    n_choices = 3

    results = {}

    for correlation in correlation_range:
        print(f"\nTesting gradient flow with correlation = {correlation}")

        # Set up separate learnable initial values and optimizers for each sampling method
        initial_value_cat = torch.tensor([10.0], requires_grad=True, device=device)
        optimizer_cat = torch.optim.Adam([initial_value_cat], lr=0.01)

        initial_value_gum = torch.tensor([10.0], requires_grad=True, device=device)
        optimizer_gum = torch.optim.Adam([initial_value_gum], lr=0.01)

        loss_history_cat = []
        loss_history_gum = []

        for trial in range(n_trials):
            # ----- Categorical Sampling Path -----
            optimizer_cat.zero_grad()
            x_cat = initial_value_cat.clone()
            # print(x_cat)
            move_history_cat = torch.zeros(memory_length, device=device)
            # Define possible moves
            if n_choices % 2 == 1:
                moves = torch.arange(-(n_choices // 2), n_choices // 2 + 1, device=device)
            else:
                moves = torch.arange(-(n_choices - 1) / 2, n_choices / 2 + 0.1, 1.0, device=device)

            for step in range(n_steps):
                base_probs = torch.zeros(n_choices, dtype=torch.float32, device=device)
                for i in range(n_choices):
                    direction = moves[i]
                    # Use torch.abs for differentiability
                    prob_factor = torch.exp(-(torch.abs(x_cat + direction) - torch.abs(x_cat)) / p)
                    base_probs[i] = prob_factor
                base_probs = base_probs / base_probs.sum()

                if step > 0:
                    weights = torch.tensor([math.pow(0.8, memory_length - i - 1) for i in range(memory_length)], device=device)
                    weighted_history = (move_history_cat * weights).sum() / weights.sum()
                    correlated_probs = torch.zeros(n_choices, dtype=torch.float32, device=device)
                    for i in range(n_choices):
                        move_val = moves[i]
                        similarity = -abs(move_val - weighted_history.item())
                        correlated_probs[i] = math.exp(similarity)
                    correlated_probs = correlated_probs / correlated_probs.sum()
                    final_probs = (1 - correlation) * base_probs + correlation * correlated_probs
                else:
                    final_probs = base_probs

                sample = Categorical.apply(final_probs.unsqueeze(0))
                move_idx = sample.item()
                move_val = moves[move_idx].item()

                x_cat = x_cat + move_val
                # Update move history
                move_history_cat = torch.cat([move_history_cat[1:], torch.tensor([move_val], device=device)])

            loss_cat = torch.abs(x_cat - target)
            loss_cat.backward()
            optimizer_cat.step()
            loss_history_cat.append(loss_cat.item())

            # ----- Gumbel Sampling Path -----
            optimizer_gum.zero_grad()
            x_gum = initial_value_gum.clone()
            move_history_gum = torch.zeros(memory_length, device=device)
            # Reuse the same moves definition
            for step in range(n_steps):
                base_probs = torch.zeros(n_choices, dtype=torch.float32, device=device)
                for i in range(n_choices):
                    direction = moves[i]
                    prob_factor = torch.exp(-(torch.abs(x_gum + direction) - torch.abs(x_gum)) / p)
                    base_probs[i] = prob_factor
                base_probs = base_probs / base_probs.sum()

                if step > 0:
                    weights = torch.tensor([math.pow(0.8, memory_length - i - 1) for i in range(memory_length)], device=device)
                    weighted_history = (move_history_gum * weights).sum() / weights.sum()
                    correlated_probs = torch.zeros(n_choices, dtype=torch.float32, device=device)
                    for i in range(n_choices):
                        move_val = moves[i]
                        similarity = -abs(move_val - weighted_history.item())
                        correlated_probs[i] = math.exp(similarity)
                    correlated_probs = correlated_probs / correlated_probs.sum()
                    final_probs = (1 - correlation) * base_probs + correlation * correlated_probs
                else:
                    final_probs = base_probs

                gumbel_sample = F.gumbel_softmax(final_probs.unsqueeze(0), tau=tau, hard=True)
                move_idx = gumbel_sample.argmax(dim=1).item()
                move_val = moves[move_idx].item()

                x_gum = x_gum + move_val
                move_history_gum = torch.cat([move_history_gum[1:], torch.tensor([move_val], device=device)])

            loss_gum = torch.abs(x_gum - target)
            loss_gum.backward()
            optimizer_gum.step()
            loss_history_gum.append(loss_gum.item())

            print(f"Trial {trial}: Categorical Loss = {loss_cat.item():.4f}, "
                  f"Gumbel Loss = {loss_gum.item():.4f}, "
                  f"Initial Cat = {initial_value_cat.item():.4f}, Initial Gum = {initial_value_gum.item():.4f}")

        # Store results for this correlation strength.
        results[correlation] = {
            'categorical': {
                'loss_history': loss_history_cat,
                'final_initial_value': initial_value_cat.item()
            },
            'gumbel': {
                'loss_history': loss_history_gum,
                'final_initial_value': initial_value_gum.item()
            }
        }

        # # Create a separate plot for the current correlation strength.
        # plt.figure(figsize=(8, 5))
        # plt.plot(loss_history_cat, label="Categorical", color='blue', marker='o')
        # plt.plot(loss_history_gum, label="Gumbel", color='red', linestyle='--', marker='x')
        # plt.title(f"Gradient Flow Optimization (Correlation={correlation})")
        # plt.xlabel("Trial")
        # plt.ylabel("Loss (Distance to Target)")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(f"gradient_flow_correlation_{correlation}.png")
        # plt.show()

    return results


if __name__ == "__main__":
    # Uncomment one of the following blocks to run your desired analysis

    # Run and visualize correlated random walks and their autocorrelation analysis
    print("Running correlated random walk analysis...")
    # analyze_correlated_walks(correlation_range=[0.0, 0.3, 0.6, 0.9], n_steps=200)
    
    # Test gradient flow through the correlated random walk
    print("Testing gradient flow...")
    results = test_gradient_flow(n_trials=100, n_steps=100)
    correlation_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for correlation in correlation_range:
        print(f"Correlation: {correlation}")
        print(f"  Categorical Final Initial Value: {results[correlation]['categorical']['final_initial_value']:.4f}")
        print(f"  Gumbel Final Initial Value: {results[correlation]['gumbel']['final_initial_value']:.4f}")
