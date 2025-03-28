from bench import *
import torch
import matplotlib.pyplot as plt

# Assuming the functions generate_long_baseline and simulate_full_trajectory are already defined
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generate baseline data for a random walk
baseline_data = generate_long_baseline(total_steps=100, num_simulations=1000, device=device)

# Set final theta values from training (modify as needed based on your experimental results)
final_theta_cat = torch.tensor([0.0334], device=device)   # Learned theta for categorical estimator
final_theta_gum = torch.tensor([-0.0601], device=device)     # Learned theta for Gumbel estimator

num_steps = 100  # Total time steps for each trajectory

# Generate and show 5 separate plots, one after the other.
for i in range(4):
    plt.figure(figsize=(6, 4))
    
    # For each trial, pick one trajectory from the baseline data.
    baseline_traj = baseline_data['full_trajectories'][i].cpu().numpy()
    
    # Generate trajectory using the categorical estimator and its learned theta
    traj_cat = simulate_full_trajectory(final_theta_cat, num_steps=num_steps, device=device, estimator='categorical').cpu().numpy()
    
    # Generate trajectory using the Gumbel-softmax estimator and its learned theta
    traj_gumbel = simulate_full_trajectory(final_theta_gum, num_steps=num_steps, device=device, estimator='gumbel').cpu().numpy()
    
    # Plot all three trajectories on the current figure
    plt.plot(range(num_steps), baseline_traj, label="Baseline", color="black", linestyle="--")
    plt.plot(range(num_steps), traj_cat, label="Categorical", color="blue")
    plt.plot(range(num_steps), traj_gumbel, label="Gumbel", color="red")
    
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)
    plt.title(f"Cat theta = {final_theta_cat.item():.4f} , Gumbel theta = {final_theta_gum.item():.4f}")
    
    # Display the current figure
    plt.show()
