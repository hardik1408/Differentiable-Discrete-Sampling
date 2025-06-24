import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import torch
import sys
import os

# Import from swarm.py
from swarm import DifferentiableBoids

class BoidsVisualizer:
    def __init__(self, n_boids=10, method="Learnable AUG", chain_length=50):
        """
        Initialize the boids visualizer
        
        Args:
            n_boids: Number of boids in the simulation
            method: Gradient estimation method to use
            chain_length: Number of simulation steps
        """
        self.n_boids = n_boids
        self.method = method
        self.chain_length = chain_length
        
        # Initialize the boids system
        self.boids_system = DifferentiableBoids(
            n_boids=self.n_boids, 
            dim=2, 
            max_speed=2.0, 
            perception_radius=3.0
        )
        
        # Storage for trajectory data
        self.positions_history = []
        self.velocities_history = []
        self.actions_history = []
        
        # Initialize the simulation
        self.run_simulation()
        
    def run_simulation(self):
        """Run the boids simulation and store trajectory data"""
        print(f"Running simulation with {self.n_boids} boids for {self.chain_length} steps...")
        print(f"Using method: {self.method}")
        
        # Initialize positions and velocities
        positions = torch.randn(self.n_boids, 2) * 5.0
        velocities = torch.randn(self.n_boids, 2) * 0.5
        
        # Policy parameters for learnable methods
        policy_params = {
            'alpha': self.boids_system.alpha,
            'beta': self.boids_system.beta
        }
        
        # Store initial state
        self.positions_history.append(positions.detach().numpy().copy())
        self.velocities_history.append(velocities.detach().numpy().copy())
        
        # Run simulation steps
        for step in range(self.chain_length):
            if step % 10 == 0:
                print(f"Step {step}/{self.chain_length}")
            
            # Sample actions for each boid
            actions = []
            for boid_idx in range(self.n_boids):
                # Create state vector
                state = torch.cat([positions[boid_idx].detach(), velocities[boid_idx].detach()])
                
                # Sample control action
                action, _ = self.boids_system.sample_control(state, policy_params, self.method)
                actions.append(action)
            
            self.actions_history.append(actions.copy())
            
            # Apply controls
            velocities = self.boids_system.apply_control(positions, velocities, actions)
            
            # Apply boids dynamics
            positions, velocities = self.boids_system.boids_update(positions, velocities)
            
            # Store current state
            self.positions_history.append(positions.detach().numpy().copy())
            self.velocities_history.append(velocities.detach().numpy().copy())
        
        print("Simulation completed!")
        
    def create_animation(self, save_path=f"boids_animation.mp4", fps=10, show_trails=True, trail_length=10):
        """
        Create an animated visualization of the boids simulation
        
        Args:
            save_path: Path to save the animation video
            fps: Frames per second for the animation
            show_trails: Whether to show trails behind boids
            trail_length: Length of trails to show
        """
        print("Creating animation...")
        
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate bounds for the plot
        all_positions = np.array(self.positions_history)
        x_min, x_max = all_positions[:, :, 0].min() - 2, all_positions[:, :, 0].max() + 2
        y_min, y_max = all_positions[:, :, 1].min() - 2, all_positions[:, :, 1].max() + 2
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Boids Simulation - Method: {self.method}\n'
                    f'Chain Length: {self.chain_length}, Number of Agents: {self.n_boids}', 
                    fontsize=14, fontweight='bold')
        
        # Colors for different boids
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_boids))
        
        # Initialize plot elements
        boid_dots = []
        # velocity_arrows = []
        trail_lines = [] if show_trails else None
        
        for i in range(self.n_boids):
            # Boid position (circle)
            dot = Circle((0, 0), 0.2, color=colors[i], alpha=0.8)
            ax.add_patch(dot)
            boid_dots.append(dot)
            
            # Velocity arrow
            # arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
            #                   arrowprops=dict(arrowstyle='->', color=colors[i], lw=2, alpha=0.7))
            # velocity_arrows.append(arrow)
            
            # Trail
            if show_trails:
                trail, = ax.plot([], [], color=colors[i], alpha=0.3, linewidth=1)
                trail_lines.append(trail)
        
        # Add legend for actions
        # action_names = ['Forward', 'Left', 'Right', 'Speed Up', 'Slow Down']
        # action_colors = ['blue', 'red', 'green', 'orange', 'purple']
        # legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
        #                             markerfacecolor=color, markersize=8, label=name)
        #                  for name, color in zip(action_names, action_colors)]
        # ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add step counter text
        step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        def animate(frame):
            """Animation function for each frame"""
            positions = self.positions_history[frame]
            velocities = self.velocities_history[frame]
            
            # Update step counter
            step_text.set_text(f'Step: {frame}/{len(self.positions_history)-1}')
            
            # Update each boid
            for i in range(self.n_boids):
                pos = positions[i]
                vel = velocities[i]
                
                # Update boid position
                boid_dots[i].center = (pos[0], pos[1])
                
                # Update velocity arrow
                arrow_end = pos + vel * 0.5  # Scale velocity for visibility
                # velocity_arrows[i].xy = arrow_end
                # velocity_arrows[i].xytext = pos
                
                # Update trail
                if show_trails and trail_lines:
                    start_idx = max(0, frame - trail_length)
                    trail_positions = self.positions_history[start_idx:frame+1]
                    if len(trail_positions) > 1:
                        trail_x = [p[i][0] for p in trail_positions]
                        trail_y = [p[i][1] for p in trail_positions]
                        trail_lines[i].set_data(trail_x, trail_y)
                
                # Color code by current action (if available)
                # if frame > 0 and frame <= len(self.actions_history):
                #     action = self.actions_history[frame-1][i]
                #     action_color = action_colors[action] if action < len(action_colors) else 'gray'
                #     boid_dots[i].set_facecolor(action_color)
                #     boid_dots[i].set_alpha(0.8)
            
            # Return all artists that need to be redrawn
            artists = boid_dots  + [step_text]
            if show_trails and trail_lines:
                artists.extend(trail_lines)
            
            return artists
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(self.positions_history),
            interval=1000//fps, blit=True, repeat=True
        )
        
        # Save animation
        print(f"Saving animation to {save_path}...")
        try:
            # Try to save as MP4
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='BoidsVisualizer'), bitrate=1800)
            anim.save(save_path, writer=writer)
            print(f"Animation saved successfully to {save_path}")
        except Exception as e:
            print(f"Error saving MP4: {e}")
            # Fallback to GIF
            gif_path = save_path.replace('.mp4', '.gif')
            try:
                anim.save(gif_path, writer='pillow', fps=fps)
                print(f"Animation saved as GIF to {gif_path}")
            except Exception as e2:
                print(f"Error saving GIF: {e2}")
                print("Please install ffmpeg or pillow for video/gif export")
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def create_static_plots(self, save_path="boids_plots.png"):
        """Create static plots showing the trajectory"""
        fig, axes = plt.subplots(figsize=(15, 12))
        
        # Plot 1: Full trajectory
        # ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_boids))
        
        for i in range(self.n_boids):
            x_traj = [pos[i][0] for pos in self.positions_history]
            y_traj = [pos[i][1] for pos in self.positions_history]
            axes.plot(x_traj, y_traj, color=colors[i], alpha=0.7, label=f'Boid {i}')
            axes.scatter(x_traj[0], y_traj[0], color=colors[i], marker='o', s=50, alpha=0.8)  # Start
            axes.scatter(x_traj[-1], y_traj[-1], color=colors[i], marker='s', s=50, alpha=0.8)  # End
        
        axes.set_title('Full Trajectory (○ = start, □ = end)')
        axes.set_xlabel('X Position')
        axes.set_ylabel('Y Position')
        axes.grid(True, alpha=0.3)
        axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # # Plot 2: Speed over time
        # ax2 = axes[0, 1]
        # for i in range(self.n_boids):
        #     speeds = [np.linalg.norm(vel[i]) for vel in self.velocities_history]
        #     ax2.plot(speeds, color=colors[i], alpha=0.7)
        
        # ax2.set_title('Speed Over Time')
        # ax2.set_xlabel('Time Step')
        # ax2.set_ylabel('Speed')
        # ax2.grid(True, alpha=0.3)
        
        # # Plot 3: Center of mass trajectory
        # ax3 = axes[1, 0]
        # center_of_mass = [np.mean(pos, axis=0) for pos in self.positions_history]
        # com_x = [com[0] for com in center_of_mass]
        # com_y = [com[1] for com in center_of_mass]
        # ax3.plot(com_x, com_y, 'k-', linewidth=2, label='Center of Mass')
        # ax3.scatter(com_x[0], com_y[0], color='green', marker='o', s=100, label='Start')
        # ax3.scatter(com_x[-1], com_y[-1], color='red', marker='s', s=100, label='End')
        # ax3.scatter(0, 0, color='blue', marker='*', s=200, label='Target (0,0)')
        
        # ax3.set_title('Center of Mass Trajectory')
        # ax3.set_xlabel('X Position')
        # ax3.set_ylabel('Y Position')
        # ax3.grid(True, alpha=0.3)
        # ax3.legend()
        
        # Plot 4: Action distribution
        # ax4 = axes[1, 1]
        # if self.actions_history:
        #     all_actions = [action for step_actions in self.actions_history for action in step_actions]
        #     action_names = ['Forward', 'Left', 'Right', 'Speed Up', 'Slow Down']
        #     action_counts = [all_actions.count(i) for i in range(5)]
            
        #     bars = ax4.bar(action_names, action_counts, color=['blue', 'red', 'green', 'orange', 'purple'])
        #     ax4.set_title('Action Distribution')
        #     ax4.set_ylabel('Count')
            
        #     # Add count labels on bars
        #     for bar, count in zip(bars, action_counts):
        #         height = bar.get_height()
        #         ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(action_counts),
        #                 f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Static plots saved to {save_path}")
        plt.show()

def main():
    """Main function to run the visualization"""
    
    # Parameters (hardcoded as requested)
    N_BOIDS = 50
    METHOD = "Fixed AUG"  # Options: "Learnable AUG", "Gumbel", "Stochastic AD", "Fixed AUG"
    CHAIN_LENGTH = 1000
    
    print("="*60)
    print("BOIDS SIMULATION VISUALIZATION")
    print("="*60)
    print(f"Number of boids: {N_BOIDS}")
    print(f"Method: {METHOD}")
    print(f"Chain length: {CHAIN_LENGTH}")
    print("="*60)
    
    # Create visualizer
    visualizer = BoidsVisualizer(
        n_boids=N_BOIDS,
        method=METHOD,
        chain_length=CHAIN_LENGTH
    )
    
    # Create static plots first
    # visualizer.create_static_plots(f"boids_analysis_{METHOD}.png")
    
    # Create animation
    animation = visualizer.create_animation(
        save_path=f"boids_simulation_{METHOD}.mp4",
        fps=8,  # Slower for better visibility
        show_trails=False,
        trail_length=15
    )
    
    print("\nVisualization complete!")
    print("Files created:")
    print("- boids_analysis.png (static analysis plots)")
    print("- boids_simulation.mp4 (animation video)")

if __name__ == "__main__":
    main()