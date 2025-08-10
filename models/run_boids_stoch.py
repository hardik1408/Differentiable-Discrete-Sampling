#!/usr/bin/env python3

import torch
from agent_torch.populations import sample2
from agent_torch.examples.models import boids_stoch
from agent_torch.core.environment import envs

def run_energy_boids_simulation_simple():
    print("Starting energy boids simulation...")
    
    try:
        runner = envs.create(
            model=boids_stoch,
            population=sample2
        )
        
        sim_steps = runner.config["simulation_metadata"]["num_steps_per_episode"]
        num_episodes = runner.config["simulation_metadata"]["num_episodes"]
        num_agents = runner.config["simulation_metadata"]["num_agents"]
        
        print(f"Simulation parameters:")
        print(f"  - Agents: {num_agents}")
        print(f"  - Episodes: {num_episodes}")
        print(f"  - Steps per episode: {sim_steps}")
        
        for episode in range(num_episodes):
            print(f"\nRunning episode {episode + 1}/{num_episodes}...")
            
            if episode > 0:
                runner.reset()
            
            positions = runner.state["agents"]["boids"]["position"]
            energy = runner.state["agents"]["boids"]["energy"]
            alive_status = runner.state["agents"]["boids"]["alive_status"]
            
            print(f"Initial state:")
            print(f"  Position shape: {positions.shape}")
            print(f"  Energy shape: {energy.shape}")
            print(f"  Alive status shape: {alive_status.shape}")
            print(f"  Avg position: ({positions.float().mean(dim=0)[0]:.1f}, {positions.float().mean(dim=0)[1]:.1f})")
            print(f"  Avg energy: {energy.mean():.2f}")
            print(f"  Boids alive: {alive_status.sum()}")
            
            for step in range(min(sim_steps, 100)):
                runner.step(1)
                
                if step % 10 == 0 or step == min(sim_steps, 100) - 1:
                    positions = runner.state["agents"]["boids"]["position"]
                    energy = runner.state["agents"]["boids"]["energy"]
                    alive_status = runner.state["agents"]["boids"]["alive_status"]
                    
                    alive_indices = alive_status.bool()
                    num_alive = alive_status.sum()
                    
                    if num_alive > 0:
                        avg_energy = energy[alive_indices].mean()
                        avg_position = positions[alive_indices].float().mean(dim=0)
                        min_energy = energy[alive_indices].min()
                        max_energy = energy[alive_indices].max()
                        
                        unique_positions = set()
                        for i in range(len(positions)):
                            if alive_status[i]:
                                pos_tuple = (positions[i, 0].item(), positions[i, 1].item())
                                unique_positions.add(pos_tuple)
                        area_covered = len(unique_positions)
                    else:
                        avg_energy = 0.0
                        avg_position = torch.zeros(2)
                        min_energy = 0.0
                        max_energy = 0.0
                        area_covered = 0
                    
                    print(f"  Step {step + 1:3d}: Alive: {num_alive:0f}, "
                          f"Avg Energy: {avg_energy:.1f}, "
                          f"Energy Range: ({min_energy:.1f}-{max_energy:.1f}), "
                          f"Area: {area_covered:3d}, "
                          f"Avg Pos: ({avg_position[0]:.1f}, {avg_position[1]:.1f})")
                    
                if num_alive == 0:
                    print(f"  All boids died at step {step+1}- ending simulation early")
                    break
        
        print("\nSimulation completed successfully!")
        return runner
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    runner = run_energy_boids_simulation_simple()