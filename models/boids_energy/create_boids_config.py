#!/usr/bin/env python3
"""
Boids Model Configuration Generator

This script generates the boids model configuration using AgentTorch's Config API.
It creates both the YAML configuration file and substep implementation templates.

Usage:
    python create_boids_config.py

The configuration includes:
    - 100 boids with position, velocity, and energy properties
    - Environment bounds and flocking parameters
    - Three main substeps: FlockingBehavior, MovementUpdate, and EnergyUpdate
    - Learnable parameters for customization
"""

from agent_torch.config import (
    ConfigBuilder,
    StateBuilder,
    AgentBuilder,
    PropertyBuilder,
    EnvironmentBuilder,
    SubstepBuilderWithImpl,
    PolicyBuilder,
    TransitionBuilder
)

def create_boids_config():
    """Create the boids flocking simulation configuration."""
    print("Creating boids model configuration...")
    
    config = ConfigBuilder()
    
    # 1. Set simulation metadata
    ### MODIFIED ###
    metadata = {
        "num_agents": 100,
        "num_episodes": 1,
        "num_steps_per_episode": 500,
        "num_substeps_per_step": 3,  # Three substeps: flocking + movement + energy
        "device": "cpu",
        "calibration": False
    }
    config.set_metadata(metadata)
    
    # 2. Build state configuration
    state_builder = StateBuilder()
    
    agent_builder = AgentBuilder("boids", metadata["num_agents"])
    
    position = PropertyBuilder("position")\
        .set_dtype("float")\
        .set_shape([metadata["num_agents"], 2])\
        .set_value([400.0, 300.0])
    agent_builder.add_property(position)
    
    velocity = PropertyBuilder("velocity")\
        .set_dtype("float")\
        .set_shape([metadata["num_agents"], 2])\
        .set_value([1.0, 0.0])
    agent_builder.add_property(velocity)
    
    energy = PropertyBuilder("energy")\
        .set_dtype("float")\
        .set_shape([metadata["num_agents"]])\
        .set_value(100.0)
    agent_builder.add_property(energy)
    
    state_builder.add_agent("boids", agent_builder)
    
    env_builder = EnvironmentBuilder()
    
    bounds = PropertyBuilder("bounds")\
        .set_dtype("float")\
        .set_shape([2])\
        .set_value([800.0, 600.0])
    env_builder.add_variable(bounds)
    
    perception_radius = PropertyBuilder("perception_radius").set_dtype("float").set_value(50.0)
    env_builder.add_variable(perception_radius)
    
    separation_distance = PropertyBuilder("separation_distance").set_dtype("float").set_value(25.0)
    env_builder.add_variable(separation_distance)
    
    max_speed = PropertyBuilder("max_speed").set_dtype("float").set_value(4.0)
    env_builder.add_variable(max_speed)
    
    max_force = PropertyBuilder("max_force").set_dtype("float").set_value(0.1)
    env_builder.add_variable(max_force)

    energy_decay_rate = PropertyBuilder("energy_decay_rate").set_dtype("float").set_value(0.5)
    env_builder.add_variable(energy_decay_rate)
    
    min_energy = PropertyBuilder("min_energy").set_dtype("float").set_value(5.0)
    env_builder.add_variable(min_energy)

    state_builder.set_environment(env_builder)
    config.set_state(state_builder.to_dict())


    
    # 3. Create Substep 0: Flocking Behavior
    flocking_substep = SubstepBuilderWithImpl(
        name="FlockingBehavior",
        description="Calculate flocking forces (separation, alignment, cohesion)",
        output_dir="agent_torch/examples/models/boids_energy/substeps"
    )
    flocking_substep.add_active_agent("boids")
    flocking_substep.config["observation"] = {"boids": None}
    
    flocking_policy = PolicyBuilder()
    speed_range = PropertyBuilder.create_argument(name="Speed range", value=[0.5, 1.5], shape=[2], learnable=True).config
    position_margin = PropertyBuilder.create_argument(name="Position margin", value=50.0, learnable=True).config
    separation_weight = PropertyBuilder.create_argument(name="Separation weight", value=1.5, learnable=True).config
    alignment_weight = PropertyBuilder.create_argument(name="Alignment weight", value=1.0, learnable=True).config
    cohesion_weight = PropertyBuilder.create_argument(name="Cohesion weight", value=1.0, learnable=True).config
    flocking_policy.add_policy(
        "calculate_flocking_forces", "CalculateFlockingForces",
        {"position": "agents/boids/position", "velocity": "agents/boids/velocity", "perception_radius": "environment/perception_radius", "separation_distance": "environment/separation_distance"},
        ["steering_force"],
        {"speed_range": speed_range, "position_margin": position_margin, "separation_weight": separation_weight, "alignment_weight": alignment_weight, "cohesion_weight": cohesion_weight}
    )
    flocking_substep.set_policy("boids", flocking_policy)
    
    flocking_transition = TransitionBuilder()
    max_force_param = PropertyBuilder.create_argument(name="Max force", value=0.1, learnable=True).config
    flocking_transition.add_transition(
        "update_velocity", "UpdateVelocity",
        {"velocity": "agents/boids/velocity", "max_force": "environment/max_force"},
        ["velocity"],
        {"max_force": max_force_param}
    )
    flocking_substep.set_transition(flocking_transition)
    
    # 4. Create Substep 1: Movement Update
    ### MODIFIED ###
    movement_substep = SubstepBuilderWithImpl(
        name="MovementUpdate",
        description="Update positions and handle boundaries",
        output_dir="agent_torch/examples/models/boids_energy/substeps"
    )
    movement_substep.add_active_agent("boids")
    movement_substep.config["observation"] = {"boids": None}
    
    movement_policy = PolicyBuilder()
    max_speed_param = PropertyBuilder.create_argument(name="Max speed", value=4.0, learnable=True).config
    movement_policy.add_policy(
        "limit_speed", "LimitSpeed",
        {"velocity": "agents/boids/velocity", "max_speed": "environment/max_speed"},
        ["limited_velocity"],
        {"max_speed": max_speed_param}
    )
    movement_substep.set_policy("boids", movement_policy)
    
    movement_transition = TransitionBuilder()
    bounds_param = PropertyBuilder.create_argument(name="World bounds", value=[800.0, 600.0], shape=[2], learnable=False).config
    movement_transition.add_transition(
        "update_position", "UpdatePosition",
        {"position": "agents/boids/position", "bounds": "environment/bounds", "velocity": "substeps/1/policy/boids/output/limited_velocity"},
        ["position"],
        {"bounds": bounds_param}
    )
    ### DELETED ### The energy update transition has been removed from this substep.
    movement_substep.set_transition(movement_transition)
    
    ### NEW ###
    # 5. Create Substep 2: Energy Update
    energy_substep = SubstepBuilderWithImpl(
        name="EnergyUpdate",
        description="Update boid energy based on movement",
        output_dir="agent_torch/examples/models/boids_energy/substeps"
    )
    energy_substep.add_active_agent("boids")
    energy_substep.config["observation"] = {"boids": None}

    energy_policy = PolicyBuilder()
    min_energy_param = PropertyBuilder.create_argument(name="Min energy", value=5.0, learnable=True).config
    energy_policy.add_policy(
        "check_min_energy", "CheckMinEnergy",
        {"energy": "agents/boids/energy", "min_energy": "environment/min_energy"},
        ["min_energy_check"],
        {"min_energy": min_energy_param}
    )
    energy_substep.set_policy("boids", energy_policy)

    energy_transition = TransitionBuilder()
    
    energy_decay_param = PropertyBuilder.create_argument(
        name="Energy decay rate",
        value=0.5,
        learnable=False
    ).config

    energy_transition.add_transition(
        "update_energy","UpdateEnergy",
        {
            "energy": "agents/boids/energy",
            "velocity": "agents/boids/velocity", # Use velocity from the movement step
            "energy_decay_rate": "environment/energy_decay_rate"
        },
        ["energy"],
        {"energy_decay_rate": energy_decay_param}
    )
    energy_substep.set_transition(energy_transition)
    
    # 6. Add all substeps to config and generate implementations
    ### MODIFIED ###
    config.add_substep("0", flocking_substep)
    config.add_substep("1", movement_substep)
    config.add_substep("2", energy_substep)
    
    print("Generating flocking behavior substep implementations...")
    flocking_files = flocking_substep.generate_implementations()
    print(f"Generated files: {flocking_files}")
    
    print("Generating movement update substep implementations...")
    movement_files = movement_substep.generate_implementations()
    print(f"Generated files: {movement_files}")
    
    ### NEW ###
    print("Generating energy update substep implementations...")
    energy_files = energy_substep.generate_implementations()
    print(f"Generated files: {energy_files}")

    config_path = "agent_torch/examples/models/boids_energy/yamls/config.yaml"
    config.save_yaml(config_path)
    print(f"Configuration saved to: {config_path}")
    
    return config

if __name__ == "__main__":
    create_boids_config()
    print("Boids model configuration and substep templates created successfully!")