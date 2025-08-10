#!/usr/bin/env python3

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

def create_energy_boids_config():
    config = ConfigBuilder()
    
    # Simulation metadata
    metadata = {
        "num_agents": 100,
        "num_episodes": 1,
        "num_steps_per_episode": 300,
        "num_substeps_per_step": 4,
        "device": "cpu",
        "calibration": False
    }
    config.set_metadata(metadata)
    
    # State configuration
    state_builder = StateBuilder()
    
    # Boid agents
    agent_builder = AgentBuilder("boids", metadata["num_agents"])
    
    position = PropertyBuilder("position")\
        .set_dtype("float")\
        .set_shape([metadata["num_agents"], 2])\
        .set_value([15, 15])
    agent_builder.add_property(position)
    
    energy = PropertyBuilder("energy")\
        .set_dtype("float")\
        .set_shape([metadata["num_agents"]])\
        .set_value(100.0)
    agent_builder.add_property(energy)
    
    alive_status = PropertyBuilder("alive_status")\
        .set_dtype("bool")\
        .set_shape([metadata["num_agents"]])\
        .set_value(True)
    agent_builder.add_property(alive_status)
    
    has_transferred_energy = PropertyBuilder("has_transferred_energy")\
        .set_dtype("bool")\
        .set_shape([metadata["num_agents"]])\
        .set_value(False)
    agent_builder.add_property(has_transferred_energy)

    area_covered_metric = PropertyBuilder("area_covered_metric")\
        .set_dtype("float")\
        .set_shape([metadata["num_agents"]])\
        .set_value(0.0)
    agent_builder.add_property(area_covered_metric)
    
    state_builder.add_agent("boids", agent_builder)
    
    # Environment variables
    env_builder = EnvironmentBuilder()
    
    grid_size = PropertyBuilder("grid_size")\
        .set_dtype("int")\
        .set_shape([2])\
        .set_value([100, 100])
    env_builder.add_variable(grid_size)
    
    energy_transfer_radius = PropertyBuilder("energy_transfer_radius")\
        .set_dtype("float")\
        .set_value(3.0)
    env_builder.add_variable(energy_transfer_radius)
    
    energy_decay_per_move = PropertyBuilder("energy_decay_per_move")\
        .set_dtype("float")\
        .set_value(2.0)
    env_builder.add_variable(energy_decay_per_move)
    
    energy_decay_per_stay = PropertyBuilder("energy_decay_per_stay")\
        .set_dtype("float")\
        .set_value(0.5)
    env_builder.add_variable(energy_decay_per_stay)
    
    energy_death_threshold = PropertyBuilder("energy_death_threshold")\
        .set_dtype("float")\
        .set_value(10.0)
    env_builder.add_variable(energy_death_threshold)
    
    min_energy_diff = PropertyBuilder("min_energy_diff")\
        .set_dtype("float")\
        .set_value(35.0)
    env_builder.add_variable(min_energy_diff)
    
    energy_transfer_percentage = PropertyBuilder("energy_transfer_percentage")\
        .set_dtype("float")\
        .set_value(0.1)
    env_builder.add_variable(energy_transfer_percentage)

    area_covered = PropertyBuilder("area_covered")\
        .set_dtype("float")\
        .set_value(0.0)
    env_builder.add_variable(area_covered)

    boids_alive = PropertyBuilder("boids_alive")\
        .set_dtype("int")\
        .set_value(metadata["num_agents"])
    env_builder.add_variable(boids_alive)

    total_positions_visited = PropertyBuilder("total_positions_visited")\
        .set_dtype("int")\
        .set_value(0)
    env_builder.add_variable(total_positions_visited)
    
    state_builder.set_environment(env_builder)
    config.set_state(state_builder.to_dict())
    
    # Substep 0: Energy Sharing
    energy_sharing_substep = SubstepBuilderWithImpl(
        name="EnergySharing",
        description="Handle energy transfer between nearby boids",
        output_dir="agent_torch/examples/models/boids_stoch/substeps"
    )
    energy_sharing_substep.add_active_agent("boids")
    energy_sharing_substep.config["observation"] = {"boids": None}
    
    energy_policy = PolicyBuilder()
    
    max_transfers_per_boid = PropertyBuilder.create_argument(
        name="Maximum transfers per boid per step",
        value=10,
        learnable=False
    ).config
    
    energy_policy.add_policy(
        "identify_energy_transfers",
        "IdentifyEnergyTransfers",
        {
            "position": "agents/boids/position",
            "energy": "agents/boids/energy",
            "alive_status": "agents/boids/alive_status",
            "energy_transfer_radius": "environment/energy_transfer_radius",
            "min_energy_diff": "environment/min_energy_diff"
        },
        ["energy_transfers_list"],
        {"max_transfers_per_boid": max_transfers_per_boid}
    )
    energy_sharing_substep.set_policy("boids", energy_policy)
    
    energy_transition = TransitionBuilder()
    
    transfer_percentage = PropertyBuilder.create_argument(
        name="Energy transfer percentage",
        value=0.1,
        learnable=False
    ).config
    
    energy_transition.add_transition(
        "execute_energy_transfers",
        "ExecuteEnergyTransfers",
        {
            "energy": "agents/boids/energy",
            "alive_status": "agents/boids/alive_status",
            "energy_transfer_percentage": "environment/energy_transfer_percentage",
            "has_transferred_energy": "agents/boids/has_transferred_energy"
        },
        ["energy", "has_transferred_energy"],
        {"transfer_percentage": transfer_percentage}
    )
    energy_sharing_substep.set_transition(energy_transition)
    
    # Substep 1: Movement Decision
    movement_decision_substep = SubstepBuilderWithImpl(
        name="MovementDecision",
        description="Decide next movement action using probabilities",
        output_dir="agent_torch/examples/models/boids_stoch/substeps"
    )
    movement_decision_substep.add_active_agent("boids")
    movement_decision_substep.config["observation"] = {"boids": None}
    
    movement_policy = PolicyBuilder()
    
    base_move_prob = PropertyBuilder.create_argument(
        name="Base movement probability",
        value=0.2,
        learnable=False
    ).config
    
    low_energy_stay_bonus = PropertyBuilder.create_argument(
        name="Stay bonus for low energy",
        value=0.4,
        learnable=False
    ).config
    
    boundary_avoidance = PropertyBuilder.create_argument(
        name="Boundary avoidance factor",
        value=0.8,
        learnable=False
    ).config
    
    movement_policy.add_policy(
        "calculate_movement_probabilities",
        "CalculateMovementProbabilities",
        {
            "position": "agents/boids/position",
            "energy": "agents/boids/energy",
            "alive_status": "agents/boids/alive_status",
            "grid_size": "environment/grid_size"
        },
        ["movement_probabilities"],
        {
            "base_move_prob": base_move_prob,
            "low_energy_stay_bonus": low_energy_stay_bonus,
            "boundary_avoidance": boundary_avoidance
        }
    )
    movement_decision_substep.set_policy("boids", movement_policy)
    
    movement_transition = TransitionBuilder()
    
    random_seed_offset = PropertyBuilder.create_argument(
        name="Random seed offset",
        value=42,
        learnable=False
    ).config
    
    movement_transition.add_transition(
        "sample_movement_action",
        "SampleMovementAction",
        {
            "position": "agents/boids/position",
            "alive_status": "agents/boids/alive_status",
            "grid_size": "environment/grid_size"
        },
        ["position"],
        {"random_seed_offset": random_seed_offset}
    )
    movement_decision_substep.set_transition(movement_transition)
    
    # Substep 2: Energy Decay and Death
    energy_decay_substep = SubstepBuilderWithImpl(
        name="EnergyDecayAndDeath",
        description="Apply energy costs and handle deaths",
        output_dir="agent_torch/examples/models/boids_stoch/substeps"
    )
    energy_decay_substep.add_active_agent("boids")
    energy_decay_substep.config["observation"] = {"boids": None}
    energy_decay_substep.config["policy"] = {"boids": None}
    
    decay_transition = TransitionBuilder()
    
    move_energy_cost = PropertyBuilder.create_argument(
        name="Energy cost for movement",
        value=2.0,
        learnable=False
    ).config
    
    stay_energy_cost = PropertyBuilder.create_argument(
        name="Energy cost for staying",
        value=0.5,
        learnable=False
    ).config
    
    death_threshold = PropertyBuilder.create_argument(
        name="Death energy threshold",
        value=5.0,
        learnable=False
    ).config
    
    decay_transition.add_transition(
        "apply_energy_decay_and_death",
        "ApplyEnergyDecayAndDeath",
        {
            "energy": "agents/boids/energy",
            "alive_status": "agents/boids/alive_status",
            "position": "agents/boids/position",
            "energy_decay_per_move": "environment/energy_decay_per_move",
            "energy_decay_per_stay": "environment/energy_decay_per_stay",
            "energy_death_threshold": "environment/energy_death_threshold"
        },
        ["energy", "alive_status"],
        {
            "move_energy_cost": move_energy_cost,
            "stay_energy_cost": stay_energy_cost,
            "death_threshold": death_threshold
        }
    )
    energy_decay_substep.set_transition(decay_transition)
    
    # Substep 3: Metrics Calculation
    metrics_substep = SubstepBuilderWithImpl(
        name="MetricsCalculation",
        description="Calculate area coverage and survival metrics",
        output_dir="substeps"
    )
    metrics_substep.add_active_agent("boids")
    metrics_substep.config["observation"] = {"boids": None}
    metrics_substep.config["policy"] = {"boids": None}
    
    metrics_transition = TransitionBuilder()
    
    coverage_decay = PropertyBuilder.create_argument(
        name="Coverage decay factor",
        value=0.99,
        learnable=False
    ).config
    
    metrics_transition.add_transition(
        "update_simulation_metrics",
        "UpdateSimulationMetrics",
        {
            "position": "agents/boids/position",
            "alive_status": "agents/boids/alive_status"
            "area_covered_metric": "agents/boids/area_covered_metric"
        },
        ["area_covered_metric"],
        {"coverage_decay": coverage_decay}
    )
    metrics_substep.set_transition(metrics_transition)
    
    # Add substeps to config
    config.add_substep("0", energy_sharing_substep)
    config.add_substep("1", movement_decision_substep)
    config.add_substep("2", energy_decay_substep)
    config.add_substep("3", metrics_substep)
    
    # Generate implementations
    energy_sharing_substep.generate_implementations()
    movement_decision_substep.generate_implementations()
    energy_decay_substep.generate_implementations()
    metrics_substep.generate_implementations()
    
    # Save configuration
    config_path = "agent_torch/examples/models/boids_stoch/yamls/config.yaml"
    config.save_yaml(config_path)
    
    return config

if __name__ == "__main__":
    config = create_energy_boids_config()
    print("Energy boids configuration created successfully!")