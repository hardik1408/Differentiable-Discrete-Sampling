from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers import get_var

@Registry.register_substep("calculate_movement_probabilities", "policy")
class CalculateMovementProbabilities(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_move_prob = self.learnable_args.get("base_move_prob", 0.2)
        self.low_energy_stay_bonus = self.learnable_args.get("low_energy_stay_bonus", 0.4)
        self.boundary_avoidance = self.learnable_args.get("boundary_avoidance", 0.8)
    
    def forward(self, state: Dict[str, Any], observations) -> Dict[str, Any]:
        position = get_var(state, self.input_variables["position"])
        energy = get_var(state, self.input_variables["energy"])
        alive_status = get_var(state, self.input_variables["alive_status"])
        grid_size = get_var(state, self.input_variables["grid_size"])
        
        num_boids = len(position)
        movement_probabilities = torch.zeros((num_boids, 5))
        
        for i in range(num_boids):
            if not alive_status[i]:
                movement_probabilities[i, 4] = 1.0  # stay
                continue
            
            x, y = position[i, 0], position[i, 1]
            probs = torch.ones(5) * self.base_move_prob
            
            if energy[i] <= 30:
                probs[4] = probs[4] + self.low_energy_stay_bonus  # stay bonus
            
            if x <= 0:
                probs[0] = probs[0]*(1 - self.boundary_avoidance)  # left
            if x >= grid_size[0] - 1:
                probs[1] = probs[1]*(1 - self.boundary_avoidance)  # right
            if y <= 0:
                probs[2] =probs[2]*(1 - self.boundary_avoidance)  # up
            if y >= grid_size[1] - 1:
                probs[3] = probs[3]*(1 - self.boundary_avoidance)  # down
            
            probs = probs / torch.sum(probs)
            movement_probabilities[i] = probs
        return {
            self.output_variables[0]: movement_probabilities
        }