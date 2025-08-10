from typing import Dict, Any
import torch
import torch.nn.functional as F
import torch.nn as nn
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_var

@Registry.register_substep("sample_movement_action", "transition")
class SampleMovementAction(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get('learnable', False):
                setattr(self, key, value['value'])
    
    def forward(self, state: Dict[str, Any], action) -> Dict[str, Any]:
        input_variables = self.input_variables
        
        position = get_var(state, input_variables["position"])
        alive_status = get_var(state, input_variables["alive_status"])
        grid_size = get_var(state, input_variables["grid_size"])
        
        movement_probabilities = action["boids"]["movement_probabilities"]
        
        new_position = position.clone()
        
        for i in range(len(position)):
            if alive_status[i]:
                action_logits = torch.nn.functional.gumbel_softmax(movement_probabilities[i], tau=1.0, hard=True)
                action_idx = torch.argmax(action_logits).item()
                
                if action_idx == 0:  # left
                    new_position[i, 0] = max(0, position[i, 0] - 1)
                elif action_idx == 1:  # right
                    new_position[i, 0] = min(grid_size[0] - 1, position[i, 0] + 1)
                elif action_idx == 2:  # up
                    new_position[i, 1] = max(0, position[i, 1] - 1)
                elif action_idx == 3:  # down
                    new_position[i, 1] = min(grid_size[1] - 1, position[i, 1] + 1)
        
        return {
            self.output_variables[0]: new_position
        }