from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_var

@Registry.register_substep("update_simulation_metrics", "transition")
class UpdateSimulationMetrics(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get('learnable', False):
                setattr(self, key, value['value'])
    
    def forward(self, state: Dict[str, Any], action) -> Dict[str, Any]:
        input_variables = self.input_variables
        
        position = get_var(state, input_variables["position"])
        alive_status = get_var(state, input_variables["alive_status"])
        
        if "covered_positions_set" not in state:
            state["covered_positions_set"] = set()
        
        for i in range(len(position)):
            if alive_status[i]:
                pos_tuple = (position[i, 0].item(), position[i, 1].item())
                state["covered_positions_set"].add(pos_tuple)
        
        area_covered_count = len(state["covered_positions_set"])
        area_covered_tensor = torch.full_like(alive_status, area_covered_count, dtype=torch.float32)
        
        return {
            self.output_variables[0]: area_covered_tensor
        }