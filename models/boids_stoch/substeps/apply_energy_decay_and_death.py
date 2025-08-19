from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_var

@Registry.register_substep("apply_energy_decay_and_death", "transition")
class ApplyEnergyDecayAndDeath(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get('learnable', False):
                setattr(self, key, value['value'])
    
    def forward(self, state: Dict[str, Any], action) -> Dict[str, Any]:
        input_variables = self.input_variables

        energy = get_var(state, input_variables["energy"])              
        alive_status = get_var(state, input_variables["alive_status"])  
        position = get_var(state, input_variables["position"])          
        energy_decay_per_move = get_var(state, input_variables["energy_decay_per_move"])
        energy_decay_per_stay = get_var(state, input_variables["energy_decay_per_stay"])
        energy_death_threshold = get_var(state, input_variables["energy_death_threshold"])
        prev_position = state.get("prev_position", position)
        print(prev_position)
        moved = (position != prev_position).any(dim=-1) if "prev_position" in state else torch.ones_like(alive_status, dtype=torch.bool)

        per_agent_decay = torch.where(moved, 
                                    torch.as_tensor(energy_decay_per_move, device=energy.device, dtype=energy.dtype),
                                    torch.as_tensor(energy_decay_per_stay, device=energy.device, dtype=energy.dtype))

        new_energy = energy - per_agent_decay * alive_status.to(energy.dtype)

        died = (new_energy <= energy_death_threshold).to(energy.dtype)
        new_alive_status = alive_status * (1.0 - died)

        state["prev_position"] = position.detach()

        return {
            self.output_variables[0]: new_energy,
            self.output_variables[1]: new_alive_status
        }
