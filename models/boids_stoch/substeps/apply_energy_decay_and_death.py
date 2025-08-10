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
        
        new_energy = energy.clone()
        new_alive_status = alive_status.clone()
        
        for i in range(len(energy)):
            if alive_status[i]:
                moved = not torch.equal(position[i], prev_position[i]) if "prev_position" in state else True
                
                if moved:
                    new_energy[i] -= energy_decay_per_move
                else:
                    new_energy[i] -= energy_decay_per_stay
                
                if new_energy[i] <= energy_death_threshold:
                    new_alive_status[i] = False
        
        state["prev_position"] = position.clone()
        return {
            self.output_variables[0]: new_energy,
            self.output_variables[1]: new_alive_status
        }