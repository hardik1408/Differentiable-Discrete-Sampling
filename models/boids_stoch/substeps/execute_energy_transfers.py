from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_var

@Registry.register_substep("execute_energy_transfers", "transition")
class ExecuteEnergyTransfers(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get('learnable', False):
                setattr(self, key, value['value'])
    
    def forward(self, state: Dict[str, Any], action) -> Dict[str, Any]:
        input_variables = self.input_variables
        
        energy = get_var(state, input_variables["energy"])
        alive_status = get_var(state, input_variables["alive_status"])
        energy_transfer_percentage = get_var(state, input_variables["energy_transfer_percentage"])
        
        energy_transfers_list = action["boids"]["energy_transfers_list"]
        
        new_energy = energy.clone()
        has_transferred = torch.zeros_like(alive_status, dtype=torch.bool)
        
        for transfer in energy_transfers_list:
            donor_idx, recipient_idx, energy_diff = transfer
            if alive_status[donor_idx] and alive_status[recipient_idx]:
                transfer_amount = energy_diff * energy_transfer_percentage
                new_energy[donor_idx] -= transfer_amount
                new_energy[recipient_idx] += transfer_amount
                has_transferred[donor_idx] = True
        
        return {
            self.output_variables[0]: new_energy,
            self.output_variables[1]: has_transferred
        }