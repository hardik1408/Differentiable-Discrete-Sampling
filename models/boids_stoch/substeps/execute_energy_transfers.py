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
        if len(energy_transfers_list) == 0:
            return {
                self.output_variables[0]: energy,
                self.output_variables[1]: torch.zeros_like(alive_status, dtype=torch.bool)
            }

        transfers = torch.tensor(energy_transfers_list, device=energy.device, dtype=energy.dtype)
        donor_idx = transfers[:, 0].long()
        recipient_idx = transfers[:, 1].long()
        energy_diff = transfers[:, 2]

        valid_mask = alive_status[donor_idx] & alive_status[recipient_idx]
        transfer_amounts = energy_diff * energy_transfer_percentage * valid_mask.to(energy.dtype)

        updates = torch.zeros_like(energy)

        updates.index_add_(0, donor_idx, -transfer_amounts)
        updates.index_add_(0, recipient_idx, transfer_amounts)

        new_energy = energy + updates

        has_transferred = torch.zeros_like(alive_status, dtype=torch.bool)
        has_transferred.index_fill_(0, donor_idx[valid_mask], True)

        return {
            self.output_variables[0]: new_energy,
            self.output_variables[1]: has_transferred
        }
