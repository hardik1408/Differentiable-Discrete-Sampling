from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers import get_var

@Registry.register_substep("identify_energy_transfers", "policy")
class IdentifyEnergyTransfers(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_transfers_per_boid = self.learnable_args.get("max_transfers_per_boid", 1)
    
    def forward(self, state: Dict[str, Any], observations) -> Dict[str, Any]:
        position = get_var(state, self.input_variables["position"])
        energy = get_var(state, self.input_variables["energy"])
        alive_status = get_var(state, self.input_variables["alive_status"])
        energy_transfer_radius = get_var(state, self.input_variables["energy_transfer_radius"])
        min_energy_diff = get_var(state, self.input_variables["min_energy_diff"])
        
        energy_transfers_list = []
        
        for i in range(len(position)):
            if not alive_status[i]:
                continue
                
            transfers_made = 0
            for j in range(len(position)):
                if i == j or not alive_status[j] or transfers_made >= self.max_transfers_per_boid:
                    continue
                
                distance = torch.norm(position[i].float() - position[j].float(), p=1)
                energy_diff = energy[i] - energy[j]
                
                if distance <= energy_transfer_radius and energy_diff >= min_energy_diff:
                    energy_transfers_list.append((i, j, energy_diff))
                    transfers_made += 1
        return {
            self.output_variables[0]: energy_transfers_list
        }