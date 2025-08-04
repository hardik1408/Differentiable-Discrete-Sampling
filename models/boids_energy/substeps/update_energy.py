from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_var

@Registry.register_substep("update_energy", "transition")
class UpdateEnergy(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize any learnable parameters from kwargs
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get('learnable', False):
                setattr(self, key, value['value'])
    
    def forward(self, state: Dict[str, Any], action) -> Dict[str, Any]:
        """
        Forward pass of the UpdateEnergy.
        
        Args:
            state: Current state of the simulation
            action: Action from policy
            
        Returns:
            Dict containing: energy
        """
        input_variables = self.input_variables
        
        # Get input variables from state
        energy = get_var(state, input_variables["energy"])
        energy_decay_rate = get_var(state, input_variables["energy_decay_rate"])
        
        new_energy = energy - energy_decay_rate
        
        return {
            self.output_variables[0]: new_energy,  # energy
        }
