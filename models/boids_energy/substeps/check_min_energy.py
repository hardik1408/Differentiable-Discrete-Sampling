from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers import get_var

@Registry.register_substep("check_min_energy", "policy")
class CheckMinEnergy(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize learnable parameters from arguments
        self.min_energy_param = self.learnable_args.get("min_energy", 5.0)
    
    def forward(self, state: Dict[str, Any], observations) -> Dict[str, Any]:
        """
        Apply speed limits to velocities.
        
        Args:
            state: Current state of the simulation
            observations: Agent observations
            
        Returns:
            Dict containing: limited_velocity
        """

        energy = get_var(state, self.input_variables["energy"])
        min_energy = get_var(state, self.input_variables["min_energy"])
        
        min_energy_check = torch.where(energy < min_energy, torch.zeros_like(energy), energy)
        # Calculate current speed (magnitude of velocity)

        
        return {
            self.output_variables[0]: min_energy_check  # limited_velocity
        } 