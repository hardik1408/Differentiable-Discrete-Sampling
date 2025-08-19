from typing import Dict, Any
import torch
import torch.nn.functional as F
import torch.nn as nn
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_var
class Categorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p):
        if p.dim() == 1:
            p = p.unsqueeze(0)
        result = torch.multinomial(p, num_samples=1)
        ctx.save_for_backward(result, p)
        return result.float()
    
    @staticmethod
    def backward(ctx, grad_output):
        result, p = ctx.saved_tensors
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(1, result.long(), 1.0)
        p_safe = p.clamp(min=1e-6, max=1.0-1e-6)
        weights = (one_hot - p) / (p_safe * (1-p_safe))
        
        return grad_output.expand_as(p) * weights

class StochasticCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, uncertainty_in=None):
        if p.dim() == 1:
            p = p.unsqueeze(0)
        
        result = torch.multinomial(p, num_samples=1)
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        
        var_chosen = (1.0 - p) / p.clamp(min=1e-6)
        var_not_chosen = p / (1.0 - p).clamp(min=1e-6)
        op_variance = one_hot * var_chosen + (1 - one_hot) * var_not_chosen
        uncertainty_out = uncertainty_in + op_variance if uncertainty_in is not None else op_variance
        
        ctx.save_for_backward(result, p, uncertainty_out)
        return result, uncertainty_out
    
    @staticmethod
    def backward(ctx, grad_output, grad_uncertainty=None):
        result, p, uncertainty = ctx.saved_tensors
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        
        w_chosen = (1.0 / p) / 2
        w_non_chosen = (1.0 / (1.0 - p)) / 2
        confidence = 1.0 / (1.0 + uncertainty.clamp(min=1e-6))
        adjusted_ws = (one_hot * w_chosen + (1 - one_hot) * w_non_chosen) * confidence
        
        grad_p = grad_output.expand_as(p) * adjusted_ws
        return grad_p, None
    
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
        # new_position = position.clone()

        new_position = position.clone()

        for i in range(len(position)):
            if alive_status[i]:
                action_logits = F.gumbel_softmax(movement_probabilities[i], tau=1.0, hard=True)
                action_idx = action_logits.argmax(dim=-1)
                # print(action_logits)
                # action_logits = Categorical.apply(movement_probabilities[i])
                # action_logits, _ = StochasticCategorical.apply(movement_probabilities[i])
                # action_idx = action_logits

                if action_idx == 0:  # left
                    candidate = position[i] + torch.tensor([-1, 0], device=position.device, dtype=position.dtype)
                elif action_idx == 1:  # right
                    candidate = position[i] + torch.tensor([1, 0], device=position.device, dtype=position.dtype)
                elif action_idx == 2:  # up
                    candidate = position[i] + torch.tensor([0, -1], device=position.device, dtype=position.dtype)
                elif action_idx == 3:  # down
                    candidate = position[i] + torch.tensor([0, 1], device=position.device, dtype=position.dtype)
                else:  # stay
                    candidate = position[i]

                candidate = torch.stack([
                    torch.clamp(candidate[0], 0, grid_size[0] - 1),
                    torch.clamp(candidate[1], 0, grid_size[1] - 1)
                ])

                new_position = torch.cat([
                    new_position[:i],
                    candidate.unsqueeze(0),
                    new_position[i+1:]
                ], dim=0)

        
        return {
            self.output_variables[0]: new_position
        }