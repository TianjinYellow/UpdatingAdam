# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version

from .galore_projector import GaLoreProjector
from .galore_projector_tensor import GaLoreProjectorTensor


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        updating_mask_method='random',
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
   
        super().__init__(params, defaults)
        self.init_masks()
        self.checksparsity()
        self.total_step=0
        self.current_step=100
        self.update_proj_gap=None
        self.updating_mask_method=updating_mask_method
    def init_masks(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "rank" in group:
                    if 'mask' not in state:
                        assert len(p.data.shape)==2
                        state['mask']=self.initialize_diagonal_rank_boolean_tensor(p.data.shape[0],p.data.shape[1],group['rank']).to(p.device)
    def checksparsity(self):
        total_num=0
        non_zero_num=0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "rank" in group:
                    total_num+=state['mask'].numel()
                    non_zero_num+=state['mask'].sum().item()
        print("density",non_zero_num/total_num)
    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        if self.total_step!=0:
            if (self.total_step+1) % self.update_proj_gap == 0:
                self.update_masks()
                print("Mask Update",flush=True)
                self.checksparsity()
                self.current_step=0
        # else:
        #     self.update_masks()
        #     print("Mask Update",flush=True)
        #     self.checksparsity()
        #     self.current_step=0
        scale=self.adjust_learning_rate(100,self.current_step)
        for group in self.param_groups:
            if "rank" in group:
                self.update_proj_gap=group["update_proj_gap"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                
                if 'dim' not in group:
                    group['dim'] = 2
                # GaLore Projection
                if "rank" in group:
                    grad=grad[state['mask']]

                # State initialization or (self.total_step+1) % self.update_proj_gap == 0
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                if "rank" in group:
                    if (self.total_step+1) % self.update_proj_gap == 0:
                        state["exp_avg"] = torch.zeros_like(grad)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                    
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom
                
                #GaLore Projection Back
                # if "rank" in group:
                #     norm_grad = state["projector"].project_back(norm_grad)
                if "rank" in group:
                    # print("norm",norm_grad)
                    # print("gradient",grad)                    
                    grad=p.grad
                    grad[state['mask']]=norm_grad
                    grad[~state['mask']]=0
                    #print("scale",scale)
                    grad=grad*scale
                    
                else:
                    grad=norm_grad*scale
                # if (self.total_step+1)%250==0 and 'rank' in group:
                #     print("step size",step_size,scale1,scale,group['lr'])
                p.add_(grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
        self.total_step+=1 
        self.current_step+=1   

        return loss

    def initialize_diagonal_rank_boolean_tensor(self,m, n, density):
        total_elements = m * n
        non_zero_count = int(density * total_elements)
        
        # Create a tensor with all False values
        tensor = torch.zeros((m, n), dtype=torch.bool)
        
        # Ensure non_zero_count is within valid range
        non_zero_count = min(non_zero_count, total_elements)
        
        # Calculate min_dim
        min_dim = min(m, n)
        
        # Initialize the counter for filled elements
        filled_count = 0
        
        # Function to fill a position if it is within bounds and needed
        def try_fill(i, j):
            nonlocal filled_count
            if 0 <= i < m and 0 <= j < n and filled_count < non_zero_count and not tensor[i, j]:
                tensor[i, j] = True
                filled_count += 1
        
        # Fill the main diagonal and adjacent diagonals uniformly
        for i in range(min_dim):
            if filled_count >= non_zero_count:
                break
            try_fill(i, i)  # Main diagonal
            if filled_count < non_zero_count:
                try_fill(i, i+1)  # Above main diagonal
            if filled_count < non_zero_count:
                try_fill(i+1, i)  # Below main diagonal
        
        # Calculate remaining non-zero elements needed
        remaining_non_zeros = non_zero_count - filled_count
        
        if remaining_non_zeros > 0:
            # Generate remaining_non_zeros unique random positions
            remaining_indices = torch.nonzero(~tensor).tolist()
            random_indices = torch.randperm(len(remaining_indices))[:remaining_non_zeros]
            
            for idx in random_indices:
                i, j = remaining_indices[idx]
                tensor[i, j] = True

        return tensor

    def initialize_partial_rank_boolean_tensor(self,m, n, density):
        total_elements = m * n
        non_zero_count = int(density * total_elements)
        
        # Create a tensor with all False values
        tensor = torch.zeros((m, n), dtype=torch.bool)
        
        # Ensure non_zero_count is within valid range
        non_zero_count = min(non_zero_count, total_elements)
        
        # Calculate min_dim
        min_dim = min(m, n)
        
        # If non_zero_count is smaller than min_dim, adjust the rank to be non_zero_count
        fill_count = min(non_zero_count, min_dim)
        
        # Fill the diagonal elements to ensure at least fill_count True elements
        if fill_count > 0:
            indices = torch.arange(fill_count)
            tensor[indices, indices] = True
        
        # Calculate remaining non-zero elements needed
        remaining_non_zeros = non_zero_count - fill_count
        
        if remaining_non_zeros > 0:
            # Generate remaining_non_zeros unique random positions
            remaining_indices = torch.nonzero(~tensor).tolist()
            random_indices = torch.randperm(len(remaining_indices))[:remaining_non_zeros]
            
            for idx in random_indices:
                i, j = remaining_indices[idx]
                tensor[i, j] = True

        return tensor
    def update_masks(self):
        for group in self.param_groups:
            print("lr", group['lr'])
            for p in group["params"]:
                state = self.state[p]
                if "rank" in group:
                    assert len(p.data.shape) == 2
                    if self.updating_mask_method == 'random':
                        new_mask, overlap_ratio = self.update_mask_random(group['rank'], p.grad, state['mask'])
                    elif self.updating_mask_method == 'grad_max':
                        new_mask, overlap_ratio = self.update_mask(group['rank'], p.grad, state['mask'])
                    elif self.updating_mask_method == 'weight_max':
                        new_mask, overlap_ratio = self.update_mask(group['rank'], p.data, state['mask'])
                    else:
                        print("Not Implemented!")
                    state['mask'] = new_mask
                    print(f"Mask overlap ratio: {overlap_ratio:.2f}")

    def update_mask(self, density, gradients, old_mask):
        m, n = gradients.shape
        total_elements = m * n
        non_zero_count = int(density * total_elements)

        # Ensure non_zero_count is within valid range
        non_zero_count = min(non_zero_count, total_elements)

        # Create a tensor with all False values
        new_mask = torch.zeros((m, n), dtype=torch.bool).to(gradients.device)

        # Calculate the absolute values of the gradients
        gradient_abs = gradients.abs()

        # Flatten the gradients to easily sort and index
        flattened_gradients = gradient_abs.view(-1)

        # Get the indices of the top non_zero_count elements
        top_indices = torch.topk(flattened_gradients, non_zero_count).indices

        # Convert the flattened indices back to 2D indices
        rows = top_indices // n
        cols = top_indices % n

        # Set the selected elements to True
        new_mask[rows, cols] = True

        # Calculate the overlap ratio
        overlap_count = (new_mask & old_mask).sum().item()
        overlap_ratio = overlap_count / non_zero_count

        return new_mask, overlap_ratio

    def update_mask_random(self, density, gradients, old_mask):
        m, n = gradients.shape
        total_elements = m * n
        non_zero_count = int(density * total_elements)

        # Create a tensor with all False values
        new_mask = torch.zeros((m, n), dtype=torch.bool)

        # Ensure non_zero_count is within valid range
        non_zero_count = min(non_zero_count, total_elements)

        if non_zero_count > 0:
            # Generate unique random positions
            indices = torch.randperm(total_elements)[:non_zero_count]

            # Convert flat indices to 2D indices
            rows = indices // n
            cols = indices % n

            # Set the corresponding positions to True
            new_mask[rows, cols] = True

        # Calculate the overlap ratio
        overlap_count = (new_mask & old_mask).sum().item()
        overlap_ratio = overlap_count / non_zero_count

        return new_mask, overlap_ratio

    def initialize_random_rank_boolean_tensor(self, m, n, density):
        total_elements = m * n
        non_zero_count = int(density * total_elements)
        
        # Create a tensor with all False values
        tensor = torch.zeros((m, n), dtype=torch.bool)
        
        # Ensure non_zero_count is within valid range
        non_zero_count = min(non_zero_count, total_elements)
        
        if non_zero_count > 0:
            # Generate unique random positions
            indices = torch.randperm(total_elements)[:non_zero_count]
            
            # Convert flat indices to 2D indices
            rows = indices // n
            cols = indices % n
            
            # Set the corresponding positions to True
            tensor[rows, cols] = True

        return tensor
    def adjust_learning_rate(self, total_steps: int,current_step:int):
        """
        Adjusts the learning rate from 0.01 to 1 over the specified total number of steps.

        Args:
            total_steps (int): The total number of steps to reach the final learning rate.
        """
        initial_lr = 0.01
        final_lr = 1.0
        step_size = (final_lr - initial_lr) / total_steps
        current_step=min(current_step,total_steps)
        scale= initial_lr + current_step * step_size
        return scale
