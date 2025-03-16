import torch
from torch.optim import Optimizer

class DiscreteStateOptimizer(Optimizer):
    def __init__(self, params, conductance_list, tau, tie_break="floor"):
        """
        Args:
            params: Iterable of trainable parameters.
            state_list: A single list of allowed discrete states shared by all parameters.
            tau: Threshold for updating states based on gradient magnitude.
            tie_break: Strategy for handling ties ("floor", "ceil", "random").
        """
        if not isinstance(conductance_list, list):
            raise ValueError("state_list should be a list of discrete states.")
        if tie_break not in ["floor", "ceil", "random"]:
            raise ValueError("tie_break must be 'floor', 'ceil', or 'random'.")
        
        def get_weights_from_conductances(finite_set):
            # min_conductance, max_conductance = torch.min(finite_set), torch.max(finite_set)
            min_conductance, max_conductance = finite_set[0], finite_set[-1]
            mid_conductance = (min_conductance + max_conductance) / 2
            min_weight, max_weight = -1.0, 1.0
            mid_weight = (max_weight + min_weight) / 2

            normalized_conductance = (finite_set - mid_conductance)/(max_conductance - mid_conductance)

            normalized_weight = normalized_conductance

            weights = normalized_weight * (max_weight - mid_weight) + mid_weight

            return weights
        
        state_list = get_weights_from_conductances(torch.tensor(conductance_list))
        defaults = {"state_list": state_list, "tau": tau, "tie_break": tie_break}
        super().__init__(params, defaults)

        # Convert state list to a tensor (shared by all params)
        if isinstance(state_list, torch.Tensor):
            self.state_tensor = state_list.clone().detach().to(torch.float32)
        else:
            self.state_tensor = torch.tensor(state_list, dtype=torch.float32)


        # Initialize parameters to the closest allowed state
        for group in self.param_groups:
            for p in group["params"]:
                self._initialize_param(p)

    def _initialize_param(self, p):
        """snaps param to nearest valid discrete state"""
        self.state_tensor = self.state_tensor.to(device=p.device)


        distances = torch.cdist(p.data.view(-1,1), self.state_tensor.view(-1, 1))

        
        # Get the index of the minimum distance for each element
        min_indices = torch.argmin(distances, dim=1)  # Shape [3*4*5]

        # Reshape back to the original dimensions (without last dim)
        min_indices = min_indices.view(*p.data.shape)

        p.data.copy_(self.state_tensor[min_indices])
        self.state[p] = {"state_idx": min_indices}


    @torch.no_grad()
    def step(self):
        """Performs a single optimization step."""
        for group in self.param_groups:
            tau = group["tau"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state_idx = self.state[p]["state_idx"]

                # Create masks for different updates
                increase_mask = (grad > tau) & (state_idx > 0)
                decrease_mask = (grad < -tau) & (state_idx < len(self.state_tensor) - 1)

                # Apply masks to update indices
                state_idx[increase_mask] -= 1
                state_idx[decrease_mask] += 1

                # Update parameter values and store new indices
                p.data.copy_(self.state_tensor[state_idx])
                self.state[p]["state_idx"] = state_idx