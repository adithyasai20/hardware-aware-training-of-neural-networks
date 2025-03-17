import torch
from torch.optim import Optimizer

class RRAMOptimizer(Optimizer):
    def __init__(self, params, potentiation, depression, tau):
        if not isinstance(potentiation, list):
            raise ValueError("potentiation should be a list")
        if not isinstance(depression, list):
            raise ValueError("depression should be a list")
        
        def map_to_nearest_indices(given_list, reference_list):
            def safe_tensor_conversion(data, dtype=torch.float32):
                if isinstance(data, torch.Tensor):
                    return data.clone().detach().to(dtype)  # Clone if it's already a tensor
                return torch.tensor(data, dtype=dtype)  # Convert if it's a list or other type

            given_tensor = safe_tensor_conversion(given_list)
            reference_tensor = safe_tensor_conversion(reference_list)

            # given_tensor = torch.tensor(given_list, dtype=torch.float32)
            # reference_tensor = torch.tensor(reference_list, dtype=torch.float32)
            
            # Compute absolute differences and find nearest indices
            indices = torch.abs(reference_tensor.unsqueeze(0) - given_tensor.unsqueeze(-1)).argmin(dim=-1)
            
            return torch.tensor(indices.tolist())
        
        def get_weights_from_conductances(finite_set):
            min_conductance, max_conductance = torch.min(torch.tensor([potentiation, depression])), torch.max(torch.tensor([potentiation, depression]))
            # min_conductance, max_conductance = finite_set[0], finite_set[-1]
            mid_conductance = (min_conductance + max_conductance) / 2
            min_weight, max_weight = -1.0, 1.0
            mid_weight = (max_weight + min_weight) / 2

            normalized_conductance = (finite_set - mid_conductance)/(max_conductance - mid_conductance)

            normalized_weight = normalized_conductance

            weights = normalized_weight * (max_weight - mid_weight) + mid_weight

            return weights
        
        potentiation_weights = get_weights_from_conductances(torch.tensor(potentiation))
        depression_weights = get_weights_from_conductances(torch.tensor(depression))

        self.potentiation_to_depression_mapping = map_to_nearest_indices(potentiation_weights, depression_weights)
        self.depression_to_potentiation_mapping = map_to_nearest_indices(depression_weights, potentiation_weights)

        # print(self.potentiation_to_depression_mapping)
        # print(self.depression_to_potentiation_mapping)

        defaults = {"potentiation":potentiation_weights, "depression":depression_weights, "tau":tau, }
        super().__init__(params, defaults)

        if isinstance(potentiation_weights, torch.Tensor):
            self.potentiation = potentiation_weights.clone().detach().to(torch.float32)
        else:
            self.potentiation = torch.tensor(potentiation_weights, dtype=torch.float32)
        
        if isinstance(depression_weights, torch.Tensor):
            self.depression = depression_weights.clone().detach().to(torch.float32)
        else:
            self.depression = torch.tensor(depression_weights, dtype=torch.float32)

        for group in self.param_groups:
            for p in group["params"]:
                self._initialize_param(p)


    def _initialize_param(self, p):
        """snaps param to nearest valid discrete state"""
        self.potentiation = self.potentiation.to(device=p.device)
        self.depression = self.depression.to(device=p.device)
        self.depression_to_potentiation_mapping = self.depression_to_potentiation_mapping.to(p.device)
        self.potentiation_to_depression_mapping = self.potentiation_to_depression_mapping.to(p.device)
        

        distances = torch.cdist(p.data.view(-1,1), self.potentiation.view(-1, 1))

        
        # Get the index of the minimum distance for each element
        min_indices = torch.argmin(distances, dim=1)  # Shape [3*4*5]

        # Reshape back to the original dimensions (without last dim)
        min_indices = min_indices.view(*p.data.shape)

        p.data.copy_(self.potentiation[min_indices])
        self.state[p] = {"state_idx": min_indices, "flags":torch.ones_like(min_indices)}

    def step(self):

        for group in self.param_groups:
            tau = group['tau']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad

                state_idx = self.state[p]['state_idx']
                flags = self.state[p]['flags']

                potentiation_to_decrease_mask = ((grad > tau) & (state_idx < len(self.potentiation) - 1) & (flags)).bool()
                potentiation_to_increase_mask = ((grad < -tau) & (state_idx < len(self.potentiation) - 1) & (flags.bool())).bool()

                depression_to_decrease_mask = ((grad > tau) & (state_idx < len(self.potentiation) - 1) & (1 - flags)).bool()
                depression_to_increase_mask = ((grad < -tau) & (state_idx < len(self.potentiation) - 1) & (~flags.bool())).bool()

                # print(potentiation_to_increase_mask, potentiation_to_decrease_mask)
                # print(depression_to_increase_mask, depression_to_decrease_mask)
                # print(flags, state_idx)
                # print("\n")


                state_idx[potentiation_to_increase_mask] += 1
                state_idx[depression_to_decrease_mask] += 1

                if potentiation_to_decrease_mask.any():
                    state_idx[potentiation_to_decrease_mask] = self.potentiation_to_depression_mapping[   state_idx[potentiation_to_decrease_mask]    ] + 1
                    flags[potentiation_to_decrease_mask] = False

                # Map potentiation to depression correctly
                # if potentiation_to_decrease_mask.any():
                #     mapped_indices = self.potentiation_to_depression_mapping[state_idx[potentiation_to_decrease_mask]]
                #     state_idx[potentiation_to_decrease_mask] = torch.clamp(mapped_indices + 1, max=len(self.depression) - 1)
                #     flags[potentiation_to_decrease_mask] = 0 

                

                if depression_to_increase_mask.any():
                    state_idx[depression_to_increase_mask] = self.depression_to_potentiation_mapping[   state_idx[depression_to_increase_mask]   ] + 1
                    flags[depression_to_increase_mask] = True

                # Map depression to potentiation correctly
                # if depression_to_increase_mask.any():
                #     mapped_indices = self.depression_to_potentiation_mapping[state_idx[depression_to_increase_mask]]
                #     state_idx[depression_to_increase_mask] = torch.clamp(mapped_indices + 1, max=len(self.potentiation) - 1)
                #     flags[depression_to_increase_mask] = 1 

                # Apply updates to parameter data
                updated_values = torch.where(
                    flags.bool(),
                    self.potentiation[state_idx],
                    self.depression[state_idx]
                )
                p.data.copy_(updated_values)



                

                







    


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