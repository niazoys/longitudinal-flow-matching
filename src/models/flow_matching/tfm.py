import ipdb
import torch
import numpy as np
from torchdyn.core import NeuralODE
from torch import nn
from src.models.flow_matching.components.grad_util import torch_wrapper_tv
from src.models.flow_matching.components.sde_func_utils import *
from src.models.flow_matching.components.fm_utils import *
from src.models.velocity_field_models.vector_field_regressor import VectorFieldRegressorBottleNeckCond # MRI
from src.models.velocity_field_models.vrf_context_cond import VectorFieldRegressor # axial/age/sex conditioned
from src.models.velocity_field_models.vrf_context_sigma_score import VectorFieldRegressorSigma
from src.models.velocity_field_models.vector_field_bnc import VectorFieldRegressorSigma_BNC

NUM_FREQS = 32

def test_trajectory_ode(batch, model, noise_prediction, sequential_traj=True, random_history=False,run_unconditional=True):
    """
    Test trajectory using ODE solver.
    
    Args:
        batch: Tuple containing data tensors
        model: Neural network model
        noise_prediction: Boolean for noise prediction
        sequential_traj: Boolean for sequential trajectory generation
        random_history: If True, randomly selects from history of predictions instead of using just the last one
    Returns:
        mse_all: Mean squared error 
        total_pred_tensor: Predicted trajectories
        mse_all: Mean squared error (repeated)
        noise_pred: Noise predictions if noise_prediction=True, else None
    """
    # Initialize ODE solver
    node_params = {
        "solver": "dopri5",
        "sensitivity": "adjoint",
        "atol": 1e-6,
        "rtol": 1e-6
    }
    
    if noise_prediction:
        node = NeuralODE(torch_wrapper_tv(model.flow_model), **node_params)
        node_noise = NeuralODE(torch_wrapper_tv(model.noise_model), **node_params)
    else:
        node = NeuralODE(torch_wrapper_tv(model), **node_params)

    # Unpack batch data
    x0_values, x0_classes, x1_values, times_x0, times_x1, x0_cls_time, label,x0_age,sex = batch
    x0_values = x0_values.squeeze(0)
    x0_classes = x0_classes.squeeze(0)
    x1_values = x1_values.squeeze(0)
    times_x0 = times_x0.squeeze().float()
    times_x1 = times_x1.squeeze().float()    
    if run_unconditional:
        label = torch.zeros_like(label).to(x0_values.device)
    else:
        label = label.to(x0_values.device)


    # Initialize storage
    total_pred, noise_pred = [], []
    mse = []
    len_path = x0_values.shape[0]

    # Initialize ref and memory 
    if model.memory > 0:
        history = x0_classes[0].squeeze(0)
        hist_time = x0_cls_time[0]
    x0 = x0_values[0]
    
    # For random history selection
    if random_history:
        history_list = [history.clone(),x0]
        hist_time_list = [hist_time.clone(),times_x0[0]] 
    
    # Generate trajectory predictions
    for i in range(len_path):
        tau_span = torch.linspace(times_x0[i], times_x1[i], 20).float().to(x0_values.device)
        time_span = torch.linspace(0, times_x1[i]-times_x0[i], 20).float().to(x0_values.device)
        # time_span= torch.zeros_like(tau_span).to(x0_values.device) # this is just dummy
        
        if sequential_traj:
            if random_history and i > 0 and len(history_list) > model.memory:
                idx = torch.randint(0, len(history_list), (model.memory,)).item()
                history = history_list[idx]
                hist_time = hist_time_list[idx]
            
            # Create test point (using either random history or latest)
            testpt = create_testpoint_ode(
                x0, history, hist_time, label, 
                times_x1[i], time_span[i],x0_age[i],sex)
        else:
            testpt = create_testpoint_ode(
                x0_values[i], x0_classes[i], x0_cls_time[i], 
                label, times_x1[i], time_span[i],x0_age[i],sex)
                
        with torch.no_grad():
            traj = node.trajectory(testpt, t_span=tau_span)
            noise_traj = node_noise.trajectory(testpt, t_span=tau_span) if noise_prediction else None

        # Process predictions
        pred_traj = traj[-1, :, :model.dim]
        total_pred.append(pred_traj)

        if noise_prediction:
            noise_traj = noise_traj[-1, :, :model.dim]
            noise_pred.append(noise_traj)

        # Calculate errors
        ground_truth_coords = x1_values[i]
        mse_traj = model.loss_fn(pred_traj, ground_truth_coords).detach().cpu().numpy()
        mse.append(mse_traj)

        if noise_prediction:
            uncertainty_traj = ground_truth_coords - pred_traj
            noise_mse_traj = model.loss_fn(noise_traj, uncertainty_traj).detach().cpu().numpy()

        # Update reference point x0 and memory(history)
        if model.memory == 1:        
            history = x0
            hist_time = tau_span[0]
        elif model.memory > 1:
            history = history.view(-1, model.dim)
            history = torch.cat([history[-1,:].view(1,-1), x0], dim=1).squeeze()
            hist_time = torch.cat([hist_time[-1].view(1,-1), tau_span[0].view(1,-1)], dim=1)

        # Update x0 for next iteration
        x0 = pred_traj
        
        # Store updated values in history lists for random selection
        if random_history:
            history_list.append(pred_traj.clone())
            hist_time_list.append(tau_span[-1])

    # Aggregate results
    mse_all = np.mean(mse)
    total_pred_tensor = torch.stack(total_pred).squeeze(1)
    if noise_prediction:
        noise_pred = torch.stack(noise_pred).squeeze(1)

    return mse_all, total_pred_tensor, mse_all, noise_pred if noise_prediction else None

def test_trajectory_sde_new(batch, model,time_scaler,noise_prediction, 
                            sequential_traj=True, random_history=False,run_unconditional=True,
                            dim=4096,
                            model_type='sde',
                            uw=0.1):
    """
    Test trajectory using ODE solver.
    
    Args:
        batch: Tuple containing data tensors
        model: Neural network model
        noise_prediction: Boolean for noise prediction
        sequential_traj: Boolean for sequential trajectory generation
        random_history: If True, randomly selects from history of predictions instead of using just the last one
    Returns:
        mse_all: Mean squared error 
        total_pred_tensor: Predicted trajectories
        mse_all: Mean squared error (repeated)
        noise_pred: Noise predictions if noise_prediction=True, else None
    """
    # sde = (SDE_func_solver(model, noise=None) 
    #        if noise_prediction else SDE(model, noise=0.1))
    sde=SDE_func_solver(model,time_scaler=time_scaler,dim=dim)

    # Unpack batch data
    x0_values, x0_classes, x1_values, times_x0, times_x1, x0_cls_time, label,x0_age,sex = batch
    x0_values = x0_values.squeeze(0)
    x0_classes = x0_classes.squeeze(0)
    x1_values = x1_values.squeeze(0)
    times_x0 = times_x0.squeeze().float()
    times_x1 = times_x1.squeeze().float()    
    if run_unconditional:
        label = torch.zeros_like(label).to(x0_values.device)
    else:
        label = label.to(x0_values.device)


    # Initialize storage
    total_pred, noise_pred = [], []
    mse = []
    len_path = x0_values.shape[0]

    # Initialize ref and memory 
    if model.memory > 0:
        history = x0_classes[0].squeeze(0)
        hist_time = x0_cls_time[0]
    x0 = x0_values[0]
    
    # For random history selection
    if random_history:
        history_list = [history.clone(),x0]
        hist_time_list = [hist_time.clone(),times_x0[0]] 
    # ipdb.set_trace()
    # Generate trajectory predictions
    for i in range(len_path):
        if time_scaler is not None:
            delta_time = time_scaler(times_x1[i]- times_x0[i])
            tau_span = torch.linspace(times_x0[i], times_x0[i] + delta_time, 25).float().to(x0_values.device)
        else:
            tau_span = torch.linspace(times_x0[i], times_x1[i], 25).float().to(x0_values.device)

        time_span = torch.linspace(0, times_x1[i]-times_x0[i], 25).float().to(x0_values.device)
        # time_span= torch.zeros_like(tau_span).to(x0_values.device) # this is just dummy
        
        if sequential_traj:
            if random_history and i > 0 and len(history_list) > model.memory:
                idx = torch.randint(0, len(history_list), (model.memory,)).item()
                history = history_list[idx]
                hist_time = hist_time_list[idx]
            
            # Create test point (using either random history or latest)
            testpt = create_testpoint_ode(
                x0, history, hist_time, label, 
                times_x1[i], time_span[i],x0_age[i],sex)
        else:
            testpt = create_testpoint_ode(
                x0_values[i], x0_classes[i], x0_cls_time[i], 
                label, times_x1[i], time_span[i],x0_age[i],sex)
                
        with torch.no_grad():
            traj , noise_traj = new_sde_solver(sde, testpt.float(), tau_span,dim=dim,int_type=model_type,uncert_w=uw)

        # Process predictions
        pred_traj = traj[-1, :, :model.dim]
        total_pred.append(pred_traj)

        if noise_prediction:
            noise_traj = noise_traj[-1, :, :model.dim]
            noise_pred.append(noise_traj)

        # Calculate errors
        ground_truth_coords = x1_values[i]
        mse_traj = model.loss_fn(pred_traj, ground_truth_coords).detach().cpu().numpy()
        mse.append(mse_traj)

        if noise_prediction:
            uncertainty_traj = ground_truth_coords - pred_traj
            noise_mse_traj = model.loss_fn(noise_traj, uncertainty_traj).detach().cpu().numpy()

        # Update reference point x0 and memory(history)
        if model.memory == 1:        
            history = x0
            hist_time = tau_span[0]
        elif model.memory > 1:
            history = history.view(-1, model.dim)
            history = torch.cat([history[-1,:].view(1,-1), x0], dim=1).squeeze()
            hist_time = torch.cat([hist_time[-1].view(1,-1), tau_span[0].view(1,-1)], dim=1)

        # Update x0 for next iteration
        x0 = pred_traj
        
        # Store updated values in history lists for random selection
        if random_history:
            history_list.append(pred_traj.clone())
            hist_time_list.append(tau_span[-1])

    # Aggregate results
    mse_all = np.mean(mse)
    total_pred_tensor = torch.stack(total_pred).squeeze(1)
    if noise_prediction:
        noise_pred = torch.stack(noise_pred).squeeze(1)

    return mse_all, total_pred_tensor, mse_all, noise_pred if noise_prediction else None

def create_testpoint_ode(x0, history, hist_time, label, 
                              times_x1, time_span,x0_age,sex):
    """Helper function to create test points for trajectory prediction"""
    # return torch.cat([
    #         x0,
    #         history.view(1,-1),
    #         hist_time.view(1,-1),
    #         label.view(1,-1),
    #         times_x1.view(1,-1),
    #         time_span.view(1,-1)
    #     ], dim=1)
    # ipdb.set_trace()
    # history = torch.zeros_like(history)
    # hist_time = torch.zeros_like(hist_time)
    return torch.cat([
            x0,
            history.view(1,-1),
            hist_time.view(1,-1),
            sex.view(1,-1),
            x0_age.view(1,-1),
            label.view(1,-1),
            torch.zeros_like(times_x1.view(1,-1)),
            torch.zeros_like(time_span.view(1,-1))
        ], dim=1)

class TimeScaling(nn.Module):
    def __init__(self, init_params=None,min_clip=0.25, max_clip=5.0):
        super().__init__()
        self.min_clip = min_clip
        self.max_clip = max_clip
        # Base sigmoid parameters (similar to original)
        self.alpha = nn.Parameter(torch.tensor(2.0))  # Sigmoid steepness
        self.beta = nn.Parameter(torch.tensor(1.0))   # Sigmoid scale
        
        # Additional shape parameters
        self.poly_coef = nn.Parameter(torch.tensor(0.2))  # Polynomial term for asymmetry
        self.exp_pos = nn.Parameter(torch.tensor(0.1))    # Exponential term for large values
        self.exp_neg = nn.Parameter(torch.tensor(0.3))    # Exponential term for small values
        
        # Center adjustment parameters
        self.center_shift = nn.Parameter(torch.tensor(0.0))  # Shift the response center
        
        # Component mixing weights
        self.w_sigmoid = nn.Parameter(torch.tensor(0.7))
        self.w_poly = nn.Parameter(torch.tensor(0.2))
        self.w_exp = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, normalized_dt):
        # Shift input relative to center (1.0 + adjustable shift)
        x = normalized_dt - (1.0 + self.center_shift)
        
        # Sigmoid component (symmetric response)
        sigmoid_comp = self.beta * (2.0 / (1.0 + torch.exp(-self.alpha * x)) - 1.0)
        
        # Polynomial component (asymmetric response)
        # x^3 term ensures the function goes through (1,0)
        poly_comp = self.poly_coef * (x**3)
        
        # Exponential components (different behavior for small vs large times)
        exp_comp_pos = self.exp_pos * (torch.exp(x) - 1.0) * (x > 0).float()
        exp_comp_neg = self.exp_neg * (1.0 - torch.exp(-x)) * (x < 0).float()
        exp_comp = exp_comp_pos + exp_comp_neg
        
        # Normalize weights to sum to 1
        total_weight = torch.abs(self.w_sigmoid) + torch.abs(self.w_poly) + torch.abs(self.w_exp)
        w_sig = torch.abs(self.w_sigmoid) / total_weight
        w_poly = torch.abs(self.w_poly) / total_weight
        w_exp = torch.abs(self.w_exp) / total_weight
        
        # Combine all components
        combined = w_sig * sigmoid_comp + w_poly * poly_comp + w_exp * exp_comp
        
        # Ensure positive output with reasonable bounds
        return torch.clamp(1.0 + combined, min=self.min_clip, max=self.max_clip)

class MLP_conditional_memory_sde_noise(torch.nn.Module):
    def __init__(self, dim, treatment_cond, memory, out_dim=None, w=64,
                 time_varying=False, conditional=False, time_dim=NUM_FREQS * 2,
                 clip=None):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            self.out_dim = 1
        self.treatment_cond = treatment_cond
        self.memory = memory
        self.dim = dim
        self.indim = dim + (time_dim if time_varying else 0) + (self.treatment_cond if conditional else 0) + (dim * memory)
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.indim, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, self.out_dim),
        )
        self.default_class = 0
        self.clip = clip

    def encoding_function(self, time_tensor):
        return positional_encoding_tensor(time_tensor)

    def forward_train(self, x):
        time_tensor = x[:, -1]
        encoded_tau_span = self.encoding_function(time_tensor).reshape(-1, NUM_FREQS * 2)
        new_x = torch.cat([x[:, :-1], encoded_tau_span], dim=1)
        result = self.net(new_x)
        return result

    def forward(self, x):
        result = self.forward_train(x)
        return torch.cat([result, torch.zeros_like(x[:, 1:-1])], dim=1)

class MLP_Cond_Memory_Module(torch.nn.Module):
    def __init__(self, treatment_cond, memory=3, dim=2, w=64, time_varying=True,
                 conditional=True, lr=1e-6, sigma=0.1, loss_fn=None,
                 metrics=['mse_loss', 'l1_loss'], implementation="ODE",
                 sde_noise=0.1, clip=None, depth=6, naming=None, velocity_net='vrf', noise_prediction=False,
                 time_prediction=False, class_conditional=False, time_dim=NUM_FREQS * 2, starman=False, gbm=False, demo_conditional=True,
                 time_scaling_min_clip=0.25):
        super().__init__()
        self.class_conditional = class_conditional 
        self.loss_fn = loss_fn 
        self.dim = dim
        self.w = w
        self.time_varying = time_varying
        self.conditional = conditional
        self.treatment_cond = treatment_cond
        self.lr = lr
        self.sigma = sigma
        self.metrics = metrics
        self.implementation = implementation
        self.memory = memory
        self.sde_noise = sde_noise
        self.clip = clip
        self.time_prediction = time_prediction
        self.out_dim = dim
        self.out_dim += 1
        self.temp = 0
        self.indim = dim + (time_dim if time_varying else 0) + (self.treatment_cond if conditional else 0) + (dim * memory)
        self.velocity_net = velocity_net
        self.use_spline_velocity = True
        self.t_prev = 0.0
        self.noise_prediction = noise_prediction
        self.demo_conditional = demo_conditional
        self.time_scaling = TimeScaling(min_clip=time_scaling_min_clip)
        self.gbm = gbm
        
        # Only VRF velocity net is supported
        if velocity_net != 'vrf':
            raise ValueError(f"Only 'vrf' velocity_net is supported, got '{velocity_net}'")
            
        if self.noise_prediction:
            if self.gbm:
                self.net = VectorFieldRegressorSigma_BNC(
                    depth=depth, 
                    mid_depth=2,
                    state_size=256,  # Match your channel count
                    state_res=(4, 4),  # Match your spatial dimensions
                    inner_dim=512,  # Adjust as needed
                    reference=False
                )
            elif starman:
                self.net = VectorFieldRegressorSigma(
                    depth=depth, 
                    mid_depth=1,
                    state_size=256,  # Match your channel count
                    state_res=(1, 1),  # Match your spatial dimensions
                    inner_dim=512,  # Adjust as needed
                    reference=False,
                    starman=True
                )
            else:
                self.net = VectorFieldRegressorSigma(
                    depth=depth, 
                    mid_depth=2,
                    state_size=256,  # Match your channel count
                    state_res=(4, 4),  # Match your spatial dimensions
                    inner_dim=512,  # Adjust as needed
                    reference=False
                )
        else:
            if not self.demo_conditional:
                self.net = VectorFieldRegressorBottleNeckCond(
                    depth=depth,
                    mid_depth=2,
                    state_size=256,  # Match your channel count
                    state_res=(4, 4),  # Match your spatial dimensions
                    inner_dim=512,  # Adjust as needed
                    reference=False
                )
            else:
                self.net = VectorFieldRegressor(
                    depth=depth, 
                    mid_depth=2,
                    state_size=256,  # Match your channel count
                    state_res=(4, 4),  # Match your spatial dimensions
                    inner_dim=512,  # Adjust as needed
                    reference=False
                )
            
        self.default_class = 0
        self.clip = clip

    def encoding_function(self, time_tensor):
        return positional_encoding_tensor(time_tensor, NUM_FREQS)

    def forward_train(self, x):
        """Forward pass during training - only supports VRF velocity net."""
        if self.noise_prediction:
            results = self.net(x)
            mu = results[:, :self.dim]
            score = results[:, self.dim:2 * self.dim]
            sigma = results[:, 2 * self.dim:-1]
            return mu, score, sigma
        else:
            result = self.net(x)
            return result



    def forward(self, x):        
        mu,score,sigma = self.forward_train(x)
        x1_coord = mu
        uncertainty = sigma
        # self.clip=0.05 # MRIs 0.01
        # self.clip=0.15 # ADNI_axial

        # pred_time_till_t1 = (x[:, -1] - self.t_prev)     
        # vt=x1_coord/torch.clip((pred_time_till_t1), min=self.clip).view(-1, 1).expand_as(x1_coord)
        if self.clip is not None:
            vt=x1_coord/torch.tensor(self.clip).view(-1, 1).expand_as(x1_coord).to(x1_coord.device)
        else:
            vt=x1_coord
        # ipdb.set_trace()
        self.t_prev = x[:, -1]
        return torch.cat([vt,score,uncertainty,torch.zeros((1,1)).to(vt.device)], dim=1)

    





    def __convert_tensor__(self, tensor):
        return tensor.to(torch.float32)

    def __x_processing__(self, x0, x1, t0, t1):
        t = torch.rand(x0.shape[0], 1).to(x0.device)
        
        # Try this out!
        # t = torch.sigmoid(t).to(x0.device)

        mu_t = x0 * (1 - t) + x1 * t
        data_t_diff = (t1 - t0).unsqueeze(1)
        x = mu_t + self.sigma * torch.randn(x0.shape[0], self.dim).to(x0.device)
        # PFSI paper
        # x = mu_t + (1-t) * torch.randn(x0.shape[0], self.dim).to(x0.device)
        tau = t * data_t_diff + t0.unsqueeze(1)
        futuretime = t1.view(-1,1) - tau
        ut= (x1 - x0) # this is equivalent to learning residual
        # ut = (x1 - x0) / (data_t_diff + 1e-8) # original
        # ut= (x1 - x) / (t1.view(-1,1) - tau) # the is following lipman
        # ut= (x1 - x) / (1 - t) # Experiment
        
        return x, ut, tau, futuretime, t
