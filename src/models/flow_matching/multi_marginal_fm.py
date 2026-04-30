from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src/mmfm"))


import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
from scipy import interpolate
from scipy.interpolate import Rbf, PchipInterpolator

def pad_a_like_b(a, b):
    """Pad a to have the same number of dimensions as b."""
    if isinstance(a, float | int):
        return a
    return a.reshape(-1, *([1] * (b.dim() - 1)))




class MultiMarginalFlowMatcher:
    """Multi-Marginal Flow Matcher.

    Structure inspired by Alex Tong's CFM implementation:
    https://github.com/atong01/conditional-flow-matching
    """

    def __init__(self, sigma: float | int | str = 0.0, interpolation: str = "cubic", mix_coeff = None,time_scaler = None,md=False):
        """Initialize the Multi-Marginal Flow Matcher.

        Args:
            sigma (float | int | str): The variance of the probability density path N(mu_t(x), sigma_t(x)).
                If sigma is an int/float, the variance is constant.
                If sigma is a string, the variance is adaptive and depends on the timepoints.
                The string should be in the format "adaptiveX-{M}-{d}" where X is the adaptive method,
                M is the maximum variance and d is the minimum variance.
            interpolation (str): The interpolation method to use for the mean function.
                Can be "linear", "lagrange" or "cubic".
        """
        self.md = md
        self.sigma = sigma
        self.interpolation = interpolation
        self.mix_coeff = mix_coeff
        self.time_scaling = time_scaler
        if self.interpolation not in ["linear", "lagrange", "cubic","pchip", "exact","sbi"]:
            raise ValueError(
                "Interpolation method must be either 'lagrange', 'cubic' or 'linear'."
            )

    def compute_mu_t(self, xs, t):
        """Compute the mean mu_t(x) of the probability density path N(mu_t(x), sigma_t(x)).

        Args:
            xs (torch.Tensor): The data points.
            t (torch.Tensor): The timepoints.
        """
        t = pad_a_like_b(t, xs)
        return self.P(t, 0)

    def compute_sigma_t(self, t, derative=0):
        """Compute the variance sigma_t(x) of the probability density path N(mu_t(x), sigma_t(x)).

        Args:
            t (torch.Tensor): The timepoints.
            derative (int): The derivative order to compute.
        """
        if derative == 0:
            # Description name: "adaptiveX-{M}-{d}"
            if isinstance(self.sigma, float | int):
                return self.sigma
            else:
                if "adaptive1" in self.sigma:
                    # Adaptive1: M * sqrt(t * (1 - t))
                    # Always reaches variance 1 between two timepoints
                    M = float(self.sigma.split("-")[1])
                    d = float(self.sigma.split("-")[2])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    # Normalize each timepoint to be between 0 and 1
                    # t_np = (t_np - lower_idx) / (upper_idx - lower_idx)
                    # std = M * np.sqrt(2)*(t_np * (1 - t_np))
                    std= M * (upper_idx-t_np) * (t_np-lower_idx) 
                    if d > 0:
                        std = np.clip(std, d, None)
                    return torch.tensor(std).to(t.device)
                elif "adaptive2" in self.sigma:
                    # Adaptive2:
                    # phi = lambda x, t0, t1: M * np.sqrt((x - t0) ** 2 * (x - t1) ** 2 / ((t1 - t0) ** 2))
                    M = float(self.sigma.split("-")[1])
                    d = float(self.sigma.split("-")[2])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    # Normalize each timepoint
                    std = M * np.sqrt(
                        (t_np - lower_idx) ** 2
                        * (t_np - upper_idx) ** 2
                        / ((upper_idx - lower_idx) ** 2)
                    )
                    if d > 0:
                        std = np.clip(std, d, None)
                    return torch.tensor(std).to(t.device)
                elif "adaptive3" in self.sigma:
                    # Adaptive3:
                    # M = 1
                    # phi = lambda x, t0, t1: M * (1 - ((2 * (x - t0) / (t1 - t0) - 1) ** 2)) * ((t1 - t0) ** 2)
                    M = float(self.sigma.split("-")[1])
                    d = float(self.sigma.split("-")[2])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    # Normalize each timepoint
                    std = M * np.sqrt(
                        (
                            1
                            - (
                                (2 * (t_np - lower_idx) / (upper_idx - lower_idx) - 1)
                                ** 2
                            )
                        )
                        * ((upper_idx - lower_idx) ** 2)
                    )
                    if d > 0:
                        std = np.clip(std, d, None)
                    return torch.tensor(std).to(t.device)
                elif "adaptive4" in self.sigma:
                    # M = 16
                    M = float(self.sigma.split("-")[1])
                    d = float(self.sigma.split("-")[2])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    std = (
                        M
                        * (t_np - lower_idx) ** 2
                        * (upper_idx - t_np) ** 2
                        / (upper_idx - lower_idx) ** 2
                    )
                    if d > 0:
                        std = np.clip(std, d, None)
                    return torch.tensor(std).to(t.device)
                elif "adaptive5" in self.sigma:
                    # M = 16
                    M = float(self.sigma.split("-")[1])
                    d = float(self.sigma.split("-")[2])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    std = (
                        M
                        * (t_np - lower_idx)
                        * (upper_idx - t_np) 
                        / (upper_idx - lower_idx) 
                    )
                    if d > 0:
                        std = np.clip(std, d, None)
                    return torch.tensor(std).to(t.device)

        elif derative == 1:
            if isinstance(self.sigma, float | int):
                return torch.tensor(0.0).to(t.device)
            else:
                if "adaptive1" in self.sigma:
                    M = float(self.sigma.split("-")[1])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    # t_np = (t_np - lower_idx) / (upper_idx - lower_idx)
                    # sigma_prime = M * (1 - 2 * t_np) / (2 * np.sqrt(t_np * (1 - t_np)))
                    sigma_prime = M * (upper_idx - 2*t_np + lower_idx)
                    return torch.tensor(sigma_prime).to(t.device)
                elif "adaptive2" in self.sigma:
                    M = float(self.sigma.split("-")[1])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    sigma_prime = (
                        M
                        * (
                            (2 * (t_np - lower_idx) ** 2 * (t_np - upper_idx))
                            / (upper_idx - lower_idx) ** 2
                            + (2 * (t_np - lower_idx) * (t_np - upper_idx) ** 2)
                            / (upper_idx - lower_idx) ** 2
                        )
                        / (
                            2
                            * np.sqrt(
                                ((t_np - lower_idx) ** 2 * (t_np - upper_idx) ** 2)
                                / (upper_idx - lower_idx) ** 2
                            )
                        )
                    )
                    # At position where t == lower_idx/upper_idx, set sigma_prime to +-M
                    sigma_prime[np.isclose(t_np, lower_idx)] = M
                    sigma_prime[np.isclose(t_np, upper_idx)] = -M
                    return torch.tensor(sigma_prime).to(t.device)
                elif "adaptive3" in self.sigma:
                    M = float(self.sigma.split("-")[1])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    sigma_prime = (
                        M
                        * 4
                        * (lower_idx + upper_idx - 2 * t_np)
                        / ((lower_idx - t_np) * (t_np - upper_idx))
                    )
                    # At position where t == lower_idx/upper_idx, set sigma_prime to +-M
                    sigma_prime[np.isclose(t_np, lower_idx)] = M
                    sigma_prime[np.isclose(t_np, upper_idx)] = -M
                    return torch.tensor(sigma_prime).to(t.device)
                elif "adaptive4" in self.sigma:
                    M = float(self.sigma.split("-")[1])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    sigma_prime = (
                        M
                        * (
                            2
                            * (t_np - lower_idx)
                            * (upper_idx - t_np)
                            * (lower_idx + upper_idx - 2 * t_np)
                        )
                        / (lower_idx - upper_idx) ** 2
                    )
                    return torch.tensor(sigma_prime).to(t.device)
                elif "adaptive5" in self.sigma:
                    M = float(self.sigma.split("-")[1])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    sigma_prime = (
                        M * (lower_idx + upper_idx - 2 * t_np) / (lower_idx - upper_idx) ** 2
                    )
                    return torch.tensor(sigma_prime).to(t.device)
        else:
            raise ValueError("Only derivatives 0 and 1 are supported")

    def sample_xt(self, xs, t, epsilon):
        """Sample from the conditional distribution N(mu_t(x), sigma_t(x)).

        Args:
            xs (torch.Tensor): The data points.
            t (torch.Tensor): The timepoints.
            epsilon (torch.Tensor): The noise to add to the sample.

        Returns:
            torch.Tensor: The sampled location x_t.
        """
        mu_t = self.compute_mu_t(xs, t)
        sigma_t = self.compute_sigma_t(t, 0)
        sigma_t = pad_a_like_b(sigma_t, xs)
        return mu_t + sigma_t * epsilon, sigma_t

    def compute_conditional_flow(self, xs, t, xt):
        """Compute the conditional flow u_t(x | z).

        Args:
            xs (torch.Tensor): The data points.
            t (torch.Tensor): The timepoints.
            xt (torch.Tensor): The sampled location x_t.

        Returns:
            torch.Tensor: The conditional flow u_t(x | z).
        """
        if isinstance(self.sigma, float | int):
            # The derivatives of a constant variance is zero, hence we only return the derivative of the
            # mean function
            t = pad_a_like_b(t, xs)
            return self.P(t, 1)
        else:
            esp= 1e-3
            # We need to evaluate the full formula
            # score= - (xt- mu_t) / sigma_t^2
            score = (self.P(t, 0)-xt) / esp + self.compute_sigma_t(t, 0).reshape(-1, 1, 1)**2
            
            drift = (self.compute_sigma_t(t, 1) / self.compute_sigma_t(t, 0)).reshape(
                -1, 1, 1
            ) * (xt - self.P(t, 0)) + self.P(t, 1)
            
            return drift, score
        
    def sample_location_and_conditional_flow(self, xs, timepoints, t=None):
        """Sample a location x_t from probability density path p_t(x) and conditional flow u_t(x | z) at time t.

        Args:
            xs (torch.Tensor): The data points.
            timepoints (torch.Tensor): The timepoints.
            t (torch.Tensor): The timepoints to sample at. If None, samples are drawn from a uniform distribution.
        """
        if self.interpolation == "basic":
            if xs.shape[1] != 2:
                raise ValueError("Basic interpolation requires exactly 2 data points")

        if t is None:
            # TODO: Make sure the values in t are not too close to the self.timepoints
            t = torch.zeros(xs.shape[0]).type_as(xs)
            for b in range(xs.shape[0]):
                # Use torch operations instead of numpy
                valid_mask = ~torch.isnan(timepoints[b])
                valid_times = timepoints[b][valid_mask]
                if len(valid_times) > 1:
                    min_time = valid_times.min()
                    max_time = valid_times.max()
                    # Sample within [min_time, max_time] range
                    t[b] = torch.rand(1).type_as(xs) * (max_time - min_time) + min_time
                else:
                    # Fallback to original behavior if only one timepoint
                    t[b] = torch.rand(1).type_as(xs)
            # because derivatives will be unstable
        assert len(t) == xs.shape[0], "t has to have batch size dimension"

        # Convert timepoints to numpy array (check if it's on cuda first)
        if isinstance(timepoints, torch.Tensor):
            timepoints = timepoints.cpu().detach().numpy()
        self.timepoints = timepoints
        eps = torch.randn_like(xs)[:, 0].unsqueeze(1)
        
       
        if self.interpolation == "cubic":
            self.P = CubicSplineInterpolation(xs, self.timepoints)
        elif self.interpolation == "lagrange":
            self.P = LagrangeInterpolation(xs, self.timepoints)
        elif self.interpolation == "linear":
            self.P = LinearInterpolation(xs, self.timepoints,self.mix_coeff)
        elif self.interpolation == "pchip":
            self.P = PCHIPInterpolation(xs, self.timepoints)
        elif self.interpolation == "exact":
            self.P = ExactVelocityInterpolation(xs=xs, t_anchor=self.timepoints,time_scaler=self.time_scaling)
        elif self.interpolation == "sbi":
            self.P = SmoothBlendedInterpolation(xs, self.timepoints)
        
        # ipdb.set_trace()   

        xt,sigma_t = self.sample_xt(xs, t, eps)
        ut,score = self.compute_conditional_flow(xs, t, xt)

        return t, xt, ut, xs, self.P,sigma_t,score,eps


class LagrangePolynomial:
    """Construct Lagrange Polynomial and its derivative.

    This class is in a batched fashion in class LagrangeInterpolation.
    """

    def __init__(self, t_anchor, x):
        """Initialize the Lagrange Polynomial."""
        self.x = x
        self.t_anchor = t_anchor
        self.n = len(t_anchor)
        self.dtype = x.dtype
        self.device = "cpu"

    def _basis_derivative(self, j, t_val, t):
        n = len(t)
        return sum(1 / (t_val - t[i]) for i in range(n) if i != j)

    def _basis_polynomial(self, j, t_val, t, n):
        L_j = 1
        for i in range(n):
            if j != i:
                L_j *= (t_val - t[i]) / (t[j] - t[i])
        return L_j

    def __call__(self, t_query, derivative):
        """Evaluate the x-derivative of the Lagrange Polynomial at t_query."""
        result = torch.zeros((1, self.x.shape[-1]), device=self.device)

        if derivative == 0:
            for i in range(self.n):
                result += self.x[i] * self._basis_polynomial(
                    i, t_query, self.t_anchor, self.n
                )
        elif derivative == 1:
            for i in range(self.n):
                result += (
                    self.x[i]
                    * self._basis_polynomial(i, t_query, self.t_anchor, self.n)
                    * self._basis_derivative(i, t_query, self.t_anchor)
                )
        else:
            raise ValueError("Only derivatives 0 and 1 are supported")
        return result

class LagrangeInterpolation:
    """Construct Lagrange Interpolation for a batch of data."""

    def __init__(self, xs, t_anchor):
        self.device = xs.device
        self.x_dim = xs.dim()
        self.dtype = xs.dtype

        self.P = self.get_lagrange_interpolation(xs, t_anchor)

    def get_lagrange_interpolation(self, xs, t_anchor):
        """Create Lagrange interpolation functions for each batch."""
        if isinstance(t_anchor, torch.Tensor):
            t_anchor = t_anchor.cpu().numpy()
        if isinstance(xs, torch.Tensor):
            xs = xs.cpu().detach().numpy()

        # Return list of polynomials for each batch
        return [
            LagrangePolynomial(
                t_anchor[b][~np.isnan(xs[b]).any(axis=1)],
                xs[b][~np.isnan(xs[b]).any(axis=1)],
            )
            for b in range(xs.shape[0])
        ]

    def __call__(self, t_query, derivative):
        """Evaluate the x-derivative of the Lagrange Interpolation at t_query."""
        return (
            torch.from_numpy(self.eval_lagrange_interpolation(t_query, derivative))
            .float()
            .to(self.device)
        )

    def eval_lagrange_interpolation(self, t_query, derivative):
        """Helper function for __call__ taking care of batching & data converting."""
        if isinstance(t_query, torch.Tensor):
            t_query = t_query.cpu().numpy()
        if isinstance(t_query, int | float):
            t_query = [np.array(t_query)]
        if len(t_query) == 1:
            t_query = np.repeat(t_query, len(self.P))
        if np.ndim(t_query) != self.x_dim:
            t_query = t_query.reshape(-1, *([1] * (self.x_dim - 1)))

        results = np.concatenate(
            [lp(t_query[k], derivative) for k, lp in enumerate(self.P)], axis=0
        )
        results = results.reshape(t_query.shape[0], 1, -1)
        return results

class CubicSplineInterpolation:
    """Construct Cubic Spline Interpolation for a batch of data."""

    def __init__(self, xs, t_anchor):
        self.splines = self.get_cubic_spline_interpolation(xs, t_anchor)
        self.device = xs.device
        self.x_dim = xs.dim()

    def __call__(self, t_query, derivative):
        """Evaluate the x-derivative of the Cubic Spline Interpolation at t_query."""
        return (
            torch.from_numpy(self.eval_cubic_spline_interpolation(t_query, derivative))
            .float()
            .to(self.device)
        )

    def get_cubic_spline_interpolation(self, xs, t_anchor):
        """Create cubic spline interpolation functions for each batch."""
        if isinstance(t_anchor, torch.Tensor):
            t_anchor = t_anchor.cpu().numpy()
        if isinstance(xs, torch.Tensor):
            xs = xs.cpu().detach().numpy()
        # Create cubic spline interpolation functions for each batch
       
        return [
            interpolate.CubicSpline(
                t_anchor[b][~np.isnan(xs[b]).any(axis=1)],
                xs[b][~np.isnan(xs[b]).any(axis=1)],
            )
            for b in range(xs.shape[0])
        ]

    def eval_cubic_spline_interpolation(self, t_query, derivative=0):
        """Evaluate the x-derivative of the Cubic Spline Interpolation at t_query."""
        if isinstance(t_query, torch.Tensor):
            t_query = t_query.cpu().numpy()
        if isinstance(t_query, int | float):
            t_query = [np.array(t_query)]
        if len(t_query) == 1:
            t_query = np.repeat(t_query, len(self.splines))
        if np.ndim(t_query) != self.x_dim:
            t_query = t_query.reshape(-1, *([1] * (self.x_dim - 1)))

        return np.concatenate(
            [
                spline(t_query[k], nu=derivative)
                for k, spline in enumerate(self.splines)
            ],
            axis=0,
        )

class LinearInterpolation:
    """Construct Linear Interpolation for a batch of data."""

    def __init__(self, xs, t_anchor,mix_coeff):
        self.linear_interpolations = self.get_linear_interpolation(xs, t_anchor)
        self.device = getattr(xs, "device", None)
        self.x_dim = xs.ndim if isinstance(xs, np.ndarray) else xs.dim()
        self.t_anchor = t_anchor
        self.xs = xs
        self.mix_coeff = mix_coeff

    def __call__(self, t_query, derivative):
        """Evaluate the Linear Interpolation at t_query."""
        return (
            torch.from_numpy(self.eval_linear_interpolation(t_query, derivative))
            .float()
            .to(self.device)
        )

    def get_linear_interpolation(self, xs, t_anchor):
        """Create linear interpolation functions for each batch."""
        if isinstance(t_anchor, torch.Tensor):
            t_anchor = t_anchor.cpu().numpy()
        if isinstance(xs, torch.Tensor):
            xs = xs.cpu().detach().numpy()

        return [
            interpolate.interp1d(
                t_anchor[b][~np.isnan(xs[b]).any(axis=1)],
                xs[b][~np.isnan(xs[b]).any(axis=1)],
                axis=0,
                fill_value="extrapolate",
            )
            for b in range(xs.shape[0])
        ]

    def eval_linear_interpolation(self, t_query, derivative=0):
        """Evaluate the Linear Interpolation or its derivative at t_query."""
        if isinstance(t_query, torch.Tensor):
            t_query = t_query.cpu().numpy()
        if isinstance(t_query, int | float):
            t_query = [np.array(t_query)]
        if len(t_query) == 1:
            t_query = np.repeat(t_query, len(self.splines))
        if np.ndim(t_query) != self.x_dim:
            t_query = t_query.reshape(-1, *([1] * (self.x_dim - 1)))

        if derivative == 0:
            results = np.concatenate(
                [
                    interp(t_query[k])
                    for k, interp in enumerate(self.linear_interpolations)
                ],
                axis=0,
            )
        elif derivative == 1:
            results = np.concatenate(
                [
                    self.compute_derivative(interp,k , t_query[k])
                    for k, interp in enumerate(self.linear_interpolations)
                ],
                axis=0,
            )
        else:
            raise ValueError(
                "Derivative order must be 0 or 1 for linear interpolation."
            )

        return results

    def compute_derivative(self, interp, k,t):
        """Compute the derivative of the linear interpolation."""
        eps = 1e-1
        return (interp(t + eps) - interp(t - eps)) / (2 * eps)

    # def compute_derivative(self,interp,k,t):
    #     """Compute the derivative of the linear interpolation.
        
    #     For linear interpolation, the derivative is the slope between the two points
    #     that bracket the query time t.
    #     """
    #     t_np = t if isinstance(t, np.ndarray) else np.array([t])
        
        
    #     time_seq = self.t_anchor[k]
            
    #     mask = time_seq <= t_np
        
    #     if not mask.any():
    #         interval_idx = 0
    #     else:
    #         interval_idx = np.max(np.where(mask))
    #         if interval_idx >= len(time_seq) :
    #             interval_idx = len(time_seq) - 1
       
    #     if self.mix_coeff is not None:
    #         # Check if we're at the last segment
    #         if interval_idx + 2 < len(self.xs[k]):
    #             # Use the mix_coeff to mix the derivative between the two intervals.
    #             derivative = ((1-self.mix_coeff) * (self.xs[k, interval_idx + 1] - self.xs[k, interval_idx]) + 
    #                       self.mix_coeff * (self.xs[k, interval_idx+2] - self.xs[k, interval_idx + 1]))   
    #         else:
    #             # For the last segment, use only the derivative of the current segment
    #             derivative = self.xs[k, interval_idx + 1] - self.xs[k, interval_idx]
    #     else:
    #         derivative = self.xs[k, interval_idx+ 1] - self.xs[k, interval_idx]

    #     return derivative.view(1, -1).cpu().detach().numpy()

class ExactVelocityInterpolation:
    """Construct interpolation with exact velocity blending for a batch of data."""

    def __init__(self, xs, t_anchor, time_scaler=None):
        """
        Initialize the interpolation.
        
        Args:
            xs: Tensor or array of shape [batch_size, num_points, feature_dim]
            t_anchor: Times corresponding to each point in xs
            time_scaled: Whether to scale velocity by time differences
        """
        self.device = getattr(xs, "device", None)
        self.x_dim = xs.ndim if isinstance(xs, np.ndarray) else xs.dim()
        self.time_scaler = time_scaler
        self.use_median=False
        # Store data as torch tensors for internal computations
        if isinstance(t_anchor, torch.Tensor):
            self.t_anchor = t_anchor
        else:
            self.t_anchor = torch.tensor(t_anchor, device=self.device)
            
        if isinstance(xs, torch.Tensor):
            self.xs = xs
        else:
            self.xs = torch.tensor(xs, device=self.device)
            
        self.feature_dim = self.xs.shape[2]  # Feature dimension
        
        # Precompute waypoint velocities for all batches
        self.way_velocities = self._precompute_velocities()

    def __call__(self, t_query, derivative):
        """Evaluate the interpolation at t_query."""
        return self.eval_interpolation(t_query, derivative)
    
    def _precompute_velocities(self):
      
        batch_size = self.xs.shape[0]
        num_points = self.xs.shape[1]
        feature_dim = self.xs.shape[2]
        
        # Initialize velocities tensor
        way_velocities = torch.zeros((batch_size, num_points, feature_dim), device=self.device)
        
        for b in range(batch_size):
            # Calculate all time differences for this batch
            time_diffs = self.t_anchor[b, 1:] - self.t_anchor[b, :-1]
            for i in range(num_points - 1):
                dt = self.t_anchor[b, i+1] - self.t_anchor[b, i]
                
                if self.time_scaler is not None:
                    if self.use_median:
                        # Use 50th percentile (median) as reference time scale
                        median_dt = torch.quantile(time_diffs, 0.5)
                        bounded_dt = self.time_scaler(dt/ median_dt)
                    else:
                        bounded_dt = self.time_scaler(dt)
                   
                    way_velocities[b, i] = (self.xs[b, i+1] - self.xs[b, i]) / bounded_dt
                else:
                    way_velocities[b, i] = (self.xs[b, i+1] - self.xs[b, i])/dt
            
            # Set the last waypoint velocity to be the same as the previous segment
            way_velocities[b, -1] = way_velocities[b, -2]
            
        return way_velocities

    def eval_interpolation(self, t_query, derivative=0):
        """Evaluate the interpolation or its derivative at t_query."""
        # Convert t_query to torch tensor
        if not isinstance(t_query, torch.Tensor):
            t_query = torch.tensor(t_query, device=self.device)
        if isinstance(t_query, (int, float)):
            t_query = torch.tensor([[t_query]], device=self.device)  # Add batch and time dimensions
        
        if len(t_query.shape) == 1:
            t_query = t_query.reshape(-1, 1)

        batch_size = t_query.shape[0]
        time_steps = t_query.shape[1]
        
        # Prepare output tensor with correct dimensions
        results = torch.zeros((batch_size, time_steps, self.feature_dim), device=self.device)
        
        # Process each batch separately
        for b in range(batch_size):
            for t in range(time_steps):
                t_value = t_query[b, t]
                
                # Find the appropriate segment
                time_seq = self.t_anchor[b]
                mask = time_seq <= t_value
                
                if not torch.any(mask):
                    segment_idx = 0
                else:
                    segment_idx = torch.max(torch.where(mask)[0])
                    if segment_idx >= len(time_seq) - 1:
                        segment_idx = len(time_seq) - 2
                
                t_i = time_seq[segment_idx]
                t_i_plus_1 = time_seq[segment_idx + 1]
                x_i = self.xs[b, segment_idx]
                v_i = self.way_velocities[b, segment_idx]
                v_i_plus_1 = self.way_velocities[b, segment_idx + 1]
                
                if derivative == 0:
                    # Compute position using exact equation
                    results[b, t] = self.compute_position(t_value, t_i, t_i_plus_1, x_i, v_i, v_i_plus_1)
                elif derivative == 1:
                    # Compute velocity
                    results[b, t] = self.compute_velocity(t_value, t_i, t_i_plus_1, v_i, v_i_plus_1)
                else:
                    raise ValueError("Derivative order must be 0 or 1 for this interpolation.")
        
        return results

    def compute_position(self, t, t_i, t_i_plus_1, x_i, v_i, v_i_plus_1):
        """
        Compute position at time t using exact derived equation with correction term.
        
        Position function (mean trajectory):
        μ_t = x_i + v_i·[(t-t_i)/(t_{i+1}-t_i)]·[t_{i+1}-(t+t_i)/2]
              + v_{i+1}·[(t-t_i)²/(2(t_{i+1}-t_i))]
              + [(v_i-v_{i+1})/2]·(t-t_i)
        """
        # Handle edge case of zero duration segment
        if t_i_plus_1 - t_i <= 0:
            return x_i
            
        # Term 1: Starting position
        term1 = x_i
        
        # Term 2: First velocity component
        term2 = v_i * ((t-t_i)/(t_i_plus_1-t_i)) * (t_i_plus_1 - (t+t_i)/2)
        
        # Term 3: Second velocity component
        term3 = v_i_plus_1 * ((t-t_i)**2/(2*(t_i_plus_1-t_i)))
        
        # Term 4: Correction term
        term4 = ((v_i-v_i_plus_1)/2) * (t-t_i)
        
        return term1 + term2 + term3 + term4

    # def compute_velocity(self, t, t_i, t_i_plus_1, v_i, v_i_plus_1):
    #     """
    #     Compute velocity at time t using exact derived equation with correction term.
        
    #     Velocity function (derivative):
    #     μ'_t = v_t + (v_i-v_{i+1})/2 = α_t·v_i + (1-α_t)·v_{i+1} + (v_i-v_{i+1})/2
    #     """
    #     # Handle edge case of zero duration segment
    #     if t_i_plus_1 - t_i <= 0:
    #         return v_i
            
    #     # Calculate alpha_t (time-dependent blending coefficient)
    #     alpha_t = (t_i_plus_1 - t) / (t_i_plus_1 - t_i)
        
    #     # Blended velocity
    #     v_t = alpha_t * v_i + (1 - alpha_t) * v_i_plus_1
        
    #     # Add correction term
    #     return v_t + (v_i - v_i_plus_1) / 2

    def compute_velocity(self, t, t_i, t_i_plus_1, v_i, v_i_plus_1):
        """
        Compute velocity at time t using exact derived equation with adaptive correction term.
        
        Velocity function (derivative):
        μ'_t = v_t + (v_i-v_{i+1})/2 = α_t·v_i + (1-α_t)·v_{i+1} + (v_i-v_{i+1})/2
        """
        # Handle edge case of zero duration segment
        dt = t_i_plus_1 - t_i
        if dt < 1e-6:
            return v_i
        
        # For segment boundary detection, use absolute time-scale
        boundary_epsilon = 1e-8
        
        # At exact boundary points, use pure waypoint velocities
        if abs(t - t_i) < boundary_epsilon:
            return v_i
        if abs(t - t_i_plus_1) < boundary_epsilon:
            return v_i_plus_1
        
        # Calculate normalized position in segment
        alpha_t = (t_i_plus_1 - t) / dt
        alpha_t = torch.clip(alpha_t, 0.0, 1.0)  # Ensure within [0,1]
        
        # Base blended velocity
        v_t = alpha_t * v_i + (1 - alpha_t) * v_i_plus_1
        
        # Correction term
        correction = (v_i - v_i_plus_1) / 2
        
        # Create smoother transition near boundaries:
        # - Use a less aggressive tapering function
        # - Apply more correction overall
        boundary_distance = min(alpha_t, 1-alpha_t)  # Distance to nearest boundary [0,0.5]
        taper_power = 0.5 # Lower power = less aggressive tapering (was effectively 2.0)
        correction_strength = 1.5  # Higher = more correction overall
        
        # Compute weight that falls off more gradually
        correction_weight = correction_strength * (2 * boundary_distance)**taper_power
        
        # Apply correction with improved weighting
        return v_t + correction_weight * correction

class SmoothBlendedInterpolation:
    """Construct interpolation with smooth blended velocities for a batch of data."""

    def __init__(self, xs, t_anchor, transition_width=0.5, scale_factor=0.05):
        """
        Initialize the interpolation.
        
        Args:
            xs: Tensor or array of shape [batch_size, num_points, feature_dim]
            t_anchor: Times corresponding to each point in xs
            transition_width: Width of the transition zone (0.0 to 1.0)
            scale_factor: Scale factor for the correction term (default 1.0)
        """
        self.linear_interpolations = self.get_linear_interpolation(xs, t_anchor)
        self.device = getattr(xs, "device", None)
        self.x_dim = xs.ndim if isinstance(xs, np.ndarray) else xs.dim()
        
        # Store data as numpy arrays for internal computations
        if isinstance(t_anchor, torch.Tensor):
            self.t_anchor = t_anchor.cpu().numpy()
        else:
            self.t_anchor = t_anchor
            
        if isinstance(xs, torch.Tensor):
            self.xs = xs.cpu().detach().numpy()
        else:
            self.xs = xs
            
        self.transition_width = transition_width
        self.scale_factor = scale_factor
        self.feature_dim = self.xs.shape[2]  # Feature dimension

    def __call__(self, t_query, derivative):
        """Evaluate the interpolation at t_query."""
        return (
            torch.from_numpy(self.eval_interpolation(t_query, derivative))
            .float()
            .to(self.device)
        )

    def get_linear_interpolation(self, xs, t_anchor):
        """Create linear interpolation functions for each batch."""
        if isinstance(t_anchor, torch.Tensor):
            t_anchor = t_anchor.cpu().numpy()
        if isinstance(xs, torch.Tensor):
            xs = xs.cpu().detach().numpy()

        return [
            interpolate.interp1d(
                t_anchor[b][~np.isnan(xs[b]).any(axis=1)],
                xs[b][~np.isnan(xs[b]).any(axis=1)],
                axis=0,
                fill_value="extrapolate",
            )
            for b in range(xs.shape[0])
        ]

    def smoothstep(self, x):
        """Smooth step function for transition blending."""
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3 - 2 * x)

    def eval_interpolation(self, t_query, derivative=0):
        """Evaluate the interpolation or its derivative at t_query."""
        # Convert t_query to numpy array
        if isinstance(t_query, torch.Tensor):
            t_query = t_query.cpu().numpy()
        if isinstance(t_query, (int, float)):
            t_query = np.array([[t_query]])  # Add batch and time dimensions
        
        if len(t_query.shape) == 1:
            t_query = t_query.reshape(-1, 1)
        
        batch_size = t_query.shape[0]
        time_steps = t_query.shape[1]
        
        # Prepare output array with correct dimensions
        results = np.zeros((batch_size, time_steps, self.feature_dim))
        
        # Process each batch separately
        for b in range(batch_size):
            for t in range(time_steps):
                t_value = t_query[b, t]
                
                # Find the appropriate segment
                time_seq = self.t_anchor[b]
                mask = time_seq <= t_value
                
                if not mask.any():
                    segment_idx = 0
                else:
                    segment_idx = np.max(np.where(mask)[0])
                    if segment_idx >= len(time_seq) - 1:
                        segment_idx = len(time_seq) - 2
                
                t_i = time_seq[segment_idx]
                t_i_plus_1 = time_seq[segment_idx + 1]
                x_i = self.xs[b, segment_idx]
                v_i = self.way_velocities[b, segment_idx]
                v_i_plus_1 = self.way_velocities[b, segment_idx + 1]
                
                if derivative == 0:
                    # Compute mean function (position)
                    results[b, t] = self.compute_mean_at_t(b, t_value)
                elif derivative == 1:
                    # Compute velocity
                    results[b, t] = self.compute_blended_velocity(b, t_value)
                else:
                    raise ValueError("Derivative order must be 0 or 1 for this interpolation.")
        
        return results

    def compute_mean_at_t(self, batch_idx, t):
        """Compute the mean function μ_t that generates the desired velocity field."""
        # Get the interpolation function for this batch
        interp = self.linear_interpolations[batch_idx]
        
        # Get the linear interpolation value (basic mean)
        linear_mean = interp(t)
        
        # Find which segment the query time falls into
        time_seq = self.t_anchor[batch_idx]
        mask = time_seq <= t
        
        if not mask.any():
            interval_idx = 0
        else:
            interval_idx = np.max(np.where(mask)[0])
            if interval_idx >= len(time_seq) - 1:
                # At the last point, just return the linear value
                return linear_mean
        
        # Check if we're in a transition zone
        t_i = time_seq[interval_idx]
        t_i_plus_1 = time_seq[interval_idx + 1]
        
        # Calculate normalized segment progress
        segment_duration = t_i_plus_1 - t_i
        if segment_duration <= 0:
            return linear_mean  # Avoid division by zero
            
        gamma = (t - t_i) / segment_duration
        blend_threshold = 1.0 - self.transition_width
        
        # If we're not in the transition zone, just return linear interpolation
        if gamma < blend_threshold or interval_idx + 2 >= len(time_seq):
            return linear_mean
        
        # We're in the transition zone, calculate correction
        xi = (gamma - blend_threshold) / self.transition_width
        s_xi = self.smoothstep(xi)
        
        # Get the points needed for correction
        x_i = self.xs[batch_idx, interval_idx]
        x_i_plus_1 = self.xs[batch_idx, interval_idx + 1]
        x_i_plus_2 = self.xs[batch_idx, interval_idx + 2]
        
        # Calculate h(t) term
        h_t = (s_xi * s_xi / 2) * segment_duration * self.scale_factor
        
        # Calculate curvature correction
        curvature = x_i_plus_2 - 2 * x_i_plus_1 + x_i
        
        # Apply correction to linear mean
        corrected_mean = linear_mean + h_t * curvature
        
        return corrected_mean

    def compute_blended_velocity(self, batch_idx, t):
        """Compute the smooth blended velocity at time t."""
        # Find which segment the query time falls into
        time_seq = self.t_anchor[batch_idx]
        mask = time_seq <= t
        
        if not mask.any():
            interval_idx = 0
        else:
            interval_idx = np.max(np.where(mask)[0])
            if interval_idx >= len(time_seq) - 1:
                # Last interval, use the velocity of the last segment
                return self.compute_segment_velocity(batch_idx, interval_idx - 1)
        
        # Calculate velocities for current and (if available) next segment
        v_current = self.compute_segment_velocity(batch_idx, interval_idx)
        
        # Check if we have a next segment
        if interval_idx + 2 < len(time_seq):
            v_next = self.compute_segment_velocity(batch_idx, interval_idx + 1)
            
            # Calculate blending based on position in segment
            t_i = time_seq[interval_idx]
            t_i_plus_1 = time_seq[interval_idx + 1]
            
            segment_duration = t_i_plus_1 - t_i
            if segment_duration <= 0:
                return v_current  # Avoid division by zero
                
            gamma = (t - t_i) / segment_duration
            blend_threshold = 1.0 - self.transition_width
            
            if gamma < blend_threshold:
                # Not in transition zone, use current velocity
                return v_current
            else:
                # In transition zone, blend velocities
                xi = (gamma - blend_threshold) / self.transition_width
                blend_factor = self.smoothstep(xi)
                return (1 - blend_factor) * v_current + blend_factor * v_next
        else:
            # No next segment, use current velocity
            return v_current
    
    def compute_segment_velocity(self, batch_idx, segment_idx):
        """Compute velocity for a specific segment."""
        if segment_idx < 0 or segment_idx >= len(self.t_anchor[batch_idx]) - 1:
            # Handle edge cases
            if segment_idx < 0:
                segment_idx = 0
            else:
                segment_idx = len(self.t_anchor[batch_idx]) - 2
        
        # Get times and positions
        t_i = self.t_anchor[batch_idx][segment_idx]
        t_i_plus_1 = self.t_anchor[batch_idx][segment_idx + 1]
        
        x_i = self.xs[batch_idx, segment_idx]
        x_i_plus_1 = self.xs[batch_idx, segment_idx + 1]
        
        # Calculate segment velocity
        segment_duration = t_i_plus_1 - t_i
        if segment_duration <= 0:
            # Handle zero duration 
            return np.zeros_like(x_i)
            
        # velocity = (x_i_plus_1 - x_i) / segment_duration
        velocity = (x_i_plus_1 - x_i) 
        return velocity

class PCHIPInterpolation:
    """Optimized PCHIP Interpolation for a batch of data."""
    
    def __init__(self, xs, t_anchor):
        """Initialize optimized PCHIP interpolation."""
        self.device = xs.device
        self.x_dim = xs.dim()
        
        # Cache data and pre-build lookup tables
        self.setup_interpolation(xs, t_anchor)
        
    def setup_interpolation(self, xs, t_anchor):
        """Prepare optimized data structures for PCHIP interpolation."""
        if isinstance(t_anchor, torch.Tensor):
            t_anchor = t_anchor.cpu().numpy()
        if isinstance(xs, torch.Tensor):
            xs = xs.cpu().detach().numpy()
        
        self.batch_size = xs.shape[0]
        self.t_data = []
        self.x_data = []
        self.interpolators = []
        
        # Process each batch element
        for b in range(self.batch_size):
            valid_mask = ~np.isnan(xs[b]).any(axis=1)
            t_valid = t_anchor[b][valid_mask]
            x_valid = xs[b][valid_mask]
            
            self.t_data.append(t_valid)
            self.x_data.append(x_valid)
            
            # Create interpolator only if we have enough points
            if len(t_valid) >= 2:
                # Create one interpolator per batch instead of per dimension
                # Store coefficients for fast lookup
                interp = PchipInterpolator(t_valid, x_valid)
                self.interpolators.append(interp)
            else:
                self.interpolators.append(None)
    
    def __call__(self, t_query, derivative):
        """Optimized evaluation of PCHIP interpolation."""
        return torch.from_numpy(self.eval_pchip(t_query, derivative)).float().to(self.device)
    
    def eval_pchip(self, t_query, derivative=0):
        """Fast PCHIP evaluation using vectorized operations."""
        if isinstance(t_query, torch.Tensor):
            t_query = t_query.cpu().numpy()
        if isinstance(t_query, (int, float)):
            t_query = [np.array(t_query)]
        if len(t_query) == 1:
            t_query = np.repeat(t_query, self.batch_size)
        if np.ndim(t_query) != self.x_dim:
            t_query = t_query.reshape(-1, *([1] * (self.x_dim - 1)))

        results = []
        
        for b in range(self.batch_size):
            if self.interpolators[b] is None:
                # Handle case with insufficient data points
                dim = self.x_data[b].shape[1] if len(self.x_data[b]) > 0 else 1
                if len(self.x_data[b]) > 0:
                    # Return constant value
                    result = np.repeat(self.x_data[b][0:1], len(t_query[b]), axis=0)
                    if derivative != 0:
                        # Zero derivative for constant function
                        result = np.zeros_like(result)
                else:
                    # No data at all
                    result = np.zeros((len(t_query[b]), dim))
            else:
                # Vectorized evaluation of PCHIP
                result = self.interpolators[b](t_query[b], nu=derivative)
            
            results.append(result)
        
        return np.vstack(results)