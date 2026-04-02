from __future__ import annotations

import gpytorch
import numpy as np
import torch


class _SingleOutputDynamicsGP(gpytorch.models.ExactGP):
    """Single-output exact GP used as one channel of the learned interaction model."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )

    def forward(self, inputs: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(inputs)
        covariance = self.covar_module(inputs)
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


class IndependentGPDynamicsModel:
    """Independent exact GPs trained with GPyTorch and queried from PyTorch rollouts."""

    def __init__(self, jitter: float = 1e-6) -> None:
        self.jitter = jitter
        self.models: list[_SingleOutputDynamicsGP] = []
        self.likelihoods: list[gpytorch.likelihoods.GaussianLikelihood] = []
        self.input_dim = 0
        self.output_dim = 0

    def fit(self, train_inputs: np.ndarray, train_targets: np.ndarray, training_steps: int = 150) -> None:
        x = torch.as_tensor(np.asarray(train_inputs, dtype=np.float64), dtype=torch.float64)
        y = torch.as_tensor(np.asarray(train_targets, dtype=np.float64), dtype=torch.float64)
        self.input_dim = x.shape[1]
        self.output_dim = y.shape[1]
        self.models = []
        self.likelihoods = []

        for output_index in range(self.output_dim):
            train_y = y[:, output_index]
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = _SingleOutputDynamicsGP(x, train_y, likelihood)

            # Sensible initialization matters a lot when each GP is trained from a
            # fairly small batch of human-robot interaction data.
            initial_lengthscale = torch.clamp(x.std(dim=0), min=0.05)
            initial_outputscale = max(float(train_y.var().item()), 0.1)
            likelihood.noise = max(initial_outputscale * 0.05, 1e-4)
            model.covar_module.base_kernel.lengthscale = initial_lengthscale
            model.covar_module.outputscale = initial_outputscale

            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for _ in range(training_steps):
                optimizer.zero_grad(set_to_none=True)
                output = model(x)
                loss = -marginal_log_likelihood(output, train_y)
                loss.backward()
                optimizer.step()

            model.eval()
            likelihood.eval()
            self.models.append(model)
            self.likelihoods.append(likelihood)

    def predict_numpy(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.as_tensor(np.asarray(inputs, dtype=np.float64), dtype=torch.float64)
        with torch.no_grad():
            means, variances = self.predict_torch(x)
        return means.cpu().numpy(), variances.cpu().numpy()

    def predict_torch(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.models:
            raise RuntimeError("Dynamics model must be fit before prediction.")

        means = []
        variances = []
        with gpytorch.settings.cholesky_jitter(self.jitter):
            for model in self.models:
                posterior = model(inputs)
                means.append(posterior.mean)
                variances.append(torch.clamp(posterior.variance, min=self.jitter))
        return torch.stack(means, dim=-1), torch.stack(variances, dim=-1)
