"""Variational Bayesian LoRA layers for uncertainty-aware adaptation.

This is the project-local, FinBERT-focused version of the implementation developed
in ``bayesian-lora-lab``.  The base projection is frozen and the low-rank matrices
are represented by independent Gaussian variational posteriors.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalGaussianParameter(nn.Module):
    """Mean-field Gaussian q(theta)=N(mu, diag(sigma^2))."""

    def __init__(
        self, shape: tuple[int, ...], *, init_std: float = 0.01, eps: float = 1e-6
    ) -> None:
        super().__init__()
        if init_std <= 0.0:
            raise ValueError("init_std must be positive")
        self.mu = nn.Parameter(torch.zeros(shape))
        rho = math.log(math.expm1(max(init_std - eps, 1e-7)))
        self.rho = nn.Parameter(torch.full(shape, rho))
        self.eps = eps

    @property
    def std(self) -> torch.Tensor:
        return F.softplus(self.rho) + self.eps

    def sample(self) -> torch.Tensor:
        return self.mu + self.std * torch.randn_like(self.mu)

    def kl_divergence(self, *, prior_std: float) -> torch.Tensor:
        if prior_std <= 0.0:
            raise ValueError("prior_std must be positive")
        sigma = self.std
        prior_var = prior_std**2
        return torch.sum(
            math.log(prior_std) - torch.log(sigma)
            + (sigma.square() + self.mu.square()) / (2.0 * prior_var)
            - 0.5
        )


class BayesianLoRALinear(nn.Module):
    """Frozen linear projection plus variational low-rank update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init_std: float = 0.01,
        prior_std: float = 1.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.prior_std = float(prior_std)
        self.sample_weight = True
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.param_a = VariationalGaussianParameter(
            (rank, base_layer.in_features), init_std=init_std
        )
        self.param_b = VariationalGaussianParameter(
            (base_layer.out_features, rank), init_std=init_std
        )
        nn.init.kaiming_uniform_(self.param_a.mu, a=math.sqrt(5))
        nn.init.zeros_(self.param_b.mu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.param_a.sample() if self.sample_weight else self.param_a.mu
        b = self.param_b.sample() if self.sample_weight else self.param_b.mu
        update = F.linear(F.linear(self.dropout(x), a), b) * self.scaling
        return self.base_layer(x) + update

    def kl_divergence(self) -> torch.Tensor:
        return self.param_a.kl_divergence(prior_std=self.prior_std) + self.param_b.kl_divergence(
            prior_std=self.prior_std
        )


def iter_bayesian_lora_layers(model: nn.Module) -> Iterator[BayesianLoRALinear]:
    for module in model.modules():
        if isinstance(module, BayesianLoRALinear):
            yield module


def set_posterior_sampling(model: nn.Module, enabled: bool) -> None:
    """Switch every Bayesian LoRA layer between posterior samples and posterior means."""
    for module in iter_bayesian_lora_layers(model):
        module.sample_weight = enabled


def model_kl_divergence(model: nn.Module) -> torch.Tensor:
    """Return summed KL(q||p) across Bayesian adapter matrices."""
    layers = list(iter_bayesian_lora_layers(model))
    if not layers:
        raise ValueError("model contains no BayesianLoRALinear layers")
    return torch.stack([layer.kl_divergence() for layer in layers]).sum()


def count_variational_parameters(model: nn.Module) -> dict[str, int]:
    mu = 0
    rho = 0
    for layer in iter_bayesian_lora_layers(model):
        mu += layer.param_a.mu.numel() + layer.param_b.mu.numel()
        rho += layer.param_a.rho.numel() + layer.param_b.rho.numel()
    return {"mu": mu, "rho": rho, "total": mu + rho}


def inject_bert_attention_bayesian_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    init_std: float,
    prior_std: float,
) -> list[str]:
    """Replace BERT attention query/value Linear modules with Bayesian LoRA wrappers."""
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (
            name.endswith("attention.self.query") or name.endswith("attention.self.value")
        ):
            replacements.append((name, module))

    for name, module in replacements:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(
            parent,
            child_name,
            BayesianLoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                init_std=init_std,
                prior_std=prior_std,
            ),
        )
    if not replacements:
        raise ValueError("no BERT attention query/value projections were found")
    return [name for name, _ in replacements]
