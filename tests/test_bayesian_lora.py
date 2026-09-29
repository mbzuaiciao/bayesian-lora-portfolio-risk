"""Tests for Bayesian LoRA layers, parameter count, and KL divergence."""

from __future__ import annotations

import torch
import torch.nn as nn

from bayesian_lora_portfolio_risk.uncertainty.bayesian_lora import (
    BayesianLoRALinear,
    VariationalGaussianParameter,
    count_variational_parameters,
    model_kl_divergence,
    set_posterior_sampling,
)


def test_variational_gaussian_parameter_sampling_and_kl() -> None:
    shape = (4, 8)
    param = VariationalGaussianParameter(shape, init_std=0.05)

    # Initial std should match init_std closely
    assert torch.allclose(param.std, torch.full(shape, 0.05), atol=1e-4)

    # Sampling with torch seed
    torch.manual_seed(42)
    s1 = param.sample()
    torch.manual_seed(42)
    s2 = param.sample()
    assert torch.equal(s1, s2)
    assert s1.shape == shape

    # KL divergence with prior std=1.0 should be positive
    kl = param.kl_divergence(prior_std=1.0)
    assert kl.item() > 0.0


def test_bayesian_lora_linear_forward_and_modes() -> None:
    in_features = 16
    out_features = 8
    rank = 4
    base_linear = nn.Linear(in_features, out_features)
    layer = BayesianLoRALinear(base_linear, rank=rank, alpha=8.0, init_std=0.01, prior_std=1.0)

    # Base parameters should be frozen
    for p in base_linear.parameters():
        assert not p.requires_grad

    # Variational parameters should require grad
    assert layer.param_a.mu.requires_grad
    assert layer.param_a.rho.requires_grad
    assert layer.param_b.mu.requires_grad
    assert layer.param_b.rho.requires_grad

    x = torch.randn(2, in_features)

    # Test sampling mode
    layer.sample_weight = True
    torch.manual_seed(10)
    out_sample1 = layer(x)
    torch.manual_seed(11)
    out_sample2 = layer(x)
    assert not torch.equal(out_sample1, out_sample2)

    # Test deterministic (mean) mode
    layer.sample_weight = False
    out_mean1 = layer(x)
    out_mean2 = layer(x)
    assert torch.equal(out_mean1, out_mean2)


def test_model_kl_and_set_posterior_sampling() -> None:
    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = BayesianLoRALinear(nn.Linear(8, 8), rank=2, alpha=2.0)
            self.l2 = BayesianLoRALinear(nn.Linear(8, 4), rank=2, alpha=2.0)

    model = SimpleModel()
    counts = count_variational_parameters(model)
    # Layer 1: A is (2, 8) -> 16; B is (8, 2) -> 16; total params = 32 for mu, 32 for rho
    # Layer 2: A is (2, 8) -> 16; B is (4, 2) -> 8; total params = 24 for mu, 24 for rho
    # Total mu = 56, rho = 56, total = 112
    assert counts["mu"] == 56
    assert counts["rho"] == 56
    assert counts["total"] == 112

    kl = model_kl_divergence(model)
    assert kl.item() > 0.0

    set_posterior_sampling(model, False)
    assert not model.l1.sample_weight
    assert not model.l2.sample_weight

    set_posterior_sampling(model, True)
    assert model.l1.sample_weight
    assert model.l2.sample_weight
