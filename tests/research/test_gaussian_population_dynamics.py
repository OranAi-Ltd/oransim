"""Mechanism and temporal-separation tests for the experimental V2 study."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from oransim.data.gaussian_population_dynamics import (
    DynamicsConfig,
    GaussianDynamicsFilter,
    difference_noise_variance,
    from_ilr,
    ilr_basis,
    to_ilr,
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_ilr_roundtrip_and_zero_stability():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(from_ilr(to_ilr(values)), values)
    basis = ilr_basis(4)
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(basis.sum(axis=1), 0.0, atol=1e-14)
    assert np.isfinite(to_ilr(np.array([0.0, 0.2, 0.8]))).all()


def test_forecast_does_not_advance_or_change_observation_state():
    config = DynamicsConfig(slope_half_life_days=2.0)
    model = GaussianDynamicsFilter(np.array([0.0, 1.0]), START, 0.1, config)
    model.update(np.array([0.1, 1.2]), START + timedelta(hours=19))
    before = (model.mean.copy(), model.covariance.copy(), model.state_time)
    late = model.forecast(START + timedelta(days=7))
    early = model.forecast(START + timedelta(days=2))
    np.testing.assert_array_equal(model.mean, before[0])
    np.testing.assert_array_equal(model.covariance, before[1])
    assert model.state_time == before[2]
    assert late.predictive_variance > early.predictive_variance
    model.update(np.array([0.2, 1.1]), START + timedelta(days=1))


def test_damped_trend_converges_and_irregular_transition_composes():
    model = GaussianDynamicsFilter(
        np.array([0.0]), START, 0.1, DynamicsConfig(slope_half_life_days=1.0)
    )
    model.mean[1] = 1.0
    one_week = model.forecast(START + timedelta(days=7)).mean[0]
    hundred_days = model.forecast(START + timedelta(days=100)).mean[0]
    assert one_week < hundred_days < 1.45
    direct_mean, direct_cov = model._project(START + timedelta(days=3))
    middle_mean, middle_cov = model._project(START + timedelta(hours=17))
    model.mean, model.covariance = middle_mean, middle_cov
    model.state_time = START + timedelta(hours=17)
    composed_mean, composed_cov = model._project(START + timedelta(days=3))
    np.testing.assert_allclose(direct_mean, composed_mean, rtol=1e-12)
    np.testing.assert_allclose(direct_cov, composed_cov, rtol=1e-12)


def test_permuting_categories_does_not_change_predictions():
    series = np.array([[0.1, 0.2, 0.3, 0.4], [0.15, 0.18, 0.31, 0.36], [0.14, 0.20, 0.30, 0.36]])
    order = np.array([2, 0, 3, 1])
    predictions = []
    for values in (series, series[:, order]):
        latent = np.stack([to_ilr(value) for value in values])
        noise = difference_noise_variance([latent])
        model = GaussianDynamicsFilter(
            latent[0], START, noise, DynamicsConfig(slope_half_life_days=2.0)
        )
        for index in (1, 2):
            model.update(latent[index], START + timedelta(hours=25 * index))
        predictions.append(from_ilr(model.forecast(START + timedelta(days=4)).mean))
    np.testing.assert_allclose(predictions[0][order], predictions[1], atol=1e-12)


def test_invalid_or_duplicate_updates_fail_without_changing_state():
    model = GaussianDynamicsFilter(np.array([0.0]), START, 0.1)
    with pytest.raises(ValueError):
        model.update(np.array([10.0]), START)
    with pytest.raises(ValueError):
        model.forecast(START - timedelta(seconds=1))
    with pytest.raises(ValueError):
        model.update(np.array([np.nan]), START + timedelta(days=1))
    np.testing.assert_array_equal(model.mean, [[0.0], [0.0]])
    assert model.state_time == START
