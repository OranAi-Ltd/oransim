from dataclasses import replace

import numpy as np
import pytest
from oransim.world_model.population_loop import (
    LoopConfig,
    PopulationWorldLoop,
    WorldAction,
    WorldObservation,
)


def model(**kwargs):
    config = replace(LoopConfig(base_arrivals=500), **kwargs)
    return PopulationWorldLoop(
        [0.6, 0.4],
        [[0.7, 0.3], [0.3, 0.7]],
        [[-1.0, 1.0], [-0.5, 1.5]],
        [[0.3, 0.2], [0.4, 0.2]],
        config,
    )


def action(content=0.0, effort=1.0):
    return WorldAction(np.full(2, content), np.zeros(2), effort)


def test_true_state_isolated_from_rollout_and_simulated_feedback():
    world = model()
    before = world.state
    branch, records = world.rollout([action()] * 8, 22)
    assert world.state.day == 0 and branch.state.day == 8
    np.testing.assert_array_equal(world.state.means, before.means)
    assert branch.state.cumulative_exposures == sum(x["exposures"] for x in records)
    with pytest.raises(ValueError):
        world.observe(
            action(), WorldObservation(0, np.array([2, 3]), np.array([1, 1]), "simulated")
        )
    with pytest.raises(RuntimeError):
        world.simulate_step(action(), np.random.default_rng(2))


def test_actual_observation_updates_posterior_and_next_campaign_arrivals():
    world = model(fatigue_gain=0)
    before = world.predict(action())
    world.observe(action(), WorldObservation(0, np.array([500, 500]), np.array([450, 450])))
    after = world.predict(action())
    assert after.expected_exposures > before.expected_exposures
    assert np.all(after.group_probabilities > before.group_probabilities)
    assert world.state.momentum > 0
    assert np.all(world.state.mean_posterior_variance < 0.1)
    with pytest.raises(ValueError):
        world.observe(action(), WorldObservation(0, np.array([1, 2]), np.array([0, 0])))
    assert world.state.day == 1


def test_action_changes_targeting_and_has_delayed_feedback_effect():
    original = model(response_process_variance=0, arrival_process_variance=0)
    targeted = WorldAction(np.zeros(2), np.array([0.0, 2.0]))
    assert (
        original.predict(targeted).exposure_weights[1]
        > original.predict(action()).exposure_weights[1]
    )
    low, _ = original.rollout([action(-2.0)], 17)
    high, _ = original.rollout([action(2.0)], 17)
    assert high.state.momentum > low.state.momentum
    assert high.predict(action()).expected_exposures > low.predict(action()).expected_exposures


def test_conditional_macro_integral_matches_latent_agent_sampling():
    world = model(base_arrivals=20000, response_process_variance=0, arrival_process_variance=0)
    rng = np.random.default_rng(90)
    branch = world.branch(rng)
    prediction, observed, batch = branch.simulate_step(action(), rng)
    assert (
        abs(batch.response.mean() - prediction.exposure_weights @ prediction.group_probabilities)
        < 0.025
    )
    assert len(batch.latent_response) == observed.exposures.sum()
    assert np.std(batch.latent_response) > 0
    assert batch.response.sum() == observed.responses.sum()


def test_zero_exposure_preserves_posterior_information_and_valid_distributions():
    world = model()
    before = world.state
    world.observe(action(effort=0), WorldObservation(0, np.zeros(2, int), np.zeros(2, int)))
    s = world.state
    expected = 0.95**2 * before.mean_posterior_variance + world.config.response_process_variance
    np.testing.assert_allclose(s.mean_posterior_variance, expected)
    np.testing.assert_allclose(s.mixture_weights.sum(1), 1)
    assert np.all(s.heterogeneity_variance >= 0)


def test_population_observation_changes_stock_without_using_exposure_as_stock():
    world = model()
    world.set_population_observation(0, [0.2, 0.8])
    world.observe(action(), WorldObservation(0, np.array([900, 100]), np.array([200, 20])))
    np.testing.assert_allclose(world.state.population_weights, [0.2, 0.8])
    with pytest.raises(ValueError):
        world.set_population_observation(0, [0.6, 0.4])


def test_disabled_feedback_removes_delayed_action_path():
    world = model(
        social_gain=0,
        fatigue_gain=0,
        mixture_gain=0,
        momentum_arrival_gain=0,
        fatigue_selection_gain=0,
        response_process_variance=0,
        arrival_process_variance=0,
    )
    left, _ = world.rollout([action(-2.0)] * 3, 90)
    right, _ = world.rollout([action(2.0)] * 3, 90)
    a, b = left.predict(action()), right.predict(action())
    assert a.expected_exposures == pytest.approx(b.expected_exposures)
    np.testing.assert_allclose(a.group_probabilities, b.group_probabilities)


def test_generative_transition_preserves_bounds_and_component_floor():
    world = model(response_process_variance=0, arrival_process_variance=0)
    world._state.mean_posterior_variance[:] = 0
    world._state.arrival_posterior_variance = 0
    branch, _ = world.rollout([action(3.0, 5.0)] * 80, 81)
    state = branch.state
    c = world.config
    assert np.all((state.fatigue >= 0) & (state.fatigue <= 1))
    assert -1 <= state.momentum <= 1
    assert np.all(state.mixture_weights >= c.mixture_refresh * world.state.mixture_weights - 1e-12)
    bound = np.max(np.abs(world.state.means)) + (abs(c.social_gain) + abs(c.fatigue_gain)) / (
        1 - c.response_retention
    )
    assert np.max(np.abs(state.means)) <= bound
