"""研究世界模型的条件似然训练、局部敏感性诊断与复现报告。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp, xlogy

from .population_loop import PopulationWorldLoop


def count_nll(n, mean, dispersion):
    if mean == 0:
        return 0.0 if n == 0 else float("inf")
    r = dispersion
    return float(
        -(
            gammaln(n + r)
            - gammaln(r)
            - gammaln(n + 1)
            + r * np.log(r / (r + mean))
            + xlogy(n, mean / (r + mean))
        )
    )


def marginal_count_nll(n, world, action):
    """Adaptive Gaussian quadrature of the NB observation over arrival uncertainty.

    Centering quadrature at the posterior mode resolves narrow count likelihoods
    that ordinary prior-centered Hermite nodes can miss at high exposure counts.
    """
    variance = world.state.arrival_posterior_variance
    central = float(world._arrival_rates(action, 0.0))
    r = world.config.arrival_dispersion
    if variance < 1e-12 or central == 0:
        return count_nll(n, central, r)

    def loss(x):
        return count_nll(n, float(world._arrival_rates(action, x)), r) + 0.5 * x * x / variance

    from scipy.optimize import minimize_scalar

    scale = max(12.0, 10 * np.sqrt(variance))
    solution = minimize_scalar(loss, bounds=(-scale, scale), method="bounded")
    mode = float(solution.x)
    mean = float(world._arrival_rates(action, mode))
    proposal_variance = 1 / (1 / variance + (n + r) * r * mean / (r + mean) ** 2)
    # Importance correction makes the approximation integrate the original
    # prior/likelihood, rather than replacing them by the mode's normal density.
    offsets = mode + np.sqrt(2 * proposal_variance) * world._nodes
    means = world._arrival_rates(action, offsets)
    log_likelihood = (
        gammaln(n + r)
        - gammaln(r)
        - gammaln(n + 1)
        + r * np.log(r / (r + means))
        + xlogy(n, means / (r + means))
    )
    log_ratio = (
        0.5 * np.log(proposal_variance / variance)
        - 0.5 * offsets**2 / variance
        + 0.5 * (offsets - mode) ** 2 / proposal_variance
    )
    return float(-logsumexp(np.log(world._weights) + log_likelihood + log_ratio))


def prediction_scores(prediction, observation, dispersion, world=None, action=None):
    n = np.asarray(observation.exposures)
    k = np.asarray(observation.responses)
    total = int(n.sum())
    p = np.clip(prediction.group_probabilities, 1e-10, 1 - 1e-10)
    response = float(-(k * np.log(p) + (n - k) * np.log1p(-p)).sum() / max(total, 1))
    composition = float(
        -(n * np.log(np.maximum(prediction.exposure_weights, 1e-12))).sum() / max(total, 1)
    )
    arrival = (
        count_nll(total, prediction.expected_exposures, dispersion)
        if world is None
        else marginal_count_nll(total, world, action)
    )
    return {
        "arrival_nll": arrival,
        "response_log_loss": response,
        "composition_log_loss": composition,
        "composite": arrival + response + composition,
        "exposure_absolute_error": abs(total - prediction.expected_exposures),
        "response_count_absolute_error": abs(int(k.sum()) - prediction.expected_responses),
    }


def replay(initial, actions, observations):
    if len(actions) != len(observations) or not actions:
        raise ValueError("aligned nonempty dated training records required")
    world = PopulationWorldLoop.from_dict(initial.to_dict())
    if world._simulation:
        raise ValueError("cannot train a simulated world as real state")
    scores = []
    sensitivity_outputs = []
    mechanism_design = []
    for action, observation in zip(actions, observations, strict=False):
        state = world.state
        prediction = world.predict(action)
        scores.append(
            prediction_scores(
                prediction, observation, world.config.arrival_dispersion, world, action
            )
        )
        w = prediction.exposure_weights
        p = np.clip(prediction.group_probabilities, 1e-8, 1 - 1e-8)
        sensitivity_outputs.extend(
            [
                np.log(max(prediction.expected_exposures, 1e-8)),
                *np.log(p / (1 - p)),
                *np.log(np.maximum(w, 1e-8)),
            ]
        )
        for g, n in enumerate(observation.exposures):
            if n > 0:
                mechanism_design.append([1.0, state.momentum, state.fatigue[g]])
        world.observe(action, observation)
    return world, scores, np.array(sensitivity_outputs), np.array(mechanism_design).reshape(-1, 3)


# Each block minimizes a conditional score. The three blocks share the same
# observed-history replay, preserving feedback timing and avoiding test-driven fits.
BLOCKS = {
    "response": [
        ("response_process_variance", np.log(1e-5), np.log(0.5), "log"),
        ("response_retention", 0.5, 0.999, "linear"),
        ("social_gain", -1.0, 1.0, "linear"),
        ("fatigue_gain", -1.0, 1.0, "linear"),
    ],
    "composition": [
        ("selection_process_variance", np.log(1e-5), np.log(0.5), "log"),
        ("recommendation_gain", -3.0, 3.0, "linear"),
        ("fatigue_selection_gain", -3.0, 3.0, "linear"),
    ],
    "arrival": [
        ("arrival_process_variance", np.log(1e-5), np.log(3.0), "log"),
        ("arrival_dispersion", np.log(2.0), np.log(1e6), "log"),
        ("momentum_arrival_gain", -3.0, 3.0, "linear"),
    ],
}
SCORE_KEYS = {
    "response": "response_log_loss",
    "composition": "composition_log_loss",
    "arrival": "arrival_nll",
}
MECHANISM = {
    "social_gain",
    "fatigue_gain",
    "recommendation_gain",
    "fatigue_selection_gain",
    "momentum_arrival_gain",
}


def fit_world(initial, actions, observations, family="mechanism", max_iterations=35):
    if family not in ["adaptive", "mechanism"]:
        raise ValueError("unknown world family")
    if len(actions) != len(observations):
        raise ValueError("actions and observations must align")
    if len(actions) < 3:
        raise ValueError("at least three training days required")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("positive iteration budget required")
    if initial.state.day != observations[0].day:
        raise ValueError("initial state and training chronology disagree")
    for action, observation in zip(actions, observations, strict=False):
        if action.exposure_effort == 0 and np.sum(observation.exposures) > 0:
            raise ValueError("zero exposure effort conflicts with observed arrivals")
    start = PopulationWorldLoop.from_dict(initial.to_dict())
    # Parameters unsupported by the current observation head stay fixed. In particular,
    # aggregate binary totals do not identify individual Gaussian component moments.
    start.config = replace(
        start.config,
        social_gain=0.0,
        fatigue_gain=0.0,
        recommendation_gain=0.0,
        fatigue_selection_gain=0.0,
        momentum_arrival_gain=0.0,
        mixture_gain=0.0,
    )
    _, _, _, design = replay(start, actions, observations)
    if not len(design):
        raise ValueError("response mechanism training requires observed exposures")
    design_scale = np.maximum(design.std(0), 1e-8)
    design_scale[0] = 1.0
    normalized = design / design_scale
    singular = np.linalg.svd(normalized, compute_uv=False)
    design_rank = int(np.linalg.matrix_rank(normalized))
    report = {
        "family": family,
        "training_days": len(actions),
        "blocks": [],
        "identification": {
            "observed_response_design_rank": design_rank,
            "columns": ["intercept", "previous_momentum", "previous_group_fatigue"],
            "design_singular_values": singular.tolist(),
            "interpretation": "Observed feature excitation and local predictive sensitivity; neither is a causal-identification proof.",
            "fixed_component_structure": "Mixture weights, baseline component moments and demographic priors are supplied; no component recovery claim from aggregate binary outcomes.",
        },
    }
    fitted_coordinates = []
    for block, definitions in BLOCKS.items():
        definitions = [x for x in definitions if family == "mechanism" or x[0] not in MECHANISM]
        if len(actions) < 8:
            definitions = [x for x in definitions if x[0] != "response_retention"]
        if design_rank < 3:
            definitions = [x for x in definitions if x[0] not in ["social_gain", "fatigue_gain"]]
        if np.std(design[:, 1]) < 1e-8:
            definitions = [x for x in definitions if x[0] != "momentum_arrival_gain"]
        if not definitions:
            continue
        initial_config = start.config
        theta = np.array(
            [
                (
                    np.log(max(getattr(initial_config, name), np.exp(lo)))
                    if kind == "log"
                    else getattr(initial_config, name)
                )
                for name, lo, _, kind in definitions
            ]
        )
        bounds = [(lo, hi) for _, lo, hi, _ in definitions]
        theta = np.array([np.clip(x, lo, hi) for x, (lo, hi) in zip(theta, bounds, strict=False)])
        evaluations = 0

        def configuration(values, initial_config=initial_config, definitions=definitions):
            return replace(
                initial_config,
                **{
                    name: float(np.exp(value) if kind == "log" else value)
                    for value, (name, _, _, kind) in zip(values, definitions, strict=False)
                },
            )

        def objective(values, block=block, definitions=definitions, configuration=configuration):
            nonlocal evaluations
            evaluations += 1
            candidate = PopulationWorldLoop.from_dict(start.to_dict())
            candidate.config = configuration(values)
            _, scores, _, _ = replay(candidate, actions, observations)
            fit = float(np.mean([s[SCORE_KEYS[block]] for s in scores]))
            # Fixed weak regularization for mechanism coefficients, chosen before test.
            penalty = 0.001 * sum(
                value**2
                for value, (name, _, _, kind) in zip(values, definitions, strict=False)
                if name in MECHANISM
            )
            return fit + penalty

        before = objective(theta)
        solved = minimize(
            objective,
            theta,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations, "ftol": 1e-8, "gtol": 1e-5, "maxls": 30},
        )
        accepted = bool(np.isfinite(solved.fun) and solved.fun <= before)
        if accepted:
            start.config = configuration(solved.x)
        report["blocks"].append(
            {
                "block": block,
                "score": SCORE_KEYS[block],
                "parameters": [d[0] for d in definitions],
                "objective_before": before,
                "objective_after": float(solved.fun),
                "accepted": accepted,
                "optimizer_success": bool(solved.success),
                "optimizer_message": str(solved.message),
                "evaluations": evaluations,
            }
        )
        fitted_coordinates.extend(definitions)
    trained, scores, center, _ = replay(start, actions, observations)
    # Numerical observability diagnostic of conditional predictive readouts.
    jacobian = []
    for name, lo, hi, kind in fitted_coordinates:
        value = getattr(start.config, name)
        coordinate = np.log(value) if kind == "log" else value
        delta = 1e-3
        shifted_coordinate = min(hi, coordinate + delta)
        if shifted_coordinate == coordinate:
            shifted_coordinate = max(lo, coordinate - delta)
        candidate = PopulationWorldLoop.from_dict(start.to_dict())
        candidate.config = replace(
            candidate.config,
            **{name: float(np.exp(shifted_coordinate) if kind == "log" else shifted_coordinate)},
        )
        _, _, changed, _ = replay(candidate, actions, observations)
        jacobian.append((changed - center) / (shifted_coordinate - coordinate))
    j = np.stack(jacobian, axis=1)
    norms = np.linalg.norm(j, axis=0)
    j_scaled = j / np.maximum(norms, 1e-12)
    values = np.linalg.svd(j_scaled, compute_uv=False)
    report["identification"].update(
        {
            "parameter_names": [x[0] for x in fitted_coordinates],
            "predictive_sensitivity_column_norms": norms.tolist(),
            "normalized_singular_values": values.tolist(),
            "numerical_rank": int(np.linalg.matrix_rank(j_scaled, tol=1e-6)),
        }
    )
    report["training_scores"] = {key: float(np.mean([s[key] for s in scores])) for key in scores[0]}
    report["fitted_config"] = vars(start.config)
    return trained, report
