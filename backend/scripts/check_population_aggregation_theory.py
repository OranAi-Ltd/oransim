#!/usr/bin/env python3
"""Independent numerical checks for the population aggregation propositions.

These finite examples check algebra and the implementation. Mathematical proofs,
including their assumptions, live in paper-readiness/theory.md. A high-order
quadrature reference is a numerical reference, not an exact integral certificate.
"""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import expit, ndtr
from torch.nn import functional as F  # noqa: N812 - standard PyTorch alias

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from oransim.world_model.differentiable_market import (
    DTYPE,
    DifferentiableMarket,
    MarketBatch,
    MarketConfig,
    MarketDay,
)


def fixture(q=15):
    model = DifferentiableMarket(MarketConfig(1, 1, 1, ("click", "like"), quadrature=q))
    with torch.no_grad():
        model.means.copy_(torch.tensor([[[-1.1], [1.3]]], dtype=DTYPE))
        model.mixture_logits.copy_(torch.tensor([[0.25, -0.25]], dtype=DTYPE))
        model.intensity_loading.fill_(0.3)
        model.intensity_features.fill_(0.15)
        model.behavior_features.copy_(torch.tensor([[0.5], [-0.2]], dtype=DTYPE))
        model.behavior_loading.fill_(0.7)
        model.behavior_bias.copy_(torch.tensor([[-0.5, -1.0]], dtype=DTYPE))
        model.dependencies[1, 0] = 0.8
        model.raw_cholesky[:, 0, 0, 0] = np.log(np.expm1(0.65 - 1e-4))
        model.raw_cholesky[:, 1, 0, 0] = np.log(np.expm1(1.1 - 1e-4))
    batch = MarketBatch.create([0] * 4, [0] * 4, [[-1.0], [0.0], [0.5], [1.5]], [100.0] * 4)
    return model, batch


def requadrature(model, q):
    result = DifferentiableMarket(replace(model.config, quadrature=q))
    values = deepcopy(model.state_dict())
    values["nodes"], values["node_weights"] = result.nodes, result.node_weights
    result.load_state_dict(values)
    return result


def untilted_probability(model, batch, state):
    """Independent readout: integrate conditional marks under population weights."""
    u, logw = model.population_nodes(batch.group, state)
    x = (batch.features - model.feature_center) / model.feature_scale
    load = torch.cat([torch.ones((1, 1), dtype=DTYPE), model.behavior_loading])
    base = (
        model.behavior_bias[batch.group]
        + model.behavior_campaign[batch.campaign]
        + x @ model.behavior_features.T
        + state.memory[batch.group] @ model.memory_coefficients.T
        + state.fatigue[batch.group, None] * model.fatigue_coefficients
        + state.common * model.environment_coefficients
    )
    logits = base[:, None, None, :] + u @ load.T
    logits = (
        logits[:, :, :, None, :]
        + (model.outcomes @ torch.tril(model.dependencies, -1).T)[None, None, None]
    )
    conditional = (model.outcomes[None, None, None] * logits - F.softplus(logits)).sum(-1).exp()
    return (conditional * logw.exp()[:, :, :, None]).sum((1, 2))


def covariance_check():
    model, batch = fixture(41)
    state = model.initial_state()
    with torch.no_grad():
        u, logw = model.population_nodes(batch.group, state)
        weights = logw.exp()
        x = batch.features
        rate = (
            (
                model.base_log_intensity[batch.campaign, batch.group, None, None]
                + (x @ model.intensity_features)[:, None, None]
                + u @ model.intensity_loading
            )
            .clamp(-16, 10)
            .exp()
        )
        response = torch.sigmoid(
            model.behavior_bias[batch.group, 0, None, None]
            + (x @ model.behavior_features[0])[:, None, None]
            + u[..., 0]
        )
        er, ep = (weights * rate).sum((1, 2)), (weights * response).sum((1, 2))
        product = (weights * rate * response).sum((1, 2))
        covariance = (weights * (rate - er[:, None, None]) * (response - ep[:, None, None])).sum(
            (1, 2)
        )
        output = model(batch)
        error = (
            (output["actions"][:, 0] / batch.population - er * ep - covariance).abs().max().item()
        )
        assert error < 1e-12
        assert covariance.min() > 0
        return {
            "identity_max_error": error,
            "covariance": covariance.tolist(),
            "exposure_conditioned_click_rate": (product / er).tolist(),
            "unweighted_click_rate": ep.tolist(),
        }


def aggregation_check():
    model, _ = fixture()
    features = [[0.4]] * 4
    batch = MarketBatch.create([0] * 4, [0] * 4, features, [1.0] * 4)
    labels = model.outcomes.clone()
    event = -model.event_log_probability(batch, labels).sum()
    cohort = MarketBatch.create([0], [0], [[0.4]], [4.0])
    aggregate = -(torch.ones((1, 4), dtype=DTYPE) * model(cohort)["joint_log_prob"]).sum()
    parameters = [model.means, model.raw_cholesky, model.mixture_logits]
    egrad = torch.cat([x.flatten() for x in torch.autograd.grad(event, parameters)])
    agrad = torch.cat([x.flatten() for x in torch.autograd.grad(aggregate, parameters)])
    assert abs(float(event.detach() - aggregate.detach())) < 1e-12
    assert (egrad - agrad).abs().max() < 1e-12
    hetero = expit(np.array([-2.0, 1.0]))
    average_feature_response = expit(-0.5)
    # With fixed distinct contexts the chance of one success is Poisson-binomial.
    exact_one_success = hetero[0] * (1 - hetero[1]) + (1 - hetero[0]) * hetero[1]
    mean = hetero.mean()
    multinomial_one_success = 2 * mean * (1 - mean)
    return {
        "same_context_nll_difference": float(event.detach() - aggregate.detach()),
        "same_context_gradient_max_difference": float((egrad - agrad).abs().max()),
        "different_context_mean_probability": float(mean),
        "probability_at_mean_feature": float(average_feature_response),
        "fixed_context_exact_one_success_probability": float(exact_one_success),
        "iid_mixture_one_success_probability": float(multinomial_one_success),
    }


def identification_check():
    support = np.array([-3.0, -1.3, 0.0, 1.1, 3.0])
    rate = np.exp(0.25 * support)
    matrix = np.stack([np.ones(5), rate, rate * expit(support - 1), rate * expit(support + 1)])
    _, _, vh = np.linalg.svd(matrix)
    null = vh[-1]
    positive, negative = np.maximum(null, 0), np.maximum(-null, 0)
    positive /= positive.sum()
    negative /= negative.sum()
    error = float(np.max(np.abs(matrix @ positive - matrix @ negative)))
    probe = rate * expit(support - 2.3)
    assert error < 1e-12
    assert abs(probe @ (positive - negative)) > 1e-4
    known = np.stack([np.ones(3), expit(support[:3] - 1), expit(support[:3] + 1)])
    truth = np.array([0.2, 0.3, 0.5])
    recovery_error = float(np.max(np.abs(np.linalg.solve(known, known @ truth) - truth)))
    model, batch = fixture()
    translated = deepcopy(model)
    shift = 0.83
    with torch.no_grad():
        translated.means.add_(shift)
        translated.intensity_bias.sub_(shift * model.intensity_loading[0])
        translated.behavior_bias[:, 0].sub_(shift)
        translated.behavior_bias[:, 1:].sub_(shift * model.behavior_loading[:, 0])
        first, second = model(batch), translated(batch)
        gauge_error = max(
            float((first[k] - second[k]).abs().max()) for k in ["exposures", "joint_log_prob"]
        )
    assert gauge_error < 1e-11
    return {
        "finite_measurements": {
            "support": support.tolist(),
            "population_a": positive.tolist(),
            "population_b": negative.tolist(),
            "same_measurement_max_error": error,
            "new_context_numerator_difference": float(probe @ (positive - negative)),
        },
        "known_support_full_rank": int(np.linalg.matrix_rank(known)),
        "known_support_weight_recovery_max_error": recovery_error,
        "latent_translation": shift,
        "translation_output_max_error": gauge_error,
    }


def exponential_tilt_check():
    model, batch = fixture(81)
    transformed = deepcopy(model)
    with torch.no_grad():
        means = model.means[..., 0]
        variance = model.cholesky()[..., 0, 0].square()
        gamma = model.intensity_loading[0]
        logs = means * gamma + 0.5 * variance * gamma.square()
        logz = torch.logsumexp(F.log_softmax(model.mixture_logits, -1) + logs, -1)
        transformed.mixture_logits.copy_(F.log_softmax(model.mixture_logits, -1) + logs)
        transformed.means[..., 0].copy_(means + variance * gamma)
        newlogs = transformed.means[..., 0] * gamma + 0.5 * variance * gamma.square()
        newz = torch.logsumexp(F.log_softmax(transformed.mixture_logits, -1) + newlogs, -1)
        transformed.intensity_bias.add_(logz - newz)
        states = [model.initial_state(), transformed.initial_state()]
        max_probability_error = max_count_error = max_state_error = 0.0
        for _ in range(8):
            first = model(batch, states[0])
            second = transformed(batch, states[1])
            probability = untilted_probability(transformed, batch, states[1])
            second["joint_log_prob"] = probability.log()
            second["marginals"] = probability @ transformed.outcomes
            second["actions"] = second["exposures"][:, None] * second["marginals"]
            max_probability_error = max(
                max_probability_error,
                float((first["joint_log_prob"].exp() - probability).abs().max()),
            )
            max_count_error = max(
                max_count_error, float((first["exposures"] - second["exposures"]).abs().max())
            )
            states = [
                model.transition(states[0], batch, first),
                transformed.transition(states[1], batch, second),
            ]
            max_state_error = max(
                max_state_error,
                *(
                    float((getattr(states[0], k) - getattr(states[1], k)).abs().max())
                    for k in ["shift", "supply", "common", "fatigue", "memory"]
                ),
            )
    assert max_probability_error < 1e-10 and max_count_error < 1e-8 and max_state_error < 1e-10
    # Analytic absolute difference between clipped and unclipped lognormal rates.
    m, sd, lower, upper = 8.0, 2.0, -16.0, 10.0
    moment = np.exp(m + 0.5 * sd * sd)
    lower_error = np.exp(lower) * ndtr((lower - m) / sd) - moment * ndtr((lower - m - sd * sd) / sd)
    upper_error = moment * ndtr((m + sd * sd - upper) / sd) - np.exp(upper) * ndtr((m - upper) / sd)
    analytic = lower_error + upper_error
    numerical = quad(
        lambda z: abs(np.exp(np.clip(m + sd * z, lower, upper)) - np.exp(m + sd * z))
        * np.exp(-z * z / 2)
        / np.sqrt(2 * np.pi),
        -12,
        12,
        points=[(lower - m) / sd, (upper - m) / sd],
        epsabs=1e-8,
    )[0]
    assert abs(analytic - numerical) / analytic < 1e-8
    return {
        "eight_step_probability_max_difference": max_probability_error,
        "eight_step_count_max_difference": max_count_error,
        "eight_step_state_max_difference": max_state_error,
        "numerical_scope": "81-point reference; chosen states have negligible lognormal clipped tails. This check is not an exact full-Gaussian identity under clamp.",
        "clipped_lognormal_tail_example": {
            "log_mean": m,
            "log_sd": sd,
            "analytic_absolute_rate_error": float(analytic),
            "numerical_reference_error": float(numerical),
        },
    }


def probability_and_gradient(model, batch):
    output = model(batch)
    d = output["exposures"][1] / batch.population[1]
    p = output["joint_log_prob"][1, -1].exp()
    a = d * p
    parameters = [model.means, model.raw_cholesky, model.mixture_logits]

    def grad(value):
        return (
            torch.cat(
                [x.flatten() for x in torch.autograd.grad(value, parameters, retain_graph=True)]
            )
            .detach()
            .numpy()
        )

    return dict(
        D=float(d.detach()),
        A=float(a.detach()),
        p=float(p.detach()),
        gD=grad(d),
        gA=grad(a),
        gp=grad(p),
    )


def quadrature_check():
    reference, batch = fixture(101)
    true = probability_and_gradient(reference, batch)
    rows = []
    for q in [3, 7, 15, 31]:
        result = probability_and_gradient(requadrature(reference, q), batch)
        ed, ea = abs(result["D"] - true["D"]), abs(result["A"] - true["A"])
        denominator = true["D"] - ed
        pbound = (ea + ed) / denominator
        delta_p = abs(result["p"] - true["p"])
        gbound = (
            np.linalg.norm(result["gA"] - true["gA"])
            + np.linalg.norm(result["gD"] - true["gD"])
            + delta_p * np.linalg.norm(true["gD"])
            + np.linalg.norm(true["gp"]) * ed
        ) / denominator
        gerror = np.linalg.norm(result["gp"] - true["gp"])
        eta = min(result["p"], true["p"])
        logerror = abs(np.log(result["p"]) - np.log(true["p"]))
        assert (
            delta_p <= pbound + 1e-13
            and gerror <= gbound + 1e-13
            and logerror <= pbound / eta + 1e-13
        )
        rows.append(
            {
                "nodes": q,
                "probability_error": delta_p,
                "probability_bound": pbound,
                "gradient_l2_error": float(gerror),
                "gradient_l2_bound": float(gbound),
                "log_probability_error": logerror,
                "log_probability_bound": pbound / eta,
            }
        )
    return {"reference_nodes": 101, "reference_is_exact_integral": False, "rows": rows}


def recursive_bound_check():
    model = DifferentiableMarket(MarketConfig(1, 1, 0, ("click",), components=1, quadrature=101))
    with torch.no_grad():
        model.means.fill_(0.7)
        model.intensity_loading.zero_()
        model.raw_cholesky.fill_(np.log(np.expm1(1.7 - 1e-4)))
        model.raw_retention[0] = np.log(0.45 / 0.55)
        model.feedback.fill_(1.2)
        model.fatigue_feedback.zero_()
    low = requadrature(model, 3)
    batch = MarketBatch.create([0], [0], [[]], [100.0])
    state, approximate = model.initial_state(), low.initial_state()
    n = float(model(batch)["exposures"][0].detach())
    lipschitz = 0.45 + 1.2 * n / (n + 2) / 4
    assert lipschitz < 1
    bound = 0.0
    rows = []
    with torch.no_grad():
        for day in range(1, 13):
            reference_at_approximate = model.transition(
                approximate, batch, model(batch, approximate)
            )
            low_next = low.transition(approximate, batch, low(batch, approximate))
            local_error = abs(float(low_next.shift - reference_at_approximate.shift))
            state = model.transition(state, batch, model(batch, state))
            approximate = low_next
            bound = lipschitz * bound + local_error
            error = abs(float(approximate.shift - state.shift))
            assert error <= bound + 1e-12
            rows.append(
                {
                    "day": day,
                    "state_shift_error": error,
                    "conditional_bound": bound,
                    "local_quadrature_residual": local_error,
                }
            )
    return {
        "scope": "Scalar state submodel with constant exposure and inactive other-state readout; not a full-model stability assertion.",
        "analytic_lipschitz_constant": lipschitz,
        "reference_nodes": 101,
        "approximate_nodes": 3,
        "rows": rows,
    }


def transformed_comparator(model, q):
    from oransim.world_model.population_comparisons import SeparatedAggregationMarket

    base = requadrature(model, q)
    other = SeparatedAggregationMarket(base.config)
    other.load_state_dict(base.state_dict())
    with torch.no_grad():
        gamma = base.intensity_loading
        covariance = base.cholesky() @ base.cholesky().transpose(-1, -2)
        delta = covariance @ gamma
        quadratic = (delta * gamma).sum(-1)
        logfactor = base.means @ gamma + 0.5 * quadratic
        logz = torch.logsumexp(F.log_softmax(base.mixture_logits, -1) + logfactor, -1)
        other.means.copy_(base.means + delta)
        other.mixture_logits.copy_(F.log_softmax(base.mixture_logits, -1) + logfactor)
        newlogz = torch.logsumexp(
            F.log_softmax(other.mixture_logits, -1) + other.means @ gamma + 0.5 * quadratic, -1
        )
        other.intensity_bias.add_(logz - newlogz)
    return base, other


def clipping_statistics(model, batch, state):
    u, logw = model.population_nodes(batch.group, state)
    x = (batch.features - model.feature_center) / model.feature_scale
    a = (
        model.base_log_intensity[batch.campaign, batch.group]
        + model.intensity_bias[batch.campaign, batch.group]
        + x @ model.intensity_features
        + state.common
        + state.supply[batch.campaign]
    )
    lograte = a[:, None, None] + u @ model.intensity_loading
    mask = (lograte < -16) | (lograte > 10)
    population_weight = logw.exp()
    selected_logw = logw + lograte.clamp(-16, 10)
    selected_weight = (
        selected_logw - torch.logsumexp(selected_logw.flatten(1), 1)[:, None, None]
    ).exp()
    mean = (
        a[:, None]
        + (model.means[batch.group] + state.shift[batch.group, None]) @ model.intensity_loading
    ).numpy()
    l = model.cholesky()[batch.group]
    sd = (l.transpose(-1, -2) @ model.intensity_loading).square().sum(-1).sqrt().numpy()
    sd = np.maximum(sd, 1e-100)
    moment = np.exp(mean + 0.5 * sd * sd)
    lower = np.exp(-16) * ndtr((-16 - mean) / sd) - moment * ndtr((-16 - mean - sd * sd) / sd)
    upper = moment * ndtr((mean + sd * sd - 10) / sd) - np.exp(10) * ndtr((mean - 10) / sd)
    pi = F.softmax(model.mixture_logits[batch.group], -1).numpy()
    error = (pi * np.maximum(lower + upper, 0)).sum(-1)
    denominator = (pi * moment).sum(-1)
    ratio = error / denominator
    return {
        "node_entries": mask.numel(),
        "clipped_node_entries": int(mask.sum()),
        "population_weight_sum": float((mask * population_weight).sum()),
        "exposure_weight_sum": float((mask * selected_weight).sum()),
        "contexts": len(batch.group),
        "max_relative_analytic_clipping_error": float(ratio.max()),
        "max_conditional_probability_clipping_bound": float(
            (2 * ratio / np.maximum(1 - ratio, 1e-100)).max()
        ),
    }


def v3_snapshot_audit():
    rows = []
    files = {}
    for policy in ["standard", "random"]:
        folder = ROOT / "models/gaussian_population/controlled" / policy
        record_path = folder / "records.json"
        records = json.loads(record_path.read_text())
        files[str(record_path.relative_to(ROOT))] = hashlib.sha256(
            record_path.read_bytes()
        ).hexdigest()
        for seed in [20260912, 20260913, 20260914]:
            path = folder / f"{seed}-joint_gaussian-train.json"
            original, _, _, _ = DifferentiableMarket.load(path)
            files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            for q in [7, 25, 61]:
                first, second = transformed_comparator(original, q)
                state, alt_state = first.initial_state(), second.initial_state()
                total = {
                    "node_entries": 0,
                    "clipped_node_entries": 0,
                    "population_weight_sum": 0.0,
                    "exposure_weight_sum": 0.0,
                    "contexts": 0,
                    "max_relative_analytic_clipping_error": 0.0,
                    "max_conditional_probability_clipping_bound": 0.0,
                }
                nll_a = nll_b = max_event_log_difference = max_cohort_count_difference = (
                    max_cohort_probability_difference
                ) = 0.0
                event_count = 0
                with torch.no_grad():
                    for split in ["train", "validation", "test"]:
                        for record in records[split]:
                            day = MarketDay.from_dict(record)
                            out, alternative = first(day.batch, state), second(day.batch, alt_state)
                            if split == "test":
                                max_cohort_count_difference = max(
                                    max_cohort_count_difference,
                                    float(
                                        (out["exposures"] - alternative["exposures"]).abs().max()
                                    ),
                                )
                                max_cohort_probability_difference = max(
                                    max_cohort_probability_difference,
                                    float(
                                        (
                                            out["joint_log_prob"].exp()
                                            - alternative["joint_log_prob"].exp()
                                        )
                                        .abs()
                                        .max()
                                    ),
                                )
                                events = day.event_batch
                                for start in range(0, len(events.group), 256):
                                    b = MarketBatch(
                                        *(
                                            getattr(events, k)[start : start + 256]
                                            for k in events.__dataclass_fields__
                                        )
                                    )
                                    y = day.event_outcomes[start : start + 256]
                                    left, right = first.event_log_probability(
                                        b, y, state
                                    ), second.event_log_probability(b, y, alt_state)
                                    nll_a -= float(left.sum())
                                    nll_b -= float(right.sum())
                                    event_count += len(y)
                                    max_event_log_difference = max(
                                        max_event_log_difference, float((left - right).abs().max())
                                    )
                                    clip = clipping_statistics(first, b, state)
                                    for key, value in clip.items():
                                        total[key] = (
                                            max(total[key], value)
                                            if key.startswith("max_")
                                            else total[key] + value
                                        )
                            state = first.transition(state, day.batch, out, day.patterns)
                            alt_state = second.transition(
                                alt_state, day.batch, alternative, day.patterns
                            )
                total["mean_population_probability_at_clipped_nodes"] = (
                    total.pop("population_weight_sum") / total["contexts"]
                )
                total["mean_exposure_probability_at_clipped_nodes"] = (
                    total.pop("exposure_weight_sum") / total["contexts"]
                )
                rows.append(
                    {
                        "policy": policy,
                        "seed": seed,
                        "quadrature": q,
                        "events": event_count,
                        "tilted_test_event_nll": nll_a / event_count,
                        "reparameterized_separated_test_event_nll": nll_b / event_count,
                        "test_event_nll_difference": (nll_b - nll_a) / event_count,
                        "max_event_log_probability_difference": max_event_log_difference,
                        "max_cohort_exposure_difference": max_cohort_count_difference,
                        "max_cohort_joint_probability_difference": max_cohort_probability_difference,
                        "clipping": total,
                    }
                )
                print(
                    f"theory snapshot {policy} {seed} q={q}: NLL delta={(nll_b-nll_a)/event_count:.3g}",
                    flush=True,
                )
    return {
        "scope": "Read-only replay from initial state with fitted v3 parameters. Train/validation/test all use observed day-end feedback. Comparator has analytic tilt parameters, no refitting. Finite quadrature differences remain.",
        "private_source_sha256": files,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/gaussian_population/paper-readiness/theory-checks.json",
    )
    parser.add_argument(
        "--audit-v3",
        action="store_true",
        help="Read-only six-snapshot real-data tilt/clipping audit at 7, 25 and 61 points.",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {
        "scope": "Numerical consistency checks; proofs and assumptions are in theory.md. No real-data performance claim.",
        "covariance_identity": covariance_check(),
        "aggregation_sufficiency": aggregation_check(),
        "identifiability_examples": identification_check(),
        "exponential_tilt_equivalence": exponential_tilt_check(),
        "quadrature_probability_gradient_bounds": quadrature_check(),
        "recursive_error_bound": recursive_bound_check(),
    }
    sources = [Path(__file__), ROOT / "backend/oransim/world_model/differentiable_market.py"]
    if args.audit_v3:
        result["v3_snapshot_tilt_clipping_audit"] = v3_snapshot_audit()
        sources.append(ROOT / "backend/oransim/world_model/population_comparisons.py")
    result["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    result["environment"] = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "dtype": "float64",
        "threads": 1,
    }
    result["all_checks_passed"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_checks_passed": True, "output": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
