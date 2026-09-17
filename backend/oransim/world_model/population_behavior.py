"""A normalized joint binary observation head over a shared Gaussian population.

The autoregressive order factorizes a joint distribution; it does not assert a
causal ordering or require a click before another logged action can occur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp


@dataclass
class BehaviorBatch:
    features: np.ndarray
    latent_nodes: np.ndarray
    latent_weights: np.ndarray
    outcomes: np.ndarray

    def take(self, indices):
        return BehaviorBatch(*(getattr(self, name)[indices] for name in self.__dataclass_fields__))

    @classmethod
    def concatenate(cls, batches):
        return cls(
            *(
                np.concatenate([getattr(b, name) for b in batches])
                for name in cls.__dataclass_fields__
            )
        )


class JointBehaviorModel:
    def __init__(self, heads, feature_names, rates=None, dependent=True):
        self.heads = list(heads)
        self.feature_names = list(feature_names)
        self.dependent = bool(dependent)
        h = len(heads)
        f = len(feature_names)
        if not 1 <= h <= 10 or len(set(heads)) != h:
            raise ValueError("one to ten distinct binary heads required")
        self.coefficients = np.zeros((h, f + 1))
        rates = np.full(h, 0.1) if rates is None else np.asarray(rates, float)
        self.coefficients[:, 0] = logit(np.clip(rates, 1e-5, 1 - 1e-5))
        self.coefficients[0, 0] = 0.0  # primary population logit is the first-head anchor
        self.dependencies = np.zeros((h, h))
        self.loadings = np.zeros(h)
        self.loadings[0] = 1.0
        self.feature_scale = np.ones(f)
        self.feature_center = np.zeros(f)

    def _validate(self, batch):
        x, u, w, y = (
            np.asarray(getattr(batch, name), float) for name in batch.__dataclass_fields__
        )
        n = len(x)
        h = len(self.heads)
        if (
            x.shape != (n, len(self.feature_names))
            or u.ndim != 2
            or u.shape != w.shape
            or len(u) != n
            or y.shape != (n, h)
        ):
            raise ValueError("incompatible behavior batch")
        if (
            not all(np.isfinite(v).all() for v in (x, u, w, y))
            or np.any(w <= 0)
            or not np.allclose(w.sum(1), 1)
        ):
            raise ValueError("invalid behavior values or quadrature weights")
        if np.any((y != 0) & (y != 1)):
            raise ValueError("binary outcomes required")
        return (
            np.column_stack([np.ones(n), (x - self.feature_center) / self.feature_scale]),
            u,
            w,
            y,
        )

    def log_probability(self, batch, shift=0.0):
        x, u, w, y = self._validate(batch)
        logits = (x @ self.coefficients.T + y @ self.dependencies.T)[:, None, :] + (u + shift)[
            :, :, None
        ] * self.loadings
        ll = (y[:, None, :] * logits - np.logaddexp(0, logits)).sum(2)
        return logsumexp(np.log(w) + ll, axis=1)

    def marginals(self, batch):
        """Exactly sum binary prefixes, with Gaussian quadrature for the latent."""
        x, u, w, _ = self._validate(batch)
        h = len(self.heads)
        base = (x @ self.coefficients.T)[:, None, :] + u[:, :, None] * self.loadings
        mass = w[:, :, None]
        prefix = np.zeros((1, 0))
        result = []
        for j in range(h):
            adjustment = prefix @ self.dependencies[j, :j]
            p = expit(base[:, :, j, None] + adjustment)
            result.append((mass * p).sum((1, 2)))
            mass = np.concatenate([mass * (1 - p), mass * p], axis=2)
            prefix = np.concatenate(
                [
                    np.column_stack([prefix, np.zeros(len(prefix))]),
                    np.column_stack([prefix, np.ones(len(prefix))]),
                ]
            )
        return np.stack(result, axis=1)

    def sample(self, features, latent, rng):
        x = np.r_[1.0, (np.asarray(features) - self.feature_center) / self.feature_scale]
        y = np.zeros(len(self.heads), int)
        base = self.coefficients @ x + self.loadings * latent
        for j in range(len(y)):
            y[j] = rng.binomial(1, expit(base[j] + self.dependencies[j, :j] @ y[:j]))
        return y

    def fit(self, batch, max_iterations=45, regularization=0.001):
        if len(batch.features) == 0:
            raise ValueError("nonempty training batch required")
        std = np.std(batch.features, axis=0)
        new_scale = np.where(std > 1e-6, std, 1.0)
        new_center = np.mean(batch.features, axis=0)
        self.coefficients[:, 0] += self.coefficients[:, 1:] @ (
            (new_center - self.feature_center) / self.feature_scale
        )
        self.coefficients[:, 1:] *= new_scale / self.feature_scale
        self.feature_scale = new_scale
        self.feature_center = new_center
        x, u, w, y = self._validate(batch)
        h = len(self.heads)
        width = x.shape[1]
        lower = np.tril_indices(h, -1) if self.dependent else (np.array([], int), np.array([], int))
        ndep = len(lower[0])
        ncoef = h * width
        theta = np.r_[self.coefficients.ravel(), self.dependencies[lower], self.loadings[1:]]

        def unpack(t):
            c = t[:ncoef].reshape(h, width)
            a = np.zeros((h, h))
            a[lower] = t[ncoef : ncoef + ndep]
            return c, a, np.r_[1.0, t[ncoef + ndep :]]

        def objective(t):
            c, a, l = unpack(t)
            logits = (x @ c.T + y @ a.T)[:, None, :] + u[:, :, None] * l
            ll = (y[:, None, :] * logits - np.logaddexp(0, logits)).sum(2) + np.log(w)
            logp = logsumexp(ll, axis=1)
            responsibility = np.exp(ll - logp[:, None])
            residual = (expit(logits) - y[:, None, :]) * responsibility[:, :, None] / len(x)
            reduced = residual.sum(1)
            gc = reduced.T @ x
            ga = (reduced.T @ y)[lower]
            gl = (residual * u[:, :, None]).sum((0, 1))[1:]
            penalty = (
                regularization
                * (np.square(c[:, 1:]).sum() + np.square(a).sum() + np.square(l[1:]).sum())
                / 2
            )
            gc[:, 1:] += regularization * c[:, 1:]
            ga += regularization * a[lower]
            gl += regularization * l[1:]
            return float(-logp.mean() + penalty), np.r_[gc.ravel(), ga, gl]

        before = objective(theta)[0]
        result = minimize(
            objective,
            theta,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-12, 12)] * len(theta),
            options={"maxiter": max_iterations, "ftol": 1e-8, "gtol": 1e-5},
        )
        accepted = bool(np.isfinite(result.fun) and result.fun <= before)
        if accepted:
            self.coefficients, self.dependencies, self.loadings = unpack(result.x)
        return {
            "rows": len(x),
            "joint_nll_before": before,
            "penalized_joint_nll_after": float(result.fun),
            "accepted": accepted,
            "optimizer_success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "regularization": regularization,
            "factorization_order": self.heads,
        }

    def to_dict(self):
        return {
            "heads": self.heads,
            "feature_names": self.feature_names,
            "dependent": self.dependent,
            **{
                name: getattr(self, name).tolist()
                for name in [
                    "coefficients",
                    "dependencies",
                    "loadings",
                    "feature_scale",
                    "feature_center",
                ]
            },
        }

    @classmethod
    def from_dict(cls, value):
        model = cls(value["heads"], value["feature_names"], dependent=value["dependent"])
        for name in ["coefficients", "dependencies", "loadings", "feature_scale", "feature_center"]:
            if name == "feature_center" and name not in value:
                continue
            array = np.asarray(value[name], float)
            if array.shape != getattr(model, name).shape or not np.isfinite(array).all():
                raise ValueError("invalid saved behavior model")
            setattr(model, name, array)
        if (
            np.any(model.feature_scale <= 0)
            or model.loadings[0] != 1
            or np.any(np.triu(model.dependencies) != 0)
        ):
            raise ValueError("invalid joint factorization")
        if not model.dependent and np.any(model.dependencies):
            raise ValueError("independent model has dependencies")
        return model
