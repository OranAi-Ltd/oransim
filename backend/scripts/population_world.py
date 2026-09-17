#!/usr/bin/env python3
"""Saved research world: forecast scenarios, assimilate dated observations or fit parameters.

Inputs are JSON lists. Each record contains an `action` object with content_logits,
targeting_logits and optional exposure_effort/platform_log_supply/attention_logit.
For observe/fit, include `observation`: {day, exposures, responses, provenance: real}.
The day is the saved model's next day, not a calendar date inferred by this CLI.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oransim.world_model.population_learning import fit_world, replay
from oransim.world_model.population_loop import PopulationWorldLoop, WorldAction, WorldObservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["forecast", "observe", "fit"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="forecast JSON or updated model JSON"
    )
    parser.add_argument("--trajectories", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--family", choices=["adaptive", "mechanism"], default="mechanism")
    parser.add_argument("--max-iterations", type=int, default=35)
    args = parser.parse_args()
    if args.operation == "forecast" and args.out.resolve() == args.model.resolve():
        parser.error("forecast output must differ from the model snapshot")
    world = PopulationWorldLoop.load(args.model)
    records = json.loads(args.records.read_text())
    actions = [WorldAction(**record["action"]) for record in records]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.operation == "forecast":
        result = world.forecast_distribution(actions, args.trajectories, args.seed)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    else:
        observations = [WorldObservation(**record["observation"]) for record in records]
        if args.operation == "fit":
            updated, report = fit_world(
                world, actions, observations, args.family, args.max_iterations
            )
        else:
            updated, scores, _, _ = replay(world, actions, observations)
            report = {"operation": "observe", "prequential_scores": scores}
        updated.save(args.out)
        args.out.with_suffix(".report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    print(str(args.out))


if __name__ == "__main__":
    main()
