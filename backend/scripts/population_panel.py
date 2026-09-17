#!/usr/bin/env python3
"""Forecast or continue a saved private population panel.

Forecast records: list of {campaign: WorldAction fields}.
Observe records: list of {day, actions, events, end_timestamp}; events contain
timestamp, user, campaign, group, outcomes and optional content logit shift.
Train complete panels with train_population_panel.py or the component fit APIs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oransim.world_model.population_loop import WorldAction
from oransim.world_model.population_panel import PanelEvent, PopulationWorldPanel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["forecast", "observe"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=200000)
    args = parser.parse_args()
    if args.operation == "forecast" and args.out.resolve() == args.model.resolve():
        parser.error("forecast output must differ from the private model")
    panel = PopulationWorldPanel.load(args.model)
    records = json.loads(args.records.read_text())
    if args.operation == "forecast":
        plans = [{c: WorldAction(**a) for c, a in day.items()} for day in records]
        report = panel.forecast(plans, args.trajectories, args.seed, args.max_events)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    else:
        rows = []
        for record in records:
            if record["day"] != panel.day:
                raise ValueError("observation day must equal the next saved panel day")
            actions = {c: WorldAction(**a) for c, a in record["actions"].items()}
            events = [PanelEvent(**e) for e in record["events"]]
            row, _ = panel.observe_day(events, actions, end_timestamp=record["end_timestamp"])
            rows.append(row)
        panel.save(args.out)
        args.out.with_suffix(".report.json").write_text(
            json.dumps(rows, indent=2, allow_nan=False) + "\n"
        )
    print(str(args.out))


if __name__ == "__main__":
    main()
