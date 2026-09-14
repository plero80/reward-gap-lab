"""Command-line controls; model libraries are imported only by relevant commands."""

import argparse
import json

from reward_gap.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="reward-gap", description="Prepare, check, and run reward-gap experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("prepare", "Prepare disjoint HH-RLHF prompt cohorts"),
                            ("preflight", "Check inputs, device and model inference"),
                            ("run", "Run or resume the two-round experiment"),
                            ("rq1", "Evaluate frozen gap predictors before and after PPO"),
                            ("gsm8k-prepare", "Prepare question-disjoint GSM8K cohorts"),
                            ("gsm8k-preflight", "Check GSM8K policy and generative graders"),
                            ("gsm8k-run", "Run or resume the three-arm GSM8K comparison"),
                            ("gsm8k-status", "Read a GSM8K run status"),
                            ("status", "Read a saved experiment status without loading models")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, help="Experiment JSON path")
        if name in ("prepare", "gsm8k-prepare"):
            command.add_argument("--download", action="store_true", help="Allow dataset downloads")
        if name in ("run", "status", "rq1", "gsm8k-run", "gsm8k-status"):
            command.add_argument("--run-name", required=True)
        if name == "rq1":
            command.add_argument("--plan", required=True, help="RQ1 prediction/evaluation plan JSON")
        if name == "run":
            command.add_argument("--until", choices=("round1", "training", "complete"), default="complete",
                                 help="Pause after a stage boundary; rerun without this option to continue")
        if name == "gsm8k-run":
            command.add_argument("--until", choices=("training", "complete"), default="complete")
    args = parser.parse_args(argv)
    try:
        if args.command.startswith("gsm8k-"):
            from reward_gap.gsm8k.config import load_gsm_config
            settings = load_gsm_config(args.config)
            if args.command == "gsm8k-prepare":
                from reward_gap.gsm8k.data import prepare
                print(f"GSM8K prepared: {prepare(settings, allow_downloads=args.download)}")
            elif args.command == "gsm8k-preflight":
                from reward_gap.gsm8k.preflight import preflight
                print(f"GSM8K preflight passed: {preflight(settings)}")
            else:
                root = settings.base.runtime.output_root.resolve()
                run_dir = (root / args.run_name).resolve()
                if run_dir == root or not run_dir.is_relative_to(root):
                    raise ValueError("Run name must select a directory inside output_root")
                if args.command == "gsm8k-status":
                    print(json.dumps(json.loads((run_dir / "status.json").read_text()), indent=2))
                else:
                    from reward_gap.gsm8k.experiment import GSMExperiment
                    result = GSMExperiment(settings, run_dir).run(until=args.until)
                    print(f"GSM8K {result.state}: {result.summary_path}")
            return
        if args.command == "prepare":
            from reward_gap.data import prepare_data
            print(f"Prepared data manifest: {prepare_data(args.config, download=args.download)}")
            return
        config = load_config(args.config)
        if args.command == "preflight":
            from reward_gap.preflight import preflight
            result = preflight(config)
            print(f"Preflight {result.state}: {result.report_path}")
            return
        root = config.runtime.output_root.resolve()
        run_dir = (root / args.run_name).resolve()
        if run_dir == root or not run_dir.is_relative_to(root):
            raise ValueError("Run name must select a directory inside output_root")
        if args.command == "status":
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            print(json.dumps(status, indent=2))
        elif args.command == "rq1":
            from reward_gap.rq1 import RQ1Experiment, load_plan
            result = RQ1Experiment(config, run_dir, plan=load_plan(args.plan)).run()
            print(f"RQ1 {result.state}: {result.summary_path}")
        else:
            from reward_gap.experiment import FollowupExperiment
            result = FollowupExperiment(config, run_dir).run(until=args.until)
            print(f"Experiment {result.state}: {result.status_path}")
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"{args.command} failed: {exc}\n")


if __name__ == "__main__":
    main()
