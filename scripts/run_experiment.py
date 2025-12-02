# scripts/run_experiment.py
import argparse

from cpdp.config.loader import load_config
from cpdp.pipeline.runner import ExperimentRunner
from cpdp.utils.io import save_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment YAML config."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/metrics.json",
        help="Path to save metrics JSON.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    runner = ExperimentRunner(config)
    metrics = runner.run()

    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    save_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
