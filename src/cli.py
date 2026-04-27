"""DAE-ELM CLI Entry Point.

Usage:
    python -m src.cli run --config config.yaml --data data.csv
    python -m src.cli optimize --model elm --input data.csv --output results.csv
"""

import argparse
import sys
import os

from src.core.runner import TaskRunner
from src.core.config_parser import YAMLConfigParser


def run_command(args):
    """Run optimization with config and data."""
    config_path = os.path.abspath(args.config)
    data_path = os.path.abspath(args.data)

    print(f"Loading config from: {config_path}")
    print(f"Loading data from: {data_path}")

    config_parser = ConfigParser()
    config = config_parser.load_and_parse(config_path)

    runner = TaskRunner(config)
    result = runner.run()

    print("Optimization completed!")
    return result


def optimize_command(args):
    """Optimize formula using specified model."""
    model = args.model
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    print(f"Using model: {model}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Load data and run optimization
    import numpy as np
    from src.data.dataset.formula_loader import FormulaDataLoader

    loader = FormulaDataLoader(input_path)
    X, y = loader.load()

    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")

    # Run with default optimization
    from src.optimization.algorithms.nsga2 import NSGA2Optimizer

    optimizer = NSGA2Optimizer(pop_size=50, n_generations=100)

    def objective(x):
        # Simple objective for demo
        return [x[0] * x[1], 1 - x[0]]

    result = optimizer.optimize([objective], [(0, 1)] * X.shape[1])

    np.savetxt(output_path, result)
    print(f"Results saved to: {output_path}")

    return result


def main():
    """Main entry point for DAE-ELM CLI."""
    parser = argparse.ArgumentParser(
        prog="daeelm",
        description="DAE-ELM - Chemical Formula Optimization System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run optimization with config and data")
    run_parser.add_argument("--config", required=True, help="Path to configuration file")
    run_parser.add_argument("--data", required=True, help="Path to data file")
    run_parser.set_defaults(func=run_command)

    # optimize command
    optimize_parser = subparsers.add_parser(
        "optimize", help="Optimize formula using specified model"
    )
    optimize_parser.add_argument("--model", default="elm", help="Model to use (default: elm)")
    optimize_parser.add_argument("--input", required=True, help="Input data file")
    optimize_parser.add_argument("--output", required=True, help="Output result file")
    optimize_parser.set_defaults(func=optimize_command)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()