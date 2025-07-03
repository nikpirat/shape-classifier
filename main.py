import argparse
import os
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualizer import run_live_visualizer

def main():
    parser = argparse.ArgumentParser(description="Shape Classification Model: Train, Evaluate, or Visualize")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--visualize", action="store_true", help="Run live visualizer on dataset")
    parser.add_argument("--data", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--model", type=str, default="model/shape_net.pt", help="Path to save/load model file")
    args = parser.parse_args()

    # Check if at least one action is specified
    if not (args.train or args.eval or args.visualize):
        parser.error("One of --train, --eval, or --visualize must be specified")

    # Validate paths
    if not os.path.exists(args.data):
        parser.error(f"Data directory '{args.data}' does not exist")
    if (args.train or args.eval or args.visualize) and not os.path.exists(os.path.dirname(args.model)) and os.path.dirname(args.model):
        parser.error(f"Model directory '{os.path.dirname(args.model)}' does not exist")

    if args.train:
        print(f"Training model with data from '{args.data}' and saving to '{args.model}'")
        train_model(args.data, args.model)
    elif args.eval:
        print(f"Evaluating model from '{args.model}' with data from '{args.data}'")
        evaluate_model(args.data, args.model)
    elif args.visualize:
        print(f"Running live visualizer with data from '{args.data}' and model from '{args.model}'")
        run_live_visualizer(args.data, args.model)

if __name__ == "__main__":
    main()