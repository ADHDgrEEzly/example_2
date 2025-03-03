import argparse
from pathlib import Path

from config import DATA_DIR, TARGET_COLUMN
from pipeline import run_pipeline
from utils.logger import setup_logger
from utils.model_io import ModelSerializer

logger = setup_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data Science Pipeline CLI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_DIR / "data.csv"),
        help="Path to training data CSV"
    )
    train_parser.add_argument(
        "--target",
        type=str,
        default=TARGET_COLUMN,
        help="Name of target column"
    )
    train_parser.add_argument(
        "--model-name",
        type=str,
        default="logistic_model",
        help="Name for the saved model"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to prediction data CSV"
    )
    predict_parser.add_argument(
        "--model-name",
        type=str,
        default="logistic_model",
        help="Model name to use for prediction"
    )
    predict_parser.add_argument(
        "--model-version",
        type=str,
        help="Specific model version to use"
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions CSV"
    )
    
    # List models command
    list_parser = subparsers.add_parser(
        "list-models",
        help="List available models"
    )
    list_parser.add_argument(
        "--model-name",
        type=str,
        help="Show versions for specific model"
    )
    
    return parser.parse_args()

def main():
    """Main CLI entry point."""
    args = parse_args()
    
    if args.command == "train":
        logger.info("Starting model training")
        accuracy, metrics = run_pipeline(
            data_path=args.data,
            target_column=args.target
        )
        
        # Save model with metrics
        serializer = ModelSerializer()
        serializer.save_model(
            model=None,  # TODO: Get model from pipeline
            model_name=args.model_name,
            metadata={"metrics": metrics}
        )
        
    elif args.command == "predict":
        logger.info("Loading model for prediction")
        serializer = ModelSerializer()
        model, metadata = serializer.load_model(
            args.model_name,
            args.model_version
        )
        
        # TODO: Implement prediction logic
        logger.info("Making predictions")
        
    elif args.command == "list-models":
        serializer = ModelSerializer()
        if args.model_name:
            versions = serializer.list_versions(args.model_name)
            print(f"\nVersions for {args.model_name}:")
            for version in versions:
                print(f"  - {version}")
        else:
            # List all model files
            print("\nAvailable models:")
            for path in Path(serializer.model_dir).glob("**/model.joblib"):
                print(f"  - {path.parent.name}")
    
    else:
        logger.error("Invalid command")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())