"""Application entrypoint configuring FastAPI server and CLI commands."""

import os
import sys
import argparse
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

from src.core import ChurnPipeline
from src.api import create_router
from src.models import CustomerRecord

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "models/metrics.json")
HOP = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

pipeline = ChurnPipeline(model_path=MODEL_PATH, metrics_path=METRICS_PATH)

app = FastAPI(
    title="Simple Churn Predictor",
    description="Lightweight machine learning service for customer churn risk prediction.",
    version="1.0.0"
)

router = create_router(pipeline)
app.include_router(router)


def run_cli():
    parser = argparse.ArgumentParser(description="Simple Churn Predictor CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    train_parser = subparsers.add_parser("train", help="Train model with synthetic or provided CSV data")
    train_parser.add_argument("--data", type=str, required=True, help="Path to CSV file containing historical training data")

    args = parser.parse_args()

    if args.command == "train":
        import pandas as pd
        if not os.path.exists(args.data):
            print(f"Error: File {args.data} not found.", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(args.data)
        records = [CustomerRecord(**row) for _, row in df.iterrows()]
        print(f"Training model on {len(records)} records...")
        metrics = pipeline.train(records)
        print("Training complete! Metrics:")
        print(metrics.model_dump_json(indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["train"]:
        run_cli()
    else:
        uvicorn.run("src.main:app", host=HOP, port=PORT, reload=True)
