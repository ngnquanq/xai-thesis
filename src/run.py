# 

import click
from datetime import datetime as dt
import os
from typing import Optional

from zenml.client import Client
from logging import getLogger
from pipelines import *
from utils import *

logger = getLogger(__name__)


@click.command(
    help="""
ZenML E2E project for Churn prediction CLI v0.0.1.

Run the ZenML E2E project for Churn prediction model training pipeline with various
options.

Examples:


  \b
  # Run the pipeline with default options
  python run.py
               
  \b
  # Run the pipeline without cache
  python run.py --no-cache

  \b
  # Run the pipeline without Hyperparameter tuning
  python run.py --no-hp-tuning

  \b
  # Run the pipeline without NA drop and normalization, 
  # but dropping columns [A,B,C] and keeping 10% of dataset 
  # as test set.
  python run.py --no-drop-na --no-normalize --drop-columns A,B,C --test-size 0.1

  \b
  # Run the pipeline with Quality Gate for accuracy set at 90% for train set 
  # and 85% for test set. If any of accuracies will be lower - pipeline will fail.
  python run.py --min-train-accuracy 0.9 --min-test-accuracy 0.85 --fail-on-accuracy-quality-gates


"""
)

@click.option(
    "--test-size",
    default=0.2,
    type=click.FloatRange(0.0, 1.0),
    help="Proportion of the dataset to include in the test split.",
)
@click.option(
    "--min-train-accuracy",
    default=0.8,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum training accuracy to pass to the model evaluator.",
)
@click.option(
    "--min-test-accuracy",
    default=0.8,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum test accuracy to pass to the model evaluator.",
)
@click.option(
    "--fail-on-accuracy-quality-gates",
    is_flag=True,
    default=False,
    help="Whether to fail the pipeline run if the model evaluation step "
    "finds that the model is not accurate enough.",
)
@click.option(
    "--only-inference",
    is_flag=True,
    default=False,
    help="Whether to run only inference pipeline.",
)
@click.option(
    "--synthesize-data",
    is_flag=True,
    help="Flag to indicate if synthetic data should be generated.",
)
@click.option(
    "--model",
    type=click.Choice(['gaussian', 'ctgan', 'copulagan'], case_sensitive=False),
    help="Model to use for data synthesis.",
)
@click.option(
    "--num-rows",
    default=100,
    type=int,
    help="Number of rows of synthetic data to generate.",
)
def main(
    no_cache: bool = False,
    no_drop_na: bool = False,
    no_normalize: bool = False,
    drop_columns: Optional[str] = None,
    test_size: float = 0.2,
    min_train_accuracy: float = 0.8,
    min_test_accuracy: float = 0.8,
    fail_on_accuracy_quality_gates: bool = False,
    only_inference: bool = False,
    synthesize_data: bool = False,
    model: Optional[str] = None,
    num_rows: int = 100,
):
    """Main entry point for the pipeline execution.
    ...
    """
    # Check if synthetic data generation is requested
    if synthesize_data:
        if model is None:
            logger.error("Model must be specified when synthesizing data.")
            return
        logger.info(f"Generating {num_rows} rows of synthetic data using the {model} model.")
        # Call your data synthesis function here
        if model == 'gaussian':
            maker = Maker(GaussianCopulaSynthesizer)
            synthetic_data = maker.create_data(num_rows=num_rows)
            print(f"Generating {num_rows} rows of synthetic data using the {model} model in data/synthetic_data.csv")
            maker.save_synthesizer(filename='GaussianCopular_synthesizer.pkl')
        elif model == 'copulagan':
            maker = Maker(CopulaGANSynthesizer)
            synthetic_data = maker.create_data(num_rows=num_rows)
            print(f"Generating {num_rows} rows of synthetic data using the {model} model in data/synthetic_data.csv")
            maker.save_synthesizer(filename='CopulaGAN_synthesizer.pkl')



if __name__ == "__main__":
    main()
