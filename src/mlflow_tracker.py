"""
MLflow Tracking Integration for Spotify Track Analytics

This module provides utilities for tracking ML experiments using MLflow with SQLite backend.
Implements Phase 3 (Section 8.3) from the ML Pipeline Improvements Specification.

Features:
- Automatic experiment creation and tracking
- Parameter, metric, and artifact logging
- Model registration with metadata
- SQLite backend configuration for local tracking
"""

import os
import mlflow
import mlflow.xgboost
from typing import Dict, Any, Optional
import json
from datetime import datetime


class MLflowTracker:
    """
    Wrapper class for MLflow experiment tracking.

    Provides simplified interface for logging parameters, metrics, and artifacts
    while maintaining consistent experiment organization.
    """

    def __init__(
        self,
        experiment_name: str = "spotify_popularity_prediction",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (defaults to ./mlruns with SQLite)
            artifact_location: Location to store artifacts (defaults to ./mlartifacts)
        """
        self.experiment_name = experiment_name

        # Set tracking URI (default: SQLite backend in mlruns/)
        if tracking_uri is None:
            # Create mlruns directory if it doesn't exist
            os.makedirs("mlruns", exist_ok=True)
            tracking_uri = "sqlite:///mlruns/mlflow.db"

        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                if artifact_location is None:
                    artifact_location = os.path.abspath("mlartifacts")
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
            else:
                experiment_id = experiment.experiment_id

            mlflow.set_experiment(experiment_name)
            self.experiment_id = experiment_id
            print(f"✅ MLflow experiment '{experiment_name}' initialized (ID: {experiment_id})")
            print(f"   Tracking URI: {tracking_uri}")
        except Exception as e:
            print(f"⚠️  Error setting up MLflow experiment: {e}")
            self.experiment_id = None

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run
            tags: Optional dictionary of tags

        Returns:
            MLflow run object
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if tags is None:
            tags = {}

        # Add default tags
        tags['pipeline_version'] = 'improved_v1.0'

        run = mlflow.start_run(run_name=run_name, tags=tags)
        print(f"🚀 Started MLflow run: {run_name} (ID: {run.info.run_id})")
        return run

    def log_params(self, params: Dict[str, Any]):
        """
        Log model hyperparameters.

        Args:
            params: Dictionary of parameters
        """
        try:
            # MLflow requires params to be strings, numbers, or booleans
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    mlflow.log_param(key, json.dumps(value))
                else:
                    mlflow.log_param(key, value)
            print(f"✅ Logged {len(params)} parameters")
        except Exception as e:
            print(f"⚠️  Error logging parameters: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log evaluation metrics.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number for tracking over iterations
        """
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            print(f"✅ Logged {len(metrics)} metrics")
        except Exception as e:
            print(f"⚠️  Error logging metrics: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a local file as an artifact.

        Args:
            local_path: Path to local file
            artifact_path: Optional subdirectory within artifact store
        """
        try:
            if os.path.exists(local_path):
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
                print(f"✅ Logged artifact: {local_path}")
            else:
                print(f"⚠️  Artifact not found: {local_path}")
        except Exception as e:
            print(f"⚠️  Error logging artifact: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log all files in a directory as artifacts.

        Args:
            local_dir: Path to local directory
            artifact_path: Optional subdirectory within artifact store
        """
        try:
            if os.path.exists(local_dir):
                mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
                print(f"✅ Logged artifacts from: {local_dir}")
            else:
                print(f"⚠️  Directory not found: {local_dir}")
        except Exception as e:
            print(f"⚠️  Error logging artifacts: {e}")

    def log_figure(self, figure, artifact_file: str):
        """
        Log a matplotlib figure.

        Args:
            figure: Matplotlib figure object
            artifact_file: Filename for the artifact (e.g., 'plot.png')
        """
        try:
            mlflow.log_figure(figure, artifact_file)
            print(f"✅ Logged figure: {artifact_file}")
        except Exception as e:
            print(f"⚠️  Error logging figure: {e}")

    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Log an XGBoost model.

        Args:
            model: Trained XGBoost model
            artifact_path: Path within artifact store
            registered_model_name: Name for model registry
            **kwargs: Additional arguments for mlflow.xgboost.log_model
        """
        try:
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                **kwargs
            )
            print(f"✅ Logged XGBoost model to: {artifact_path}")
            if registered_model_name:
                print(f"   Registered as: {registered_model_name}")
        except Exception as e:
            print(f"⚠️  Error logging model: {e}")

    def log_dict(self, dictionary: Dict, artifact_file: str):
        """
        Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Filename for the artifact (e.g., 'metadata.json')
        """
        try:
            mlflow.log_dict(dictionary, artifact_file)
            print(f"✅ Logged dictionary: {artifact_file}")
        except Exception as e:
            print(f"⚠️  Error logging dictionary: {e}")

    def log_text(self, text: str, artifact_file: str):
        """
        Log text as an artifact.

        Args:
            text: Text content
            artifact_file: Filename for the artifact
        """
        try:
            mlflow.log_text(text, artifact_file)
            print(f"✅ Logged text: {artifact_file}")
        except Exception as e:
            print(f"⚠️  Error logging text: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for the current run.

        Args:
            tags: Dictionary of tag name -> value
        """
        try:
            mlflow.set_tags(tags)
            print(f"✅ Set {len(tags)} tags")
        except Exception as e:
            print(f"⚠️  Error setting tags: {e}")

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        try:
            mlflow.end_run(status=status)
            print(f"✅ MLflow run ended with status: {status}")
        except Exception as e:
            print(f"⚠️  Error ending run: {e}")

    @staticmethod
    def get_run_info(run_id: str) -> Dict:
        """
        Get information about a specific run.

        Args:
            run_id: MLflow run ID

        Returns:
            Dictionary with run information
        """
        try:
            run = mlflow.get_run(run_id)
            return {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'artifact_uri': run.info.artifact_uri,
                'params': run.data.params,
                'metrics': run.data.metrics,
                'tags': run.data.tags
            }
        except Exception as e:
            print(f"⚠️  Error getting run info: {e}")
            return {}

    @staticmethod
    def search_runs(
        experiment_names: Optional[list] = None,
        filter_string: str = "",
        order_by: Optional[list] = None
    ):
        """
        Search for runs across experiments.

        Args:
            experiment_names: List of experiment names to search
            filter_string: Filter query (e.g., "metrics.rmse < 10")
            order_by: List of columns to order by

        Returns:
            Pandas DataFrame with run information
        """
        try:
            if experiment_names is None:
                experiment_names = ["spotify_popularity_prediction"]

            experiments = [mlflow.get_experiment_by_name(name) for name in experiment_names]
            experiment_ids = [exp.experiment_id for exp in experiments if exp is not None]

            if not experiment_ids:
                print("⚠️  No experiments found")
                return None

            runs = mlflow.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                order_by=order_by
            )

            print(f"✅ Found {len(runs)} runs")
            return runs
        except Exception as e:
            print(f"⚠️  Error searching runs: {e}")
            return None


def track_experiment(
    experiment_name: str,
    model,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Dict[str, str] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    registered_model_name: Optional[str] = None
):
    """
    Convenience function to track a complete experiment.

    Args:
        experiment_name: Name of the experiment
        model: Trained model
        params: Model hyperparameters
        metrics: Evaluation metrics
        artifacts: Dictionary of artifact_type -> path
        run_name: Optional run name
        tags: Optional tags
        registered_model_name: Optional name for model registry

    Returns:
        MLflow run ID
    """
    tracker = MLflowTracker(experiment_name=experiment_name)

    run = tracker.start_run(run_name=run_name, tags=tags)

    try:
        # Log parameters and metrics
        tracker.log_params(params)
        tracker.log_metrics(metrics)

        # Log model
        tracker.log_model(
            model,
            registered_model_name=registered_model_name
        )

        # Log artifacts if provided
        if artifacts:
            for artifact_type, path in artifacts.items():
                if os.path.isdir(path):
                    tracker.log_artifacts(path, artifact_path=artifact_type)
                else:
                    tracker.log_artifact(path, artifact_path=artifact_type)

        tracker.end_run(status="FINISHED")
        print(f"\n✅ Experiment tracked successfully!")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   View in UI: mlflow ui --backend-store-uri {tracker.tracking_uri}")

        return run.info.run_id

    except Exception as e:
        print(f"❌ Error tracking experiment: {e}")
        tracker.end_run(status="FAILED")
        return None
