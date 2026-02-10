"""
Run Storage Management.

Provides functionality to create, manage, and persist kernel generation runs with
automatic directory structure creation, configuration management, and result storage.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, asdict

import yaml


@dataclass
class GenerationMetadata:
    """Metadata associated with a generated kernel."""
    level: int
    problem_id: int
    attempt: int
    model_name: str
    timestamp: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    finish_reason: Optional[str] = None
    prompt_hash: Optional[str] = None


class RunStore:
    """
    Manages run directories and persists kernel generations and results.

    Creates organized directory structure:
    - runs/
      - run_name/
        - config.yaml (frozen run config)
        - generations/ (generated kernel code)
        - results/ (evaluation results)
    """

    def __init__(self, base_dir: str = "runs"):
        """
        Initialize RunStore.

        Args:
            base_dir: Root directory for all runs

        Raises:
            ValueError: If base_dir is invalid
        """
        self.base_dir = Path(base_dir)
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Failed to create base directory {base_dir}: {e}")

    def create_run(
        self,
        run_name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Create a new run directory with optional configuration.

        Auto-generates run_name using timestamp if not provided.

        Args:
            run_name: Name for the run. If None, generates "run_YYYYMMDD_HHMMSS"
            config: Optional configuration dict to save as config.yaml

        Returns:
            The run_name (either provided or auto-generated)

        Raises:
            OSError: If directory creation fails
            yaml.YAMLError: If config serialization fails
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        run_dir = self.base_dir / run_name
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "generations").mkdir(exist_ok=True)
            (run_dir / "results").mkdir(exist_ok=True)

            if config:
                config_path = run_dir / "config.yaml"
                config_path.write_text(yaml.dump(config, default_flow_style=False))

            return run_name
        except OSError as e:
            raise OSError(f"Failed to create run directory {run_dir}: {e}")

    def save_generation(
        self,
        run_name: str,
        level: int,
        problem_id: int,
        attempt: int,
        code: str,
        metadata: dict[str, Any],
    ) -> Path:
        """
        Save a generated kernel code with metadata.

        Creates files:
        - {level}_{problem_id}_{attempt}.py (kernel code)
        - {level}_{problem_id}_{attempt}.json (metadata)

        Args:
            run_name: Name of the run
            level: Problem difficulty level
            problem_id: Unique problem identifier
            attempt: Attempt number for this problem
            code: Generated kernel code
            metadata: Dictionary with generation metadata (model_name, timestamp, etc.)

        Returns:
            Path to saved code file

        Raises:
            ValueError: If run_name doesn't exist
            OSError: If file write fails
            json.JSONDecodeError: If metadata serialization fails
        """
        run_dir = self.base_dir / run_name
        if not run_dir.exists():
            raise ValueError(f"Run '{run_name}' does not exist")

        gen_dir = run_dir / "generations"
        base_filename = f"{level}_{problem_id}_{attempt}"

        try:
            # Save code file
            code_path = gen_dir / f"{base_filename}.py"
            code_path.write_text(code, encoding="utf-8")

            # Save metadata
            metadata_path = gen_dir / f"{base_filename}.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)

            return code_path
        except (OSError, TypeError) as e:
            raise OSError(
                f"Failed to save generation for {run_name}/{level}/{problem_id}/{attempt}: {e}"
            )

    def save_result(
        self,
        run_name: str,
        level: int,
        problem_id: int,
        result: dict[str, Any],
    ) -> Path:
        """
        Save evaluation result for a problem.

        Saves to: results/{level}_{problem_id}_result.json

        Args:
            run_name: Name of the run
            level: Problem difficulty level
            problem_id: Unique problem identifier
            result: Result dictionary (should include passed, speedup, error, etc.)

        Returns:
            Path to saved result file

        Raises:
            ValueError: If run_name doesn't exist
            OSError: If file write fails
            TypeError: If result contains non-serializable objects
        """
        run_dir = self.base_dir / run_name
        if not run_dir.exists():
            raise ValueError(f"Run '{run_name}' does not exist")

        results_dir = run_dir / "results"
        result_filename = f"{level}_{problem_id}_result.json"
        result_path = results_dir / result_filename

        try:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            return result_path
        except (OSError, TypeError) as e:
            raise OSError(
                f"Failed to save result for {run_name}/{level}/{problem_id}: {e}"
            )

    def load_results(self, run_name: str) -> list[dict[str, Any]]:
        """
        Load all evaluation results for a run.

        Reads all result files from results/ directory.

        Args:
            run_name: Name of the run

        Returns:
            List of result dictionaries, sorted by (level, problem_id)

        Raises:
            ValueError: If run_name doesn't exist
            json.JSONDecodeError: If result file is malformed
        """
        run_dir = self.base_dir / run_name
        if not run_dir.exists():
            raise ValueError(f"Run '{run_name}' does not exist")

        results_dir = run_dir / "results"
        if not results_dir.exists():
            return []

        results = []
        try:
            for result_file in sorted(results_dir.glob("*_result.json")):
                with open(result_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
                    results.append(result)
            return results
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse result file: {e.msg}", e.doc, e.pos
            )

    def list_runs(self) -> list[dict[str, Any]]:
        """
        List all runs with basic information.

        Returns:
            List of run info dicts with keys: name, created, config_exists, num_results

        Raises:
            OSError: If directory reading fails
        """
        try:
            runs = []
            for run_dir in sorted(self.base_dir.iterdir()):
                if not run_dir.is_dir():
                    continue

                config_path = run_dir / "config.yaml"
                results_dir = run_dir / "results"
                num_results = (
                    len(list(results_dir.glob("*_result.json")))
                    if results_dir.exists()
                    else 0
                )

                # Extract timestamp from directory name if possible
                created = run_dir.stat().st_mtime
                created_str = datetime.fromtimestamp(created).isoformat()

                runs.append(
                    {
                        "name": run_dir.name,
                        "created": created_str,
                        "config_exists": config_path.exists(),
                        "num_results": num_results,
                    }
                )

            return runs
        except OSError as e:
            raise OSError(f"Failed to list runs: {e}")

    def get_run_config(self, run_name: str) -> Optional[dict[str, Any]]:
        """
        Load frozen configuration for a run.

        Args:
            run_name: Name of the run

        Returns:
            Config dictionary or None if config.yaml doesn't exist

        Raises:
            ValueError: If run_name doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        run_dir = self.base_dir / run_name
        if not run_dir.exists():
            raise ValueError(f"Run '{run_name}' does not exist")

        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse config.yaml for {run_name}: {e}")

    def get_run_generations(
        self, run_name: str, level: Optional[int] = None, problem_id: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """
        Load generations for a run, optionally filtered by level and problem_id.

        Args:
            run_name: Name of the run
            level: Optional difficulty level filter
            problem_id: Optional problem ID filter

        Returns:
            List of generation dicts with keys: level, problem_id, attempt, code_path, metadata

        Raises:
            ValueError: If run_name doesn't exist
            json.JSONDecodeError: If metadata file is malformed
        """
        run_dir = self.base_dir / run_name
        if not run_dir.exists():
            raise ValueError(f"Run '{run_name}' does not exist")

        gen_dir = run_dir / "generations"
        if not gen_dir.exists():
            return []

        generations = []
        try:
            for metadata_file in sorted(gen_dir.glob("*.json")):
                # Parse filename: {level}_{problem_id}_{attempt}.json
                parts = metadata_file.stem.split("_")
                if len(parts) < 3:
                    continue

                gen_level = int(parts[0])
                gen_problem_id = int(parts[1])
                gen_attempt = int(parts[2])

                # Apply filters
                if level is not None and gen_level != level:
                    continue
                if problem_id is not None and gen_problem_id != problem_id:
                    continue

                # Load metadata
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                # Find corresponding code file
                code_file = metadata_file.with_suffix(".py")
                code = code_file.read_text(encoding="utf-8") if code_file.exists() else ""

                generations.append(
                    {
                        "level": gen_level,
                        "problem_id": gen_problem_id,
                        "attempt": gen_attempt,
                        "code_path": str(code_file),
                        "code": code,
                        "metadata": metadata,
                    }
                )

            return generations
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse metadata file in {run_name}: {e.msg}", e.doc, e.pos
            )
