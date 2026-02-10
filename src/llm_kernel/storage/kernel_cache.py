"""
Kernel Cache Management.

Caches successful kernel generations for fast retrieval and stores metadata
about speedup, model used, and optimization level.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Any


class KernelCache:
    """
    Fast cache for successful kernel generations.

    Stores successful kernels with metadata (speedup, model, level) for quick
    retrieval and avoids re-generation of previously successful optimizations.

    Directory structure:
    - .kernel_cache/
      - index.json (in-memory index of all cached kernels)
      - {problem_id}/
        - {level}_kernels.json (all attempts for this level)
        - {level}_best.py (fastest kernel for this level)
    """

    def __init__(self, cache_dir: str = ".kernel_cache"):
        """
        Initialize KernelCache.

        Args:
            cache_dir: Directory to store cached kernels

        Raises:
            OSError: If cache directory creation fails
        """
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create cache directory {cache_dir}: {e}")

        self._index_path = self.cache_dir / "index.json"
        self._index: dict[str, Any] = self._load_index()

    def _load_index(self) -> dict[str, Any]:
        """
        Load the cache index from disk.

        Index format:
        {
            "last_updated": "ISO timestamp",
            "kernels": {
                "{problem_id}_{level}": {
                    "speedup": 1.5,
                    "model_name": "Qwen2.5-Coder-32B",
                    "timestamp": "ISO timestamp",
                    "file": "path/to/best.py"
                }
            }
        }

        Returns:
            Index dictionary, empty if doesn't exist

        Raises:
            json.JSONDecodeError: If index.json is malformed
        """
        if not self._index_path.exists():
            return {"last_updated": datetime.now().isoformat(), "kernels": {}}

        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse cache index: {e.msg}", e.doc, e.pos
            )

    def _save_index(self) -> None:
        """
        Save the cache index to disk.

        Raises:
            OSError: If file write fails
            TypeError: If index contains non-serializable objects
        """
        self._index["last_updated"] = datetime.now().isoformat()

        try:
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2, default=str)
        except (OSError, TypeError) as e:
            raise OSError(f"Failed to save cache index: {e}")

    def _get_problem_dir(self, problem_id: int) -> Path:
        """
        Get or create problem-specific cache directory.

        Args:
            problem_id: Problem identifier

        Returns:
            Path to problem cache directory

        Raises:
            OSError: If directory creation fails
        """
        problem_dir = self.cache_dir / str(problem_id)
        try:
            problem_dir.mkdir(parents=True, exist_ok=True)
            return problem_dir
        except OSError as e:
            raise OSError(f"Failed to create problem cache directory: {e}")

    def get(self, problem_id: int, level: int) -> Optional[str]:
        """
        Get cached kernel code for a problem at a specific level.

        Returns the fastest known kernel for the problem/level combination.

        Args:
            problem_id: Problem identifier
            level: Optimization level

        Returns:
            Kernel code as string, or None if not cached

        Raises:
            OSError: If file read fails
        """
        key = f"{problem_id}_{level}"
        entry = self._index.get("kernels", {}).get(key)

        if entry is None:
            return None

        cache_file = Path(entry["file"])
        if not cache_file.exists():
            # Entry is stale, remove it
            del self._index["kernels"][key]
            self._save_index()
            return None

        try:
            return cache_file.read_text(encoding="utf-8")
        except OSError as e:
            raise OSError(f"Failed to read cached kernel {cache_file}: {e}")

    def put(
        self,
        problem_id: int,
        level: int,
        code: str,
        speedup: float,
        model_name: str,
    ) -> None:
        """
        Cache a successful kernel generation.

        If a better kernel (higher speedup) already exists, keeps the better one.

        Args:
            problem_id: Problem identifier
            level: Optimization level
            code: Generated kernel code
            speedup: Performance improvement ratio (e.g., 1.5 for 50% faster)
            model_name: Name of model that generated the kernel

        Raises:
            OSError: If directory or file write fails
            ValueError: If speedup is invalid
        """
        if speedup < 0:
            raise ValueError(f"Invalid speedup value: {speedup}")

        key = f"{problem_id}_{level}"
        problem_dir = self._get_problem_dir(problem_id)

        # Check if we already have a better kernel
        existing = self._index.get("kernels", {}).get(key, {})
        if existing and existing.get("speedup", 0) > speedup:
            # Keep the existing, better kernel
            return

        # Determine filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_file = problem_dir / f"{level}_best.py"
        history_file = problem_dir / f"{level}_kernels.json"

        try:
            # Save the code as the best kernel
            best_file.write_text(code, encoding="utf-8")

            # Update index
            if "kernels" not in self._index:
                self._index["kernels"] = {}

            self._index["kernels"][key] = {
                "speedup": speedup,
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "file": str(best_file),
            }

            # Append to history
            history = []
            if history_file.exists():
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)

            history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "speedup": speedup,
                    "model_name": model_name,
                    "file": str(best_file),
                }
            )

            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            self._save_index()
        except (OSError, TypeError) as e:
            raise OSError(
                f"Failed to cache kernel for problem {problem_id} level {level}: {e}"
            )

    def get_best(self, problem_id: int, level: int) -> Optional[str]:
        """
        Get the fastest cached kernel for a problem/level.

        Convenience alias for get() - returns the best known kernel.

        Args:
            problem_id: Problem identifier
            level: Optimization level

        Returns:
            Best kernel code or None if not cached
        """
        return self.get(problem_id, level)

    def list_cached(self) -> list[dict[str, Any]]:
        """
        List all cached kernels with metadata.

        Returns:
            List of cache entry dicts with keys: problem_id, level, speedup,
            model_name, timestamp, file

        Raises:
            OSError: If directory reading fails
        """
        try:
            cached = []
            for key, entry in self._index.get("kernels", {}).items():
                parts = key.split("_")
                if len(parts) != 2:
                    continue

                problem_id = int(parts[0])
                level = int(parts[1])

                cached.append(
                    {
                        "problem_id": problem_id,
                        "level": level,
                        "speedup": entry.get("speedup", 0),
                        "model_name": entry.get("model_name", "unknown"),
                        "timestamp": entry.get("timestamp", ""),
                        "file": entry.get("file", ""),
                    }
                )

            return sorted(cached, key=lambda x: (x["problem_id"], x["level"]))
        except (ValueError, KeyError) as e:
            raise OSError(f"Failed to list cached kernels: {e}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with keys: total_cached, avg_speedup, by_model, last_updated
        """
        kernels = self._index.get("kernels", {})
        if not kernels:
            return {
                "total_cached": 0,
                "avg_speedup": 0.0,
                "by_model": {},
                "last_updated": self._index.get("last_updated", ""),
            }

        speedups = [entry.get("speedup", 0) for entry in kernels.values()]
        model_counts = {}
        for entry in kernels.values():
            model = entry.get("model_name", "unknown")
            model_counts[model] = model_counts.get(model, 0) + 1

        return {
            "total_cached": len(kernels),
            "avg_speedup": sum(speedups) / len(speedups) if speedups else 0.0,
            "by_model": model_counts,
            "last_updated": self._index.get("last_updated", ""),
        }

    def clear(self) -> None:
        """
        Clear all cached kernels.

        Raises:
            OSError: If file deletion fails
        """
        try:
            # Remove all problem directories
            for problem_dir in self.cache_dir.iterdir():
                if problem_dir.is_dir() and problem_dir.name != "__pycache__":
                    for file in problem_dir.iterdir():
                        file.unlink()
                    problem_dir.rmdir()

            # Reset index
            self._index = {"last_updated": datetime.now().isoformat(), "kernels": {}}
            self._save_index()
        except OSError as e:
            raise OSError(f"Failed to clear cache: {e}")

    def remove(self, problem_id: int, level: int) -> None:
        """
        Remove specific cached kernel.

        Args:
            problem_id: Problem identifier
            level: Optimization level

        Raises:
            OSError: If file deletion fails
        """
        key = f"{problem_id}_{level}"
        entry = self._index.get("kernels", {}).get(key)

        if entry is None:
            return

        try:
            cache_file = Path(entry["file"])
            if cache_file.exists():
                cache_file.unlink()

            del self._index["kernels"][key]
            self._save_index()
        except (OSError, KeyError) as e:
            raise OSError(f"Failed to remove cached kernel {key}: {e}")
