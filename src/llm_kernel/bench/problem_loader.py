"""
KernelBench Problem Loader.

Loads Triton kernel optimization problems from HuggingFace datasets
or local KernelBench repository clone.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


@dataclass
class Problem:
    """Represents a single KernelBench problem."""
    problem_id: int
    level: int
    name: str
    pytorch_code: str
    operation_description: str

    @property
    def full_name(self) -> str:
        """Get full problem identifier."""
        return f"L{self.level}_P{self.problem_id}_{self.name}"


class ProblemLoader:
    """
    Load KernelBench problems from HuggingFace or local repository.

    Supports two data sources:
    1. HuggingFace: ScalingIntelligence/KernelBench dataset
    2. Local: Clone of KernelBench GitHub repository

    The loader automatically extracts operation descriptions from code docstrings
    and file names for context-aware kernel generation.
    """

    def __init__(self, dataset_src: str = "huggingface", local_path: Optional[Path] = None):
        """
        Initialize problem loader.

        Args:
            dataset_src: Data source - "huggingface" or "local"
            local_path: Path to local KernelBench clone (required for "local" mode)

        Raises:
            ValueError: If dataset_src is invalid or required dependencies missing
        """
        self.dataset_src = dataset_src.lower()
        self.local_path = local_path

        if self.dataset_src == "huggingface":
            if not HF_DATASETS_AVAILABLE:
                raise ValueError(
                    "HuggingFace datasets library not installed. "
                    "Install with: pip install datasets"
                )
            self._dataset = None  # Lazy load

        elif self.dataset_src == "local":
            if local_path is None:
                raise ValueError("local_path must be provided for local dataset source")
            if not local_path.exists():
                raise ValueError(f"Local path does not exist: {local_path}")
            self.local_path = Path(local_path)

        else:
            raise ValueError(
                f"Invalid dataset_src: {dataset_src}. Must be 'huggingface' or 'local'"
            )

    def _lazy_load_hf_dataset(self):
        """Lazy load HuggingFace dataset."""
        if self._dataset is None and self.dataset_src == "huggingface":
            self._dataset = load_dataset("ScalingIntelligence/KernelBench")

    def _extract_operation_description(self, code: str, filename: str = "") -> str:
        """
        Extract operation description from code or filename.

        Attempts to extract description from:
        1. Class docstring (if available)
        2. Module docstring
        3. Filename parsing

        Args:
            code: PyTorch reference code
            filename: Optional filename for fallback extraction

        Returns:
            Extracted operation description
        """
        # Try to extract from class docstring
        class_docstring_match = re.search(
            r'class\s+\w+.*?:\s*"""(.*?)"""',
            code,
            re.DOTALL
        )
        if class_docstring_match:
            docstring = class_docstring_match.group(1).strip()
            # Get first line as description
            first_line = docstring.split('\n')[0].strip()
            if first_line:
                return first_line

        # Try to extract from module docstring
        module_docstring_match = re.search(
            r'^"""(.*?)"""',
            code,
            re.DOTALL | re.MULTILINE
        )
        if module_docstring_match:
            docstring = module_docstring_match.group(1).strip()
            first_line = docstring.split('\n')[0].strip()
            if first_line:
                return first_line

        # Fallback: parse filename
        if filename:
            # Remove extension and convert underscores to spaces
            name = Path(filename).stem.replace('_', ' ')
            # Capitalize first letter of each word
            return name.title()

        # Final fallback
        return "Unknown operation"

    def _load_problem_from_hf(self, level: int, problem_id: int) -> Optional[Problem]:
        """
        Load problem from HuggingFace dataset.

        Args:
            level: Problem difficulty level (1-4)
            problem_id: Problem ID within level

        Returns:
            Problem instance or None if not found
        """
        self._lazy_load_hf_dataset()

        # HuggingFace KernelBench dataset structure:
        # Splits: level_1, level_2, level_3, level_4
        # Fields: code, level, name, problem_id
        split_name = f"level_{level}"

        if split_name not in self._dataset:
            # Fallback: try common split names
            for alt_split in ["train", "test", "validation", "all"]:
                if alt_split in self._dataset:
                    split_name = alt_split
                    break
            else:
                return None

        dataset = self._dataset[split_name]

        # Filter by problem_id
        for item in dataset:
            item_problem_id = item.get("problem_id")
            # Handle both int and str problem_id
            if str(item_problem_id) == str(problem_id) or item_problem_id == problem_id:
                # KernelBench uses 'code' field, not 'pytorch_code'
                pytorch_code = item.get("code", item.get("pytorch_code", ""))
                name = item.get("name", f"problem_{problem_id}")

                description = self._extract_operation_description(
                    pytorch_code,
                    filename=name
                )

                return Problem(
                    problem_id=int(item_problem_id) if item_problem_id else problem_id,
                    level=level,
                    name=name,
                    pytorch_code=pytorch_code,
                    operation_description=description
                )

        return None

    def _load_problem_from_local(self, level: int, problem_id: int) -> Optional[Problem]:
        """
        Load problem from local KernelBench clone.

        Expected structure:
        kernelbench/
          level1/
            problem_001_add.py
            problem_002_mul.py
          level2/
            ...

        Args:
            level: Problem difficulty level (1-4)
            problem_id: Problem ID within level

        Returns:
            Problem instance or None if not found
        """
        # Search for problem file
        level_dir = self.local_path / f"level{level}"
        if not level_dir.exists():
            # Try alternative naming
            level_dir = self.local_path / f"Level{level}"
            if not level_dir.exists():
                return None

        # Find file matching problem_id
        # Expected pattern: problem_{id:03d}_name.py
        pattern = f"problem_{problem_id:03d}_*.py"
        matching_files = list(level_dir.glob(pattern))

        if not matching_files:
            # Try alternative pattern
            pattern = f"{problem_id:03d}_*.py"
            matching_files = list(level_dir.glob(pattern))

        if not matching_files:
            return None

        problem_file = matching_files[0]

        # Read code
        with open(problem_file, "r") as f:
            pytorch_code = f.read()

        # Extract name from filename
        name = problem_file.stem
        # Remove problem_id prefix if present
        name = re.sub(r'problem_\d+_', '', name)
        name = re.sub(r'^\d+_', '', name)

        description = self._extract_operation_description(
            pytorch_code,
            filename=problem_file.name
        )

        return Problem(
            problem_id=problem_id,
            level=level,
            name=name,
            pytorch_code=pytorch_code,
            operation_description=description
        )

    def load_problem(self, level: int, problem_id: int) -> Problem:
        """
        Load a single problem by level and ID.

        Args:
            level: Problem difficulty level (1-4)
            problem_id: Problem ID within level

        Returns:
            Problem instance

        Raises:
            ValueError: If problem not found or invalid parameters
        """
        if level < 1 or level > 4:
            raise ValueError(f"Invalid level: {level}. Must be 1-4")

        if problem_id < 0:
            raise ValueError(f"Invalid problem_id: {problem_id}. Must be >= 0")

        if self.dataset_src == "huggingface":
            problem = self._load_problem_from_hf(level, problem_id)
        else:
            problem = self._load_problem_from_local(level, problem_id)

        if problem is None:
            raise ValueError(
                f"Problem not found: level={level}, problem_id={problem_id}"
            )

        return problem

    def load_level(self, level: int) -> List[Problem]:
        """
        Load all problems from a specific level.

        Args:
            level: Problem difficulty level (1-4)

        Returns:
            List of Problem instances

        Raises:
            ValueError: If level is invalid
        """
        if level < 1 or level > 4:
            raise ValueError(f"Invalid level: {level}. Must be 1-4")

        problems = []

        if self.dataset_src == "huggingface":
            self._lazy_load_hf_dataset()
            # KernelBench splits: level_1, level_2, level_3, level_4
            split_name = f"level_{level}"
            if split_name not in self._dataset:
                for alt_split in ["train", "test", "validation", "all"]:
                    if alt_split in self._dataset:
                        split_name = alt_split
                        break

            if split_name not in self._dataset:
                return problems

            dataset = self._dataset[split_name]

            for item in dataset:
                # For level-specific splits, all items are from that level
                item_level = item.get("level", level)
                if item_level == level or split_name == f"level_{level}":
                    problem_id = item.get("problem_id")
                    # KernelBench uses 'code' field
                    pytorch_code = item.get("code", item.get("pytorch_code", ""))
                    name = item.get("name", f"problem_{problem_id}")

                    description = self._extract_operation_description(
                        pytorch_code,
                        filename=name
                    )

                    problems.append(
                        Problem(
                            problem_id=problem_id,
                            level=level,
                            name=name,
                            pytorch_code=pytorch_code,
                            operation_description=description
                        )
                    )

        else:
            # Local loading
            level_dir = self.local_path / f"level{level}"
            if not level_dir.exists():
                level_dir = self.local_path / f"Level{level}"

            if level_dir.exists():
                for problem_file in sorted(level_dir.glob("*.py")):
                    # Extract problem_id from filename
                    match = re.search(r'(\d+)', problem_file.stem)
                    if match:
                        problem_id = int(match.group(1))

                        with open(problem_file, "r") as f:
                            pytorch_code = f.read()

                        name = problem_file.stem
                        name = re.sub(r'problem_\d+_', '', name)
                        name = re.sub(r'^\d+_', '', name)

                        description = self._extract_operation_description(
                            pytorch_code,
                            filename=problem_file.name
                        )

                        problems.append(
                            Problem(
                                problem_id=problem_id,
                                level=level,
                                name=name,
                                pytorch_code=pytorch_code,
                                operation_description=description
                            )
                        )

        return problems

    def load_all(self) -> List[Problem]:
        """
        Load all problems from all levels.

        Returns:
            List of all Problem instances sorted by level and problem_id
        """
        all_problems = []

        for level in range(1, 5):
            try:
                level_problems = self.load_level(level)
                all_problems.extend(level_problems)
            except Exception:
                # Skip levels that don't exist
                continue

        # Sort by level, then problem_id
        all_problems.sort(key=lambda p: (p.level, p.problem_id))

        return all_problems

    def get_level_count(self, level: int) -> int:
        """
        Get number of problems in a level.

        Args:
            level: Problem difficulty level (1-4)

        Returns:
            Number of problems in level
        """
        return len(self.load_level(level))

    def get_total_count(self) -> int:
        """
        Get total number of problems across all levels.

        Returns:
            Total problem count
        """
        return len(self.load_all())
