"""
Response Parser for LLM-Generated Triton Kernels.

Extracts and validates Triton code from LLM responses, ensuring
KernelBench compatibility with required components.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result from parsing LLM response."""
    code: str
    parse_errors: List[str] = field(default_factory=list)
    is_valid: bool = False
    missing_components: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ResponseParser:
    """
    Parses LLM responses to extract valid Triton kernel code.

    Validates KernelBench requirements:
    - Model class with forward method
    - @triton.jit decorated kernel
    - get_inputs() function
    - get_init_inputs() function
    """

    def __init__(self):
        """Initialize parser with regex patterns."""
        # Pattern for code blocks
        self.code_block_pattern = re.compile(
            r"```(?:python)?\s*\n(.*?)```",
            re.DOTALL | re.MULTILINE
        )

        # Pattern for imports
        self.import_pattern = re.compile(
            r"^import\s+triton|^from\s+triton",
            re.MULTILINE
        )

        # Pattern for Model class
        self.model_class_pattern = re.compile(
            r"class\s+Model\s*\(",
            re.MULTILINE
        )

        # Pattern for @triton.jit
        self.triton_jit_pattern = re.compile(
            r"@triton\.jit",
            re.MULTILINE
        )

        # Pattern for get_inputs function
        self.get_inputs_pattern = re.compile(
            r"def\s+get_inputs\s*\(",
            re.MULTILINE
        )

        # Pattern for get_init_inputs function
        self.get_init_inputs_pattern = re.compile(
            r"def\s+get_init_inputs\s*\(",
            re.MULTILINE
        )

    def parse(
        self,
        llm_response: str,
        original_pytorch_code: Optional[str] = None,
    ) -> ParseResult:
        """
        Parse LLM response and extract valid Triton code.

        Args:
            llm_response: Raw response from LLM
            original_pytorch_code: Original PyTorch code (for extracting
                                  get_inputs/get_init_inputs if missing)

        Returns:
            ParseResult with extracted code and validation status
        """
        errors = []
        missing = []
        warnings = []

        # Step 1: Extract code from markdown blocks
        code = self._extract_code_blocks(llm_response)

        if not code:
            # No code block found, try to find code by import statement
            code = self._extract_by_import(llm_response)

        if not code:
            errors.append("No Python code block found in response")
            return ParseResult(
                code="",
                parse_errors=errors,
                is_valid=False,
            )

        # Step 2: Post-processing - clean up common LLM output issues
        code = self._postprocess_code(code, warnings)

        # Step 3: Validate required components
        has_triton_import = bool(self.import_pattern.search(code))
        has_model_class = bool(self.model_class_pattern.search(code))
        has_triton_jit = bool(self.triton_jit_pattern.search(code))
        has_get_inputs = bool(self.get_inputs_pattern.search(code))
        has_get_init_inputs = bool(self.get_init_inputs_pattern.search(code))

        # Check critical components
        if not has_triton_import:
            missing.append("triton import")

        if not has_model_class:
            missing.append("Model class")
            errors.append("Model class not found (required by KernelBench)")

        if not has_triton_jit:
            missing.append("@triton.jit kernel")
            errors.append("No @triton.jit decorated function found")

        # Check helper functions (can be recovered from original)
        if not has_get_inputs:
            missing.append("get_inputs()")
            if original_pytorch_code:
                # Try to extract from original
                extracted = self._extract_helper_function(
                    original_pytorch_code, "get_inputs"
                )
                if extracted:
                    code = code + "\n\n" + extracted
                    has_get_inputs = True
                    missing.remove("get_inputs()")
                else:
                    errors.append("get_inputs() not found and cannot be extracted from original")
            else:
                errors.append("get_inputs() not found (required by KernelBench)")

        if not has_get_init_inputs:
            missing.append("get_init_inputs()")
            if original_pytorch_code:
                # Try to extract from original
                extracted = self._extract_helper_function(
                    original_pytorch_code, "get_init_inputs"
                )
                if extracted:
                    code = code + "\n\n" + extracted
                    has_get_init_inputs = True
                    missing.remove("get_init_inputs()")
                else:
                    # Provide default implementation
                    code = code + "\n\ndef get_init_inputs():\n    return ()\n"
                    has_get_init_inputs = True
                    missing.remove("get_init_inputs()")

        # Step 4: Check get_inputs() returns CUDA tensors
        if has_get_inputs:
            cuda_warnings = self._check_get_inputs_cuda(code)
            warnings.extend(cuda_warnings)

        # Determine validity
        is_valid = (
            has_model_class and
            has_triton_jit and
            has_get_inputs and
            has_get_init_inputs
        )

        return ParseResult(
            code=code,
            parse_errors=errors,
            is_valid=is_valid,
            missing_components=missing,
            warnings=warnings,
        )

    def _extract_code_blocks(self, text: str) -> str:
        """
        Extract code from markdown code blocks.

        Args:
            text: Raw text with potential code blocks

        Returns:
            Concatenated code from all blocks
        """
        matches = self.code_block_pattern.findall(text)

        if not matches:
            return ""

        # Concatenate all code blocks
        return "\n\n".join(matches)

    def _extract_by_import(self, text: str) -> str:
        """
        Extract code by finding import triton statement.

        Fallback when code blocks are not properly formatted.

        Args:
            text: Raw text

        Returns:
            Code starting from first triton import, with trailing prose removed
        """
        match = self.import_pattern.search(text)

        if not match:
            return ""

        # Extract from import onwards
        start_pos = match.start()
        code = text[start_pos:]

        # Remove trailing explanation / prose sections that LLMs commonly append
        code = self._remove_trailing_prose(code)

        return code.strip()

    def _remove_trailing_prose(self, code: str) -> str:
        """
        Remove trailing explanation text that LLMs append after code.

        Detects patterns like:
        - ### Explanation:
        - # Explanation
        - Blocks of consecutive comment-only lines at the end
        - Markdown headers after code

        Args:
            code: Extracted code that may contain trailing prose

        Returns:
            Code with trailing prose removed
        """
        # Remove everything after common explanation markers
        explanation_patterns = [
            r'\n### Explanation.*$',
            r'\n## Explanation.*$',
            r'\n# Explanation.*$',
            r'\n\*\*Explanation\*\*.*$',
            r'\nExplanation:.*$',
            r'\n### How it works.*$',
            r'\n## How it works.*$',
            r'\n### Notes?:.*$',
            r'\n## Notes?:.*$',
            r'\n### Key (?:points?|changes?|differences?|features?).*$',
            r'\n---\n.*$',  # Markdown horizontal rule followed by text
            r'\nThis (?:code|implementation|kernel|solution).*$',
            r'\nThe above (?:code|implementation).*$',
            r'\nHere\'s (?:a |an |the )?(?:brief |short )?(?:explanation|breakdown|summary).*$',
        ]

        for pattern in explanation_patterns:
            code = re.sub(pattern, '', code, flags=re.DOTALL | re.IGNORECASE)

        return code

    def _postprocess_code(self, code: str, warnings: List[str]) -> str:
        """
        Post-process extracted code to fix common LLM output issues.

        Fixes:
        - TritonModel → Model class name
        - class Model: → class Model(nn.Module):
        - Removes if __name__ == '__main__' blocks
        - Removes trailing explanation text

        Args:
            code: Extracted code
            warnings: List to append warnings to

        Returns:
            Cleaned-up code
        """
        # Fix 1: Rename TritonModel (or other Model variants) to Model
        tritonmodel_pattern = re.compile(
            r'class\s+(TritonModel|Triton_Model|MyModel|CustomModel)\s*\(',
            re.MULTILINE,
        )
        match = tritonmodel_pattern.search(code)
        if match:
            old_name = match.group(1)
            warnings.append(f"Renamed class '{old_name}' to 'Model'")
            logger.warning("Renamed class '%s' to 'Model'", old_name)
            # Replace class definition
            code = re.sub(
                rf'class\s+{old_name}\s*\(',
                'class Model(',
                code,
            )
            # Replace all references to old class name (e.g., instantiation)
            code = re.sub(
                rf'\b{old_name}\b',
                'Model',
                code,
            )

        # Fix 2: Ensure Model inherits from nn.Module
        bare_model_pattern = re.compile(
            r'class\s+Model\s*:\s*$',
            re.MULTILINE,
        )
        if bare_model_pattern.search(code):
            warnings.append("Fixed 'class Model:' to 'class Model(nn.Module):'")
            logger.warning("Fixed bare 'class Model:' to inherit from nn.Module")
            code = bare_model_pattern.sub('class Model(nn.Module):', code)

        # Fix 3: Remove if __name__ == '__main__' blocks
        main_block_pattern = re.compile(
            r"\nif\s+__name__\s*==\s*['\"]__main__['\"]\s*:.*$",
            re.DOTALL | re.MULTILINE,
        )
        if main_block_pattern.search(code):
            warnings.append("Removed if __name__ == '__main__' block")
            code = main_block_pattern.sub('', code)

        # Fix 4: Remove trailing prose (in case it survived code block extraction)
        code = self._remove_trailing_prose(code)

        return code.strip()

    def _check_get_inputs_cuda(self, code: str) -> List[str]:
        """
        Check if get_inputs() returns tensors on CUDA device.

        Args:
            code: Full Python code

        Returns:
            List of warning messages
        """
        warnings = []

        # Extract get_inputs function body
        get_inputs_body = self._extract_helper_function(code, "get_inputs")
        if not get_inputs_body:
            return warnings

        # Check for tensor creation without .cuda() or device='cuda'
        has_cuda = (
            '.cuda()' in get_inputs_body
            or "device='cuda'" in get_inputs_body
            or 'device="cuda"' in get_inputs_body
            or "device=torch.device('cuda')" in get_inputs_body
            or 'device=torch.device("cuda")' in get_inputs_body
        )

        has_tensor_creation = any(
            pattern in get_inputs_body
            for pattern in ['torch.randn', 'torch.rand', 'torch.zeros', 'torch.ones',
                            'torch.empty', 'torch.tensor', 'torch.full', 'torch.arange']
        )

        if has_tensor_creation and not has_cuda:
            warnings.append(
                "get_inputs() creates tensors without .cuda() or device='cuda'. "
                "KernelBench requires CUDA tensors."
            )
            logger.warning("get_inputs() may return CPU tensors — missing .cuda() or device='cuda'")

        return warnings

    def _extract_helper_function(
        self,
        code: str,
        function_name: str
    ) -> Optional[str]:
        """
        Extract a specific function from code.

        Args:
            code: Python code to search
            function_name: Function name (e.g., "get_inputs")

        Returns:
            Function code if found, None otherwise
        """
        pattern = re.compile(
            rf"(def\s+{function_name}\s*\(.*?\):.*?)(?=\ndef\s|\nclass\s|\Z)",
            re.DOTALL | re.MULTILINE
        )

        match = pattern.search(code)

        if match:
            return match.group(1).strip()

        return None

    def validate_syntax(self, code: str) -> List[str]:
        """
        Validate Python syntax without executing.

        Args:
            code: Python code to validate

        Returns:
            List of syntax errors (empty if valid)
        """
        errors = []

        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Compilation error: {str(e)}")

        return errors

    def extract_imports(self, code: str) -> List[str]:
        """
        Extract all import statements from code.

        Args:
            code: Python code

        Returns:
            List of import statements
        """
        import_pattern = re.compile(
            r"^(?:import\s+.+|from\s+.+\s+import\s+.+)$",
            re.MULTILINE
        )

        return import_pattern.findall(code)

    def has_required_imports(self, code: str) -> bool:
        """
        Check if code has all required imports for Triton kernels.

        Args:
            code: Python code

        Returns:
            True if torch and triton are imported
        """
        imports = self.extract_imports(code)
        imports_str = "\n".join(imports)

        has_torch = "import torch" in imports_str or "from torch" in imports_str
        has_triton = "import triton" in imports_str or "from triton" in imports_str

        return has_torch and has_triton

    def extract_model_class(self, code: str) -> Optional[str]:
        """
        Extract the Model class definition from code.

        Args:
            code: Python code

        Returns:
            Model class code if found
        """
        # Find Model class
        pattern = re.compile(
            r"(class\s+Model\s*\(.*?\):.*?)(?=\nclass\s|\ndef\s+get_|\Z)",
            re.DOTALL | re.MULTILINE
        )

        match = pattern.search(code)

        if match:
            return match.group(1).strip()

        return None

    def extract_kernel_functions(self, code: str) -> List[str]:
        """
        Extract all @triton.jit decorated functions.

        Args:
            code: Python code

        Returns:
            List of kernel function names
        """
        # Find all @triton.jit decorators followed by function definitions
        pattern = re.compile(
            r"@triton\.jit.*?\ndef\s+(\w+)\s*\(",
            re.DOTALL | re.MULTILINE
        )

        return pattern.findall(code)
