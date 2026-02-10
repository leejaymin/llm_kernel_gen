"""
KernelBench Submission Formatter.

Converts LLM-generated code to KernelBench-compliant submission format
with validation and auto-completion of required components.
"""

import re
from typing import Tuple, List, Optional


class SubmissionFormatter:
    """
    Format and validate generated Triton kernels for KernelBench submission.

    KernelBench requires specific structure:
    1. import statements
    2. @triton.jit decorated kernel function(s)
    3. Model class with forward() method
    4. get_inputs() function returning test inputs
    5. get_init_inputs() function returning initialization parameters
    """

    def __init__(self):
        """Initialize submission formatter."""
        self.required_imports = {
            "triton",
            "torch"
        }

    def _extract_from_reference(self, reference_code: str, function_name: str) -> Optional[str]:
        """
        Extract a function definition from reference PyTorch code.

        Args:
            reference_code: PyTorch reference implementation
            function_name: Name of function to extract

        Returns:
            Function code or None if not found
        """
        # Pattern: def function_name(...): with body
        pattern = rf'def {function_name}\([^)]*\):.*?(?=\ndef |\nclass |\Z)'
        match = re.search(pattern, reference_code, re.DOTALL)

        if match:
            return match.group(0).strip()

        return None

    def _has_triton_kernel(self, code: str) -> bool:
        """Check if code contains @triton.jit decorated function."""
        return bool(re.search(r'@triton\.jit', code))

    def _has_model_class(self, code: str) -> bool:
        """Check if code contains Model class with forward() method."""
        has_class = bool(re.search(r'class\s+Model\s*[:(]', code))
        has_forward = bool(re.search(r'def\s+forward\s*\(', code))
        return has_class and has_forward

    def _has_get_inputs(self, code: str) -> bool:
        """Check if code contains get_inputs() function."""
        return bool(re.search(r'def\s+get_inputs\s*\(', code))

    def _has_get_init_inputs(self, code: str) -> bool:
        """Check if code contains get_init_inputs() function."""
        return bool(re.search(r'def\s+get_init_inputs\s*\(', code))

    def _extract_imports(self, code: str) -> List[str]:
        """Extract all import statements from code."""
        imports = []
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                imports.append(stripped)
        return imports

    def _ensure_imports(self, code: str) -> str:
        """
        Ensure required imports are present.

        Args:
            code: Generated code

        Returns:
            Code with guaranteed imports
        """
        existing_imports = self._extract_imports(code)
        existing_import_text = '\n'.join(existing_imports)

        # Check for required imports
        has_triton = 'triton' in existing_import_text
        has_torch = 'torch' in existing_import_text

        new_imports = []

        if not has_torch:
            new_imports.append('import torch')

        if not has_triton:
            new_imports.append('import triton')
            new_imports.append('import triton.language as tl')

        if new_imports:
            # Add new imports at the top
            import_block = '\n'.join(new_imports)
            if existing_imports:
                return f"{import_block}\n{code}"
            else:
                # Find first non-comment, non-docstring line
                lines = code.split('\n')
                insert_idx = 0
                in_docstring = False

                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        if in_docstring:
                            in_docstring = False
                            insert_idx = i + 1
                        else:
                            in_docstring = True
                    elif not in_docstring and not stripped.startswith('#') and stripped:
                        insert_idx = i
                        break

                lines.insert(insert_idx, import_block)
                return '\n'.join(lines)

        return code

    def _inject_get_inputs(self, code: str, reference_code: str) -> str:
        """
        Inject get_inputs() from reference code if missing.

        Args:
            code: Generated code
            reference_code: PyTorch reference

        Returns:
            Code with get_inputs() added
        """
        get_inputs_code = self._extract_from_reference(reference_code, 'get_inputs')

        if get_inputs_code:
            # Add at the end
            return f"{code}\n\n{get_inputs_code}"

        # Generate a basic placeholder
        placeholder = """
def get_inputs():
    \"\"\"Generate test inputs.\"\"\"
    # TODO: Implement based on kernel requirements
    return []
"""
        return f"{code}\n{placeholder}"

    def _inject_get_init_inputs(self, code: str, reference_code: str) -> str:
        """
        Inject get_init_inputs() from reference code if missing.

        Args:
            code: Generated code
            reference_code: PyTorch reference

        Returns:
            Code with get_init_inputs() added
        """
        get_init_inputs_code = self._extract_from_reference(reference_code, 'get_init_inputs')

        if get_init_inputs_code:
            # Add at the end
            return f"{code}\n\n{get_init_inputs_code}"

        # Generate a basic placeholder
        placeholder = """
def get_init_inputs():
    \"\"\"Generate initialization inputs.\"\"\"
    return {}
"""
        return f"{code}\n{placeholder}"

    def format(self, generated_code: str, reference_code: str) -> str:
        """
        Format generated code to meet KernelBench requirements.

        This method:
        1. Ensures required imports are present
        2. Validates Triton kernel exists
        3. Injects missing get_inputs() from reference
        4. Injects missing get_init_inputs() from reference
        5. Ensures proper structure

        Args:
            generated_code: LLM-generated Triton kernel code
            reference_code: PyTorch reference implementation

        Returns:
            Formatted code ready for KernelBench submission

        Raises:
            ValueError: If code cannot be formatted (missing critical components)
        """
        code = generated_code.strip()

        # 1. Ensure imports
        code = self._ensure_imports(code)

        # 2. Validate kernel exists
        if not self._has_triton_kernel(code):
            raise ValueError(
                "Generated code must contain at least one @triton.jit kernel"
            )

        # 3. Inject get_inputs if missing
        if not self._has_get_inputs(code):
            code = self._inject_get_inputs(code, reference_code)

        # 4. Inject get_init_inputs if missing
        if not self._has_get_init_inputs(code):
            code = self._inject_get_init_inputs(code, reference_code)

        # 5. Validate Model class (warn if missing but don't fail)
        if not self._has_model_class(code):
            # Many KernelBench problems may not need Model class
            # Just log a warning rather than failing
            pass

        return code

    def validate(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate code against KernelBench requirements.

        Checks:
        1. Has required imports (torch, triton)
        2. Has @triton.jit kernel
        3. Has get_inputs() function
        4. Has get_init_inputs() function
        5. Has Model class with forward() (optional warning)

        Args:
            code: Code to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check imports
        has_torch = 'import torch' in code
        has_triton = 'import triton' in code or 'from triton' in code

        if not has_torch:
            errors.append("Missing required import: torch")

        if not has_triton:
            errors.append("Missing required import: triton")

        # Check kernel
        if not self._has_triton_kernel(code):
            errors.append("Missing @triton.jit decorated kernel function")

        # Check get_inputs
        if not self._has_get_inputs(code):
            errors.append("Missing required function: get_inputs()")

        # Check get_init_inputs
        if not self._has_get_init_inputs(code):
            errors.append("Missing required function: get_init_inputs()")

        # Check Model class (warning only)
        if not self._has_model_class(code):
            # Don't add as error, some problems may not need it
            pass

        # Check for syntax errors (basic check)
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            errors.append(f"Syntax error: {str(e)}")

        is_valid = len(errors) == 0

        return is_valid, errors

    def extract_kernel_name(self, code: str) -> Optional[str]:
        """
        Extract the main kernel function name.

        Args:
            code: Triton kernel code

        Returns:
            Kernel function name or None
        """
        # Find @triton.jit decorated function
        pattern = r'@triton\.jit\s+def\s+(\w+)\s*\('
        match = re.search(pattern, code)

        if match:
            return match.group(1)

        return None

    def extract_block_sizes(self, code: str) -> List[str]:
        """
        Extract block size parameters from kernel signature.

        Args:
            code: Triton kernel code

        Returns:
            List of block size parameter names
        """
        block_sizes = []

        # Find tl.constexpr parameters (typically block sizes)
        pattern = r'(\w+)\s*:\s*tl\.constexpr'
        matches = re.findall(pattern, code)

        for match in matches:
            if 'BLOCK' in match.upper():
                block_sizes.append(match)

        return block_sizes

    def get_formatting_stats(self, original: str, formatted: str) -> dict:
        """
        Get statistics about formatting changes.

        Args:
            original: Original generated code
            formatted: Formatted code

        Returns:
            Dictionary with formatting statistics
        """
        return {
            'lines_added': formatted.count('\n') - original.count('\n'),
            'imports_added': formatted.count('import') - original.count('import'),
            'functions_added': formatted.count('def ') - original.count('def '),
            'has_kernel': self._has_triton_kernel(formatted),
            'has_model': self._has_model_class(formatted),
            'has_get_inputs': self._has_get_inputs(formatted),
            'has_get_init_inputs': self._has_get_init_inputs(formatted),
        }

    def clean_generated_code(self, code: str) -> str:
        """
        Clean up common LLM generation artifacts.

        Removes:
        - Markdown code fences
        - Explanatory text before/after code
        - Excessive blank lines

        Args:
            code: Raw LLM output

        Returns:
            Cleaned Python code
        """
        # Remove markdown code fences
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)

        # Remove common prefixes
        code = re.sub(r'^Here is.*?:\s*', '', code, flags=re.MULTILINE | re.IGNORECASE)
        code = re.sub(r'^The code.*?:\s*', '', code, flags=re.MULTILINE | re.IGNORECASE)

        # Remove excessive blank lines (3+ in a row)
        code = re.sub(r'\n{4,}', '\n\n\n', code)

        return code.strip()
