"""Common Triton kernel patterns and error handling guide.

This module provides reference patterns for common Triton operations, best practices,
and common error messages with their solutions. Used for kernel code generation
and debugging assistance.
"""

from typing import Optional

# Common Triton errors and their solutions
TRITON_COMMON_ERRORS = {
    "out_of_bounds": (
        "Out of bounds memory access detected. "
        "Always use mask = offsets < n_elements with tl.load/tl.store to prevent invalid access. "
        "Example: tl.load(ptr + offsets, mask=mask, other=0.0)"
    ),
    "type_mismatch": (
        "Type mismatch in operation. "
        "Use explicit casting with .to(tl.float32) for accumulation buffers. "
        "Don't mix different dtypes without explicit conversion. "
        "Example: acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)"
    ),
    "reduction_axis": (
        "Reduction axis not specified or invalid. "
        "Specify axis explicitly in tl.sum/tl.max/tl.min operations. "
        "Example: tl.sum(tensor, axis=1)"
    ),
    "grid_size": (
        "Invalid grid size calculation. "
        "Use triton.cdiv(N, BLOCK_SIZE) for proper grid dimension calculation. "
        "Example: grid = (triton.cdiv(N, BLOCK_SIZE),)"
    ),
    "constexpr": (
        "Block sizes must be tl.constexpr parameters. "
        "Declare block dimensions as constexpr in function signature. "
        "Example: def kernel(x, BLOCK_SIZE: tl.constexpr):"
    ),
    "shared_memory": (
        "Shared memory exceeded or not properly allocated. "
        "Ensure total shared memory per block <= 96KB (sm_90) or 98KB (sm_120). "
        "Use tl.zeros/tl.ones for dynamic allocation in shared scope."
    ),
    "warp_synchronization": (
        "Missing or improper warp synchronization. "
        "Use tl.barrier() to synchronize threads within a block. "
        "Needed when writing to shared memory and reading it from other threads."
    ),
    "memory_aliasing": (
        "Memory aliasing detected - same pointer accessed with different types. "
        "Ensure load/store operations use consistent data types. "
        "Cast pointers explicitly if needed: ptr.reinterpret_cast(tl.float32)"
    ),
}

# Triton kernel patterns for common operations
TRITON_PATTERNS = {
    "elementwise": '''
def elementwise_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Element-wise operation template."""
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask)
    y = tl.exp(x)  # Apply operation
    tl.store(y_ptr + idx, y, mask=mask)
''',
    "reduction": '''
def reduction_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Reduction operation template."""
    idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx, mask=idx < N)
    y = tl.sum(x, axis=0)  # Reduction
    tl.store(y_ptr, y)
''',
    "matmul": '''
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Matrix multiplication template using tensor cores."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * K + (offs_k[None, :] + k))
        b = tl.load(b_ptr + (offs_k[:, None] + k) * N + offs_n[None, :])
        acc = tl.dot(a, b, acc=acc)  # Use tensor cores

    c_offsets = offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptr + c_offsets, acc)
''',
    "softmax": '''
def softmax_kernel(x_ptr, y_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    """Softmax with numerical stability."""
    row_idx = tl.program_id(axis=0)

    col_idx = tl.arange(0, BLOCK_SIZE)
    mask = col_idx < M

    x = tl.load(x_ptr + row_idx * M + col_idx, mask=mask, other=float('-inf'))
    x_max = tl.max(x, axis=0)

    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    y = x_exp / x_sum

    tl.store(y_ptr + row_idx * M + col_idx, y, mask=mask)
''',
    "attention": '''
def attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, L, S,
                    BLOCK_L: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr):
    """Flash Attention style kernel."""
    batch_idx = tl.program_id(axis=0)
    query_idx = tl.program_id(axis=1)

    # Load query block
    q_offs = tl.arange(0, BLOCK_D)
    q = tl.load(q_ptr + batch_idx * L * BLOCK_D + query_idx * BLOCK_D + q_offs)

    # Compute attention scores
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for k_block in range(0, S, BLOCK_S):
        k_offs = tl.arange(0, BLOCK_D)
        k = tl.load(k_ptr + batch_idx * S * BLOCK_D + k_block * BLOCK_D + k_offs)

        scores = tl.dot(q, k)  # Query-key dot product
        v = tl.load(v_ptr + batch_idx * S * BLOCK_D + k_block * BLOCK_D + k_offs)

        acc += scores * v

    o_offs = tl.arange(0, BLOCK_D)
    tl.store(o_ptr + batch_idx * L * BLOCK_D + query_idx * BLOCK_D + o_offs, acc)
''',
    "normalization": '''
def layer_norm_kernel(x_ptr, y_ptr, w_ptr, b_ptr, N, D, eps, BLOCK_SIZE: tl.constexpr):
    """Layer normalization kernel."""
    row_idx = tl.program_id(axis=0)
    col_idx = tl.arange(0, BLOCK_SIZE)

    mask = col_idx < D
    x = tl.load(x_ptr + row_idx * D + col_idx, mask=mask)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / D
    var = tl.sum((x - mean) ** 2, axis=0) / D

    # Normalize
    y = (x - mean) / tl.sqrt(var + eps)

    # Scale and shift
    w = tl.load(w_ptr + col_idx, mask=mask)
    b = tl.load(b_ptr + col_idx, mask=mask)
    y = y * w + b

    tl.store(y_ptr + row_idx * D + col_idx, y, mask=mask)
''',
}

# Best practices and guidelines
TRITON_BEST_PRACTICES = [
    "Always use mask parameter in tl.load/tl.store for boundary handling",
    "Use tl.constexpr for block size parameters for compile-time optimization",
    "Prefer tl.dot() over manual matrix multiplication for tensor core utilization",
    "Use float32 accumulation buffer even with float16 inputs for numerical stability",
    "Keep register usage under 256 per thread to avoid spilling",
    "Use @triton.autotune decorator to find optimal block configurations",
    "Profile with triton.testing to ensure kernel correctness and performance",
    "Use triton.cdiv() for proper grid dimension calculation",
    "Add data type assertions using tl.assert_dequals for debugging",
    "Minimize HBM accesses by using shared memory effectively",
    "Use tl.barrier() when multiple threads need synchronized access",
    "Declare loop bounds as constexpr for unrolling opportunities",
    "Use different block sizes for different dimensions (BLOCK_M, BLOCK_N, BLOCK_K)",
    "Prefetch next iteration's data before finishing current computation",
    "Use pack/unpack for efficient tensor reshape operations",
]

# Memory access patterns guide
MEMORY_PATTERNS = {
    "coalesced": (
        "Threads 0-31 access consecutive memory addresses. "
        "Example: thread i loads from base + i"
    ),
    "strided": (
        "Threads access memory with fixed stride. "
        "Example: thread i loads from base + i*stride"
    ),
    "gathered": (
        "Each thread loads from arbitrary address. "
        "Less efficient; use only when necessary with proper masking"
    ),
    "shared_memory": (
        "Stage data in shared memory to reduce HBM bandwidth. "
        "Use padding to avoid bank conflicts in shared memory arrays"
    ),
}


def get_error_solution(error_type: str) -> Optional[str]:
    """Get solution for a common Triton error.

    Args:
        error_type: Type of error - one of:
            - "out_of_bounds": Memory access boundary violation
            - "type_mismatch": Data type mismatch
            - "reduction_axis": Invalid reduction axis
            - "grid_size": Incorrect grid dimensions
            - "constexpr": Missing tl.constexpr
            - "shared_memory": Shared memory issues
            - "warp_synchronization": Missing synchronization
            - "memory_aliasing": Pointer aliasing issues

    Returns:
        Solution string or None if error type not found

    Examples:
        >>> solution = get_error_solution("out_of_bounds")
        >>> "mask" in solution.lower()
        True
    """
    return TRITON_COMMON_ERRORS.get(error_type.lower())


def get_pattern_for_operation(op_type: str) -> Optional[str]:
    """Get Triton kernel pattern template for a given operation.

    Args:
        op_type: Operation type - one of:
            - "elementwise": Element-wise operations
            - "reduction": Reduction operations (sum, max, etc.)
            - "matmul": Matrix multiplication
            - "softmax": Softmax normalization
            - "attention": Attention mechanism
            - "normalization": Layer/batch normalization

    Returns:
        Code template string or None if operation not found

    Examples:
        >>> pattern = get_pattern_for_operation("elementwise")
        >>> "BLOCK_SIZE" in pattern
        True
    """
    return TRITON_PATTERNS.get(op_type.lower())


def get_all_patterns() -> dict[str, str]:
    """Get all available kernel pattern templates.

    Returns:
        Dictionary mapping operation name to code template

    Examples:
        >>> patterns = get_all_patterns()
        >>> "matmul" in patterns
        True
    """
    return TRITON_PATTERNS.copy()


def get_best_practices() -> list[str]:
    """Get list of Triton best practices.

    Returns:
        List of best practice strings

    Examples:
        >>> practices = get_best_practices()
        >>> len(practices) > 0
        True
    """
    return TRITON_BEST_PRACTICES.copy()


def format_best_practices(prefix: str = "Triton Best Practices:\n") -> str:
    """Format best practices as a readable string for LLM prompts.

    Args:
        prefix: Optional prefix string

    Returns:
        Formatted string with bullet points

    Examples:
        >>> formatted = format_best_practices()
        >>> "Triton Best Practices" in formatted
        True
    """
    practices = get_best_practices()
    bullet_list = "\n".join(f"- {practice}" for practice in practices)
    return f"{prefix}{bullet_list}"


def get_memory_access_guide(pattern_type: str) -> Optional[str]:
    """Get explanation of a memory access pattern.

    Args:
        pattern_type: Pattern type - one of:
            - "coalesced": Sequential memory access
            - "strided": Fixed-stride access
            - "gathered": Random access
            - "shared_memory": Shared memory usage

    Returns:
        Pattern explanation or None if not found

    Examples:
        >>> guide = get_memory_access_guide("coalesced")
        >>> "consecutive" in guide.lower()
        True
    """
    return MEMORY_PATTERNS.get(pattern_type.lower())


def format_pattern_as_prompt(op_type: str, prefix: str = "Triton Pattern:\n") -> str:
    """Format operation pattern as a code example for LLM prompts.

    Args:
        op_type: Operation type
        prefix: Optional prefix string

    Returns:
        Formatted code example

    Examples:
        >>> formatted = format_pattern_as_prompt("softmax")
        >>> "softmax_kernel" in formatted
        True
    """
    pattern = get_pattern_for_operation(op_type)
    if not pattern:
        return ""
    return f"{prefix}{pattern}"


def get_error_resolution_guide() -> dict[str, str]:
    """Get comprehensive error resolution guide.

    Returns:
        Dictionary mapping error types to solutions

    Examples:
        >>> guide = get_error_resolution_guide()
        >>> "out_of_bounds" in guide
        True
    """
    return TRITON_COMMON_ERRORS.copy()


def suggest_block_size(
    operation_type: str, input_size: int, target_bandwidth_gb_per_sec: float = 1000
) -> int:
    """Suggest optimal block size for an operation based on heuristics.

    Args:
        operation_type: Type of operation
        input_size: Total elements to process
        target_bandwidth_gb_per_sec: Target bandwidth for device

    Returns:
        Suggested block size (power of 2)

    Examples:
        >>> size = suggest_block_size("matmul", 1024)
        >>> size in [32, 64, 128, 256, 512, 1024]
        True
    """
    # Heuristic: 128-512 for most operations
    base_block_size = 128

    if operation_type.lower() == "reduction":
        # Reductions benefit from larger blocks
        return 512
    elif operation_type.lower() in ["softmax", "normalization"]:
        # These can use smaller blocks
        return 256
    elif operation_type.lower() == "matmul":
        # Matrix mult with 128x128 or 256x256
        return 128
    else:
        return base_block_size


def get_kernel_complexity_info(op_type: str) -> dict[str, int | str]:
    """Get complexity metrics for a kernel type.

    Args:
        op_type: Operation type

    Returns:
        Dictionary with complexity metrics

    Examples:
        >>> info = get_kernel_complexity_info("matmul")
        >>> info["arithmetic_intensity"]
        "High"
    """
    complexity_map = {
        "elementwise": {
            "arithmetic_intensity": "Low",
            "memory_bound": True,
            "suggested_block_size": 256,
        },
        "reduction": {
            "arithmetic_intensity": "Low",
            "memory_bound": True,
            "suggested_block_size": 512,
        },
        "matmul": {
            "arithmetic_intensity": "High",
            "memory_bound": False,
            "suggested_block_size": 128,
        },
        "softmax": {
            "arithmetic_intensity": "Medium",
            "memory_bound": True,
            "suggested_block_size": 256,
        },
        "attention": {
            "arithmetic_intensity": "High",
            "memory_bound": False,
            "suggested_block_size": 128,
        },
        "normalization": {
            "arithmetic_intensity": "Low",
            "memory_bound": True,
            "suggested_block_size": 256,
        },
    }

    return complexity_map.get(op_type.lower(), {})
