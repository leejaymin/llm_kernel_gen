"""Blackwell-specific GPU optimization hints.

This module provides detailed optimization guidance for NVIDIA Blackwell architecture
(SM 12.0, compute capability 12.0), including tensor core features, memory optimization,
and common patterns for kernel development.
"""

from typing import Any


# General Blackwell optimization hints
BLACKWELL_GENERAL_HINTS = [
    "Use tl.dot() for matrix operations to trigger tcgen05.mma instructions on SM 12.0",
    "Blackwell Tensor Cores support FP8, FP16, BF16 data types - prefer BF16 for training, FP16 for inference",
    "Target peak memory bandwidth of 1792 GB/s with coalesced memory access patterns",
    "Minimum BLOCK_SIZE should be 128 for optimal GDDR7 memory utilization",
    "Use num_warps=8 as starting point for Blackwell, adjust based on shared memory usage",
    "Leverage @triton.autotune with multiple configurations for BLOCK_M, BLOCK_N, BLOCK_K",
    "Blackwell supports TMA (Tensor Memory Accelerator) for efficient asynchronous memory transfers",
    "Bank conflict reduction: Use padding for shared memory arrays when accessing via different patterns",
    "Avoid register spilling by keeping thread-local working set under 256 registers",
    "Use tensor cores for all heavy compute, especially matrix multiplication and element-wise operations",
]

# Operation-specific optimization hints
MATMUL_HINTS = [
    "Use tl.dot(a, b) for 2D matrix multiplication on tensor cores",
    "Transpose input if memory layout is column-major to improve cache efficiency",
    "Block size strategy: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32 for BF16 inputs",
    "For large matrices, use block-level pipelining with overlapped compute and memory transfers",
    "Enable @triton.autotune to find optimal blocking factors for your specific problem",
    "Consider transposed layout (N, M) if you have more work in one dimension",
    "Use fp32 accumulation even with fp16/bf16 inputs to avoid numerical issues",
]

SOFTMAX_HINTS = [
    "Use tl.max() followed by tl.sum(exp(x - max)) for numerical stability",
    "One warp per row is often optimal; adjust num_warps based on BLOCK_SIZE",
    "Mask out padding positions before computing max to avoid -inf issues",
    "Stage max computation separately from exp+sum for better pipelining",
    "Consider online (Welford) softmax for long sequences to reduce memory access",
    "Use bf16 for softmax output to match common training precision",
]

ATTENTION_HINTS = [
    "Block-wise attention: Process queries in blocks (BLOCK_SIZE_Q), keys/values in blocks (BLOCK_SIZE_KV)",
    "Use numerically stable log-sum-exp trick for attention scores",
    "Prefetch key/value blocks ahead of query processing for better memory overlap",
    "Apply causal mask efficiently with conditional masking, not full matrix",
    "Consider flash-attention pattern: process in blocks to minimize HBM traffic",
    "Use tl.store() with mask parameter to handle variable sequence lengths efficiently",
]

REDUCTION_HINTS = [
    "Use tl.sum(), tl.max(), tl.min() with explicit axis parameter",
    "For multi-step reductions, use shared memory to aggregate warp results",
    "Specify axis explicitly in reduction calls (e.g., axis=1)",
    "Consider persistent kernels for reductions requiring multiple grid launches",
    "Use mask parameter in reductions to handle variable input sizes",
    "Combine reduction with other ops to amortize memory access overhead",
]

ELEMENTWISE_HINTS = [
    "One thread per element is typical; adjust BLOCK_SIZE for memory coalescing",
    "Use tl.load/tl.store with proper masking for variable-sized inputs",
    "Vectorize operations where possible (tl.float4 instead of 4x float)",
    "Apply all transformations in a single pass to minimize memory bandwidth",
    "Consider fused operations: combine multiple element-wise ops together",
    "Memory access should be coalesced: consecutive threads access consecutive memory",
]

CONVOLUTION_HINTS = [
    "Blocking strategy: BLOCK_H x BLOCK_W for output spatial dimensions",
    "Unroll inner kernel loops when kernel size is small (3x3, 5x5)",
    "Use shared memory to cache input tile with padding for efficient access",
    "Separable convolutions: decompose into horizontal and vertical passes",
    "For grouped convolutions, ensure each block handles complete group",
    "Consider im2col transformation for better GEMM-like performance",
]

NORMALIZATION_HINTS = [
    "Layer norm: Compute mean and variance per feature across batch dimension",
    "Use Welford's algorithm for numerically stable mean/variance computation",
    "Combined forward: Normalize and scale in one pass to reduce memory access",
    "Batch norm: Use persistent kernel for computing population statistics",
    "Group norm: Organize threads to cover group dimensions efficiently",
    "Avoid redundant exp() calls in normalization with numerical tricks",
]

MEMORY_HINTS = [
    "Coalesce memory access: threads 0, 1, 2... should access consecutive memory addresses",
    "Use shared memory to reduce HBM (high-bandwidth memory) bandwidth consumption",
    "Enable async copy with TMA for pipelined memory operations",
    "Arrange data layout (transposition) to match access patterns",
    "Prefetch next data block before previous computation completes",
    "Avoid scattered memory access; use gather only when necessary",
    "Utilize L1 and L2 caches effectively by temporal locality",
]

TENSOR_CORE_HINTS = [
    "Tensor cores require specific data types: FP16, BF16, FP8 (not FP32)",
    "Use tl.dot() to automatically route to tensor cores",
    "Multiply small matrices in one tensor core operation (16x16x16 for BF16)",
    "Accumulate in higher precision (FP32) to maintain accuracy",
    "TMA (Tensor Memory Accelerator) handles efficient data movement to tensor cores",
    "Enable tensor float 32 (TF32) mode for BF16/FP16 computations when higher precision not needed",
]

BANK_CONFLICT_HINTS = [
    "Shared memory has 128 banks; access pattern determines conflicts",
    "Stride accesses by bank: if data_type=4 bytes, stride by 32 for no conflicts",
    "Transpose shared memory access pattern to avoid conflicts",
    "Add padding bytes to shared memory arrays to shift access patterns",
    "Adjacent threads should access adjacent banks when possible",
]

# Comprehensive hint mapping
_OPERATION_HINTS = {
    "matmul": MATMUL_HINTS,
    "softmax": SOFTMAX_HINTS,
    "attention": ATTENTION_HINTS,
    "reduction": REDUCTION_HINTS,
    "elementwise": ELEMENTWISE_HINTS,
    "convolution": CONVOLUTION_HINTS,
    "normalization": NORMALIZATION_HINTS,
    "memory": MEMORY_HINTS,
    "tensor_core": TENSOR_CORE_HINTS,
    "bank_conflict": BANK_CONFLICT_HINTS,
}


def get_optimization_hints(operation_type: str) -> list[str]:
    """Return Blackwell-specific optimization hints for a given operation type.

    Args:
        operation_type: Type of operation - one of:
            - "matmul": Matrix multiplication
            - "softmax": Softmax normalization
            - "attention": Attention mechanism
            - "reduction": Reduction operations (sum, max, min)
            - "elementwise": Element-wise operations
            - "convolution": Convolutional operations
            - "normalization": Layer/batch/group norm
            - "memory": Memory optimization
            - "tensor_core": Tensor core usage
            - "bank_conflict": Shared memory bank conflicts

    Returns:
        List of optimization hints as strings

    Examples:
        >>> hints = get_optimization_hints("matmul")
        >>> len(hints) > 0
        True
        >>> "tl.dot()" in hints[0]
        True
    """
    operation_type = operation_type.lower().strip()
    return _OPERATION_HINTS.get(operation_type, [])


def get_all_optimization_hints() -> dict[str, list[str]]:
    """Get all available optimization hints organized by operation type.

    Returns:
        Dictionary mapping operation type to list of hints

    Examples:
        >>> hints = get_all_optimization_hints()
        >>> "matmul" in hints
        True
    """
    return _OPERATION_HINTS.copy()


def add_custom_hints(operation_type: str, hints: list[str]) -> None:
    """Add custom optimization hints for an operation type.

    Args:
        operation_type: Type of operation
        hints: List of custom hint strings

    Examples:
        >>> add_custom_hints("custom_op", ["Do X", "Do Y"])
        >>> get_optimization_hints("custom_op")
        ["Do X", "Do Y"]
    """
    operation_type = operation_type.lower().strip()
    if operation_type not in _OPERATION_HINTS:
        _OPERATION_HINTS[operation_type] = []
    _OPERATION_HINTS[operation_type].extend(hints)


def format_hints_for_prompt(
    operation_type: str, prefix: str = "Optimization hints:\n"
) -> str:
    """Format optimization hints as a readable string for LLM prompts.

    Args:
        operation_type: Type of operation
        prefix: Optional prefix string

    Returns:
        Formatted hint string with bullet points

    Examples:
        >>> formatted = format_hints_for_prompt("softmax")
        >>> "Optimization hints:" in formatted
        True
    """
    hints = get_optimization_hints(operation_type)
    if not hints:
        return ""

    bullet_list = "\n".join(f"- {hint}" for hint in hints)
    return f"{prefix}{bullet_list}"


def get_general_hints() -> list[str]:
    """Get general Blackwell optimization hints applicable to all kernels.

    Returns:
        List of general optimization hints

    Examples:
        >>> hints = get_general_hints()
        >>> len(hints) > 0
        True
    """
    return BLACKWELL_GENERAL_HINTS.copy()


def format_general_hints(prefix: str = "Blackwell General Hints:\n") -> str:
    """Format general hints as a readable string for LLM prompts.

    Args:
        prefix: Optional prefix string

    Returns:
        Formatted hint string with bullet points

    Examples:
        >>> formatted = format_general_hints()
        >>> "Blackwell" in formatted
        True
    """
    hints = get_general_hints()
    bullet_list = "\n".join(f"- {hint}" for hint in hints)
    return f"{prefix}{bullet_list}"


def get_combined_hints(
    operation_type: str, include_general: bool = True
) -> list[str]:
    """Get combined general and operation-specific hints.

    Args:
        operation_type: Type of operation
        include_general: Whether to include general hints first

    Returns:
        Combined list of hints

    Examples:
        >>> hints = get_combined_hints("matmul")
        >>> len(hints) > len(get_optimization_hints("matmul"))
        True
    """
    hints = []
    if include_general:
        hints.extend(BLACKWELL_GENERAL_HINTS)
    hints.extend(get_optimization_hints(operation_type))
    return hints


def get_architecture_info() -> dict[str, Any]:
    """Get comprehensive Blackwell architecture information.

    Returns:
        Dictionary with architecture specifications and capabilities

    Examples:
        >>> info = get_architecture_info()
        >>> info["compute_capability"]
        "12.0"
    """
    return {
        "name": "NVIDIA Blackwell",
        "arch_code": "sm_120",
        "compute_capability": "12.0",
        "tensor_core_gen": 5,
        "supported_dtypes": ["FP32", "BF16", "FP16", "FP8", "TF32"],
        "tensor_core_matrix_size": "16x16x16",
        "max_warps_per_block": 32,
        "max_registers_per_thread": 255,
        "shared_memory_per_block_bytes": 98304,
        "max_shared_memory_banks": 128,
        "warp_size": 32,
        "peak_memory_bandwidth_gbs": 1792,
        "supports_tma": True,
        "supports_tf32": True,
    }
