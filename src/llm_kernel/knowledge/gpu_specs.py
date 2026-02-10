"""GPU specification database for kernel optimization.

This module provides detailed GPU specifications including architectural details,
memory characteristics, and compute capabilities. Used for generating
Blackwell-optimized kernel hints and performance predictions.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUSpec:
    """Comprehensive GPU specification."""
    name: str
    arch: str  # e.g., "sm_100", "sm_120"
    compute_capability: str  # e.g., "10.0", "12.0"
    vram_gb: int
    memory_bandwidth_gbs: float  # GB/s
    fp16_tflops: float  # Teraflops
    bf16_tflops: float  # Teraflops
    fp32_tflops: float  # Teraflops
    cuda_cores: int
    tensor_cores: int
    tensor_core_gen: int  # Generation of tensor cores
    tdp_watts: int  # Thermal Design Power


# Blackwell Architecture (12.0)
RTX_PRO_6000_BLACKWELL = GPUSpec(
    name="NVIDIA RTX PRO 6000 Blackwell",
    arch="sm_120",
    compute_capability="12.0",
    vram_gb=96,
    memory_bandwidth_gbs=1792,
    fp16_tflops=251.9,
    bf16_tflops=251.9,
    fp32_tflops=126.0,
    cuda_cores=24064,
    tensor_cores=752,
    tensor_core_gen=5,
    tdp_watts=600,
)

RTX_6000_ADA_BLACKWELL = GPUSpec(
    name="NVIDIA RTX 6000 Ada Blackwell",
    arch="sm_120",
    compute_capability="12.0",
    vram_gb=48,
    memory_bandwidth_gbs=960,
    fp16_tflops=251.9,
    bf16_tflops=251.9,
    fp32_tflops=126.0,
    cuda_cores=24064,
    tensor_cores=752,
    tensor_core_gen=5,
    tdp_watts=320,
)

# Hopper Architecture (9.0)
H100_PCIe = GPUSpec(
    name="NVIDIA H100 PCIe",
    arch="sm_90",
    compute_capability="9.0",
    vram_gb=80,
    memory_bandwidth_gbs=2039,
    fp16_tflops=1979,
    bf16_tflops=1979,
    fp32_tflops=989.5,
    cuda_cores=14080,
    tensor_cores=568,
    tensor_core_gen=4,
    tdp_watts=700,
)

H100_NVL = GPUSpec(
    name="NVIDIA H100 NVL",
    arch="sm_90",
    compute_capability="9.0",
    vram_gb=192,
    memory_bandwidth_gbs=4080,
    fp16_tflops=1979,
    bf16_tflops=1979,
    fp32_tflops=989.5,
    cuda_cores=14080,
    tensor_cores=568,
    tensor_core_gen=4,
    tdp_watts=1400,
)

# Ampere Architecture (8.0)
A100_PCIe = GPUSpec(
    name="NVIDIA A100 PCIe",
    arch="sm_80",
    compute_capability="8.0",
    vram_gb=80,
    memory_bandwidth_gbs=2039,
    fp16_tflops=312,
    bf16_tflops=312,
    fp32_tflops=156,
    cuda_cores=6912,
    tensor_cores=432,
    tensor_core_gen=3,
    tdp_watts=250,
)

A100_SXM4 = GPUSpec(
    name="NVIDIA A100 SXM4",
    arch="sm_80",
    compute_capability="8.0",
    vram_gb=80,
    memory_bandwidth_gbs=2039,
    fp16_tflops=312,
    bf16_tflops=312,
    fp32_tflops=156,
    cuda_cores=6912,
    tensor_cores=432,
    tensor_core_gen=3,
    tdp_watts=400,
)

# Ada Architecture (8.9)
RTX_6000_Ada = GPUSpec(
    name="NVIDIA RTX 6000 Ada",
    arch="sm_89",
    compute_capability="8.9",
    vram_gb=48,
    memory_bandwidth_gbs=960,
    fp16_tflops=352,
    bf16_tflops=352,
    fp32_tflops=176,
    cuda_cores=18176,
    tensor_cores=568,
    tensor_core_gen=4,
    tdp_watts=320,
)

L40S = GPUSpec(
    name="NVIDIA L40S",
    arch="sm_89",
    compute_capability="8.9",
    vram_gb=48,
    memory_bandwidth_gbs=960,
    fp16_tflops=352,
    bf16_tflops=352,
    fp32_tflops=176,
    cuda_cores=18176,
    tensor_cores=568,
    tensor_core_gen=4,
    tdp_watts=350,
)


# GPU Specification Registry
_GPU_REGISTRY = {
    "rtx_pro_6000_blackwell": RTX_PRO_6000_BLACKWELL,
    "rtx_6000_ada_blackwell": RTX_6000_ADA_BLACKWELL,
    "h100_pcie": H100_PCIe,
    "h100_nvl": H100_NVL,
    "a100_pcie": A100_PCIe,
    "a100_sxm4": A100_SXM4,
    "rtx_6000_ada": RTX_6000_Ada,
    "l40s": L40S,
}


def get_gpu_spec(name: str) -> Optional[GPUSpec]:
    """Retrieve GPU specification by registered name.

    Args:
        name: GPU identifier (e.g., "rtx_pro_6000_blackwell", "h100_pcie")

    Returns:
        GPUSpec instance or None if not found

    Examples:
        >>> spec = get_gpu_spec("rtx_pro_6000_blackwell")
        >>> spec.vram_gb
        96
    """
    return _GPU_REGISTRY.get(name.lower())


def get_gpu_by_compute_capability(compute_cap: str) -> list[GPUSpec]:
    """Get all GPUs matching a compute capability.

    Args:
        compute_cap: Compute capability version (e.g., "12.0", "9.0")

    Returns:
        List of matching GPUSpec instances

    Examples:
        >>> specs = get_gpu_by_compute_capability("12.0")
        >>> len(specs)
        2
    """
    return [
        spec
        for spec in _GPU_REGISTRY.values()
        if spec.compute_capability == compute_cap
    ]


def get_gpu_by_architecture(arch: str) -> list[GPUSpec]:
    """Get all GPUs matching an architecture codename.

    Args:
        arch: Architecture string (e.g., "sm_120", "sm_90")

    Returns:
        List of matching GPUSpec instances

    Examples:
        >>> specs = get_gpu_by_architecture("sm_90")
        >>> len(specs)
        2
    """
    return [spec for spec in _GPU_REGISTRY.values() if spec.arch == arch]


def list_all_gpus() -> list[GPUSpec]:
    """Get all registered GPU specifications.

    Returns:
        List of all GPUSpec instances
    """
    return list(_GPU_REGISTRY.values())


def register_custom_gpu(name: str, spec: GPUSpec) -> None:
    """Register a custom GPU specification.

    Args:
        name: Unique identifier for the GPU
        spec: GPUSpec instance

    Raises:
        ValueError: If name already registered
    """
    if name.lower() in _GPU_REGISTRY:
        raise ValueError(f"GPU '{name}' already registered")
    _GPU_REGISTRY[name.lower()] = spec


def detect_current_gpu() -> Optional[GPUSpec]:
    """Detect current GPU from nvidia-smi and return matching spec.

    This attempts to parse nvidia-smi output and match against known GPUs.
    Falls back to None if nvidia-smi unavailable or GPU not recognized.

    Returns:
        Matched GPUSpec or None if detection fails

    Note:
        Requires nvidia-smi to be available in PATH
    """
    try:
        import subprocess

        # Query GPU name and compute capability
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split("\n")
        if not lines:
            return None

        # Parse first GPU
        parts = lines[0].split(",")
        if len(parts) < 2:
            return None

        gpu_name = parts[0].strip()

        # Try exact name match first
        for spec in _GPU_REGISTRY.values():
            if spec.name.lower() == gpu_name.lower():
                return spec

        # Try partial name match
        gpu_name_lower = gpu_name.lower()
        for spec in _GPU_REGISTRY.values():
            if gpu_name_lower in spec.name.lower():
                return spec

        return None

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None
