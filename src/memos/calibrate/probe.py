"""
Per-node local probe: measures HBM, peer GPU, host DRAM, and NVMe tiers.

Designed to run once per node (via srun --ntasks-per-node=1 or equivalent).
Writes a JSON result that _assemble later aggregates across nodes.

Primary tools (all publicly available):
  - nvbandwidth (github.com/NVIDIA/nvbandwidth) - GPU<->GPU, GPU<->Host
  - fio          (open source)                   - NVMe / storage
  - libcudart    (CUDA toolkit)                  - fallback memcpy
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TierResult:
    name: str
    scope: str  # "device" | "node"
    capacity_gb: float
    bandwidth_gbps: float
    latency_us: float
    multiplicity: int = 1
    method: str = ""


@dataclass
class ProbeResult:
    hostname: str
    gpu_name: str
    gpu_count: int
    tiers: list[TierResult] = field(default_factory=list)
    software: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GPU / topology detection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GPUInfo:
    index: int
    name: str
    hbm_capacity_mb: int
    numa_node: int
    bus_id: str


def detect_gpus() -> list[GPUInfo]:
    """Query nvidia-smi for GPU names, memory, bus ID, and NUMA affinity."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,gpu_bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        idx = int(parts[0])
        name = parts[1]
        mem_mb = int(parts[2])
        bus_id = parts[3].lower()
        numa_node = _gpu_numa_node(bus_id)
        gpus.append(
            GPUInfo(
                index=idx,
                name=name,
                hbm_capacity_mb=mem_mb,
                numa_node=numa_node,
                bus_id=bus_id,
            )
        )
    return gpus


def _gpu_numa_node(bus_id: str) -> int:
    for prefix in (bus_id, f"0000:{bus_id.split(':', 1)[-1]}"):
        p = Path(f"/sys/bus/pci/devices/{prefix}/numa_node")
        if p.exists():
            val = int(p.read_text().strip())
            return val if val >= 0 else 0
    return 0


@dataclass(slots=True)
class NUMANode:
    node_id: int
    memory_mb: int
    distances: list[int]


def detect_numa() -> list[NUMANode]:
    """Read NUMA topology from sysfs."""
    base = Path("/sys/devices/system/node")
    if not base.exists():
        return []
    nodes = []
    for nd in sorted(base.glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
        node_id = int(nd.name[4:])
        mem_mb = 0
        meminfo = nd / "meminfo"
        if meminfo.exists():
            for line in meminfo.read_text().splitlines():
                if "MemTotal" in line:
                    mem_mb = int(line.split()[3]) // 1024
                    break
        distances: list[int] = []
        dist_file = nd / "distance"
        if dist_file.exists():
            distances = [int(x) for x in dist_file.read_text().split()]
        nodes.append(NUMANode(node_id=node_id, memory_mb=mem_mb, distances=distances))
    return nodes


def detect_gpu_interconnect(gpu_count: int) -> dict[str, Any]:
    """Parse nvidia-smi topo to determine GPU interconnect type and pairs."""
    try:
        out = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"type": "unknown", "pairs": [], "raw": ""}

    nvlink_pairs: list[tuple[int, int]] = []
    pcie_pairs: list[tuple[int, int]] = []
    lines = out.strip().splitlines()

    header_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("GPU"):
            header_idx = i
            break

    if header_idx >= 0:
        for i in range(gpu_count):
            row_idx = header_idx + 1 + i
            if row_idx >= len(lines):
                break
            cols = lines[row_idx].split()
            for j in range(i + 1, gpu_count):
                col_idx = 1 + j
                if col_idx < len(cols):
                    val = cols[col_idx]
                    if "NV" in val:
                        nvlink_pairs.append((i, j))
                    elif "PIX" in val or "PHB" in val or "PXB" in val:
                        pcie_pairs.append((i, j))

    if nvlink_pairs:
        return {
            "type": "nvlink",
            "pairs": nvlink_pairs,
            "pcie_pairs": pcie_pairs,
            "raw": out,
        }
    elif pcie_pairs:
        return {
            "type": "pcie",
            "pairs": pcie_pairs,
            "pcie_pairs": pcie_pairs,
            "raw": out,
        }
    else:
        all_pairs = [(i, j) for i in range(gpu_count) for j in range(i + 1, gpu_count)]
        return {"type": "unknown", "pairs": all_pairs, "pcie_pairs": [], "raw": out}


# ---------------------------------------------------------------------------
# Software version detection
# ---------------------------------------------------------------------------


def detect_software_versions() -> dict[str, str]:
    """Capture CUDA, driver, NCCL, and other relevant versions."""
    versions: dict[str, str] = {}

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        )
        versions["driver_version"] = out.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        for line in out.split("\n"):
            if "release" in line:
                versions["cuda_version"] = (
                    line.split("release")[-1].split(",")[0].strip()
                )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        import torch

        versions["nccl_version"] = ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception:
        pass

    versions["python_version"] = platform.python_version()

    container_image = os.environ.get("NVIDIA_PYTORCH_VERSION") or os.environ.get(
        "ENROOT_IMAGE", ""
    )
    if container_image:
        versions["container_image"] = container_image

    return versions


# ---------------------------------------------------------------------------
# nvbandwidth integration (primary measurement tool)
# ---------------------------------------------------------------------------


def _nvbandwidth_available() -> bool:
    return shutil.which("nvbandwidth") is not None


def _run_nvbandwidth(
    testcase: str, extra_args: list[str] | None = None
) -> dict[str, Any]:
    cmd = ["nvbandwidth", "--testcase", testcase, "--json"]
    if extra_args:
        cmd.extend(extra_args)
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        return json.loads(out)
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        return {}


def _extract_nvbw_bandwidth(result: dict[str, Any]) -> float:
    try:
        testcases = result.get("testcases", [])
        if not testcases:
            return 0.0
        results = testcases[0].get("results", [])
        if not results:
            return 0.0
        bandwidths = []
        for r in results:
            bw = r.get("bandwidth")
            if bw is not None:
                bandwidths.append(float(bw))
        return max(bandwidths) if bandwidths else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def _extract_nvbw_bandwidth_for_pair(
    result: dict[str, Any], src: int, dst: int
) -> float:
    try:
        testcases = result.get("testcases", [])
        if not testcases:
            return 0.0
        results = testcases[0].get("results", [])
        for r in results:
            if r.get("src") == src and r.get("dst") == dst:
                return float(r.get("bandwidth", 0.0))
            if r.get("device") == src:
                return float(r.get("bandwidth", 0.0))
        bandwidths = [
            float(r.get("bandwidth", 0)) for r in results if r.get("bandwidth")
        ]
        return max(bandwidths) if bandwidths else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def measure_hbm_nvbandwidth() -> float:
    result = _run_nvbandwidth("device_to_device_memcpy_read_sm")
    return _extract_nvbw_bandwidth(result)


def measure_peer_nvbandwidth(src: int, dst: int) -> float:
    result = _run_nvbandwidth("device_to_device_bidirectional_memcpy_read_sm")
    bw = _extract_nvbw_bandwidth_for_pair(result, src, dst)
    if bw == 0.0:
        result = _run_nvbandwidth("device_to_device_memcpy_read_sm")
        bw = _extract_nvbw_bandwidth_for_pair(result, src, dst)
    return bw


def measure_host_device_nvbandwidth() -> float:
    result = _run_nvbandwidth("host_to_device_memcpy_sm")
    return _extract_nvbw_bandwidth(result)


# ---------------------------------------------------------------------------
# CUDA runtime bindings (fallback when nvbandwidth unavailable)
# ---------------------------------------------------------------------------

_libcudart: ctypes.CDLL | None = None

_BW_BUF_BYTES = 256 * 1024 * 1024
_BW_ITERATIONS = 20
_BW_WARMUP = 5
_LAT_BUF_BYTES = 4
_LAT_ITERATIONS = 200
_LAT_WARMUP = 50


def _cuda_rt() -> ctypes.CDLL:
    global _libcudart
    if _libcudart is not None:
        return _libcudart
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
        try:
            _libcudart = ctypes.CDLL(name)
            return _libcudart
        except OSError:
            continue
    path = ctypes.util.find_library("cudart")
    if path:
        _libcudart = ctypes.CDLL(path)
        return _libcudart
    raise RuntimeError("Cannot find libcudart. Is CUDA installed?")


def _check_cuda(err: int) -> None:
    if err != 0:
        rt = _cuda_rt()
        rt.cudaGetErrorString.restype = ctypes.c_char_p
        msg = rt.cudaGetErrorString(err).decode()
        raise RuntimeError(f"CUDA error {err}: {msg}")


@dataclass(slots=True)
class BandwidthResult:
    bandwidth_gbps: float
    latency_us: float


def _measure_memcpy_bw(
    rt: ctypes.CDLL,
    dst: ctypes.c_void_p,
    src: ctypes.c_void_p,
    nbytes: int,
    kind: int,
) -> float:
    start_ev = ctypes.c_void_p()
    end_ev = ctypes.c_void_p()
    _check_cuda(rt.cudaEventCreate(ctypes.byref(start_ev)))
    _check_cuda(rt.cudaEventCreate(ctypes.byref(end_ev)))

    for _ in range(_BW_WARMUP):
        _check_cuda(rt.cudaMemcpy(dst, src, nbytes, kind))
    _check_cuda(rt.cudaDeviceSynchronize())

    _check_cuda(rt.cudaEventRecord(start_ev, None))
    for _ in range(_BW_ITERATIONS):
        _check_cuda(rt.cudaMemcpy(dst, src, nbytes, kind))
    _check_cuda(rt.cudaEventRecord(end_ev, None))
    _check_cuda(rt.cudaEventSynchronize(end_ev))

    elapsed_ms = ctypes.c_float()
    _check_cuda(rt.cudaEventElapsedTime(ctypes.byref(elapsed_ms), start_ev, end_ev))
    rt.cudaEventDestroy(start_ev)
    rt.cudaEventDestroy(end_ev)

    total_bytes = nbytes * _BW_ITERATIONS
    return (total_bytes / (elapsed_ms.value / 1000.0)) / 1e9


def _measure_memcpy_latency(
    rt: ctypes.CDLL,
    dst: ctypes.c_void_p,
    src: ctypes.c_void_p,
    kind: int,
) -> float:
    start_ev = ctypes.c_void_p()
    end_ev = ctypes.c_void_p()
    _check_cuda(rt.cudaEventCreate(ctypes.byref(start_ev)))
    _check_cuda(rt.cudaEventCreate(ctypes.byref(end_ev)))

    for _ in range(_LAT_WARMUP):
        _check_cuda(rt.cudaMemcpy(dst, src, _LAT_BUF_BYTES, kind))
    _check_cuda(rt.cudaDeviceSynchronize())

    _check_cuda(rt.cudaEventRecord(start_ev, None))
    for _ in range(_LAT_ITERATIONS):
        _check_cuda(rt.cudaMemcpy(dst, src, _LAT_BUF_BYTES, kind))
    _check_cuda(rt.cudaEventRecord(end_ev, None))
    _check_cuda(rt.cudaEventSynchronize(end_ev))

    elapsed_ms = ctypes.c_float()
    _check_cuda(rt.cudaEventElapsedTime(ctypes.byref(elapsed_ms), start_ev, end_ev))
    rt.cudaEventDestroy(start_ev)
    rt.cudaEventDestroy(end_ev)

    return (elapsed_ms.value * 1000.0) / _LAT_ITERATIONS


def measure_hbm_cuda(device: int = 0) -> BandwidthResult:
    rt = _cuda_rt()
    _check_cuda(rt.cudaSetDevice(device))
    src = ctypes.c_void_p()
    dst = ctypes.c_void_p()
    _check_cuda(rt.cudaMalloc(ctypes.byref(src), _BW_BUF_BYTES))
    _check_cuda(rt.cudaMalloc(ctypes.byref(dst), _BW_BUF_BYTES))
    try:
        bw = _measure_memcpy_bw(rt, dst, src, _BW_BUF_BYTES, 3)
        lat = _measure_memcpy_latency(rt, dst, src, 3)
    finally:
        rt.cudaFree(src)
        rt.cudaFree(dst)
    return BandwidthResult(bandwidth_gbps=bw, latency_us=lat)


def measure_peer_cuda(src_device: int, dst_device: int) -> BandwidthResult:
    rt = _cuda_rt()
    _check_cuda(rt.cudaSetDevice(src_device))
    err = rt.cudaDeviceEnablePeerAccess(dst_device, 0)
    if err not in (0, 704):
        _check_cuda(err)
    _check_cuda(rt.cudaSetDevice(dst_device))
    err = rt.cudaDeviceEnablePeerAccess(src_device, 0)
    if err not in (0, 704):
        _check_cuda(err)

    _check_cuda(rt.cudaSetDevice(src_device))
    src_buf = ctypes.c_void_p()
    _check_cuda(rt.cudaMalloc(ctypes.byref(src_buf), _BW_BUF_BYTES))
    _check_cuda(rt.cudaSetDevice(dst_device))
    dst_buf = ctypes.c_void_p()
    _check_cuda(rt.cudaMalloc(ctypes.byref(dst_buf), _BW_BUF_BYTES))

    try:
        _check_cuda(rt.cudaSetDevice(src_device))
        bw = _measure_memcpy_bw(rt, dst_buf, src_buf, _BW_BUF_BYTES, 3)
        lat = _measure_memcpy_latency(rt, dst_buf, src_buf, 3)
    finally:
        _check_cuda(rt.cudaSetDevice(src_device))
        rt.cudaFree(src_buf)
        _check_cuda(rt.cudaSetDevice(dst_device))
        rt.cudaFree(dst_buf)
    return BandwidthResult(bandwidth_gbps=bw, latency_us=lat)


def measure_host_device_cuda(
    device: int = 0, numa_node: int | None = None
) -> BandwidthResult:
    rt = _cuda_rt()
    _check_cuda(rt.cudaSetDevice(device))
    dev_buf = ctypes.c_void_p()
    _check_cuda(rt.cudaMalloc(ctypes.byref(dev_buf), _BW_BUF_BYTES))

    host_buf, numa_allocated = _alloc_host_on_numa(rt, _BW_BUF_BYTES, numa_node)

    try:
        bw = _measure_memcpy_bw(rt, dev_buf, host_buf, _BW_BUF_BYTES, 1)
        lat = _measure_memcpy_latency(rt, dev_buf, host_buf, 1)
    finally:
        _free_host_numa(rt, host_buf, _BW_BUF_BYTES, numa_allocated)
        rt.cudaFree(dev_buf)
    return BandwidthResult(bandwidth_gbps=bw, latency_us=lat)


def _alloc_host_on_numa(
    rt: ctypes.CDLL, size: int, numa_node: int | None
) -> tuple[ctypes.c_void_p, bool]:
    """Allocate host memory on a specific NUMA node and register it with CUDA.

    If numa_node is specified, uses libnuma's numa_alloc_onnode to place
    memory on the correct node BEFORE pinning it with CUDA. This is
    necessary because cudaHostAlloc pins pages in place, making them
    immovable by mbind.

    Returns (pointer, True) if numa-allocated, or (pointer, False) if
    falling back to cudaHostAlloc.
    """
    if numa_node is not None:
        try:
            numa = ctypes.CDLL("libnuma.so.1")
            numa.numa_alloc_onnode.restype = ctypes.c_void_p
            ptr = numa.numa_alloc_onnode(ctypes.c_size_t(size), ctypes.c_int(numa_node))
            if ptr:
                host_buf = ctypes.c_void_p(ptr)
                cudaHostRegisterDefault = 0
                err = rt.cudaHostRegister(
                    host_buf, ctypes.c_size_t(size), cudaHostRegisterDefault
                )
                if err == 0:
                    return host_buf, True
                numa.numa_free(host_buf, ctypes.c_size_t(size))
        except (OSError, AttributeError):
            pass

    host_buf = ctypes.c_void_p()
    _check_cuda(rt.cudaHostAlloc(ctypes.byref(host_buf), size, 0))
    return host_buf, False


def _free_host_numa(
    rt: ctypes.CDLL, buf: ctypes.c_void_p, size: int, numa_allocated: bool
) -> None:
    if numa_allocated:
        rt.cudaHostUnregister(buf)
        try:
            numa = ctypes.CDLL("libnuma.so.1")
            numa.numa_free(buf, ctypes.c_size_t(size))
        except (OSError, AttributeError):
            pass
    else:
        rt.cudaFreeHost(buf)


# ---------------------------------------------------------------------------
# NVMe / storage measurement
# ---------------------------------------------------------------------------


def _find_storage_mount() -> str | None:
    try:
        out = subprocess.check_output(
            ["df", "-B1", "--output=target,size,source"], text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    candidates: list[tuple[str, int]] = []
    for line in out.strip().splitlines()[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount, size_str, source = parts[0], parts[1], parts[2]
        if "nvme" in source or mount in ("/raid", "/scratch", "/local", "/tmp"):
            try:
                candidates.append((mount, int(size_str)))
            except ValueError:
                continue

    if not candidates:
        for line in out.strip().splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 3 and parts[0] not in ("/", "/boot", "/boot/efi"):
                try:
                    size = int(parts[1])
                    if size > 50 * (1024**3):
                        candidates.append((parts[0], size))
                except ValueError:
                    continue

    if candidates:
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]
    return None


def _find_shared_fs_mount() -> str | None:
    try:
        out = subprocess.check_output(
            ["mount", "-t", "lustre,gpfs,nfs,nfs4"], text=True
        )
        for line in out.strip().splitlines():
            parts = line.split()
            if len(parts) >= 3 and "on" in parts:
                mount_idx = parts.index("on") + 1
                if mount_idx < len(parts):
                    return parts[mount_idx]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        mounts = Path("/proc/mounts").read_text()
        for line in mounts.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[2] in ("lustre", "gpfs", "nfs", "nfs4"):
                return parts[1]
    except OSError:
        pass
    return None


def measure_storage_bandwidth(mount_point: str) -> BandwidthResult:
    if shutil.which("fio"):
        bw = _fio_seq_read_bw(mount_point)
        lat = _fio_rand_read_latency(mount_point)
    else:
        bw = _direct_io_seq_read_bw(mount_point)
        lat = _direct_io_rand_read_latency(mount_point)
    return BandwidthResult(bandwidth_gbps=bw, latency_us=lat)


def _fio_seq_read_bw(mount_point: str) -> float:
    try:
        out = subprocess.check_output(
            [
                "fio",
                "--name=seqread",
                "--rw=read",
                "--bs=1M",
                "--size=256M",
                "--numjobs=1",
                "--direct=1",
                "--ioengine=libaio",
                "--iodepth=32",
                f"--directory={mount_point}",
                "--output-format=json",
                "--time_based",
                "--runtime=5",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
        return data["jobs"][0]["read"]["bw_bytes"] / 1e9
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        KeyError,
        json.JSONDecodeError,
    ):
        return 0.0


def _fio_rand_read_latency(mount_point: str) -> float:
    try:
        out = subprocess.check_output(
            [
                "fio",
                "--name=randread",
                "--rw=randread",
                "--bs=4k",
                "--size=64M",
                "--numjobs=1",
                "--direct=1",
                "--ioengine=libaio",
                "--iodepth=1",
                f"--directory={mount_point}",
                "--output-format=json",
                "--time_based",
                "--runtime=5",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
        return data["jobs"][0]["read"]["lat_ns"]["mean"] / 1000.0
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        KeyError,
        json.JSONDecodeError,
    ):
        return 0.0


def _direct_io_seq_read_bw(mount_point: str) -> float:
    test_file = os.path.join(mount_point, ".memos_calibrate_bw")
    buf_size = 128 * 1024 * 1024
    iterations = 10
    try:
        with open(test_file, "wb") as f:
            f.write(os.urandom(buf_size))
            f.flush()
            os.fsync(f.fileno())
        _drop_file_cache(test_file)

        fd = os.open(test_file, os.O_RDONLY | os.O_DIRECT)
        aligned_buf = _aligned_alloc(buf_size, 4096)
        try:
            for _ in range(2):
                os.lseek(fd, 0, os.SEEK_SET)
                os.readv(fd, [aligned_buf])
            start = time.perf_counter()
            for _ in range(iterations):
                os.lseek(fd, 0, os.SEEK_SET)
                os.readv(fd, [aligned_buf])
            elapsed = time.perf_counter() - start
        finally:
            os.close(fd)
        return (buf_size * iterations / elapsed) / 1e9
    except OSError:
        return 0.0
    finally:
        try:
            os.unlink(test_file)
        except OSError:
            pass


def _direct_io_rand_read_latency(mount_point: str) -> float:
    import random as _random

    test_file = os.path.join(mount_point, ".memos_calibrate_lat")
    file_size = 64 * 1024 * 1024
    block_size = 4096
    iterations = 500
    try:
        with open(test_file, "wb") as f:
            f.write(os.urandom(file_size))
            f.flush()
            os.fsync(f.fileno())
        _drop_file_cache(test_file)

        fd = os.open(test_file, os.O_RDONLY | os.O_DIRECT)
        aligned_buf = _aligned_alloc(block_size, 4096)
        max_off = (file_size // block_size) - 1
        offsets = [_random.randint(0, max_off) * block_size for _ in range(iterations)]
        try:
            for off in offsets[:50]:
                os.lseek(fd, off, os.SEEK_SET)
                os.readv(fd, [aligned_buf])
            start = time.perf_counter()
            for off in offsets:
                os.lseek(fd, off, os.SEEK_SET)
                os.readv(fd, [aligned_buf])
            elapsed = time.perf_counter() - start
        finally:
            os.close(fd)
        return (elapsed / iterations) * 1e6
    except OSError:
        return 0.0
    finally:
        try:
            os.unlink(test_file)
        except OSError:
            pass


def _drop_file_cache(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass


def _aligned_alloc(size: int, alignment: int) -> memoryview:
    buf = bytearray(size + alignment)
    addr = ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))
    offset = (alignment - addr % alignment) % alignment
    return memoryview(buf)[offset : offset + size]


def _storage_capacity_gb(mount_point: str) -> float:
    try:
        stat = os.statvfs(mount_point)
        return (stat.f_blocks * stat.f_frsize) / (1024**3)
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# Probe orchestrator
# ---------------------------------------------------------------------------


def probe_node(verbose: bool = True) -> ProbeResult:
    """Run full local probe: measure all locally reachable tiers."""
    from datetime import datetime

    cal_start = time.perf_counter()
    use_nvbw = _nvbandwidth_available()

    if verbose:
        _log(
            f"nvbandwidth: {'available' if use_nvbw else 'NOT FOUND (using cudaMemcpy fallback)'}"
        )
        _log(
            f"fio:         {'available' if shutil.which('fio') else 'NOT FOUND (using O_DIRECT fallback)'}"
        )
        _log("")

    gpus = detect_gpus()
    if not gpus:
        raise RuntimeError("No GPUs detected. Is nvidia-smi available?")
    gpu_count = len(gpus)
    gpu_name = gpus[0].name
    if verbose:
        _log(f"  {gpu_count}x {gpu_name} ({gpus[0].hbm_capacity_mb} MiB HBM each)")

    interconnect = detect_gpu_interconnect(gpu_count)
    numa_nodes = detect_numa()
    cpu_nodes = [n for n in numa_nodes if n.memory_mb > 1024]
    if verbose:
        _log(
            f"  Interconnect: {interconnect['type']} ({len(interconnect['pairs'])} pairs)"
        )
        _log(f"  NUMA: {len(numa_nodes)} total nodes, {len(cpu_nodes)} CPU nodes")
        _log("")

    tiers: list[TierResult] = []

    # ===== TIER: HBM =====
    if verbose:
        _log("Measuring HBM bandwidth...")
    if use_nvbw:
        hbm_bw = measure_hbm_nvbandwidth()
        hbm_method = "nvbandwidth:device_to_device_memcpy_read_sm"
        hbm_lat_result = measure_hbm_cuda(device=0)
        hbm_lat = hbm_lat_result.latency_us
        if hbm_bw == 0.0:
            hbm_bw = hbm_lat_result.bandwidth_gbps
            hbm_method = "cudaMemcpy:device_to_device"
    else:
        result = measure_hbm_cuda(device=0)
        hbm_bw = result.bandwidth_gbps
        hbm_lat = result.latency_us
        hbm_method = "cudaMemcpy:device_to_device"

    tiers.append(
        TierResult(
            name="hbm",
            scope="device",
            capacity_gb=round(gpus[0].hbm_capacity_mb / 1024, 1),
            bandwidth_gbps=round(hbm_bw, 1),
            latency_us=round(hbm_lat, 3),
            multiplicity=1,
            method=hbm_method,
        )
    )
    if verbose:
        _log(f"  BW={hbm_bw:.1f} GB/s  Lat={hbm_lat:.3f} us  [{hbm_method}]")

    # ===== TIER: Peer HBM =====
    if gpu_count > 1 and interconnect["pairs"]:
        if verbose:
            _log("Measuring peer GPU bandwidth...")
        src_dev, dst_dev = interconnect["pairs"][0]

        if use_nvbw:
            peer_bw = measure_peer_nvbandwidth(src_dev, dst_dev)
            peer_method = "nvbandwidth:device_to_device_bidirectional_memcpy_read_sm"
            peer_lat_result = measure_peer_cuda(src_dev, dst_dev)
            peer_lat = peer_lat_result.latency_us
            if peer_bw == 0.0:
                peer_bw = peer_lat_result.bandwidth_gbps
                peer_method = "cudaMemcpy:peer_device_to_device"
        else:
            result = measure_peer_cuda(src_dev, dst_dev)
            peer_bw = result.bandwidth_gbps
            peer_lat = result.latency_us
            peer_method = "cudaMemcpy:peer_device_to_device"

        tiers.append(
            TierResult(
                name="peer_hbm",
                scope="node",
                capacity_gb=round(gpus[1].hbm_capacity_mb / 1024, 1),
                bandwidth_gbps=round(peer_bw, 1),
                latency_us=round(peer_lat, 3),
                multiplicity=gpu_count - 1,
                method=peer_method,
            )
        )
        if verbose:
            _log(f"  BW={peer_bw:.1f} GB/s  Lat={peer_lat:.3f} us  [{peer_method}]")

    # ===== TIER: Host DRAM =====
    if cpu_nodes:
        gpu_numa = gpus[0].numa_node
        local_nodes = [n for n in cpu_nodes if n.node_id == gpu_numa]
        remote_nodes = [n for n in cpu_nodes if n.node_id != gpu_numa]
        if not local_nodes:
            local_nodes = [cpu_nodes[0]]
            remote_nodes = cpu_nodes[1:]

        if local_nodes:
            node = local_nodes[0]
            if verbose:
                _log(f"Measuring host DRAM bandwidth (NUMA {node.node_id}, local)...")

            if use_nvbw:
                host_bw = measure_host_device_nvbandwidth()
                host_method = "nvbandwidth:host_to_device_memcpy_sm"
                host_lat_result = measure_host_device_cuda(
                    device=0, numa_node=node.node_id
                )
                host_lat = host_lat_result.latency_us
                if host_bw == 0.0:
                    host_bw = host_lat_result.bandwidth_gbps
                    host_method = "cudaMemcpy:host_to_device"
            else:
                result = measure_host_device_cuda(device=0, numa_node=node.node_id)
                host_bw = result.bandwidth_gbps
                host_lat = result.latency_us
                host_method = "cudaMemcpy:host_to_device:numa_local"

            tiers.append(
                TierResult(
                    name="host_dram",
                    scope="node",
                    capacity_gb=round(node.memory_mb / 1024, 1),
                    bandwidth_gbps=round(host_bw, 1),
                    latency_us=round(host_lat, 3),
                    multiplicity=1,
                    method=host_method,
                )
            )
            if verbose:
                _log(f"  BW={host_bw:.1f} GB/s  Lat={host_lat:.3f} us  [{host_method}]")

        if remote_nodes:
            node = remote_nodes[0]
            if verbose:
                _log(f"Measuring host DRAM bandwidth (NUMA {node.node_id}, remote)...")

            result = measure_host_device_cuda(device=0, numa_node=node.node_id)
            remote_bw = result.bandwidth_gbps
            remote_lat = result.latency_us
            remote_method = "cudaMemcpy:host_to_device:numa_remote"

            tiers.append(
                TierResult(
                    name="host_dram_remote",
                    scope="node",
                    capacity_gb=round(node.memory_mb / 1024, 1),
                    bandwidth_gbps=round(remote_bw, 1),
                    latency_us=round(remote_lat, 3),
                    multiplicity=len(remote_nodes),
                    method=remote_method,
                )
            )
            if verbose:
                _log(
                    f"  BW={remote_bw:.1f} GB/s  Lat={remote_lat:.3f} us  [{remote_method}]"
                )

    # ===== TIER: NVMe =====
    nvme_mount = _find_storage_mount()
    if nvme_mount:
        if verbose:
            _log(f"Measuring NVMe bandwidth ({nvme_mount})...")
        nvme_result = measure_storage_bandwidth(nvme_mount)
        nvme_cap = _storage_capacity_gb(nvme_mount)
        nvme_method = (
            "fio:seqread_1M+randread_4k"
            if shutil.which("fio")
            else "direct_io:seqread+randread"
        )
        tiers.append(
            TierResult(
                name="nvme",
                scope="node",
                capacity_gb=round(nvme_cap, 0),
                bandwidth_gbps=round(nvme_result.bandwidth_gbps, 2),
                latency_us=round(nvme_result.latency_us, 1),
                method=nvme_method,
            )
        )
        if verbose:
            _log(
                f"  BW={nvme_result.bandwidth_gbps:.2f} GB/s  Lat={nvme_result.latency_us:.1f} us  [{nvme_method}]"
            )
    elif verbose:
        _log("No local NVMe/scratch mount found, skipping.")

    # ===== TIER: Shared filesystem =====
    shared_mount = _find_shared_fs_mount()
    if shared_mount and shared_mount != nvme_mount:
        if verbose:
            _log(f"Measuring shared filesystem ({shared_mount})...")
        shared_result = measure_storage_bandwidth(shared_mount)
        shared_cap = _storage_capacity_gb(shared_mount)
        shared_method = (
            "fio:seqread_1M+randread_4k"
            if shutil.which("fio")
            else "direct_io:seqread+randread"
        )
        tiers.append(
            TierResult(
                name="shared_storage",
                scope="cluster",
                capacity_gb=round(shared_cap, 0),
                bandwidth_gbps=round(shared_result.bandwidth_gbps, 2),
                latency_us=round(shared_result.latency_us, 1),
                method=shared_method,
            )
        )
        if verbose:
            _log(
                f"  BW={shared_result.bandwidth_gbps:.2f} GB/s  Lat={shared_result.latency_us:.1f} us"
            )
    elif verbose:
        _log("No shared filesystem detected, skipping.")

    cal_elapsed = time.perf_counter() - cal_start
    software = detect_software_versions()
    hostname = os.environ.get("HOSTNAME", os.environ.get("HOST", _get_hostname()))

    result = ProbeResult(
        hostname=hostname,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        tiers=tiers,
        software=software,
        metadata={
            "calibrated_at": datetime.now().isoformat(),
            "calibration_duration_sec": round(cal_elapsed, 1),
            "interconnect_type": interconnect["type"],
            "numa_nodes_total": len(numa_nodes),
            "numa_nodes_cpu": len(cpu_nodes),
            "tools_used": {
                "nvbandwidth": use_nvbw,
                "fio": shutil.which("fio") is not None,
            },
        },
    )

    if verbose:
        _log("")
        _log(f"Probe complete in {cal_elapsed:.1f}s - {len(tiers)} tiers measured")

    return result


def probe_to_json(result: ProbeResult) -> str:
    """Serialize ProbeResult to JSON string."""
    doc = {
        "hostname": result.hostname,
        "gpu_name": result.gpu_name,
        "gpu_count": result.gpu_count,
        "tiers": [asdict(t) for t in result.tiers],
        "software": result.software,
        "metadata": result.metadata,
    }
    return json.dumps(doc, indent=2)


def _get_hostname() -> str:
    try:
        return subprocess.check_output(["hostname"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _log(msg: str) -> None:
    import click

    click.echo(f"[probe] {msg}")
