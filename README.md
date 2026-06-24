# memos

A benchmark suite, memory roofline model, and scheduling toolkit for AI inference memory systems.

## What is this?

Modern AI inference is increasingly constrained by memory, not compute. Systems like PagedAttention, Mooncake, HiCache, LMCache, and NVIDIA Dynamo already manage KV cache across tiered memory hierarchies (HBM, DRAM, NVMe, remote storage). But the field lacks three things:

1. **A formal cost model for placement decisions.** Existing systems use heuristic policies (LRU, LFU) to decide what stays in fast memory and what gets evicted. None of them model recomputation as an alternative to fetching - even though regenerating KV cache from tokens is sometimes cheaper than loading it from a slow tier. No system formally reasons about the time-dependent cost of restoring evicted data.

2. **A memory roofline model.** The compute roofline (Williams et al., 2009) transformed how we reason about whether workloads are compute-bound or memory-bandwidth-bound. There is no equivalent for AI memory systems. We don't have standardized ways to ask: "Is this workload stalling on tier-crossing transfers? Would more DRAM bandwidth help, or is it already compute-bound? How much effective capacity does tiered memory actually buy?"

3. **Cross-hardware reference measurements.** The H100 accesses CPU DRAM at ~25 GB/s over PCIe. The GB200 accesses it at 900 GB/s over NVLink-C2C - a 36x increase that fundamentally changes whether DRAM is "slow swap" or "fast L2." Nobody has systematically measured how this shift changes the memory roofline, the recompute-vs-offload crossover point, or the capacity amplification factor across workloads.

memos fills these gaps.

## What does memos provide?

### Metrics

Novel metrics for memory-centric AI inference, implemented as instrumentation for inference engines:

- **bytes/token (BPT)** - Total bytes of data movement per generated token. Analogous to arithmetic intensity, but for memory. Reveals whether a workload is dominated by data movement.
- **stall_time/token (STPT)** - Microseconds spent waiting for tier-crossing transfers per token. Analogous to pipeline bubbles. Reveals the latency cost of memory virtualization.
- **page_faults/token (PFPT)** - Tier misses (block needed but not resident in HBM) per token. Analogous to cache miss rate. Reveals eviction policy effectiveness.
- **capacity amplification factor (CAF)** - Effective memory capacity divided by physical HBM. How much does tiered memory amplify what you can serve?
- **tier hit rates** - Per-tier (HBM, peer GPU, DRAM, remote DRAM, NVMe) hit rates over time.

### Benchmarks

Workloads designed to stress memory systems in ways existing benchmarks don't:

- **Context length sweep** - Scale context from 4K to 256K+ tokens, measuring how memory behavior changes.
- **Thrashing sweep** - Ramp concurrent agent sessions until throughput collapses. Find the thrashing onset point.
- **Sleeping agents** - Many sessions, few active. Simulate agentic workloads with tool-call pauses and resumption.
- **Recompute crossover** - Find the context length where offloading becomes cheaper than recomputation, for each tier, on each hardware platform.

### Memory roofline model

A roofline model for AI memory systems:

```
tokens/sec = min(
    compute_ceiling:    peak_flops / flops_per_token,
    bandwidth_ceiling:  effective_bw / bytes_per_token,
    fault_ceiling:      1 / (fault_rate * fault_latency)
)
```

The roofline classifies workloads as compute-bound, bandwidth-bound, or fault-bound, and visualizes where the optimization opportunity lies. Running the same workload on H100 vs GB200 shows different bottleneck regimes because the DRAM bandwidth ceiling shifts by 36x.

### Cost-model-driven eviction

An eviction policy that treats recomputation as a memory tier:

```
evict(block) = argmin over candidates: future_restoration_cost(block)

restoration_cost(block) = min(
    fetch_from_dram(block),
    fetch_from_peer_gpu(block),
    fetch_from_nvme(block),
    recompute(block)        # cost depends on tokens since eviction
)
```

Evicts the block that is cheapest to restore later - not the least recently used. Recently generated blocks near the end of a sequence are cheap to recompute; old system-prompt blocks are expensive. This naturally aligns with the inverted age-importance pattern observed in agentic workloads (cf. RoleKV), but for the right reason (cost) rather than a heuristic (role tags).

### Copy-on-write contexts

Agent-level fork, snapshot, and rollback over KV cache block tables:

- `fork(context)` - Create a branch that shares all existing blocks (zero-copy). New tokens on the branch allocate new blocks.
- `snapshot(context)` - Save a restore point.
- `rollback(context, snapshot)` - Revert to a prior state.

Enables speculative tool calls, branching agent execution, and failure recovery without duplicating the entire KV cache.

## Hardware progression

memos benchmarks are designed to run across a hardware progression, where each step adds memory tiers:

```
Single H100          HBM <-> DRAM (PCIe, ~25 GB/s)
    |
Multi H100           HBM <-> Peer HBM (NVLink) <-> DRAM (PCIe)
    |
Multi-node H100      HBM <-> Peer HBM <-> DRAM <-> Remote DRAM (RDMA)
    |
Single GB200         HBM <-> DRAM (C2C, 900 GB/s)
    |
Multi GB200          HBM <-> DRAM (C2C) <-> Peer HBM (NVLink)
    |
Multi-node GB200     HBM <-> DRAM (C2C) <-> Peer HBM <-> Remote DRAM (RDMA)
```

The H100-to-GB200 comparison at each scale level is independently interesting. On GB200, the DRAM tier may be faster than peer GPU access for some patterns, inverting the traditional memory hierarchy.

## Scope

memos does **not** replace systems like Mooncake, HiCache, LMCache, or Dynamo KVBM. It does not rebuild transport layers (use NIXL) or block allocators (use vLLM's). It provides what they are missing:

- A formal cost model for placement decisions
- Recomputation as a first-class alternative to fetching
- A memory roofline framework with cross-hardware reference data
- Agent-aware memory semantics (fork/snapshot/rollback)
- Standardized benchmarks and metrics for memory-centric inference

## Engine support

- **vLLM** - primary, instrumented first
- **SGLang** - planned
- **TensorRT-LLM** - planned

The runner interface is engine-agnostic from day one. Adding an engine means implementing one class.

## Roadmap

- [ ] Metric collectors (bytes/token, stall_time/token, page_faults/token, CAF, tier hit rates)
- [ ] Workload generators (context sweep, thrashing sweep, sleeping agents, recompute crossover)
- [ ] Memory roofline model and plotting tool
- [ ] Single H100 reference measurements
- [ ] Single GB200 reference measurements and cross-architecture comparison
- [ ] Multi H100 measurements (peer HBM as L2 hypothesis)
- [ ] Cost-model library (tier specs, recompute cost functions, placement oracle)
- [ ] Recompute-aware eviction policy for vLLM
- [ ] Multi-node measurements (remote DRAM tier via RDMA)
- [ ] Copy-on-write context manager (fork/snapshot/rollback)
- [ ] Branching agent workload and CoW benchmarks
- [ ] Multi GB200 measurements (tier-ordering inversion analysis)
- [ ] Additional engine support (SGLang, TRT-LLM)
- [ ] Learned placement/eviction policies via RL (using memos benchmarks as training environment, cost-model policy as baseline)

## Research

This section collects the systems, papers, and observations that inform the design of memos.

### Existing tiered KV cache systems

Five systems already manage KV cache across memory hierarchies. memos builds on their infrastructure rather than replacing it, and focuses on the scheduling/policy gaps they leave open.

**PagedAttention / vLLM** ([paper](https://arxiv.org/abs/2309.06180), [code](https://github.com/vllm-project/vllm)). Introduced paged KV cache management inspired by OS virtual memory. Block tables map logical token positions to physical block locations - these are literally page tables. Achieved 2-4x throughput over FasterTransformer and Orca by eliminating fragmentation and enabling block sharing. The foundation that all subsequent systems build on. Recent work (PR #22706) explores elastic KV cache memory pools for dynamic GPU memory sharing across co-located models.

**Mooncake** ([paper](https://www.usenix.org/system/files/fast25-qin.pdf), [code](https://github.com/kvcache-ai/Mooncake)). FAST 2025 Best Paper. Production system for Kimi (Moonshot AI), processing 100B+ tokens/day across thousands of nodes. Separates prefill and decode clusters. Builds a cluster-wide disaggregated KV cache pool by aggregating underutilized CPU, DRAM, and SSD resources via RDMA. Global scheduler (Conductor) selects instances based on cache locality. Achieves up to 525% throughput increase in long-context scenarios. Integrated into vLLM and SGLang. Key observation: KV cache is not local state - it's a cluster-level shareable resource.

**HiCache** ([blog](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/), [design doc](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/hicache_design.md)). SGLang's hierarchical cache, inspired by CPU L1/L2/L3 architecture. GPU memory is L1, host memory is L2, distributed storage (Mooncake, 3FS, NIXL) is L3. Implements layer-wise compute/transfer overlap: loads KV cache for layer N+1 while layer N executes, hiding transfer latency behind computation. Configurable prefetch policies: best-effort mode vs. aggressive staging. Supports cross-TP-size KV reuse (e.g., tp=4 and tp=8 sharing the same storage backend).

**LMCache** ([paper](https://arxiv.org/abs/2510.09665), [code](https://github.com/LMCache/LMCache)). Engine-independent KV cache management daemon. Runs as a standalone process with no fate-sharing with the inference engine - cache survives engine crashes and restarts. Supports tiered offloading (GPU -> CPU -> disk -> remote) with persistence. Non-prefix KV reuse via CacheBlend: reuses cached KV blocks at arbitrary positions, not just prefix matches. First-class control APIs: pin, lookup, cleanup, movement, compression. Demonstrated 15x throughput improvement on multi-turn QA workloads. Key observation: KV cache should be treated as persistent, reusable data, not ephemeral state.

**NVIDIA Dynamo KVBM** ([design doc](https://docs.nvidia.com/dynamo/design-docs/component-design/kvbm-design), [code](https://github.com/ai-dynamo/dynamo)). 4-tier KV block manager: GPU (G1), CPU pinned memory (G2), local SSD (G3), remote storage (G4). Written in Rust with RAII memory lifecycle. NUMA-aware allocation via worker pools with first-touch placement and CUDA context binding. Event-based block state machine (allocate -> register -> match). Connector APIs for vLLM, TRT-LLM, SGLang. Uses NIXL for transport. Recent work adds prefetch hooks for agentic workloads: the harness can signal "bring these blocks from storage to GPU ahead of the next request," giving the orchestrator lifecycle control over cache placement.

### Transport

**NIXL** ([code](https://github.com/ai-dynamo/nixl), [docs](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md)). NVIDIA Inference Xfer Library. Provides a unified abstraction over heterogeneous memory types (DRAM, VRAM, file, object store) and transports (UCX/RDMA, GPUDirect Storage, NVLink, EFA/Libfabric, TCP). Pluggable backend architecture via `dlopen()`. Automatically selects the optimal backend based on source/destination memory types. C++17 core with Python and Rust bindings. v1.2.0 as of mid-2026, with UCCL backend for GPU memory transfers and NUMA-aware rail selection. memos should use NIXL for data movement rather than reimplementing transport.

### Agentic workload characterization

**Agentic AI Workload Characteristics** (Yuan et al., [arxiv 2605.26297](https://arxiv.org/abs/2605.26297), June 2026). The most comprehensive characterization of ReAct-style agent workloads to date. Key findings:

- With effective context caching, 84.6-99.5% of input tokens are reused across turns, making execution decode-dominated (91.0-98.6% of LLM time in decode). Agentic serving is not long-prompt serving - it is repeated decode over a growing cached context.
- The raw input-to-output ratio (53.9x to 559.8x) mischaracterizes agent workloads as prefill-heavy. The append-to-output ratio (1.5x to 7.3x) better reflects actual per-turn pressure.
- When aggregate agent context exceeds GPU memory, the system enters a thrashing regime. Concur measured 49.1% of end-to-end latency lost to recomputation in the middle phase of thrashing.
- Failed agents accumulate larger contexts and more turns (up to 1.8x mean context), amplifying memory pressure without producing useful work.
- Tool calls are not uniform: Agent/TaskOutput calls dominate latency (up to 916s mean), while Bash/Edit/Read are more failure-prone and drive retry loops.
- Agents exhibit temporal structure: early turns are exploration-heavy (Read, Grep, WebFetch), later turns shift to execution-heavy (Bash, Edit, Write).

**Implication for memos:** The sleeping-agents and thrashing-sweep benchmarks should model these observed patterns - decode-dominated execution, high cache reuse, tool-call pauses, and the thrashing cliff.

**RoleKV** ([OpenReview](https://openreview.net/forum?id=of1W47Odj1)). Demonstrates an inverted age-importance phenomenon in agent workloads: the oldest KV blocks (system prompts, tool definitions) are the most persistently attended, while the newest blocks (chain-of-thought reasoning) decay fastest. LRU eviction is therefore systematically wrong for agents - it evicts the blocks that matter most. RoleKV assigns semantic roles to blocks and evicts by role-aware decay. Achieved up to 31% preemption reduction, 35% PCIe traffic reduction, and 1.34x end-to-end speedup on 8B-70B models.

**Implication for memos:** Cost-model-driven eviction should naturally reproduce RoleKV's results for the right reason. System-prompt blocks have high recomputation cost (many tokens to regenerate) and therefore high `future_restoration_cost`. Chain-of-thought blocks have low recomputation cost (few tokens, recently generated). A cost-aware policy evicts cheap-to-restore blocks first, which aligns with role-aware retention without requiring explicit role tagging.

**Concur** ([arxiv 2601.22705](https://arxiv.org/abs/2601.22705)). Identifies the fundamental mismatch between long-lived, stateful agent execution and request-centric LLM serving abstractions. Proposes agent-level admission control: regulate how many agents can execute simultaneously so that their combined KV cache footprint fits in GPU memory, preventing thrashing proactively rather than reacting to it. During the middle phase of unconstrained execution, KV cache usage stays near saturation (80-100%) but cache hit rate drops precipitously - the system is trapped in an eviction-recomputation cycle that consumes 49.1% of latency.

**Implication for memos:** The thrashing-sweep benchmark should reproduce and extend this finding. memos adds the question: if you have cost-aware eviction + tiered memory, does the thrashing onset point shift? Can you serve more concurrent agents before hitting the cliff?

### Recomputation vs. offloading

**LLM KV Cache Performance Study** (Phan et al., [TU Delft 2025](https://atlarge-research.com/pdfs/phan2025isp.pdf)). Systematic measurement of recomputation vs. CPU offloading vs. disk offloading on vLLM:

- At 129K tokens, recomputation is 50x slower than GPU caching (54.38s vs 1.10s). CPU offloading is 1.44x slower (1.57s). Disk offloading is 6.6x slower (7.20s).
- Recomputation is only competitive with disk offloading below ~1K tokens.
- CPU DRAM offloading captures most of the benefit of GPU caching with minimal overhead.

**Implication for memos:** These numbers are the baseline memos must beat and extend. The crossover point (where recompute becomes cheaper than offloading) shifts with hardware: on GB200 with 900 GB/s C2C bandwidth, CPU offloading becomes even cheaper relative to recomputation, potentially pushing the crossover to near-zero. Conversely, on multi-node setups where the lowest tier is remote DRAM at ~25-50 GB/s, the crossover point rises. memos should produce crossover curves for each tier on each hardware configuration.

**Cost-Efficient LLM Serving with KV Cache Offloading** (InferSave, [arxiv 2504.11816](https://arxiv.org/abs/2504.11816)). Proposes SLO-based VM instance selection that factors in KV cache offloading. Achieves up to 73.7% cost reduction for online workloads by selecting cheaper GPU instances when offloading can meet the SLO. Introduces the Compute Time Calibration Function (CTCF) for adjusting between theoretical and actual GPU performance.

**Implication for memos:** Confirms that the recompute-vs-offload decision has economic dimensions beyond raw latency. The cost model should include `cost_per_gb_hour` per tier.

**Dell RDMA-Accelerated Architecture** ([Dell InfoHub](https://infohub.delltechnologies.com/p/scaling-multi-turn-llm-inference-with-kv-cache-storage-offload-and-dell-rdma-accelerated-architecture/)). Measured up to 3.7x higher token throughput with storage-offloaded KV cache vs. HBM-only on DGX H100. As prompt sizes grow, the scalability gap widens: at high concurrency, HBM-only vLLM plateaus at ~14K tokens/sec while storage-offloaded reaches ~43K tokens/sec. Fetching KV cache from storage is more efficient than recomputing it on the GPU for long contexts.

**Implication for memos:** Provides multi-node baseline numbers to compare against. The CAF measurement should capture this throughput scaling effect.

### Hardware architecture

**GB200 Grace Blackwell Superchip** ([architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/), [NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)). Pairs one Grace CPU (72 Arm Neoverse V2 cores) with two Blackwell GPUs via NVLink-C2C at 900 GB/s bidirectional bandwidth. Key architectural features:

- **Hardware Address Translation Service (ATS):** CPU and GPU share a single per-process page table. Translation between virtual and physical addresses is hardware-accelerated. This eliminates page faulting and minimizes page migration - the hardware provides what a software memory manager would try to build.
- **Cache-line coherence:** NVLink-C2C coherence operates at cache-line granularity, not page granularity. When different processors access different cache lines in the same page, there is no contention. Only conflicting accesses to the same cache line trigger exchange.
- **Unified memory capacity:** ~192 GB HBM3e + ~480 GB LPDDR5X = ~672 GB coherent memory per superchip. Models that once required multi-GPU tensor parallelism fit on a single superchip.
- **NVL72 rack scale:** 36 Grace CPUs + 72 Blackwell GPUs with 130 TB/s aggregate NVLink bandwidth. The rack functions as a single logical GPU with 30 TB coherent memory.

**Implication for memos:** On GB200, software-managed tiering between HBM and DRAM may be unnecessary for single-node workloads because hardware ATS handles it transparently at 900 GB/s. The software contribution shifts to: (1) multi-node memory management where hardware coherence doesn't reach, (2) policy decisions like recompute-vs-fetch that hardware can't make, (3) agent lifecycle semantics (fork/snapshot/rollback) that are above the hardware abstraction layer. The H100-to-GB200 comparison at each hardware step is critical: it shows where software memory management matters and where hardware makes it irrelevant.

### Where the OS analogy holds

The virtual memory analogy is not just a metaphor - several concepts map directly:

- **Page tables.** PagedAttention's block tables map logical token positions to physical block locations. This is a page table.
- **Working set.** Active sequences define a working set; dormant agents don't. Thrashing occurs when the aggregate working set exceeds physical capacity, exactly as Denning described in 1968.
- **Swapping.** Moving KV cache from HBM to DRAM is swapping. The performance characteristics (high latency, high capacity) match.
- **Copy-on-write.** Shared prefixes in prompt caching are CoW pages. When two requests share a system prompt, they point to the same physical blocks. Extending this to agent fork/snapshot is a natural next step.

### Where the OS analogy breaks

These divergences are what make the problem interesting and where memos's contributions lie:

- **Recomputation has no OS equivalent.** You cannot "recompute" a traditional memory page. KV cache can be regenerated from input tokens at known cost. This adds a dimension to the placement decision that does not exist in traditional VM.
- **Access patterns are deterministic.** During attention, the model accesses ALL KV cache for a sequence. There are no page fault surprises. The scheduler knows exactly what data will be needed before execution begins. Demand paging is the wrong model; proactive placement is correct.
- **The scheduler has oracle knowledge.** The AI memory scheduler knows which sequences are active, which are waiting for tool calls, expected token budgets, request priorities, SLOs, and whether a sequence will need its context again. Traditional VM has none of this information.
- **Objects are semantically typed.** KV cache, weights, and activations have different access patterns, lifetimes, and recomputation costs. A uniform "page" abstraction that ignores type leaves optimization on the table.
- **The cost function is multi-dimensional.** Traditional VM minimizes page faults. AI memory management must minimize a compound objective: latency, throughput, memory utilization, bandwidth consumption, and compute waste - simultaneously, subject to per-request SLOs.

### Summary of gaps memos addresses

| Gap | Existing state | memos contribution |
|-----|---------------|-------------------|
| Cost model for placement | Heuristic policies (LRU, LFU, role-aware) | Formal cost model with recomputation as a tier |
| Memory roofline | No equivalent exists | Roofline model classifying workloads as compute/BW/fault-bound |
| Cross-hardware measurements | Each system benchmarks on its own hardware | Standardized benchmarks across H100 -> GB200, single -> multi-node |
| Agent memory semantics | Prefix sharing only | Fork/snapshot/rollback with CoW over block tables |
| Recompute-vs-fetch decision | Implicit fallback when everything else fails | Explicit, calibrated, per-block cost comparison |

## Related work

- [PagedAttention / vLLM](https://github.com/vllm-project/vllm) - Paged KV cache management, the foundation for block-table-based memory
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - KVCache-centric disaggregated architecture (FAST 2025 Best Paper)
- [HiCache / SGLang](https://docs.sglang.ai/advanced_features/hicache_design.html) - Hierarchical L1/L2/L3 KV cache for SGLang
- [LMCache](https://github.com/LMCache/LMCache) - Engine-independent KV cache management layer
- [NVIDIA Dynamo KVBM](https://docs.nvidia.com/dynamo/design-docs/component-design/kvbm-design) - 4-tier KV block manager with NIXL transport
- [NIXL](https://github.com/ai-dynamo/nixl) - Transport abstraction for heterogeneous memory (UCX, GPUDirect, NVLink, EFA)
- [RoleKV](https://openreview.net/forum?id=of1W47Odj1) - Role-aware KV cache eviction for agent workloads
- [Concur](https://arxiv.org/abs/2601.22705) - Agent-level admission control for agentic batch inference
- [Agentic AI Workload Characteristics](https://arxiv.org/abs/2605.26297) - Characterization of ReAct-style agent memory and tool-use patterns
