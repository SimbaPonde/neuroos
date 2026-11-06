# Related work vs. NeuroOS (checked Nov 2026)

| Work | What it does | Energy measured? | Tasks | Platform | Gap NeuroOS targets |
|---|---|---|---|---|---|
| **AIOS** (Mei et al., arXiv 2403.16971) | "Kernel" for agents: scheduling, context, memory, storage, access control; up to 2.1× faster agent serving | No | Agent frameworks | Server, user space | No energy objective; no local-vs-cloud decision |
| **AgentOS** (Liu et al., arXiv 2603.08938) | Vision: natural-language "single port" replaces desktop; Agent Kernel over legacy OS; skills as modules; semantic firewall; rollback | No (position paper, no implementation) | — | — | Gives the vision; NeuroOS gives a measured implementation |
| **HybridInfer** (Koul, arXiv 2609.30270) | Q-learning router: phone / edge 8B+RAG / cloud, state = thermal headroom + complexity | Only a current×time proxy, not tabulated | 210 single prompts | Android phone | Single queries, not tool-using agents; no true joules; no laptop/PC |
| **Carbon-aware function-calling routing** (Panteleaki et al., arXiv 2609.13559) | k-NN predicts per-tier success/delay/power; picks lowest-carbon tier meeting accuracy threshold; ~4× carbon cut | Power per tier from calibration; cloud energy from published estimates | Function-calling benchmarks (BFCL, GeoEngine) | Jetson edge + cloud H100 | Per-query, not whole multi-step agent tasks with real side effects; edge boards, not personal computers; no OS resource control |
| FrugalGPT, Hybrid LLM, RouteLLM | Cost/quality cascades and learned routers | No | Single queries | Cloud | Classic baselines |
| Windows agent workspace + MCP | Agents in an isolated OS workspace with connectors | No | Product | Windows | Industry proof the direction matters |

## Checking the original NeuroOS ideas

| Original idea | Verdict | Why |
|---|---|---|
| LLM-driven natural-language OS interface | **Not novel alone** | AgentOS, Windows, OpenClaw, Open Interpreter |
| Agent "kernel" services (scheduling, memory, access control) | **Not novel alone** | AIOS |
| Routing tasks across small/large/cloud models | **Not novel alone** | HybridInfer, carbon-aware routing, RouteLLM |
| Energy per *successful multi-step agent task*, **measured** on a personal computer | **Open** | Prior work routes single queries, uses proxies or published estimates |
| Router that includes **deterministic tools** (zero-model pathway) and escalation | **Partly open** | Cascades exist; tool pathway + escalation for agents is rarely measured |
| Joint routing + **model residency** (which models stay in RAM) | **Open** | Not addressed in the routing papers |
| OS-level resource control (cgroups, sched_ext) for agent workloads | **Open, but prove need first** | AgenticOS 2026 workshop lists this as an open topic |
| Brain-inspired framing | **Drop or make concrete** | Doesn't change any design decision yet |
| Continual LoRA adaptation | **Defer** | Out of scope for a first paper |

## Lessons to adopt

- **Value locality explicitly** (HybridInfer): a router optimizing only quality/latency/cost learns to offload everything. NeuroOS's objective is joules per success with a local preference; state this in the method.
- **Success must be verified** (carbon-aware paper uses binary success): NeuroOS checkers inspect the filesystem, not the model's claim.
- **On-device can be unreliable, not just slow** (HybridInfer): log crashes/timeouts as failures and report reliability.
- **Report cloud energy as an estimate with a sensitivity range**, never as a measurement.

## Positioning sentence (draft)

> Prior routers select a model per query and approximate energy; NeuroOS routes whole multi-step
> agent tasks on a personal computer across a zero-model tool pathway, local models of two sizes,
> and the cloud, and evaluates them by measured joules per verified successful task.

## Target venue

AgenticOS workshop (co-located with ASPLOS; 2026 edition held Mar 22–23) — watch for 2027 CFP.
Also: HotCarbon, EuroMLSys, NeurIPS/ICML efficiency workshops.
