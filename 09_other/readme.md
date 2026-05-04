# End-to-End Architecture

This document traces the complete path from model loading through backend operator dispatch and execution.

## System overview

```mermaid
sequenceDiagram
    participant User as User / Client
    participant API as API Server
    participant Engine as Engine Core
    participant Worker as GPU Worker
    participant Runner as Model Runner
    participant Compile as torch.compile / Inductor
    participant Backend as Backend Kernels
    participant Replay as CUDA Graph

    User->>API: send request
    API->>Engine: route request
    Engine->>Worker: schedule step
    Worker->>Runner: prepare inputs
    Runner->>Compile: execute model.forward()
    Compile->>Compile: trace to FX graph
    Compile->>Compile: normalize & partition
    Compile->>Compile: match backend rules
    Compile->>Backend: lower to kernels
    Backend->>Replay: capture stable regions
    Replay-->>Runner: execute or replay
    Runner-->>Worker: return outputs
    Worker-->>Engine: step complete
    Engine-->>API: stream results
    API-->>User: deliver response
```

## Layered architecture

```mermaid
flowchart TB
    subgraph Entrypoints["Entrypoints"]
        LLM["LLM class<br/>(offline)"]
        OpenAI["OpenAI API<br/>(online)"]
    end

    subgraph Orchestration["Request Orchestration"]
        APIServer["API Server<br/>Process"]
        EngineCore["Engine Core<br/>Process"]
    end

    subgraph DeviceExecution["Device Execution"]
        Worker["GPU Worker<br/>Process"]
        Runner["Model Runner"]
        Model["torch.nn.Module"]
    end

    subgraph Compilation["Compilation & Lowering"]
        Dynamo["TorchDynamo<br/>Trace"]
        FX["FX Graph"]
        Normalize["Normalize &<br/>Partition"]
        Match["Match Backend<br/>Rules"]
        Lower["Inductor<br/>Lowering"]
    end

    subgraph Execution["Execution"]
        CustomOp["Custom Ops"]
        VLLMIR["vLLM IR"]
        Kernels["Triton/CUDA<br/>Kernels"]
        CudaGraph["CUDA Graph<br/>Replay"]
    end

    LLM --> APIServer
    OpenAI --> APIServer
    APIServer --> EngineCore
    EngineCore --> Worker
    Worker --> Runner
    Runner --> Model
    Model --> Dynamo
    Dynamo --> FX
    FX --> Normalize
    Normalize --> Match
    Match --> Lower
    Lower --> CustomOp
    Lower --> VLLMIR
    CustomOp --> Kernels
    VLLMIR --> Kernels
    Kernels --> CudaGraph
    CudaGraph --> Runner
```

## Data flow: model to kernels

```mermaid
flowchart TD
    A["Model inputs<br/>token_ids, positions<br/>multimodal_data"] 
    B["Model wrapper<br/>torch.nn.Module"]
    C["Python forward()<br/>execution"]
    D["TorchDynamo<br/>trace"]
    E["FX graph<br/>ATen ops"]
    F["Normalize &<br/>preserve boundaries"]
    G["Partition into<br/>subgraphs"]


    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
  


```
```mermaid
flowchart TD
G["Partition into<br/>subgraphs"]
    H["Summarize<br/>signatures"]
    I["Match backend<br/>rules"]
    J["Select impl:<br/>CustomOp/IR/Inductor"]
    K["Lower to<br/>kernels"]
    L["Capture or<br/>execute"]
    M["Backend<br/>kernels"]
    N["Outputs<br/>hidden_states"]
  G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
```

## Ownership and responsibility

| Layer | Owner | Responsibility |
| --- | --- | --- |
| **Entrypoint** | LLM class / API server | Accept requests, manage lifecycle |
| **Orchestration** | Engine core | Schedule work, coordinate execution |
| **Device execution** | Worker + model runner | Load model, prepare inputs, manage state |
| **Model wrapper** | Model registry | Select correct Python model class |
| **Trace capture** | TorchDynamo | Convert Python forward to FX graph |
| **Graph normalization** | vLLM passes | Preserve boundaries, functionalize |
| **Partitioning** | vLLM + Inductor scheduler | Split into reusable regions |
| **Signature matching** | vLLM backend matcher | Summarize operator and tensor metadata |
| **Backend selection** | CustomOp / vLLM IR / registries | Choose implementation by priority |
| **Lowering** | Inductor | Generate kernel code |
| **Replay** | CUDA graph manager | Capture and reuse stable regions |
| **Execution** | GPU worker | Run kernels or replay graphs |

## Key decision points

### 1. Model adaptation boundary

**Question**: Which Python model class should we use?

**Answer**: Model registry looks up the architecture name and selects the correct wrapper.

**Owner**: Hugging Face integration + model registry

**Example**: `Qwen/Qwen2-7B` → `Qwen2ForCausalLM` class

### 2. Compilation boundary

**Question**: Which regions can be compiled together?

**Answer**: TorchDynamo traces executable regions; custom ops and graph breaks define boundaries.

**Owner**: TorchDynamo + custom-op registration

**Example**: Attention is wrapped as a custom op so Dynamo doesn't inspect its internals.

### 3. Backend selection boundary

**Question**: Which kernel should execute this subgraph?

**Answer**: Match operator structure and tensor metadata against registered implementations.

**Owner**: CustomOp / vLLM IR / backend registries

**Example**: If the region contains `rms_norm` + `matmul`, check if a fused kernel is available.

### 4. Replay boundary

**Question**: Can this region be captured as a CUDA graph?

**Answer**: Yes, if shapes are stable and memory behavior is predictable.

**Owner**: CUDA graph manager

**Example**: Decode-phase regions are usually replayable; prefill regions may not be.

## Patterns and design principles

| Pattern | Where | Why |
| --- | --- | --- |
| **Adapter** | Model registry, Hugging Face integration | Normalize many model families to one runtime interface |
| **Pipeline** | Trace → normalize → partition → match → lower → replay | Separate compiler stages cleanly |
| **Strategy** | Backend implementation priority lists | Choose best implementation for platform and shape |
| **Registry** | CustomOp and vLLM IR registries | Extend without changing orchestrator |
| **Facade** | Engine and runner APIs | Hide distributed execution and compilation details |

## Related docs

- [Architecture overview](overview.md)
- [Boundaries and contexts](boundaries.md)
- [Process model](process-model.md)
- [Execution pipeline](execution-pipeline.md)
- [Patterns](patterns.md)
- [Backend operator mapping](../backend_operator_mapping.md)
- [torch.compile integration](../torch_compile.md)
- [CustomOp](../custom_op.md)
- [vLLM IR](../vllm_ir.md)
