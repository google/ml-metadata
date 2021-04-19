# mlmd_bench

`mlmd_bench` is a MLMD benchmark tool that can measure the MLMD query facility
performance and scalability for different backends and deployment settings.
The performance metrics of interests include throughputs of concurrent
operations and data sizes.

## Introduction

MLMD exposes a provenance graph data model, which consists of {Artifact,
Execution, Context, Type} as nodes, and {Event, Association, Attribution,
Instance} as edges. Based on the data model, it defines a set of APIs, such as
creating and updating nodes and edges, listing nodes by types, traversing nodes
via edges.

The APIs are implemented on various backends (e.g., MySQL, Sqlite) and deployed
in different modes (e.g., library, grpc server). On the other hand, the
integration of MLMD with different partners have different use cases and
exhibit different read/write workloads. To improve API performance, it often
requires proper optimization in MLMD query implementation or schema
reorganization.

To better support users and guide the performance tuning of MLMD, we provide a
benchmarking tool, `mlmd_bench`, which can:

*   Compose different workloads mimicking use cases of different integration
    partners.
*   Measure the MLMD operation performance on different backends and deployment
    modes.

## Benchmark coverage

| Workload      | Benchmark APIs | Specification
| :----------- | :----------- | :----------- |
| FillTypes  | PutArtifactType /<br> PutExecutionType /<br> PutContextType | Insert / Update <br> Artifact Type / Execution Type / Context Type <br> Number of properties for each type|
| FillNodes   | PutArtifact / PutExecution /<br> PutContext        | Insert / Update<br>Artifact / Execution / Context<br>Number of properties for each node <br> Length for string properties of each node<br>APIs’ specification(e.g. number of nodes per request)|
| FillContextEdges      | PutAttributionsAndAssociation       | Attribution / Association<br>Context / Non-context popularity<br>APIs’ specification(e.g. number of context edges per request)|
| FillEvents      | PutEvent       | Input / Output Event<br>Artifact / Execution popularity<br>APIs’ specification(e.g. number of events per request)|
| ReadTypes      | GetArtifactTypes /<br> GetArtifactTypesByID /<br> GetArtifactType /<br> GetExecutionTypes /<br> GetExecutionTypesByID /<br> GetExecutionType /<br> GetContextTypes /<br> GetContextTypesByID /<br> GetContextType  | The type listing / querying APIs<br>APIs’ specification(e.g. number of ids per request)|
| ReadNodesByProperties      | GetArtifactsByID /<br> GetArtifactsByType /<br> GetArtifactByTypeAndName /<br> GetArtifactsByURI /<br> GetExecutionsByID /<br> GetExecutionsByType /<br> GetExecutionByTypeAndName /<br> GetContextsByID /<br> GetContextsByType /<br> GetContextByTypeAndName | The nodes listing / querying APIs<br>APIs’ specification(e.g. number of ids per request)|
| ReadNodesViaContextEdges      | GetArtifactsByContext /<br> GetContextsByArtifact /<br> GetExecutionsByContext /<br> GetContextsByExecution| The nodes traversal APIs|
| ReadEvents      | GetEventsByArtifactIDs /<br> GetEventsByExecutionIDs       | The events listing / querying APIs<br>APIs’ specification(e.g. number of ids per request)|

## How to use

### 1. Build from source:

```shell
bazel build -c opt --define grpc_no_ares=true //ml_metadata/tools/mlmd_bench:mlmd_bench
```

### 2. Run the binary:

```shell
cd bazel-bin/ml_metadata/tools/mlmd_bench/
./mlmd_bench --config_file_path=<input mlmd_bench config .pbtxt file path> --output_report_path=<output mlmd_bench summary report file path>
```

The input should be a `MLMDBenchConfig` Protocol Buffers message in text format, e.g.:

```shell
mlmd_config: {
  sqlite: {
    filename_uri: "mlmd-bench.db"
    connection_mode: READWRITE_OPENCREATE
  }
}
workload_configs: {
  fill_types_config: {
    update: false
    specification: ARTIFACT_TYPE
    num_properties: { minimum: 1 maximum: 10 }
  }
  num_operations: 100
}
thread_env_config: { num_threads: 10 }
```

The output is a `MLMDBenchReport` Protocol Buffers message in text format, e.g.:

```shell
summaries {
  workload_config {
    fill_types_config {
      update: false
      specification: ARTIFACT_TYPE
      num_properties {
        minimum: 1
        maximum: 10
      }
    }
    num_operations: 100
  }
  microseconds_per_operation: 193183
  bytes_per_second: 351
}
```
