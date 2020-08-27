# mlmd_bench

*mlmd_bench* is a MLMD benchmark tool that can measure the MLMD query facility
performance and scalability for different backends and deployment settings.
The performance metrics of interests include throughputs of concurrent
operations and data sizes.

## Motivation

MLMD exposes a provenance graph data model, which consists of {Artifact, Execution, Context, Type} as nodes, and {Event, Association, Attribution, Instance} as edges. Based on the data model, it defines a set of APIs, such as creating and updating nodes and edges, listing nodes by types, traversing nodes via edges.  

The APIs are implemented on various backends (e.g., MySQL, Sqlite, CNS, Spanner) and deployed in different modes (e.g., library, grpc server, lamprey server etc). On the other hand, the integration of MLMD with different partners have different use cases and exhibit different read/write workloads. To improve API performance, it often requires proper optimization in MLMD query implementation or schema reorganization.    

To guide the performance tuning of MLMD and better support users, we propose a a benchmarking tool, *mlmd_bench*, which can:
*   Compose different workloads mimicking use cases of integration partners. 
*   Measure the MLMD operation performance on different backends and deployment modes.

The tool abstraction and design are inspired by [leveldb](https://github.com/google/leveldb) benchmarking tools, and tailored to the MLMD provenance graph data model and workloads. We envision a list of follow-ups to tune query implementations and evolve MLMD schema guided by the benchmark results.

## Benchmark coverage

|  Workload   | Benchmark MLMD APIs  | Specification
|  ----  | ----  |
| FillTypes  | PutArtifactType / PutExecutionType / PutContextType | Insert / Update
Artifact Type / Execution Type / Context Type
Number of Properties for each type
| 单元格  | 单元格 |

## How to use

### 1. Prerequisites

To compile and use *mlmd_bench*, you need to set up some prerequisites.

#### Install Bazel

If Bazel is not installed on your system, install it now by following [these
directions](https://bazel.build/versions/master/docs/install.html).

#### Install cmake
If cmake is not installed on your system, install it now by following [these
directions](https://cmake.org/install/).

### 2. Build from source:

```shell
bazel build -c opt --define grpc_no_ares=true //ml_metadata/tools/mlmd_bench:all
```

### 3. Run the binary:

```shell
cd bazel-bin/ml_metadata/tools/mlmd_bench/
./mlmd_bench --config_file_path=<input mlmd_bench config .pbtxt file path> --output_report_path=<output mlmd_bench summary report file path>
```

The input mlmd_bench 

## INPUT OUTPUT FORMAT

### 1. Input is a MLMDBenchConfig protocol buffer message in text format:

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
    specification: EXECUTION_TYPE
    num_properties: { minimum: 1 maximum: 10 }
  }
  num_operations: 100
}
workload_configs: {
  fill_types_config: {
    update: true
    specification: EXECUTION_TYPE
    num_properties: { minimum: 1 maximum: 10 }
  }
  num_operations: 200
}
thread_env_config: { num_threads: 10 }
```

### 2. Output is a MLMDBenchReport protocol buffer message in text format:

```shell
summaries {
  workload_config {
    fill_types_config {
      update: false
      specification: EXECUTION_TYPE
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
summaries {
  workload_config {
    fill_types_config {
      update: true
      specification: EXECUTION_TYPE
      num_properties {
        minimum: 1
        maximum: 10
      }
    }
    num_operations: 200
  }
  microseconds_per_operation: 119221
  bytes_per_second: 1110
}
```

## How to test

```shell
bazel test -c opt --define grpc_no_ares=true //ml_metadata/tools/mlmd_bench:all
```

## 