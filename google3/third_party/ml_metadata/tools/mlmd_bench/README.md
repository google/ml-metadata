# mlmd_bench

*mlmd_bench* is a MLMD benchmark tool that can measure the MLMD query facility
performance and scalability for different backends and deployment settings.
The performance metrics of interests include throughputs of concurrent
operations and data sizes.

## HOW TO USE

### 1. Build from source:

```shell
bazel build -c opt --define grpc_no_ares=true //ml_metadata/tools/mlmd_bench:mlmd_bench
```

### 2. Run the binary:

```shell
cd bazel-bin/ml_metadata/tools/mlmd_bench/
./mlmd_bench --config_file_path=<input mlmd_bench config .pbtxt file path> --output_report_path=<output mlmd_bench summary report file path>
```

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
