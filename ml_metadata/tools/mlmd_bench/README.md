
# mlmd_bench

*mlmd_bench* is a MLMD benchmark tool that can measure the MLMD query facility performance and scalability for different backends and deployment settings. The performance metrics of interests include throughputs of concurrent operations and data sizes.

## HOW TO USE

### 1. Build

```shell
bazel build -c opt //ml_metadata/tools/mlmd_bench:mlmd_bench
```

### 2. Run the binary

```shell
cd bazel-bin/ml_metadata/tools/mlmd_bench/
./mlmd_bench <input configuration .pbtxt file path> <output report directory>
```

## INPUT OUTPUT FORMAT

### 1. Input should be under the following format:

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

### 2. Output will be under the following format:

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
