/* Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_FACTORY_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_FACTORY_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/proto/metadata_source.pb.h"

namespace ml_metadata {

// Factory method, if the return value is ok, 'result' is populated with an
// object that can be used to access metadata with the given config and
// metadata_source. The caller is responsible to own a MetadataSource, and the
// MetadataAccessObject connects and execute queries with the MetadataSource.
// Returns INVALID_ARGUMENT error, if query_config is not valid.
// Returns detailed INTERNAL error, if the MetadataSource cannot be connected.
absl::Status CreateMetadataAccessObject(
    const MetadataSourceQueryConfig& query_config,
    MetadataSource* const metadata_source,
    std::unique_ptr<MetadataAccessObject>* result);

// For multi-tenant applications to have better availability when configured
// with a set of existing backends with different schema versions, an
// `schema_version` can be set to allow the MetadataAccessObject
// works with an existing db having that particular schema version.
absl::Status CreateMetadataAccessObject(
    const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, std::optional<int64_t> schema_version,
    std::unique_ptr<MetadataAccessObject>* result);

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_FACTORY_H_
