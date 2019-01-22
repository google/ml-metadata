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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_STORE_FACTORY_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_STORE_FACTORY_H_

#include <memory>

#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {
// Creates a MetadataStore.
// If the method returns OK, the method MUST set result to contain
// a non-null pointer.
tensorflow::Status CreateMetadataStore(
    const ConnectionConfig& config,
    std::unique_ptr<MetadataStore>* result);

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_STORE_FACTORY_H_
