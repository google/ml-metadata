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

// Package mlmetadata provides access to the metadata_store shared library.
package mlmetadata

import (
	"errors"
	"log"

	"github.com/golang/protobuf/proto"
	wrap "ml_metadata/metadata_store/metadata_store_go_wrap"
	mdpb "ml_metadata/proto/metadata_store_go_proto"
	apipb "ml_metadata/proto/metadata_store_service_go_proto"
)

// Store type provides a list of Go functions to access the methods defined in
// metadata_store/metadata_store.h and proto/metadata_store_service.proto
// It contains a pointer to the shared metadata_store cc library. The instance
// of it should be created with NewStore and destroyed with store.Close to avoid
// memory leak allocated in cc library.
type Store struct {
	ptr wrap.Ml_metadata_MetadataStore
}

// NewStore creates Store instance given a connection config.
func NewStore(config *mdpb.ConnectionConfig) (*Store, error) {
	status := wrap.NewStatus()
	defer wrap.DeleteStatus(status)

	b, err := proto.Marshal(config)
	if err != nil {
		log.Printf("Cannot marshal given connection config: %v. Error: %v\n", config, err)
		return nil, err
	}
	s := wrap.CreateMetadataStore(string(b), status)
	if !status.Ok() {
		return nil, errors.New(status.Error_message())
	}
	return &Store{s}, nil
}

// Close frees allocated memory in cc library of the Store instance.
func (store *Store) Close() {
	if store.ptr != nil {
		wrap.DestroyMetadataStore(store.ptr)
		store.ptr = nil
	}
}

// ArtifactTypeID refers the id space of ArtifactType
type ArtifactTypeID int64

// ExecutionTypeID refers the id space of ExecutionType
type ExecutionTypeID int64

// ArtifactID refers the id space of Artifact
type ArtifactID int64

// ExecutionID refers the id space of Execution
type ExecutionID int64

// PutTypeOptions defines options for PutArtifactType request.
type PutTypeOptions struct {
	CanAddFields       bool
	CanDeleteFields    bool
	AllFieldsMustMatch bool
}

// PutArtifactType inserts or updates an artifact type.
// If no artifact type exists in the database with the given name,
// it creates a new artifact type and returns the type_id.
// If an artifact type with the same name already exists, and the given artifact
// type match all properties (both name and value type) with the existing type,
// it returns the original type_id.
//
// Valid `atype` should not include a TypeID. `opts`.AllFieldsMustMatch must be
// true, `opts`.CanAddFields and `opts`.CanDeleteFields must be false; otherwise
// error is returned.
func (store *Store) PutArtifactType(atype *mdpb.ArtifactType, opts *PutTypeOptions) (ArtifactTypeID, error) {
	req := &apipb.PutArtifactTypeRequest{
		ArtifactType:    atype,
		CanAddFields:    &opts.CanAddFields,
		CanDeleteFields: &opts.CanDeleteFields,
		AllFieldsMatch:  &opts.AllFieldsMustMatch,
	}
	resp := &apipb.PutArtifactTypeResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.PutArtifactType, req, resp)
	return ArtifactTypeID(resp.GetTypeId()), err
}

// GetArtifactType gets an artifact type by name. If no type exists or query
// execution fails, error is returned.
func (store *Store) GetArtifactType(typeName string) (*mdpb.ArtifactType, error) {
	req := &apipb.GetArtifactTypeRequest{
		TypeName: proto.String(typeName),
	}
	resp := &apipb.GetArtifactTypeResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetArtifactType, req, resp)
	return resp.GetArtifactType(), err
}

// GetArtifactTypesByID gets a list of artifact types by ID. If no type with an
// ID exists, the artifact type is skipped. If the query execution fails, error is
// returned.
func (store *Store) GetArtifactTypesByID(tids []ArtifactTypeID) ([]*mdpb.ArtifactType, error) {
	req := &apipb.GetArtifactTypesByIDRequest{
		TypeIds: convertToInt64ArrayFromArtifactTypeIDs(tids),
	}
	resp := &apipb.GetArtifactTypesByIDResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetArtifactTypesByID, req, resp)
	return resp.GetArtifactTypes(), err
}

// PutExecutionType inserts or updates an execution type.
// If no execution type exists in the database with the given name,
// it creates a new execution type and returns the type_id.
// If an execution type with the same name already exists, and the given
// execution type match all properties (both name and value type) with the
// existing type, it returns the original type_id.
//
// Valid `etype` should not include a TypeID. `opts`.AllFieldsMustMatch must be
// true, `opts`.CanAddFields and `opts`.CanDeleteFields must be false; otherwise
// error is returned.
func (store *Store) PutExecutionType(etype *mdpb.ExecutionType, opts *PutTypeOptions) (ExecutionTypeID, error) {
	req := &apipb.PutExecutionTypeRequest{
		ExecutionType:   etype,
		CanAddFields:    &opts.CanAddFields,
		CanDeleteFields: &opts.CanDeleteFields,
		AllFieldsMatch:  &opts.AllFieldsMustMatch,
	}
	resp := &apipb.PutExecutionTypeResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.PutExecutionType, req, resp)
	return ExecutionTypeID(resp.GetTypeId()), err
}

// GetExecutionType gets an execution type by name. If no type exists or query
// execution fails, error is returned.
func (store *Store) GetExecutionType(typeName string) (*mdpb.ExecutionType, error) {
	req := &apipb.GetExecutionTypeRequest{
		TypeName: proto.String(typeName),
	}
	resp := &apipb.GetExecutionTypeResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetExecutionType, req, resp)
	return resp.GetExecutionType(), err
}

// GetExecutionTypesByID gets a list of execution types by ID. If no type with
// an ID exists, the execution type is skipped. If the query execution fails, error
// is returned.
func (store *Store) GetExecutionTypesByID(tids []ExecutionTypeID) ([]*mdpb.ExecutionType, error) {
	req := &apipb.GetExecutionTypesByIDRequest{
		TypeIds: convertToInt64ArrayFromExecutionTypeIDs(tids),
	}
	resp := &apipb.GetExecutionTypesByIDResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetExecutionTypesByID, req, resp)
	return resp.GetExecutionTypes(), err
}

// PutArtifacts inserts and updates artifacts into the store.
//
// In `artifacts`, if Id is specified, an existing artifact is updated;
// if Id is not specified, a new artifact is created. It returns a list of
// artifact ids index-aligned with the input.
//
// It returns an error if a) no artifact is found with the given id, b) the given
// TypeId is different from the one stored, or c) given property names and types
// do not align with the ArtifactType on file.
func (store *Store) PutArtifacts(artifacts []*mdpb.Artifact) ([]ArtifactID, error) {
	req := &apipb.PutArtifactsRequest{
		Artifacts: artifacts,
	}
	resp := &apipb.PutArtifactsResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.PutArtifacts, req, resp)
	rst := make([]ArtifactID, len(resp.GetArtifactIds()))
	for i, v := range resp.GetArtifactIds() {
		rst[i] = ArtifactID(v)
	}
	return rst, err
}

// PutExecutions inserts and updates executions into the store.
//
// In `executions`, if Id is specified, an existing execution is updated;
// if Id is not specified, a new execution is created. It returns a list
// of execution ids index-aligned with the input.
//
// It returns an error if a) no execution is found with the given id, b) the given
// TypeId is different from the one stored, or c) given property names and types
// do not align with the ExecutionType on file.
func (store *Store) PutExecutions(executions []*mdpb.Execution) ([]ExecutionID, error) {
	req := &apipb.PutExecutionsRequest{
		Executions: executions,
	}
	resp := &apipb.PutExecutionsResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.PutExecutions, req, resp)
	rst := make([]ExecutionID, len(resp.GetExecutionIds()))
	for i, v := range resp.GetExecutionIds() {
		rst[i] = ExecutionID(v)
	}
	return rst, err
}

// GetArtifactsByID gets a list of artifacts by ID.
// If no artifact with an ID exists, the artifact id is skipped.
// It returns an error if the query execution fails.
func (store *Store) GetArtifactsByID(aids []ArtifactID) ([]*mdpb.Artifact, error) {
	req := &apipb.GetArtifactsByIDRequest{
		ArtifactIds: convertToInt64ArrayFromArtifactIDs(aids),
	}
	resp := &apipb.GetArtifactsByIDResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetArtifactsByID, req, resp)
	return resp.GetArtifacts(), err
}

// GetExecutionsByID gets a list of executions by ID.
// If no execution with an ID exists, the execution id is skipped.
// It returns an error if the query execution fails.
func (store *Store) GetExecutionsByID(eids []ExecutionID) ([]*mdpb.Execution, error) {
	req := &apipb.GetExecutionsByIDRequest{
		ExecutionIds: convertToInt64ArrayFromExecutionIDs(eids),
	}
	resp := &apipb.GetExecutionsByIDResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetExecutionsByID, req, resp)
	return resp.GetExecutions(), err
}

// PutEvents inserts events into the store.
//
// In `events`, the ExecutionId and ArtifactId must already exist. Once created,
// events cannot be modified. If MillisecondsSinceEpoch is not set, it will be
// set to the current time.
//
// It returns an error if a) no artifact exists with the given ArtifactId, b) no
// execution exists with the given ExecutionId, c) the Type field is UNKNOWN.
func (store *Store) PutEvents(events []*mdpb.Event) error {
	req := &apipb.PutEventsRequest{
		Events: events,
	}
	resp := &apipb.PutEventsResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.PutEvents, req, resp)
	return err
}

// GetEventsByArtifactIDs gets all events with matching artifact ids.
// It returns an error if the query execution fails.
func (store *Store) GetEventsByArtifactIDs(aids []ArtifactID) ([]*mdpb.Event, error) {
	req := &apipb.GetEventsByArtifactIDsRequest{
		ArtifactIds: convertToInt64ArrayFromArtifactIDs(aids),
	}
	resp := &apipb.GetEventsByArtifactIDsResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetEventsByArtifactIDs, req, resp)
	return resp.GetEvents(), err
}

// GetEventsByExecutionIDs gets all events with matching execution ids.
// It returns an error if the query execution fails.
func (store *Store) GetEventsByExecutionIDs(eids []ExecutionID) ([]*mdpb.Event, error) {
	req := &apipb.GetEventsByExecutionIDsRequest{
		ExecutionIds: convertToInt64ArrayFromExecutionIDs(eids),
	}
	resp := &apipb.GetEventsByExecutionIDsResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetEventsByExecutionIDs, req, resp)
	return resp.GetEvents(), err
}

// GetArtifacts gets all artifacts.
// It returns an error if the query execution fails.
func (store *Store) GetArtifacts() ([]*mdpb.Artifact, error) {
	req := &apipb.GetArtifactsRequest{}
	resp := &apipb.GetArtifactsResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetArtifacts, req, resp)
	return resp.GetArtifacts(), err
}

// GetArtifactsByType gets all artifacts of a given type.
// It returns an error if the query execution fails.
func (store *Store) GetArtifactsByType(typeName string) ([]*mdpb.Artifact, error) {
	req := &apipb.GetArtifactsByTypeRequest{
		TypeName: proto.String(typeName),
	}
	resp := &apipb.GetArtifactsByTypeResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetArtifactsByType, req, resp)
	return resp.GetArtifacts(), err
}

// GetArtifactsByURI gets all artifacts of a given uri.
// It returns an error if the query execution fails.
func (store *Store) GetArtifactsByURI(uri string) ([]*mdpb.Artifact, error) {
	req := &apipb.GetArtifactsByURIRequest{
		Uri: proto.String(uri),
	}
	resp := &apipb.GetArtifactsByURIResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetArtifactsByURI, req, resp)
	return resp.GetArtifacts(), err
}

// GetExecutions gets all executions.
// It returns an error if the query execution fails.
func (store *Store) GetExecutions() ([]*mdpb.Execution, error) {
	req := &apipb.GetExecutionsRequest{}
	resp := &apipb.GetExecutionsResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetExecutions, req, resp)
	return resp.GetExecutions(), err
}

// GetExecutionsByType gets all executions of a given type.
// It returns an error if the query execution fails.
func (store *Store) GetExecutionsByType(typeName string) ([]*mdpb.Execution, error) {
	req := &apipb.GetExecutionsByTypeRequest{
		TypeName: proto.String(typeName),
	}
	resp := &apipb.GetExecutionsByTypeResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetExecutionsByType, req, resp)
	return resp.GetExecutions(), err
}

type metadataStoreMethod func(wrap.Ml_metadata_MetadataStore, string, wrap.Status) string

// callMetadataStoreWrapMethod calls a `metadataStoreMethod` in cc library.
// It returns an error if the cc library method returns non-ok status, or `req`,
// `resp` proto message cannot be serialized/parsed correctly.
func (store *Store) callMetadataStoreWrapMethod(fn metadataStoreMethod, req proto.Message, resp proto.Message) error {
	status := wrap.NewStatus()
	defer wrap.DeleteStatus(status)

	b, err := proto.Marshal(req)
	if err != nil {
		return err
	}
	wrt := fn(store.ptr, string(b), status)
	if !status.Ok() {
		return errors.New(status.Error_message())
	}
	err = proto.Unmarshal(([]byte)(wrt), resp)
	if err != nil {
		return err
	}
	return nil
}

func convertToInt64ArrayFromArtifactTypeIDs(tids []ArtifactTypeID) []int64 {
	ids := make([]int64, len(tids))
	for i, v := range tids {
		ids[i] = int64(v)
	}
	return ids
}

func convertToInt64ArrayFromExecutionTypeIDs(tids []ExecutionTypeID) []int64 {
	ids := make([]int64, len(tids))
	for i, v := range tids {
		ids[i] = int64(v)
	}
	return ids
}

func convertToInt64ArrayFromArtifactIDs(aids []ArtifactID) []int64 {
	ids := make([]int64, len(aids))
	for i, v := range aids {
		ids[i] = int64(v)
	}
	return ids
}

func convertToInt64ArrayFromExecutionIDs(eids []ExecutionID) []int64 {
	ids := make([]int64, len(eids))
	for i, v := range eids {
		ids[i] = int64(v)
	}
	return ids
}
