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
	status := wrap.CreateABSLStatus()
	defer wrap.DestroyABSLStatus(status)

	b, err := proto.Marshal(config)
	if err != nil {
		log.Printf("Cannot marshal given connection config: %v. Error: %v\n", config, err)
		return nil, err
	}
	s := wrap.CreateMetadataStore(string(b), status)
	if !wrap.IsOk(status) {
		return nil, errors.New(wrap.ErrorMessage(status))
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

// ContextTypeID refers the id space of ContextType
type ContextTypeID int64

// ArtifactID refers the id space of Artifact
type ArtifactID int64

// ExecutionID refers the id space of Execution
type ExecutionID int64

// ContextID refers the id space of Context
type ContextID int64

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
// true, `opts`.CanAddFields should be true when update an stored type
// and `opts`.CanDeleteFields must be false; otherwise error is returned.
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

// GetArtifactTypes gets all artifact types. If query execution fails, error
// is returned.
func (store *Store) GetArtifactTypes() ([]*mdpb.ArtifactType, error) {
	req := &apipb.GetArtifactTypesRequest{}
	resp := &apipb.GetArtifactTypesResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetArtifactTypes, req, resp)
	return resp.GetArtifactTypes(), err
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
// true, `opts`.CanAddFields should be true when update an stored type
// and `opts`.CanDeleteFields must be false; otherwise error is returned.
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

// GetExecutionTypes gets all execution types. If query execution fails, error
// is returned.
func (store *Store) GetExecutionTypes() ([]*mdpb.ExecutionType, error) {
	req := &apipb.GetExecutionTypesRequest{}
	resp := &apipb.GetExecutionTypesResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetExecutionTypes, req, resp)
	return resp.GetExecutionTypes(), err
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

// PutContextType inserts or updates a context type.
// If no context type exists in the database with the given name,
// it creates a new context type and returns the type_id.
// If an context type with the same name already exists, and the given context
// type match all properties (both name and value type) with the existing type,
// it returns the original type_id.
//
// Valid `ctype` should not include a TypeID. `opts`.AllFieldsMustMatch must be
// true, `opts`.CanAddFields should be true when update an stored type and
// `opts`.CanDeleteFields must be false; otherwise error is returned.
func (store *Store) PutContextType(ctype *mdpb.ContextType, opts *PutTypeOptions) (ContextTypeID, error) {
	req := &apipb.PutContextTypeRequest{
		ContextType:     ctype,
		CanAddFields:    &opts.CanAddFields,
		CanDeleteFields: &opts.CanDeleteFields,
		AllFieldsMatch:  &opts.AllFieldsMustMatch,
	}
	resp := &apipb.PutContextTypeResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.PutContextType, req, resp)
	return ContextTypeID(resp.GetTypeId()), err
}

// GetContextType gets a context type by name. If no type exists or query
// execution fails, error is returned.
func (store *Store) GetContextType(typeName string) (*mdpb.ContextType, error) {
	req := &apipb.GetContextTypeRequest{
		TypeName: proto.String(typeName),
	}
	resp := &apipb.GetContextTypeResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetContextType, req, resp)
	return resp.GetContextType(), err
}

// GetContextTypes gets all context types. If query execution fails, error
// is returned.
func (store *Store) GetContextTypes() ([]*mdpb.ContextType, error) {
	req := &apipb.GetContextTypesRequest{}
	resp := &apipb.GetContextTypesResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetContextTypes, req, resp)
	return resp.GetContextTypes(), err
}

// GetContextTypesByID gets a list of context types by ID. If no type with an
// ID exists, the context type is skipped. If the query execution fails, error is
// returned.
func (store *Store) GetContextTypesByID(tids []ContextTypeID) ([]*mdpb.ContextType, error) {
	req := &apipb.GetContextTypesByIDRequest{
		TypeIds: convertToInt64ArrayFromContextTypeIDs(tids),
	}
	resp := &apipb.GetContextTypesByIDResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetContextTypesByID, req, resp)
	return resp.GetContextTypes(), err
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

// PutContexts inserts and updates contexts into the store.
//
// In `contexts`, if Id is specified, an existing context is updated;
// if Id is not specified, a new context is created. It returns a list
// of context ids index-aligned with the input.
//
// It returns an error if
// a) no context is found with the given id,
// b) the given TypeId is different from the one stored,
// c) given property names and types, do not align with the ContextType on file,
// d) context name is empty,
// e) the given name already exists in the contexts of the given TypeId.
func (store *Store) PutContexts(contexts []*mdpb.Context) ([]ContextID, error) {
	req := &apipb.PutContextsRequest{
		Contexts: contexts,
	}
	resp := &apipb.PutContextsResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.PutContexts, req, resp)
	rst := make([]ContextID, len(resp.GetContextIds()))
	for i, v := range resp.GetContextIds() {
		rst[i] = ContextID(v)
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

// GetContextsByID gets a list of contexts by ID.
// If no context with an ID exists, the context id is skipped.
// It returns an error if the query execution fails.
func (store *Store) GetContextsByID(cids []ContextID) ([]*mdpb.Context, error) {
	req := &apipb.GetContextsByIDRequest{
		ContextIds: convertToInt64ArrayFromContextIDs(cids),
	}
	resp := &apipb.GetContextsByIDResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetContextsByID, req, resp)
	return resp.GetContexts(), err
}

// PutAttributionsAndAssociations inserts attribution and association relationships in the database.
//
// In `attributions` and `associations`, the ArtifactId, ExecutionId and ConextId must already exist.
// Once added, the relationships cannot be modified. If the relationship exists, this call does nothing.
//
// It returns an error if any artifact, execution or context cannot be found with the given id.
func (store *Store) PutAttributionsAndAssociations(attributions []*mdpb.Attribution, associations []*mdpb.Association) error {
	req := &apipb.PutAttributionsAndAssociationsRequest{
		Attributions: attributions,
		Associations: associations,
	}
	resp := &apipb.PutAttributionsAndAssociationsResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.PutAttributionsAndAssociations, req, resp)
	return err
}

// GetContextsByArtifact gets all context that an artifact is attributed to.
// It returns an error if query execution fails.
func (store *Store) GetContextsByArtifact(aid ArtifactID) ([]*mdpb.Context, error) {
	rid := int64(aid)
	req := &apipb.GetContextsByArtifactRequest{
		ArtifactId: &rid,
	}
	resp := &apipb.GetContextsByArtifactResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetContextsByArtifact, req, resp)
	return resp.GetContexts(), err
}

// GetContextsByExecution gets all context that an execution is associated with.
// It returns an error if query execution fails.
func (store *Store) GetContextsByExecution(eid ExecutionID) ([]*mdpb.Context, error) {
	rid := int64(eid)
	req := &apipb.GetContextsByExecutionRequest{
		ExecutionId: &rid,
	}
	resp := &apipb.GetContextsByExecutionResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetContextsByExecution, req, resp)
	return resp.GetContexts(), err
}

// GetArtifactsByContext gets all direct artifacts that a context attributes to.
// It returns an error if query execution fails.
func (store *Store) GetArtifactsByContext(cid ContextID) ([]*mdpb.Artifact, error) {
	rid := int64(cid)
	req := &apipb.GetArtifactsByContextRequest{
		ContextId: &rid,
	}
	resp := &apipb.GetArtifactsByContextResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetArtifactsByContext, req, resp)
	return resp.GetArtifacts(), err
}

// GetExecutionsByContext gets all direct executions that a context associates with.
// It returns an error if query execution fails.
func (store *Store) GetExecutionsByContext(cid ContextID) ([]*mdpb.Execution, error) {
	rid := int64(cid)
	req := &apipb.GetExecutionsByContextRequest{
		ContextId: &rid,
	}
	resp := &apipb.GetExecutionsByContextResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetExecutionsByContext, req, resp)
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

// ArtifactAndEvent defines a pair of artifact and event for PutExecution request.
// Event's artifact_id or execution_id can be empty, as the artifact or execution
// may not be stored beforehand.
type ArtifactAndEvent struct {
	Artifact *mdpb.Artifact
	Event    *mdpb.Event
}

// PutExecution inserts or updates atomically
//   - an Execution
//   - its input and output artifacts and events
//   - its related contexts
// The request includes the state changes of the Artifacts used or generated by the
// Execution, as well as the input/output Event. Optionally, the contexts can be given
// in the request to capture the attribution and association of the artifacts and the
// execution.
//
// If an execution_id, artifact_id, or context_id is specified, it is an update, otherwise
// it does an insertion. For insertion, type must be specified.
// If event.timestamp is not set, it will be set to the current time.
//
// It returns an error if
// a) no artifact, execution, or context exists with the given id,
// b) artifact, execution, or context's type id or property type does not align with the stored ones.
// c) the event.type field is UNKNOWN.
func (store *Store) PutExecution(e *mdpb.Execution, aep []*ArtifactAndEvent, ctxs []*mdpb.Context) (ExecutionID, []ArtifactID, []ContextID, error) {
	raep := make([]*apipb.PutExecutionRequest_ArtifactAndEvent, len(aep))
	for i, v := range aep {
		raep[i] = &apipb.PutExecutionRequest_ArtifactAndEvent{
			Artifact: v.Artifact,
			Event:    v.Event,
		}
	}
	req := &apipb.PutExecutionRequest{
		Execution:          e,
		ArtifactEventPairs: raep,
		Contexts:           ctxs,
	}
	resp := &apipb.PutExecutionResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.PutExecution, req, resp)
	aids := make([]ArtifactID, len(resp.GetArtifactIds()))
	cids := make([]ContextID, len(resp.GetContextIds()))
	for i, v := range resp.GetArtifactIds() {
		aids[i] = ArtifactID(v)
	}
	for i, v := range resp.GetContextIds() {
		cids[i] = ContextID(v)
	}
	return ExecutionID(resp.GetExecutionId()), aids, cids, err
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

// GetArtifactByTypeAndName gets the artifact of a given type and artifact name.
// It returns an error if the query execution fails.
func (store *Store) GetArtifactByTypeAndName(typeName, artifactName string) (*mdpb.Artifact, error) {
	req := &apipb.GetArtifactByTypeAndNameRequest{
		TypeName:     proto.String(typeName),
		ArtifactName: proto.String(artifactName),
	}
	resp := &apipb.GetArtifactByTypeAndNameResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetArtifactByTypeAndName, req, resp)
	return resp.GetArtifact(), err
}

// GetArtifactsByURI gets all artifacts of a given uri.
// It returns an error if the query execution fails.
func (store *Store) GetArtifactsByURI(uri string) ([]*mdpb.Artifact, error) {
	req := &apipb.GetArtifactsByURIRequest{
		Uris: []string{uri},
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

// GetExecutionByTypeAndName gets the execution of a given type and execution name.
// It returns an error if the query execution fails.
func (store *Store) GetExecutionByTypeAndName(typeName string, executionName string) (*mdpb.Execution, error) {
	req := &apipb.GetExecutionByTypeAndNameRequest{
		TypeName:      proto.String(typeName),
		ExecutionName: proto.String(executionName),
	}
	resp := &apipb.GetExecutionByTypeAndNameResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetExecutionByTypeAndName, req, resp)
	return resp.GetExecution(), err
}

// GetContexts gets all contexts.
// It returns an error if the query execution fails.
func (store *Store) GetContexts() ([]*mdpb.Context, error) {
	req := &apipb.GetContextsRequest{}
	resp := &apipb.GetContextsResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetContexts, req, resp)
	return resp.GetContexts(), err
}

// GetContextsByType gets all contexts of a given type.
// It returns an error if the query execution fails.
func (store *Store) GetContextsByType(typeName string) ([]*mdpb.Context, error) {
	req := &apipb.GetContextsByTypeRequest{
		TypeName: proto.String(typeName),
	}
	resp := &apipb.GetContextsByTypeResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetContextsByType, req, resp)
	return resp.GetContexts(), err
}

// GetContextByTypeAndName gets the context of a given type and context name.
// It returns an error if the query execution fails.
func (store *Store) GetContextByTypeAndName(typeName string, contextName string) (*mdpb.Context, error) {
	req := &apipb.GetContextByTypeAndNameRequest{
		TypeName:    proto.String(typeName),
		ContextName: proto.String(contextName),
	}
	resp := &apipb.GetContextByTypeAndNameResponse{}
	err := store.callMetadataStoreWrapMethod(wrap.GetContextByTypeAndName, req, resp)
	return resp.GetContext(), err
}

type metadataStoreMethod func(wrap.Ml_metadata_MetadataStore, string, wrap.Absl_Status) string

// callMetadataStoreWrapMethod calls a `metadataStoreMethod` in cc library.
// It returns an error if the cc library method returns non-ok status, or `req`,
// `resp` proto message cannot be serialized/parsed correctly.
func (store *Store) callMetadataStoreWrapMethod(fn metadataStoreMethod, req proto.Message, resp proto.Message) error {
	status := wrap.CreateABSLStatus()
	defer wrap.DestroyABSLStatus(status)

	b, err := proto.Marshal(req)
	if err != nil {
		return err
	}
	wrt := fn(store.ptr, string(b), status)
	if !wrap.IsOk(status) {
		return errors.New(wrap.ErrorMessage(status))
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

func convertToInt64ArrayFromContextTypeIDs(tids []ContextTypeID) []int64 {
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

func convertToInt64ArrayFromContextIDs(cids []ContextID) []int64 {
	ids := make([]int64, len(cids))
	for i, v := range cids {
		ids[i] = int64(v)
	}
	return ids
}
