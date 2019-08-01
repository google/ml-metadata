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

package mlmetadata

import (
	"log"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/google/go-cmp/cmp"
	mdpb "ml_metadata/proto/metadata_store_go_proto"
)

func createConnectionConfig(textConfig string) *mdpb.ConnectionConfig {
	config := &mdpb.ConnectionConfig{}
	if err := proto.UnmarshalText(textConfig, config); err != nil {
		log.Fatalf("Cannot create parse connection config: %v", textConfig)
	}
	return config
}

func fakeDatabaseConfig() *mdpb.ConnectionConfig {
	return createConnectionConfig(`fake_database {}`)
}

func TestPutArtifactType(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	defer store.Close()
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}

	typeName := `test_type_name`
	aType := &mdpb.ArtifactType{Name: &typeName}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := store.PutArtifactType(aType, opts)
	if err != nil {
		t.Fatalf("PutArtifactType failed: %v", err)
	}
	if typeID <= 0 {
		t.Errorf("expected type ID should be positive, got %v", typeID)
	}

	typeID2, err := store.PutArtifactType(aType, opts)
	if err != nil {
		t.Fatalf("PutArtifactType failed: %v", err)
	}
	if typeID2 != typeID {
		t.Errorf("given the same type name, type IDs should be the same. want: %v, got %v", typeID, typeID2)
	}

	newTypeName := "another_type_name"
	newType := &mdpb.ArtifactType{Name: &newTypeName}
	typeID3, err := store.PutArtifactType(newType, opts)
	if err != nil {
		t.Fatalf("PutArtifactType failed: %v", err)
	}
	if typeID3 == typeID {
		t.Errorf("given different type name, type IDs should be different. want: %v, got %v", typeID, typeID3)
	}
}

func TestPutAndUpdateArtifactType(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	defer store.Close()
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}

	textType := `name: 'test_type_name' properties { key: 'p1' value: INT } `
	aType := &mdpb.ArtifactType{}
	if err := proto.UnmarshalText(textType, aType); err != nil {
		t.Fatalf("Cannot parse text for ArtifactType proto: %v", err)
	}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := store.PutArtifactType(aType, opts)
	if err != nil {
		t.Fatalf("PutArtifactType failed: %v", err)
	}
	if typeID <= 0 {
		t.Errorf("expected type ID should be positive, got %v", typeID)
	}

	wantTextType := `name: 'test_type_name' properties { key: 'p1' value: INT } properties { key: 'p2' value: DOUBLE } `
	wantType := &mdpb.ArtifactType{}
	if err := proto.UnmarshalText(wantTextType, wantType); err != nil {
		t.Fatalf("Cannot parse text for ArtifactType proto: %v", err)
	}
	opts = &PutTypeOptions{AllFieldsMustMatch: true, CanAddFields: true}
	typeID2, err := store.PutArtifactType(wantType, opts)
	if err != nil {
		t.Fatalf("PutArtifactType failed: %v", err)
	}
	if typeID2 != typeID {
		t.Errorf("update the type, type IDs should be the same. want: %v, got %v", typeID, typeID2)
	}

	tids := make([]ArtifactTypeID, 1)
	tids[0] = typeID
	gotTypesByID, err := store.GetArtifactTypesByID(tids)
	if err != nil {
		t.Fatalf("GetArtifactTypesByID failed: %v", err)
	}
	tid := int64(typeID)
	wantType.Id = &tid
	if len(gotTypesByID) < 1 || !proto.Equal(wantType, gotTypesByID[0]) {
		t.Errorf("put and get type by id mismatch, want: %v, got: %v", wantType, gotTypesByID)
	}
}

func TestGetArtifactType(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	defer store.Close()
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}

	typeName := `test_type_name`
	wantType := &mdpb.ArtifactType{Name: &typeName}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := store.PutArtifactType(wantType, opts)
	if err != nil {
		t.Fatalf("PutArtifactType failed: %v", err)
	}
	tid := int64(typeID)
	wantType.Id = &tid

	// get artifact type by name
	gotType, err := store.GetArtifactType(typeName)
	if err != nil {
		t.Fatalf("GetArtifactType failed: %v", err)
	}
	if !proto.Equal(wantType, gotType) {
		t.Errorf("put and get type mismatch, want: %v, got: %v", wantType, gotType)
	}

	// get artifact type by id
	tids := make([]ArtifactTypeID, 1)
	tids[0] = typeID
	gotTypesByID, err := store.GetArtifactTypesByID(tids)
	if err != nil {
		t.Fatalf("GetArtifactTypesByID failed: %v", err)
	}
	if len(gotTypesByID) < 1 || !proto.Equal(wantType, gotTypesByID[0]) {
		t.Errorf("put and get type by id mismatch, want: %v, got: %v", wantType, gotTypesByID)
	}
}

func TestGetArtifactTypes(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	defer store.Close()
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}

	wantType1 := &mdpb.ArtifactType{Name: proto.String("test_type_1")}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := store.PutArtifactType(wantType1, opts)
	if err != nil {
		t.Fatalf("PutArtifactType failed: %v", err)
	}
	wantType1.Id = proto.Int64(int64(typeID))

	wantType2 := &mdpb.ArtifactType{Name: proto.String("test_type_2")}
	typeID, err = store.PutArtifactType(wantType2, opts)
	if err != nil {
		t.Fatalf("PutArtifactType failed: %v", err)
	}
	wantType2.Id = proto.Int64(int64(typeID))

	wantTypes := []*mdpb.ArtifactType{wantType1, wantType2}
	gotTypes, err := store.GetArtifactTypes()
	if err != nil {
		t.Fatalf("GetArtifactTypes failed: %v", err)
	}

	if !cmp.Equal(wantTypes, gotTypes, cmp.Comparer(proto.Equal)) {
		t.Errorf("GetArtifactTypes() mismatch, want: %v\n got: %v\nDiff:\n%s", wantTypes, gotTypes, cmp.Diff(gotTypes, wantTypes))
	}
}

func TestGetExecutionTypes(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	defer store.Close()
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}

	wantType1 := &mdpb.ExecutionType{Name: proto.String("test_type_1")}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := store.PutExecutionType(wantType1, opts)
	if err != nil {
		t.Fatalf("PutExecutionType failed: %v", err)
	}
	wantType1.Id = proto.Int64(int64(typeID))

	wantType2 := &mdpb.ExecutionType{Name: proto.String("test_type_2")}
	typeID, err = store.PutExecutionType(wantType2, opts)
	if err != nil {
		t.Fatalf("PutExecutionType failed: %v", err)
	}
	wantType2.Id = proto.Int64(int64(typeID))

	wantTypes := []*mdpb.ExecutionType{wantType1, wantType2}
	gotTypes, err := store.GetExecutionTypes()
	if err != nil {
		t.Fatalf("GetExecutionTypes failed: %v", err)
	}

	if !cmp.Equal(wantTypes, gotTypes, cmp.Comparer(proto.Equal)) {
		t.Errorf("GetExecutionTypes() mismatch, want: %v\n got: %v\nDiff:\n%s", wantTypes, gotTypes, cmp.Diff(gotTypes, wantTypes))
	}
}

func TestPutAndGetExecutionType(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	defer store.Close()
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	textType := `name: 'test_type_name' properties { key: 'p1' value: INT } `
	wantType := &mdpb.ExecutionType{}
	if err := proto.UnmarshalText(textType, wantType); err != nil {
		t.Fatalf("Cannot parse text for ExecutionType proto: %v", err)
	}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := store.PutExecutionType(wantType, opts)
	if err != nil {
		t.Fatalf("PutExecutionType failed: %v", err)
	}
	tid := int64(typeID)
	wantType.Id = &tid

	typeName := wantType.GetName()
	gotType, err := store.GetExecutionType(typeName)
	if err != nil {
		t.Fatalf("GetExecutionType failed: %v", err)
	}
	if !proto.Equal(wantType, gotType) {
		t.Errorf("put and get type mismatch, want: %v, got: %v", wantType, gotType)
	}

	tids := make([]ExecutionTypeID, 1)
	tids[0] = typeID
	gotTypesByID, err := store.GetExecutionTypesByID(tids)
	if err != nil {
		t.Fatalf("GetExecutionTypesByID failed: %v", err)
	}
	if len(gotTypesByID) < 1 || !proto.Equal(wantType, gotTypesByID[0]) {
		t.Errorf("put and get type by id mismatch, want: %v, got: %v", wantType, gotTypesByID)
	}
}

func TestPutAndUpdateExecutionType(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	defer store.Close()
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}

	textType := `name: 'test_type_name' properties { key: 'p1' value: INT } `
	eType := &mdpb.ExecutionType{}
	if err := proto.UnmarshalText(textType, eType); err != nil {
		t.Fatalf("Cannot parse text for ExecutionType proto: %v", err)
	}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := store.PutExecutionType(eType, opts)
	if err != nil {
		t.Fatalf("PutExecutionType failed: %v", err)
	}
	if typeID <= 0 {
		t.Errorf("expected type ID should be positive, got %v", typeID)
	}

	wantTextType := `name: 'test_type_name' properties { key: 'p1' value: INT } properties { key: 'p2' value: DOUBLE } `
	wantType := &mdpb.ExecutionType{}
	if err := proto.UnmarshalText(wantTextType, wantType); err != nil {
		t.Fatalf("Cannot parse text for ExecutionType proto: %v", err)
	}
	opts = &PutTypeOptions{AllFieldsMustMatch: true, CanAddFields: true}
	typeID2, err := store.PutExecutionType(wantType, opts)
	if err != nil {
		t.Fatalf("PutExecutionType failed: %v", err)
	}
	if typeID2 != typeID {
		t.Errorf("update the type, type IDs should be the same. want: %v, got %v", typeID, typeID2)
	}

	tids := make([]ExecutionTypeID, 1)
	tids[0] = typeID
	gotTypesByID, err := store.GetExecutionTypesByID(tids)
	if err != nil {
		t.Fatalf("GetExecutionTypesByID failed: %v", err)
	}
	tid := int64(typeID)
	wantType.Id = &tid
	if len(gotTypesByID) < 1 || !proto.Equal(wantType, gotTypesByID[0]) {
		t.Errorf("put and get type by id mismatch, want: %v, got: %v", wantType, gotTypesByID)
	}
}

func insertArtifactType(s *Store, textType string) (int64, error) {
	rst := &mdpb.ArtifactType{}
	if err := proto.UnmarshalText(textType, rst); err != nil {
		return 0, err
	}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	tid, err := s.PutArtifactType(rst, opts)
	if err != nil {
		return 0, err
	}
	return int64(tid), nil
}

// TODO(b/124764089) Separate the test.
func TestPutAndGetArtifacts(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	defer store.Close()
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	tid, err := insertArtifactType(store, `name: 'test_type_name' properties { key: 'p1' value: INT } `)
	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}

	// put artifacts
	artifacts := make([]*mdpb.Artifact, 2)
	uri1, uri2 := "test_uri1", "test_uri2"
	artifacts[0] = &mdpb.Artifact{
		TypeId: &tid,
		Uri:    &uri1,
		CustomProperties: map[string]*mdpb.Value{
			`p1`: &mdpb.Value{Value: &mdpb.Value_StringValue{StringValue: `val`}},
		},
	}
	artifacts[1] = &mdpb.Artifact{
		TypeId: &tid,
		Uri:    &uri2,
		Properties: map[string]*mdpb.Value{
			`p1`: &mdpb.Value{Value: &mdpb.Value_IntValue{IntValue: 1}},
		},
	}

	aids, err := store.PutArtifacts(artifacts)
	if err != nil {
		t.Fatalf("PutArtifacts failed: %v", err)
	}
	if len(aids) != len(artifacts) {
		t.Errorf("PutArtifacts number of artifacts mismatch, want: %v, got: %v", len(artifacts), len(aids))
	}
	if aids[0] == aids[1] {
		t.Errorf("PutArtifacts should not return two identical id, id1: %v, id2: %v", aids[0], aids[1])
	}

	// query artifacts by ids
	wantArtifact := artifacts[0]
	waid := int64(aids[0])
	wantArtifact.Id = &waid
	gotArtifacts, err := store.GetArtifactsByID(aids[0:1])
	if err != nil {
		t.Fatalf("GetArtifactsByID failed: %v", err)
	}
	if len(gotArtifacts) != 1 {
		t.Errorf("GetArtifactsByID cannot find result with id: %v", waid)
	}
	if !proto.Equal(wantArtifact, gotArtifacts[0]) {
		t.Errorf("GetArtifactsByID returned result is incorrect. want: %v, got: %v", wantArtifact, gotArtifacts[0])
	}

	// query all artifacts
	gotStoredArtifacts, err := store.GetArtifacts()
	if err != nil {
		t.Fatalf("GetArtifacts failed: %v", err)
	}
	if len(gotStoredArtifacts) != len(artifacts) {
		t.Errorf("GetArtifacts number of artifacts mismatch, want: %v, got: %v", len(artifacts), len(gotStoredArtifacts))
	}
	if proto.Equal(gotStoredArtifacts[0], gotStoredArtifacts[1]) {
		t.Errorf("GetArtifacts returns duplicated artifacts: %v. want: %v, %v", gotStoredArtifacts[0], artifacts[0], artifacts[1])
	}

	// query artifacts by URI
	gotArtifactsOfURI, err := store.GetArtifactsByURI(uri1)
	if err != nil {
		t.Fatalf("GetArtifactsByURI failed: %v", err)
	}
	if len(gotArtifactsOfURI) != 1 {
		t.Errorf("GetArtifactsByURI number of artifacts mismatch, want: %v, got: %v", 1, len(gotArtifactsOfURI))
	}
	if !proto.Equal(wantArtifact, gotArtifactsOfURI[0]) {
		t.Errorf("GetArtifactsByURI returned result is incorrect. want: %v, got: %v", wantArtifact, gotArtifactsOfURI[0])
	}
	unknownURI := "unknown_uri"
	gotArtifactsOfUnknownURI, err := store.GetArtifactsByURI(unknownURI)
	if err != nil {
		t.Fatalf("GetArtifactsByURI failed: %v", err)
	}
	if len(gotArtifactsOfUnknownURI) != 0 {
		t.Errorf("GetArtifactsByURI number of artifacts mismatch, want: %v, got: %v", 0, len(gotArtifactsOfUnknownURI))
	}

	// query artifacts of a particular type
	typeName := "test_type_name"
	gotArtifactsOfType, err := store.GetArtifactsByType(typeName)
	if err != nil {
		t.Fatalf("GetArtifactsByType failed: %v", err)
	}
	if len(gotArtifactsOfType) != len(artifacts) {
		t.Errorf("GetArtifactsByType number of artifacts mismatch, want: %v, got: %v", len(artifacts), len(gotArtifactsOfType))
	}
	if proto.Equal(gotArtifactsOfType[0], gotArtifactsOfType[1]) {
		t.Errorf("GetArtifactsByType returns duplicated artifacts: %v. want: %v, %v", gotArtifactsOfType[0], artifacts[0], artifacts[1])
	}

	// query artifacts of a non-exist type
	notExistTypeName := "not_exist_type_name"
	gotArtifactsOfNotExistType, err := store.GetArtifactsByType(notExistTypeName)
	if err != nil {
		t.Fatalf("GetArtifactsByType failed: %v", err)
	}
	if len(gotArtifactsOfNotExistType) != 0 {
		t.Errorf("GetArtifactsByType number of artifacts mismatch of non-exist type, want: 0, got: %v", len(gotArtifactsOfNotExistType))
	}

	// test querying artifacts of an empty type having no artifacts
	_, err = insertArtifactType(store, `name: 'test_type_name_no_artifacts' properties { key: 'p1' value: INT } `)
	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}
	typeNameNoArtifacts := "test_type_name_no_artifacts"
	gotEmptyTypeArtifacts, err := store.GetArtifactsByType(typeNameNoArtifacts)
	if err != nil {
		t.Fatalf("GetArtifactsByType failed: %v", err)
	}
	if len(gotEmptyTypeArtifacts) != 0 {
		t.Errorf("GetArtifactsByType number of artifacts mismatch of an empty type, want: 0, got: %v", len(gotEmptyTypeArtifacts))
	}
}

func insertExecutionType(s *Store, textType string) (int64, error) {
	rst := &mdpb.ExecutionType{}
	if err := proto.UnmarshalText(textType, rst); err != nil {
		return 0, err
	}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	tid, err := s.PutExecutionType(rst, opts)
	if err != nil {
		return 0, err
	}
	return int64(tid), nil
}

// TODO(b/124764089) Separate the test.
func TestPutAndGetExecutions(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	defer store.Close()
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	tid, err := insertExecutionType(store, `name: 'test_type_name' properties { key: 'p1' value: DOUBLE } `)
	if err != nil {
		t.Fatalf("Cannot create execution type: %v", err)
	}
	wantExecution := &mdpb.Execution{
		TypeId: &tid,
		Properties: map[string]*mdpb.Value{
			`p1`: &mdpb.Value{Value: &mdpb.Value_DoubleValue{DoubleValue: 1.0}},
		},
		CustomProperties: map[string]*mdpb.Value{
			`p1`: &mdpb.Value{Value: &mdpb.Value_IntValue{IntValue: 1}},
		},
	}
	// insert 1 execution
	executions := make([]*mdpb.Execution, 1)
	executions[0] = wantExecution
	eids, err := store.PutExecutions(executions)
	if err != nil {
		t.Fatalf("PutExecutions failed: %v", err)
	}
	if len(eids) != len(executions) {
		t.Errorf("PutExecutions number of executions mismatch, want: %v, got: %v", len(executions), len(eids))
	}
	weid := int64(eids[0])
	wantExecution.Id = &weid

	// query executions by ids
	gotExecutions, err := store.GetExecutionsByID(eids[0:1])
	if err != nil {
		t.Fatalf("GetExecutionsByID failed: %v", err)
	}
	if len(gotExecutions) != 1 {
		t.Errorf("GetExecutionsByID cannot find result with id: %v", weid)
	}
	if !proto.Equal(wantExecution, gotExecutions[0]) {
		t.Errorf("GetExecutionsByID returned result is incorrect. want: %v, got: %v", wantExecution, gotExecutions[0])
	}

	// query all executions
	gotStoredExecutions, err := store.GetExecutions()
	if err != nil {
		t.Fatalf("GetExecutions failed: %v", err)
	}
	if len(gotStoredExecutions) != len(executions) {
		t.Errorf("GetExecutions number of executions mismatch, want: %v, got: %v", len(executions), len(gotStoredExecutions))
	}
	if !proto.Equal(wantExecution, gotStoredExecutions[0]) {
		t.Errorf("GetExecutions returned result is incorrect. want: %v, got: %v", wantExecution, gotStoredExecutions[0])
	}

	// query executions of a particular type
	typeName := "test_type_name"
	gotExecutionsOfType, err := store.GetExecutionsByType(typeName)
	if err != nil {
		t.Fatalf("GetExecutionsByType failed: %v", err)
	}
	if len(gotExecutionsOfType) != len(executions) {
		t.Errorf("GetExecutionsByType number of executions mismatch, want: %v, got: %v", len(executions), len(gotExecutionsOfType))
	}
	if !proto.Equal(wantExecution, gotExecutionsOfType[0]) {
		t.Errorf("GetExecutionsByType returned result is incorrect. want: %v, got: %v", wantExecution, gotExecutionsOfType[0])
	}

	// query executions of a non-existent type
	notExistTypeName := "not_exist_type_name"
	gotExecutionsOfNotExistType, err := store.GetExecutionsByType(notExistTypeName)
	if err != nil {
		t.Fatalf("GetExecutionsByType failed: %v", err)
	}
	if len(gotExecutionsOfNotExistType) != 0 {
		t.Errorf("GetExecutionsByType number of executions mismatch of non-exist type, want: 0, got: %v", len(gotExecutionsOfNotExistType))
	}

	// test querying executions of an empty type having no execution
	_, err = insertArtifactType(store, `name: 'test_type_name_no_execution' properties { key: 'p1' value: INT } `)
	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}
	typeNameNoExecutions := "test_type_name_no_execution"
	gotEmptyTypeArtifacts, err := store.GetExecutionsByType(typeNameNoExecutions)
	if err != nil {
		t.Fatalf("GetExecutionsByType failed: %v", err)
	}
	if len(gotEmptyTypeArtifacts) != 0 {
		t.Errorf("GetExecutionsByType number of artifacts mismatch of an empty type, want: 0, got: %v", len(gotEmptyTypeArtifacts))
	}
}

func TestPutAndGetEvents(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	defer store.Close()
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	atid, err := insertArtifactType(store, `name: 'artifact_type_name' `)
	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}
	as := make([]*mdpb.Artifact, 2)
	a1 := &mdpb.Artifact{TypeId: &atid}
	as[0] = a1
	a2 := &mdpb.Artifact{TypeId: &atid}
	as[1] = a2
	aids, err := store.PutArtifacts(as)
	if err != nil {
		t.Fatalf("PutArtifacts failed: %v", err)
	}
	a1id, a2id := int64(aids[0]), int64(aids[1])

	etid, err := insertExecutionType(store, `name: 'execution_type_name' `)
	if err != nil {
		t.Fatalf("Cannot create execution type: %v", err)
	}
	es := make([]*mdpb.Execution, 2)
	e1 := &mdpb.Execution{TypeId: &etid}
	es[0] = e1
	e2 := &mdpb.Execution{TypeId: &etid}
	es[1] = e2
	eids, err := store.PutExecutions(es)
	if err != nil {
		t.Fatalf("PutExecutions failed: %v", err)
	}
	e1id := int64(eids[0])

	// insert events
	wantEvents := make([]*mdpb.Event, 2)
	inputType := mdpb.Event_Type(mdpb.Event_INPUT)
	outputType := mdpb.Event_Type(mdpb.Event_OUTPUT)
	// event: a1 - e1
	inTime := int64(100000)
	wantEvents[0] = &mdpb.Event{
		ArtifactId:  &a1id,
		ExecutionId: &e1id,
		Path: &mdpb.Event_Path{
			Steps: []*mdpb.Event_Path_Step{
				&mdpb.Event_Path_Step{
					Value: &mdpb.Event_Path_Step_Key{Key: `param1`},
				},
			},
		},
		Type:                   &inputType,
		MillisecondsSinceEpoch: &inTime,
	}
	// event: a2 - e1
	outTime := int64(200000)
	wantEvents[1] = &mdpb.Event{
		ArtifactId:  &a2id,
		ExecutionId: &e1id,
		Path: &mdpb.Event_Path{
			Steps: []*mdpb.Event_Path_Step{
				&mdpb.Event_Path_Step{
					Value: &mdpb.Event_Path_Step_Index{Index: 1},
				},
			},
		},
		Type:                   &outputType,
		MillisecondsSinceEpoch: &outTime,
	}
	err = store.PutEvents(wantEvents)
	if err != nil {
		t.Fatalf("PutEvents failed: %v", err)
	}

	// query events via a1
	gotEvents, err := store.GetEventsByArtifactIDs(aids[0:1])
	if err != nil {
		t.Fatalf("GetEventsByArtifactIDs failed: %v", err)
	}
	if len(gotEvents) != 1 {
		t.Errorf("GetEventsByArtifactIDs number of events mismatch, want: %v, got: %v", 1, len(gotEvents))
	}
	if !proto.Equal(gotEvents[0], wantEvents[0]) {
		t.Errorf("GetEventsByArtifactIDs returned events mismatch, want: %v, got: %v", wantEvents[0], gotEvents[0])
	}

	// query events via a2
	gotEvents, err = store.GetEventsByArtifactIDs(aids[1:2])
	if err != nil {
		t.Fatalf("GetEventsByArtifactIDs failed: %v", err)
	}
	if len(gotEvents) != 1 {
		t.Errorf("GetEventsByArtifactIDs number of events mismatch, want: %v, got: %v", 1, len(gotEvents))
	}
	if !proto.Equal(gotEvents[0], wantEvents[1]) {
		t.Errorf("GetEventsByArtifactIDs returned events mismatch, want: %v, got: %v", wantEvents[1], gotEvents[0])
	}

	// query events via e1
	gotEvents, err = store.GetEventsByExecutionIDs(eids[0:1])
	if err != nil {
		t.Fatalf("GetEventsByExecutionIDs failed: %v", err)
	}
	if len(gotEvents) != 2 {
		t.Errorf("GetEventsByExecutionIDs number of events mismatch, want: %v, got: %v", 2, len(gotEvents))
	}
	if gotEvents[0].GetArtifactId() == a1id {
		if !proto.Equal(gotEvents[0], wantEvents[0]) || !proto.Equal(gotEvents[1], wantEvents[1]) {
			t.Errorf("GetEventsByExecutionIDs returned events mismatch, want: %v, got: %v", gotEvents, wantEvents)
		}
	} else if gotEvents[0].GetArtifactId() == a2id {
		if !proto.Equal(gotEvents[0], wantEvents[1]) || !proto.Equal(gotEvents[1], wantEvents[0]) {
			t.Errorf("GetEventsByExecutionIDs returned events mismatch, want: %v, got: %v", gotEvents, wantEvents)
		}
	} else {
		t.Errorf("GetEventsByExecutionIDs returned events mismatch, want: %v, got: %v", gotEvents, wantEvents)
	}

	// query events by e2
	gotEvents, err = store.GetEventsByExecutionIDs(eids[1:2])
	if err != nil {
		t.Fatalf("GetEventsByExecutionIDs failed: %v", err)
	}
	if len(gotEvents) > 0 {
		t.Errorf("GetEventsByExecutionIDs number of events mismatch, want: %v, got: %v", 0, len(gotEvents))
	}
}
