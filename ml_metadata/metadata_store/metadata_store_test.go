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

	"github.com/google/go-cmp/cmp"
	"github.com/golang/protobuf/proto"
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
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()

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
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()

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
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()

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
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()

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
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()

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
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()

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
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()

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

func TestGetContextTypes(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()

	wantType1 := &mdpb.ContextType{Name: proto.String("test_type_1")}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := store.PutContextType(wantType1, opts)
	if err != nil {
		t.Fatalf("PutContextType failed: %v", err)
	}
	wantType1.Id = proto.Int64(int64(typeID))

	wantType2 := &mdpb.ContextType{Name: proto.String("test_type_2")}
	typeID, err = store.PutContextType(wantType2, opts)
	if err != nil {
		t.Fatalf("PutContextType failed: %v", err)
	}
	wantType2.Id = proto.Int64(int64(typeID))

	wantTypes := []*mdpb.ContextType{wantType1, wantType2}
	gotTypes, err := store.GetContextTypes()
	if err != nil {
		t.Fatalf("GetContextTypes failed: %v", err)
	}

	if !cmp.Equal(wantTypes, gotTypes, cmp.Comparer(proto.Equal)) {
		t.Errorf("GetContextTypes() mismatch, want: %v\n got: %v\nDiff:\n%s", wantTypes, gotTypes, cmp.Diff(gotTypes, wantTypes))
	}
}

func TestPutAndGetContextType(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()

	textType := `name: 'test_type_name' properties { key: 'p1' value: INT } `
	wantType := &mdpb.ContextType{}
	if err := proto.UnmarshalText(textType, wantType); err != nil {
		t.Fatalf("Cannot parse text for ContextType proto: %v", err)
	}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := store.PutContextType(wantType, opts)
	if err != nil {
		t.Fatalf("PutContextType failed: %v", err)
	}
	tid := int64(typeID)
	wantType.Id = &tid

	typeName := wantType.GetName()
	gotType, err := store.GetContextType(typeName)
	if err != nil {
		t.Fatalf("GetContextType failed: %v", err)
	}
	if !proto.Equal(wantType, gotType) {
		t.Errorf("put and get type mismatch, want: %v, got: %v", wantType, gotType)
	}

	tids := make([]ContextTypeID, 1)
	tids[0] = typeID
	gotTypesByID, err := store.GetContextTypesByID(tids)
	if err != nil {
		t.Fatalf("GetContextTypesByID failed: %v", err)
	}
	if len(gotTypesByID) < 1 || !proto.Equal(wantType, gotTypesByID[0]) {
		t.Errorf("put and get type by id mismatch, want: %v, got: %v", wantType, gotTypesByID)
	}
}

func TestPutAndUpdateContextType(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()

	textType := `name: 'test_type_name' properties { key: 'p1' value: INT } `
	cType := &mdpb.ContextType{}
	if err := proto.UnmarshalText(textType, cType); err != nil {
		t.Fatalf("Cannot parse text for ContextType proto: %v", err)
	}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := store.PutContextType(cType, opts)
	if err != nil {
		t.Fatalf("PutContextType failed: %v", err)
	}

	wantTextType := `name: 'test_type_name' properties { key: 'p1' value: INT } properties { key: 'p2' value: DOUBLE } `
	wantType := &mdpb.ContextType{}
	if err := proto.UnmarshalText(wantTextType, wantType); err != nil {
		t.Fatalf("Cannot parse text for ContextType proto: %v", err)
	}
	opts = &PutTypeOptions{AllFieldsMustMatch: true, CanAddFields: true}
	typeID2, err := store.PutContextType(wantType, opts)
	if err != nil {
		t.Fatalf("PutContextType failed: %v", err)
	}
	if typeID2 != typeID {
		t.Errorf("update the type, type IDs should be the same. want: %v, got %v", typeID, typeID2)
	}

	tids := make([]ContextTypeID, 1)
	tids[0] = typeID
	gotTypesByID, err := store.GetContextTypesByID(tids)
	if err != nil {
		t.Fatalf("GetContextTypesByID failed: %v", err)
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
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()
	tid, err := insertArtifactType(store, `name: 'test_type_name' properties { key: 'p1' value: INT } `)
	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}

	// put artifacts
	artifacts := make([]*mdpb.Artifact, 2)
	uri1, uri2 := "test_uri1", "test_uri2"
	createTimeSinceEpoch, lastUpdateTimeSinceEpoch := int64(1), int64(2)
	artifacts[0] = &mdpb.Artifact{
		TypeId:                   &tid,
		Uri:                      &uri1,
		CreateTimeSinceEpoch:     &createTimeSinceEpoch,
		LastUpdateTimeSinceEpoch: &lastUpdateTimeSinceEpoch,
		CustomProperties: map[string]*mdpb.Value{
			`p1`: &mdpb.Value{Value: &mdpb.Value_StringValue{StringValue: `val`}},
		},
	}
	artifacts[1] = &mdpb.Artifact{
		TypeId:                   &tid,
		Uri:                      &uri2,
		CreateTimeSinceEpoch:     &createTimeSinceEpoch,
		LastUpdateTimeSinceEpoch: &lastUpdateTimeSinceEpoch,
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
	// skip comparing create/update timestamps
	wantArtifact.CreateTimeSinceEpoch = gotArtifacts[0].CreateTimeSinceEpoch
	wantArtifact.LastUpdateTimeSinceEpoch = gotArtifacts[0].LastUpdateTimeSinceEpoch
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
	// skip comparing create/update timestamps
	wantArtifact.CreateTimeSinceEpoch = gotArtifactsOfURI[0].CreateTimeSinceEpoch
	wantArtifact.LastUpdateTimeSinceEpoch = gotArtifactsOfURI[0].LastUpdateTimeSinceEpoch
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

func TestPutArtifactAndGetArtifactByTypeAndName(t *testing.T) {
	// prepare test data
	store, err := NewStore(fakeDatabaseConfig())
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()
	tid, err := insertArtifactType(store, `name: 'test_type_name'`)
	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}
	artifactName := "test_artifact"
	artifactTypeName := "test_type_name"
	uri := "/test/uri"
	wantArtifact := &mdpb.Artifact{
		TypeId: &tid,
		Name:   &artifactName,
		Uri:    &uri,
	}
	// insert 1 artifact
	artifacts := []*mdpb.Artifact{wantArtifact}
	aids, err := store.PutArtifacts(artifacts)
	if err != nil {
		t.Fatalf("PutArtifacts failed: %v", err)
	}
	if len(aids) != len(artifacts) {
		t.Errorf("PutArtifacts number of artifacts mismatch, want: %v, got: %v", len(artifacts), len(aids))
	}
	waid := int64(aids[0])
	wantArtifact.Id = &waid

	// test GetArtifactByTypeAndName functionality
	// query artifact by both type name and artifact name
	gotStoredArtifact, err := store.GetArtifactByTypeAndName(artifactTypeName, artifactName)
	if err != nil {
		t.Fatalf("GetArtifactByTypeAndName failed: %v", err)
	}
	// skip comparing create/update timestamps
	wantArtifact.CreateTimeSinceEpoch = gotStoredArtifact.CreateTimeSinceEpoch
	wantArtifact.LastUpdateTimeSinceEpoch = gotStoredArtifact.LastUpdateTimeSinceEpoch
	if !proto.Equal(wantArtifact, gotStoredArtifact) {
		t.Errorf("GetArtifactByTypeAndName returned result is incorrect. want: %v, got: %v", wantArtifact, gotStoredArtifact)
	}

	// query artifact with either artifactTypeName or artifactName that doesn't exist
	tests := []struct {
		aname  string
		atname string
	}{
		{
			aname:  artifactName,
			atname: "random_type_name",
		},
		{
			aname:  "random_artifact_name",
			atname: artifactTypeName,
		},
		{
			aname:  "random_artifact_name",
			atname: "random_type_name",
		},
	}

	for _, tc := range tests {
		gotEmptyArtifact, err := store.GetArtifactByTypeAndName(tc.atname, tc.aname)
		if err != nil {
			t.Errorf("GetArtifactByTypeAndName failed with input type name: %v, artifact name: %v and got error: %v", tc.atname, tc.aname, err)
			continue
		}
		if gotEmptyArtifact != nil {
			t.Errorf("GetArtifactByTypeAndName returned result is incorrect. want: %v, got: %v. Input type name: %v, artifact name: %v", nil, gotEmptyArtifact, tc.atname, tc.aname)
		}
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
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()
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
	// skip comparing create/update timestamps
	wantExecution.CreateTimeSinceEpoch = gotExecutions[0].CreateTimeSinceEpoch
	wantExecution.LastUpdateTimeSinceEpoch = gotExecutions[0].LastUpdateTimeSinceEpoch
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
	// skip comparing create/update timestamps
	wantExecution.CreateTimeSinceEpoch = gotStoredExecutions[0].CreateTimeSinceEpoch
	wantExecution.LastUpdateTimeSinceEpoch = gotStoredExecutions[0].LastUpdateTimeSinceEpoch
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
	// skip comparing create/update timestamps
	wantExecution.CreateTimeSinceEpoch = gotExecutionsOfType[0].CreateTimeSinceEpoch
	wantExecution.LastUpdateTimeSinceEpoch = gotExecutionsOfType[0].LastUpdateTimeSinceEpoch
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
	_, err = insertExecutionType(store, `name: 'test_type_name_no_execution' properties { key: 'p1' value: INT } `)
	if err != nil {
		t.Fatalf("Cannot create execution type: %v", err)
	}
	typeNameNoExecutions := "test_type_name_no_execution"
	gotEmptyTypeExecutions, err := store.GetExecutionsByType(typeNameNoExecutions)
	if err != nil {
		t.Fatalf("GetExecutionsByType failed: %v", err)
	}
	if len(gotEmptyTypeExecutions) != 0 {
		t.Errorf("GetExecutionsByType number of artifacts mismatch of an empty type, want: 0, got: %v", len(gotEmptyTypeExecutions))
	}
}

func TestPutExecutionAndGetExecutionByTypeAndName(t *testing.T) {
	// prepare test data
	store, err := NewStore(fakeDatabaseConfig())
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()
	tid, err := insertExecutionType(store, `name: 'test_type_name'`)
	if err != nil {
		t.Fatalf("Cannot create execution type: %v", err)
	}
	executionName := "test_execution"
	executionTypeName := "test_type_name"
	wantExecution := &mdpb.Execution{
		TypeId: &tid,
		Name:   &executionName,
	}
	// insert 1 execution
	executions := make([]*mdpb.Execution, 1)
	executions[0] = wantExecution
	cids, err := store.PutExecutions(executions)
	if err != nil {
		t.Fatalf("PutExecutions failed: %v", err)
	}
	if len(cids) != len(executions) {
		t.Errorf("PutExecutions number of executions mismatch, got: %v, want: %v", len(executions), len(cids))
	}
	wcid := int64(cids[0])
	wantExecution.Id = &wcid

	// test GetExecutionByTypeAndName functionality
	// query execution by both type name and execution name
	gotStoredExecution, err := store.GetExecutionByTypeAndName(executionTypeName, executionName)
	if err != nil {
		t.Fatalf("GetExecutionByTypeAndName failed: %v", err)
	}
	// skip comparing create/update timestamps
	wantExecution.CreateTimeSinceEpoch = gotStoredExecution.CreateTimeSinceEpoch
	wantExecution.LastUpdateTimeSinceEpoch = gotStoredExecution.LastUpdateTimeSinceEpoch
	if !proto.Equal(wantExecution, gotStoredExecution) {
		t.Errorf("GetExecutionByTypeAndName returned result is incorrect. got: %v, want: %v", gotStoredExecution, wantExecution)
	}

	// query execution with either executionTypeName or executionName that doesn't exist
	tests := []struct {
		cname  string
		ctname string
	}{
		{
			cname:  executionName,
			ctname: "random_type_name",
		},
		{
			cname:  "random_execution_name",
			ctname: executionTypeName,
		},
		{
			cname:  "random_execution_name",
			ctname: "random_type_name",
		},
	}

	for _, tc := range tests {
		gotEmptyExecution, err := store.GetExecutionByTypeAndName(tc.ctname, tc.cname)
		if err != nil {
			t.Fatalf("GetExecutionByTypeAndName failed: %v", err)
		}
		if gotEmptyExecution != nil {
			t.Errorf("GetExecutionByTypeAndName returned result is incorrect. got: %v, want: %v", gotEmptyExecution, nil)
		}
	}
}

func insertContextType(s *Store, textType string) (int64, error) {
	rst := &mdpb.ContextType{}
	if err := proto.UnmarshalText(textType, rst); err != nil {
		return 0, err
	}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	tid, err := s.PutContextType(rst, opts)
	if err != nil {
		return 0, err
	}
	return int64(tid), nil
}

// TODO(b/124764089) Separate the test.
func TestPutAndGetContexts(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()
	tid, err := insertContextType(store, `name: 'test_type_name' properties { key: 'p1' value: STRING } `)
	if err != nil {
		t.Fatalf("Cannot create context type: %v", err)
	}
	cname := "context"
	wantContext := &mdpb.Context{
		TypeId: &tid,
		Name:   &cname,
		Properties: map[string]*mdpb.Value{
			`p1`: &mdpb.Value{Value: &mdpb.Value_StringValue{StringValue: `val`}},
		},
		CustomProperties: map[string]*mdpb.Value{
			`p1`: &mdpb.Value{Value: &mdpb.Value_IntValue{IntValue: 1}},
		},
	}
	// insert 1 context
	contexts := make([]*mdpb.Context, 1)
	contexts[0] = wantContext
	cids, err := store.PutContexts(contexts)
	if err != nil {
		t.Fatalf("PutContexts failed: %v", err)
	}
	if len(cids) != len(contexts) {
		t.Errorf("PutContexts number of contexts mismatch, want: %v, got: %v", len(contexts), len(cids))
	}
	wcid := int64(cids[0])
	wantContext.Id = &wcid

	// query contexts by ids
	gotContexts, err := store.GetContextsByID(cids[0:1])
	if err != nil {
		t.Fatalf("GetContextsByID failed: %v", err)
	}
	if len(gotContexts) != 1 {
		t.Errorf("GetContextsByID cannot find result with id: %v", wcid)
	}
	// skip comparing create/update timestamps
	wantContext.CreateTimeSinceEpoch = gotContexts[0].CreateTimeSinceEpoch
	wantContext.LastUpdateTimeSinceEpoch = gotContexts[0].LastUpdateTimeSinceEpoch
	if !proto.Equal(wantContext, gotContexts[0]) {
		t.Errorf("GetContextsByID returned result is incorrect. want: %v, got: %v", wantContext, gotContexts[0])
	}

	// query all contexts
	gotStoredContexts, err := store.GetContexts()
	if err != nil {
		t.Fatalf("GetContexts failed: %v", err)
	}
	if len(gotStoredContexts) != len(contexts) {
		t.Errorf("GetContexts number of contexts mismatch, want: %v, got: %v", len(contexts), len(gotStoredContexts))
	}
	// skip comparing create/update timestamps
	wantContext.CreateTimeSinceEpoch = gotStoredContexts[0].CreateTimeSinceEpoch
	wantContext.LastUpdateTimeSinceEpoch = gotStoredContexts[0].LastUpdateTimeSinceEpoch
	if !proto.Equal(wantContext, gotStoredContexts[0]) {
		t.Errorf("GetContexts returned result is incorrect. want: %v, got: %v", wantContext, gotStoredContexts[0])
	}

	// query contexts of a particular type
	typeName := "test_type_name"
	gotContextsOfType, err := store.GetContextsByType(typeName)
	if err != nil {
		t.Fatalf("GetContextsByType failed: %v", err)
	}
	if len(gotContextsOfType) != len(contexts) {
		t.Errorf("GetContextsByType number of contexts mismatch, want: %v, got: %v", len(contexts), len(gotContextsOfType))
	}
	// skip comparing create/update timestamps
	wantContext.CreateTimeSinceEpoch = gotContextsOfType[0].CreateTimeSinceEpoch
	wantContext.LastUpdateTimeSinceEpoch = gotContextsOfType[0].LastUpdateTimeSinceEpoch
	if !proto.Equal(wantContext, gotContextsOfType[0]) {
		t.Errorf("GetContextsByType returned result is incorrect. want: %v, got: %v", wantContext, gotContextsOfType[0])
	}
}

func TestPutContextAndGetContextByTypeAndName(t *testing.T) {
	// prepare test data
	store, err := NewStore(fakeDatabaseConfig())
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()
	tid, err := insertContextType(store, `name: 'test_type_name'`)
	if err != nil {
		t.Fatalf("Cannot create context type: %v", err)
	}
	contextName := "test_context"
	contextTypeName := "test_type_name"
	wantContext := &mdpb.Context{
		TypeId: &tid,
		Name:   &contextName,
	}
	// insert 1 context
	contexts := make([]*mdpb.Context, 1)
	contexts[0] = wantContext
	cids, err := store.PutContexts(contexts)
	if err != nil {
		t.Fatalf("PutContexts failed: %v", err)
	}
	if len(cids) != len(contexts) {
		t.Errorf("PutContexts number of contexts mismatch, want: %v, got: %v", len(contexts), len(cids))
	}
	wcid := int64(cids[0])
	wantContext.Id = &wcid

	// test GetContextByTypeAndName functionality
	// query context by both type name and context name
	gotStoredContext, err := store.GetContextByTypeAndName(contextTypeName, contextName)
	if err != nil {
		t.Fatalf("GetContextByTypeAndName failed: %v", err)
	}
	// skip comparing create/update timestamps
	wantContext.CreateTimeSinceEpoch = gotStoredContext.CreateTimeSinceEpoch
	wantContext.LastUpdateTimeSinceEpoch = gotStoredContext.LastUpdateTimeSinceEpoch
	if !proto.Equal(wantContext, gotStoredContext) {
		t.Errorf("GetContextByTypeAndName returned result is incorrect. want: %v, got: %v", wantContext, gotStoredContext)
	}

	// query context with either contextTypeName or contextName that doesn't exist
	tests := []struct {
		cname  string
		ctname string
	}{
		{
			cname:  contextName,
			ctname: "random_type_name",
		},
		{
			cname:  "random_context_name",
			ctname: contextTypeName,
		},
		{
			cname:  "random_context_name",
			ctname: "random_type_name",
		},
	}

	for _, tc := range tests {
		gotEmptyContext, err := store.GetContextByTypeAndName(tc.ctname, tc.cname)
		if err != nil {
			t.Fatalf("GetContextByTypeAndName failed: %v", err)
		}
		if gotEmptyContext != nil {
			t.Errorf("GetContextByTypeAndName returned result is incorrect. want: %v, got: %v", nil, gotEmptyContext)
		}
	}
}

func TestPutAndGetEvents(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()
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

func TestPutExecutionWithoutContext(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()
	// create test types
	atid, err := insertArtifactType(store, `name: 'artifact_type_name' `)
	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}
	etid, err := insertExecutionType(store, `name: 'execution_type_name' `)
	if err != nil {
		t.Fatalf("Cannot create execution type: %v", err)
	}
	// create an stored input artifact ia
	as := make([]*mdpb.Artifact, 1)
	ia := &mdpb.Artifact{TypeId: &atid}
	as[0] = ia
	aids, err := store.PutArtifacts(as)
	if err != nil {
		t.Fatalf("PutArtifacts failed: %v", err)
	}
	aid := int64(aids[0])
	ia.Id = &aid
	// prepare an execution and an output artifact, and publish input and output together with events
	// input has no event update, output has a new event
	e := &mdpb.Execution{TypeId: &etid}
	aep := make([]*ArtifactAndEvent, 2)
	aep[0] = &ArtifactAndEvent{
		Artifact: ia,
	}
	oa := &mdpb.Artifact{TypeId: &atid}
	oet := mdpb.Event_Type(mdpb.Event_OUTPUT)
	ot := int64(100000)
	aep[1] = &ArtifactAndEvent{
		Artifact: oa,
		Event: &mdpb.Event{
			Type:                   &oet,
			MillisecondsSinceEpoch: &ot,
		},
	}
	// publish the execution and examine the results
	reid, raids, rcids, err := store.PutExecution(e, aep, nil)
	if err != nil {
		t.Fatalf("PutExecution failed: %v", err)
	}
	if len(rcids) != 0 {
		t.Fatalf("PutExecution number of contexts mismatch, want: %v, got: %v", 1, len(rcids))
	}
	if len(raids) != 2 {
		t.Errorf("PutExecution number of artifacts mismatch, want: %v, got: %v", 2, len(raids))
	}
	if raids[0] != ArtifactID(aid) {
		t.Errorf("PutExecution returned Id for stored Artifact mismatch, want: %v, got: %v", aid, raids[0])
	}
	// query execution that is just stored by returned eid
	eids := make([]ExecutionID, 1)
	eids[0] = reid
	gotEvents, err := store.GetEventsByExecutionIDs(eids)
	if err != nil {
		t.Fatalf("GetEventsByExecutionIDs failed: %v", err)
	}
	if len(gotEvents) != 1 {
		t.Fatalf("GetEventsByExecutionIDs number of events mismatch, want: %v, got: %v", 1, len(gotEvents))
	}
	wantEvent := aep[1].Event
	oaid := int64(raids[1])
	eid := int64(reid)
	wantEvent.ArtifactId = &oaid
	wantEvent.ExecutionId = &eid
	if !proto.Equal(gotEvents[0], wantEvent) {
		t.Errorf("GetEventsByExecutionIDs returned events mismatch, want: %v, got: %v", wantEvent, gotEvents[0])
	}
}

func TestPutExecutionWithContext(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()
	// create test types
	atid, err := insertArtifactType(store, `name: 'artifact_type_name' `)
	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}
	etid, err := insertExecutionType(store, `name: 'execution_type_name' `)
	if err != nil {
		t.Fatalf("Cannot create execution type: %v", err)
	}
	ctid, err := insertContextType(store, `name: 'context_type_name' `)
	if err != nil {
		t.Fatalf("Cannot create context type: %v", err)
	}
	// create an stored input artifact ia
	as := make([]*mdpb.Artifact, 1)
	ia := &mdpb.Artifact{TypeId: &atid}
	as[0] = ia
	aids, err := store.PutArtifacts(as)
	if err != nil {
		t.Fatalf("PutArtifacts failed: %v", err)
	}
	aid := int64(aids[0])
	ia.Id = &aid
	// prepare an execution and an output artifact, and publish input and output together with events
	// input has no event update, output has a new event
	e := &mdpb.Execution{TypeId: &etid}
	aep := make([]*ArtifactAndEvent, 1)
	aep[0] = &ArtifactAndEvent{
		Artifact: ia,
	}
	// prepare an context.
	cname := "context_name"
	c := &mdpb.Context{TypeId: &ctid, Name: &cname}
	ic := make([]*mdpb.Context, 1)
	ic[0] = c
	// publish the execution and examine the results
	reid, raids, rcids, err := store.PutExecution(e, aep, ic)
	if err != nil {
		t.Fatalf("PutExecution failed: %v", err)
	}
	if len(rcids) != 1 {
		t.Fatalf("PutExecution number of contexts mismatch, want: %v, got: %v", 1, len(rcids))
	}
	if len(raids) != 1 {
		t.Errorf("PutExecution number of artifacts mismatch, want: %v, got: %v", 2, len(raids))
	}
	if raids[0] != ArtifactID(aid) {
		t.Errorf("PutExecution returned Id for stored Artifact mismatch, want: %v, got: %v", aid, raids[0])
	}
	// test the attribution links between artifacts and the context are correct.
	rcid := int64(rcids[0])
	c.Id = &rcid
	gotContexts, err := store.GetContextsByArtifact(aids[0])
	if err != nil {
		t.Fatalf("GetContextsByArtifact failed: %v", err)
	}
	if len(gotContexts) != 1 {
		t.Errorf("GetContextsByArtifact returned number of results is incorrect. want: %v, got: %v", 1, len(gotContexts))
	}
	// skip comparing create/update timestamps
	c.CreateTimeSinceEpoch = gotContexts[0].CreateTimeSinceEpoch
	c.LastUpdateTimeSinceEpoch = gotContexts[0].LastUpdateTimeSinceEpoch
	if !proto.Equal(c, gotContexts[0]) {
		t.Errorf("GetContextsByArtifact returned result is incorrect. want: %v, got: %v", c, gotContexts[0])
	}
	// test the association link between the execution and the context is correct.
	gotContexts, err = store.GetContextsByExecution(reid)
	if err != nil {
		t.Fatalf("GetContextsByExecution failed: %v", err)
	}
	if len(gotContexts) != 1 {
		t.Errorf("GetContextsByExecution returned number of results is incorrect. want: %v, got: %v", 1, len(gotContexts))
	}
	// skip comparing create/update timestamps
	c.CreateTimeSinceEpoch = gotContexts[0].CreateTimeSinceEpoch
	c.LastUpdateTimeSinceEpoch = gotContexts[0].LastUpdateTimeSinceEpoch
	if !proto.Equal(c, gotContexts[0]) {
		t.Errorf("GetContextsByExecution returned result is incorrect. want: %v, got: %v", c, gotContexts[0])
	}
	eid := int64(reid)
	e.Id = &eid
	gotExecutions, err := store.GetExecutionsByContext(rcids[0])
	if err != nil {
		t.Fatalf("GetExecutionsByContext failed: %v", err)
	}
	if len(gotExecutions) != 1 {
		t.Errorf("GetExecutionsByContext returned number of results is incorrect. want: %v, got: %v", 1, len(gotContexts))
	}
	// skip comparing create/update timestamps
	e.CreateTimeSinceEpoch = gotExecutions[0].CreateTimeSinceEpoch
	e.LastUpdateTimeSinceEpoch = gotExecutions[0].LastUpdateTimeSinceEpoch
	if !proto.Equal(e, gotExecutions[0]) {
		t.Errorf("GetExecutionsByContext returned result is incorrect. want: %v, got: %v", e, gotExecutions[0])
	}
}

func insertContext(s *Store, ctid int64, textContext string) (*mdpb.Context, error) {
	c := &mdpb.Context{}
	if err := proto.UnmarshalText(textContext, c); err != nil {
		return nil, err
	}
	c.TypeId = &ctid
	contexts := make([]*mdpb.Context, 1)
	contexts[0] = c
	cids, err := s.PutContexts(contexts)
	if err != nil {
		return nil, err
	}
	cid := int64(cids[0])
	c.Id = &cid
	return c, nil
}

func insertExecution(s *Store, etid int64, textExecution string) (*mdpb.Execution, error) {
	e := &mdpb.Execution{}
	if err := proto.UnmarshalText(textExecution, e); err != nil {
		return nil, err
	}
	e.TypeId = &etid
	executions := make([]*mdpb.Execution, 1)
	executions[0] = e
	eids, err := s.PutExecutions(executions)
	if err != nil {
		return nil, err
	}
	eid := int64(eids[0])
	e.Id = &eid
	return e, nil
}

func insertArtifact(s *Store, atid int64, textArtifact string) (*mdpb.Artifact, error) {
	a := &mdpb.Artifact{}
	if err := proto.UnmarshalText(textArtifact, a); err != nil {
		return nil, err
	}
	a.TypeId = &atid
	artifacts := make([]*mdpb.Artifact, 1)
	artifacts[0] = a
	aids, err := s.PutArtifacts(artifacts)
	if err != nil {
		return nil, err
	}
	aid := int64(aids[0])
	a.Id = &aid
	return a, nil
}

func TestPutAndUseAttributionsAndAssociations(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()
	// prepare types
	ctid, err := insertContextType(store, `name: 'context_type_name'`)
	if err != nil {
		t.Fatalf("Cannot create context type: %v", err)
	}
	etid, err := insertExecutionType(store, `name: 'execution_type_name' `)
	if err != nil {
		t.Fatalf("Cannot create execution type: %v", err)
	}
	atid, err := insertArtifactType(store, `name: 'artifact_type_name' properties { key: 'p1' value: STRING } `)
	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}
	// prepare instances
	wantContext, err := insertContext(store, ctid, ` name: 'context' `)
	if err != nil {
		t.Fatalf("Cannot create context: %v", err)
	}
	wantExecution, err := insertExecution(store, etid, `custom_properties { key: 'p1' value: { int_value: 1 } }`)
	if err != nil {
		t.Fatalf("Cannot create execution: %v", err)
	}
	wantArtifact, err := insertArtifact(store, atid, ` uri: 'test uri' properties { key: 'p1' value: { string_value: 's' } }`)
	if err != nil {
		t.Fatalf("Cannot create execution: %v", err)
	}
	// insert attributions and associations
	attributions := make([]*mdpb.Attribution, 1)
	attributions[0] = &mdpb.Attribution{
		ArtifactId: wantArtifact.Id,
		ContextId:  wantContext.Id,
	}
	associations := make([]*mdpb.Association, 1)
	associations[0] = &mdpb.Association{
		ExecutionId: wantExecution.Id,
		ContextId:   wantContext.Id,
	}
	if err = store.PutAttributionsAndAssociations(attributions, associations); err != nil {
		t.Fatalf("PutAttributionsAndAssociations failed: %v", err)
	}

	// query contexts from artifact and execution
	gotContexts, err := store.GetContextsByArtifact(ArtifactID(*wantArtifact.Id))
	if err != nil {
		t.Fatalf("GetContextsByArtifact failed: %v", err)
	}
	if len(gotContexts) != 1 {
		t.Errorf("GetContextsByArtifact returned number of results is incorrect. want: %v, got: %v", 1, len(gotContexts))
	}
	// skip comparing create/update timestamps
	wantContext.CreateTimeSinceEpoch = gotContexts[0].CreateTimeSinceEpoch
	wantContext.LastUpdateTimeSinceEpoch = gotContexts[0].LastUpdateTimeSinceEpoch
	if !proto.Equal(wantContext, gotContexts[0]) {
		t.Errorf("GetContextsByArtifact returned result is incorrect. want: %v, got: %v", wantContext, gotContexts[0])
	}
	gotContexts, err = store.GetContextsByExecution(ExecutionID(*wantExecution.Id))
	if err != nil {
		t.Fatalf("GetContextsByExecution failed: %v", err)
	}
	if len(gotContexts) != 1 {
		t.Errorf("GetContextsByExecution returned number of results is incorrect. want: %v, got: %v", 1, len(gotContexts))
	}
	// skip comparing create/update timestamps
	wantContext.CreateTimeSinceEpoch = gotContexts[0].CreateTimeSinceEpoch
	wantContext.LastUpdateTimeSinceEpoch = gotContexts[0].LastUpdateTimeSinceEpoch
	if !proto.Equal(wantContext, gotContexts[0]) {
		t.Errorf("GetContextsByExecution returned result is incorrect. want: %v, got: %v", wantContext, gotContexts[0])
	}
	// query execution and artifact from context
	gotArtifacts, err := store.GetArtifactsByContext(ContextID(*wantContext.Id))
	if err != nil {
		t.Fatalf("GetArtifactsByContext failed: %v", err)
	}
	if len(gotArtifacts) != 1 {
		t.Errorf("GetArtifactsByContext returned number of results is incorrect. want: %v, got: %v", 1, len(gotArtifacts))
	}
	// skip comparing create/update timestamps
	wantArtifact.CreateTimeSinceEpoch = gotArtifacts[0].CreateTimeSinceEpoch
	wantArtifact.LastUpdateTimeSinceEpoch = gotArtifacts[0].LastUpdateTimeSinceEpoch
	if !proto.Equal(wantArtifact, gotArtifacts[0]) {
		t.Errorf("GetArtifactsByContext returned result is incorrect. want: %v, got: %v", wantArtifact, gotArtifacts[0])
	}
	gotExecutions, err := store.GetExecutionsByContext(ContextID(*wantContext.Id))
	if err != nil {
		t.Fatalf("GetExecutionsByContext failed: %v", err)
	}
	if len(gotExecutions) != 1 {
		t.Errorf("GetExecutionsByContext returned number of results is incorrect. want: %v, got: %v", 1, len(gotArtifacts))
	}
	// skip comparing create/update timestamps
	wantExecution.CreateTimeSinceEpoch = gotExecutions[0].CreateTimeSinceEpoch
	wantExecution.LastUpdateTimeSinceEpoch = gotExecutions[0].LastUpdateTimeSinceEpoch
	if !proto.Equal(wantExecution, gotExecutions[0]) {
		t.Errorf("GetExecutionsByContext returned result is incorrect. want: %v, got: %v", wantExecution, gotExecutions[0])
	}
}

func TestPutDuplicatedAttributionsAndEmptyAssociations(t *testing.T) {
	store, err := NewStore(fakeDatabaseConfig())
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}
	defer store.Close()
	ctid, err := insertContextType(store, `name: 'context_type_name'`)
	if err != nil {
		t.Fatalf("Cannot create context type: %v", err)
	}
	atid, err := insertArtifactType(store, `name: 'artifact_type_name' properties { key: 'p1' value: STRING } `)
	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}
	wantContext, err := insertContext(store, ctid, ` name: 'context' `)
	if err != nil {
		t.Fatalf("Cannot create context: %v", err)
	}
	wantArtifact, err := insertArtifact(store, atid, ` uri: 'test uri' properties { key: 'p1' value: { string_value: 's' } }`)
	if err != nil {
		t.Fatalf("Cannot create execution: %v", err)
	}
	// insert duplicated attributions and no associations
	attributions := make([]*mdpb.Attribution, 2)
	attributions[0] = &mdpb.Attribution{
		ArtifactId: wantArtifact.Id,
		ContextId:  wantContext.Id,
	}
	attributions[1] = &mdpb.Attribution{
		ArtifactId: wantArtifact.Id,
		ContextId:  wantContext.Id,
	}
	if err = store.PutAttributionsAndAssociations(attributions, nil); err != nil {
		t.Fatalf("PutAttributionsAndAssociations failed: %v", err)
	}
	// query contexts and artifacts
	gotContexts, err := store.GetContextsByArtifact(ArtifactID(*wantArtifact.Id))
	if err != nil {
		t.Fatalf("GetContextsByArtifact failed: %v", err)
	}
	if len(gotContexts) != 1 {
		t.Errorf("GetContextsByArtifact returned number of results is incorrect. want: %v, got: %v", 1, len(gotContexts))
	}
	// skip comparing create/update timestamps
	wantContext.CreateTimeSinceEpoch = gotContexts[0].CreateTimeSinceEpoch
	wantContext.LastUpdateTimeSinceEpoch = gotContexts[0].LastUpdateTimeSinceEpoch
	if !proto.Equal(wantContext, gotContexts[0]) {
		t.Errorf("GetContextsByArtifact returned result is incorrect. want: %v, got: %v", wantContext, gotContexts[0])
	}
	// query execution and artifact from context
	gotArtifacts, err := store.GetArtifactsByContext(ContextID(*wantContext.Id))
	if err != nil {
		t.Fatalf("GetArtifactsByContext failed: %v", err)
	}
	if len(gotArtifacts) != 1 {
		t.Errorf("GetArtifactsByContext returned number of results is incorrect. want: %v, got: %v", 1, len(gotArtifacts))
	}
	// skip comparing create/update timestamps
	wantArtifact.CreateTimeSinceEpoch = gotArtifacts[0].CreateTimeSinceEpoch
	wantArtifact.LastUpdateTimeSinceEpoch = gotArtifacts[0].LastUpdateTimeSinceEpoch
	if !proto.Equal(wantArtifact, gotArtifacts[0]) {
		t.Errorf("GetArtifactsByContext returned result is incorrect. want: %v, got: %v", wantArtifact, gotArtifacts[0])
	}
}
