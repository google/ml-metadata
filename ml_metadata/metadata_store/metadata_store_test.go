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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"
	mdpb "ml_metadata/proto/metadata_store_go_proto"
)

var artifactCmpOpts = []cmp.Option{
	protocmp.Transform(),
	protocmp.IgnoreFields(&mdpb.Artifact{}, "create_time_since_epoch", "last_update_time_since_epoch"),
}

var executionCmpOpts = []cmp.Option{
	protocmp.Transform(),
	protocmp.IgnoreFields(&mdpb.Execution{}, "create_time_since_epoch", "last_update_time_since_epoch"),
}

var contextCmpOpts = []cmp.Option{
	protocmp.Transform(),
	protocmp.IgnoreFields(&mdpb.Context{}, "create_time_since_epoch", "last_update_time_since_epoch"),
}

// createStore creates a store and returns it if there is no error.
// It also takes care of closing it automatically.
func createStore(t *testing.T) *Store {
	t.Helper()

	store, err := NewStore(&mdpb.ConnectionConfig{
		Config: &mdpb.ConnectionConfig_FakeDatabase{FakeDatabase: &mdpb.FakeDatabaseConfig{}},
	})
	if err != nil {
		t.Fatalf("Cannot create Store: %v", err)
	}

	t.Cleanup(func() { store.Close() })
	return store
}

func TestPutArtifactType(t *testing.T) {
	store := createStore(t)

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
		t.Errorf("Given the same type name, type IDs should be the same. store.PutArtifactType(%v, %v) = %v, want %v", aType, opts, typeID2, typeID)
	}

	newTypeName := "another_type_name"
	newType := &mdpb.ArtifactType{Name: &newTypeName}
	typeID3, err := store.PutArtifactType(newType, opts)
	if err != nil {
		t.Fatalf("PutArtifactType failed: %v", err)
	}
	if typeID3 == typeID {
		t.Errorf("Given different type name, type IDs should be different. store.PutArtifactType(%v, %v) = %v, want %v", newType, opts, typeID3, typeID)
	}
}

func TestPutAndUpdateArtifactType(t *testing.T) {
	store := createStore(t)

	aType := &mdpb.ArtifactType{
		Name:       proto.String("test_type_name"),
		Properties: map[string]mdpb.PropertyType{"p1": mdpb.PropertyType_INT},
	}
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := store.PutArtifactType(aType, opts)
	if err != nil {
		t.Fatalf("PutArtifactType failed: %v", err)
	}
	if typeID <= 0 {
		t.Errorf("expected type ID should be positive, got %v", typeID)
	}

	wantType := &mdpb.ArtifactType{
		Name:       proto.String("test_type_name"),
		Properties: map[string]mdpb.PropertyType{"p1": mdpb.PropertyType_INT, "p2": mdpb.PropertyType_DOUBLE},
	}

	opts = &PutTypeOptions{AllFieldsMustMatch: true, CanAddFields: true}
	typeID2, err := store.PutArtifactType(wantType, opts)
	if err != nil {
		t.Fatalf("PutArtifactType failed: %v", err)
	}
	if typeID2 != typeID {
		t.Errorf("Update the type, type IDs should be the same. store.PutArtifactType(%v, %v) = %v, want: %v", wantType, opts, typeID2, typeID)
	}

	tids := []ArtifactTypeID{typeID}
	gotTypesByID, err := store.GetArtifactTypesByID(tids)
	if err != nil {
		t.Fatalf("GetArtifactTypesByID failed: %v", err)
	}

	wantType.Id = proto.Int64(int64(typeID))
	wantTypes := []*mdpb.ArtifactType{wantType}
	if !cmp.Equal(wantTypes, gotTypesByID, cmp.Comparer(proto.Equal)) {
		t.Errorf("Put and get type by id mismatch. store.GetArtifactTypesByID(%v) = %v, want: %v", tids, gotTypesByID, wantTypes)
	}
}

func TestGetArtifactType(t *testing.T) {
	store := createStore(t)

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
	tids := []ArtifactTypeID{typeID}
	gotTypesByID, err := store.GetArtifactTypesByID(tids)
	if err != nil {
		t.Fatalf("GetArtifactTypesByID failed: %v", err)
	}
	if len(gotTypesByID) < 1 || !proto.Equal(wantType, gotTypesByID[0]) {
		t.Errorf("put and get type by id mismatch, want: %v, got: %v", wantType, gotTypesByID)
	}
}

func TestGetArtifactTypes(t *testing.T) {
	store := createStore(t)

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
	store := createStore(t)

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
	store := createStore(t)

	wantType := &mdpb.ExecutionType{
		Name:       proto.String("test_type_name"),
		Properties: map[string]mdpb.PropertyType{"p1": mdpb.PropertyType_INT},
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

	tids := []ExecutionTypeID{typeID}
	gotTypesByID, err := store.GetExecutionTypesByID(tids)
	if err != nil {
		t.Fatalf("GetExecutionTypesByID failed: %v", err)
	}
	if len(gotTypesByID) < 1 || !proto.Equal(wantType, gotTypesByID[0]) {
		t.Errorf("put and get type by id mismatch, want: %v, got: %v", wantType, gotTypesByID)
	}
}

func TestPutAndUpdateExecutionType(t *testing.T) {
	store := createStore(t)

	eType := &mdpb.ExecutionType{
		Name:       proto.String("test_type_name"),
		Properties: map[string]mdpb.PropertyType{"p1": mdpb.PropertyType_INT},
	}

	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := store.PutExecutionType(eType, opts)
	if err != nil {
		t.Fatalf("PutExecutionType failed: %v", err)
	}
	if typeID <= 0 {
		t.Errorf("expected type ID should be positive, got %v", typeID)
	}

	wantType := &mdpb.ExecutionType{
		Name:       proto.String("test_type_name"),
		Properties: map[string]mdpb.PropertyType{"p1": mdpb.PropertyType_INT, "p2": mdpb.PropertyType_DOUBLE},
	}

	opts = &PutTypeOptions{AllFieldsMustMatch: true, CanAddFields: true}
	typeID2, err := store.PutExecutionType(wantType, opts)
	if err != nil {
		t.Fatalf("PutExecutionType failed: %v", err)
	}
	if typeID2 != typeID {
		t.Errorf("Update the type, type IDs should be the same. store.PutExecutionType(%v, %v) = %v, want: %v", wantType, opts, typeID2, typeID)
	}

	tids := []ExecutionTypeID{typeID}
	gotTypesByID, err := store.GetExecutionTypesByID(tids)
	if err != nil {
		t.Fatalf("GetExecutionTypesByID failed: %v", err)
	}

	wantType.Id = proto.Int64(int64(typeID))
	wantTypes := []*mdpb.ExecutionType{wantType}
	if !cmp.Equal(wantTypes, gotTypesByID, cmp.Comparer(proto.Equal)) {
		t.Errorf("Put and get type by id mismatch, store.GetExecutionTypesByID(%v) = %v, want: %v", tids, gotTypesByID, wantTypes)
	}
}

func TestGetContextTypes(t *testing.T) {
	store := createStore(t)

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
	store := createStore(t)

	wantType := &mdpb.ContextType{
		Name:       proto.String("test_type_name"),
		Properties: map[string]mdpb.PropertyType{"p1": mdpb.PropertyType_INT},
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

	tids := []ContextTypeID{typeID}
	gotTypesByID, err := store.GetContextTypesByID(tids)
	if err != nil {
		t.Fatalf("GetContextTypesByID failed: %v", err)
	}
	if len(gotTypesByID) < 1 || !proto.Equal(wantType, gotTypesByID[0]) {
		t.Errorf("put and get type by id mismatch, want: %v, got: %v", wantType, gotTypesByID)
	}
}

func TestPutAndUpdateContextType(t *testing.T) {
	store := createStore(t)

	cType := &mdpb.ContextType{
		Name:       proto.String("test_type_name"),
		Properties: map[string]mdpb.PropertyType{"p1": mdpb.PropertyType_INT},
	}

	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := store.PutContextType(cType, opts)
	if err != nil {
		t.Fatalf("PutContextType failed: %v", err)
	}

	wantType := &mdpb.ContextType{
		Name:       proto.String("test_type_name"),
		Properties: map[string]mdpb.PropertyType{"p1": mdpb.PropertyType_INT, "p2": mdpb.PropertyType_DOUBLE},
	}

	opts = &PutTypeOptions{AllFieldsMustMatch: true, CanAddFields: true}
	typeID2, err := store.PutContextType(wantType, opts)
	if err != nil {
		t.Fatalf("PutContextType failed: %v", err)
	}
	if typeID2 != typeID {
		t.Errorf("Update the type, type IDs should be the same. store.PutContextType(%v, %v) = %v, want: %v", wantType, opts, typeID2, typeID)
	}

	tids := []ContextTypeID{typeID}
	gotTypesByID, err := store.GetContextTypesByID(tids)
	if err != nil {
		t.Fatalf("GetContextTypesByID failed: %v", err)
	}

	wantType.Id = proto.Int64(int64(typeID))
	wantTypes := []*mdpb.ContextType{wantType}
	if !cmp.Equal(wantTypes, gotTypesByID, cmp.Comparer(proto.Equal)) {
		t.Errorf("Put and get type by id mismatch. store.GetContextTypesByID(%v) = %v, want: %v", tids, gotTypesByID, wantTypes)
	}
}

func insertArtifactType(s *Store, rst *mdpb.ArtifactType) (int64, error) {
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	tid, err := s.PutArtifactType(rst, opts)
	if err != nil {
		return 0, err
	}
	return int64(tid), nil
}

// putAndGetArtifactTypeID makes an Artifact Type and returns its ID.
func putAndGetArtifactTypeID(t *testing.T, store *Store, artifactTypeName string) int64 {
	t.Helper()

	typeID, err := insertArtifactType(store, &mdpb.ArtifactType{
		Name:       proto.String(artifactTypeName),
		Properties: map[string]mdpb.PropertyType{"p1": mdpb.PropertyType_INT},
	})

	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}
	return int64(typeID)
}

// makeArtifact makes an Artifact given its uri, typeID and custom properties.
func makeArtifact(uri string, typeID int64, customProperties map[string]*mdpb.Value) *mdpb.Artifact {
	return &mdpb.Artifact{
		TypeId:                   proto.Int64(typeID),
		Uri:                      proto.String(uri),
		CreateTimeSinceEpoch:     proto.Int64(1),
		LastUpdateTimeSinceEpoch: proto.Int64(2),
		CustomProperties:         customProperties,
	}
}

// makeArtifactsData makes data for 2 Artifacts (does not create them).
func makeArtifactsData(typeID int64) []*mdpb.Artifact {
	return []*mdpb.Artifact{
		makeArtifact("test_uri1", typeID,
			map[string]*mdpb.Value{`p1`: {Value: &mdpb.Value_StringValue{StringValue: `val`}}}),
		makeArtifact("test_uri2", typeID,
			map[string]*mdpb.Value{`p1`: {Value: &mdpb.Value_IntValue{IntValue: 1}}}),
	}
}

func insertArtifacts(t *testing.T, store *Store, typeID int64, artifacts []*mdpb.Artifact) []ArtifactID {
	t.Helper()

	artifactIDs, err := store.PutArtifacts(artifacts)
	if err != nil {
		t.Fatalf("PutArtifacts failed: %v", err)
	}

	if len(artifactIDs) != len(artifacts) {
		t.Errorf("PutArtifacts number of artifacts mismatch, want: %v, got: %v", len(artifacts), len(artifactIDs))
	}
	if len(artifactIDs) > 1 && artifactIDs[0] == artifactIDs[1] {
		t.Errorf("PutArtifacts should not return two identical id, id1: %v, id2: %v", artifactIDs[0], artifactIDs[1])
	}
	return artifactIDs
}

func TestPutAndGetArtifactsByID(t *testing.T) {
	store := createStore(t)

	typeID := putAndGetArtifactTypeID(t, store, "test_type_name")

	artifacts := makeArtifactsData(typeID)
	artifactIDs := insertArtifacts(t, store, typeID, artifacts)

	wantArtifacts := artifacts[0:1]
	wantArtifacts[0].Id = proto.Int64(int64(artifactIDs[0]))
	wantArtifacts[0].Type = proto.String("test_type_name")

	gotArtifacts, err := store.GetArtifactsByID(artifactIDs[0:1])
	if err != nil {
		t.Fatalf("GetArtifactsByID failed: %v", err)
	}

	if !cmp.Equal(wantArtifacts, gotArtifacts, artifactCmpOpts...) {
		t.Errorf("store.GetArtifactsByID(%v) = %v, want: %v", artifactIDs[0:1], gotArtifacts, wantArtifacts)
	}
}

func TestGetAllArtifacts(t *testing.T) {
	store := createStore(t)

	typeID := putAndGetArtifactTypeID(t, store, "test_type_name")
	artifacts := makeArtifactsData(typeID)
	insertArtifacts(t, store, typeID, artifacts)

	gotStoredArtifacts, err := store.GetArtifacts()
	if err != nil {
		t.Fatalf("GetArtifacts failed: %v", err)
	}
	if len(gotStoredArtifacts) != len(artifacts) {
		t.Errorf("GetArtifacts number of artifacts mismatch, got: %v, want: %v", len(gotStoredArtifacts), len(artifacts))
	}
	if proto.Equal(gotStoredArtifacts[0], gotStoredArtifacts[1]) {
		t.Errorf("GetArtifacts returns duplicated artifacts: %v. want: %v, %v", gotStoredArtifacts[0], artifacts[0], artifacts[1])
	}
}

func TestGetArtifactsByURI(t *testing.T) {
	store := createStore(t)

	typeID := putAndGetArtifactTypeID(t, store, "test_type_name")

	uri1 := "test_uri1"
	artifacts := []*mdpb.Artifact{
		makeArtifact(uri1, typeID,
			map[string]*mdpb.Value{`p1`: {Value: &mdpb.Value_StringValue{StringValue: `val`}}}),
	}
	artifactIDs := insertArtifacts(t, store, typeID, artifacts)

	wantArtifacts := artifacts[0:1]
	wantArtifacts[0].Id = proto.Int64(int64(artifactIDs[0]))
	wantArtifacts[0].Type = proto.String("test_type_name")

	gotArtifactsOfURI, err := store.GetArtifactsByURI(uri1)
	if err != nil {
		t.Fatalf("GetArtifactsByURI failed: %v", err)
	}

	if !cmp.Equal(wantArtifacts, gotArtifactsOfURI, artifactCmpOpts...) {
		t.Errorf("store.GetArtifactsByURI(%v) = %v, want: %v", uri1, gotArtifactsOfURI, wantArtifacts)
	}

	unknownURI := "unknown_uri"
	gotArtifactsOfUnknownURI, err := store.GetArtifactsByURI(unknownURI)
	if err != nil {
		t.Fatalf("GetArtifactsByURI failed: %v", err)
	}
	if len(gotArtifactsOfUnknownURI) != 0 {
		t.Errorf("GetArtifactsByURI number of artifacts mismatch, got: %v, want: 0", len(gotArtifactsOfUnknownURI))
	}
}

func TestGetArtifactsByType(t *testing.T) {
	store := createStore(t)

	typeName := "test_type_name"
	typeID := putAndGetArtifactTypeID(t, store, typeName)

	artifacts := makeArtifactsData(typeID)
	insertArtifacts(t, store, typeID, artifacts)

	gotArtifactsOfType, err := store.GetArtifactsByType(typeName)
	if err != nil {
		t.Fatalf("GetArtifactsByType failed: %v", err)
	}
	if len(gotArtifactsOfType) != len(artifacts) {
		t.Errorf("GetArtifactsByType number of artifacts mismatch, got: %v, want: %v", len(gotArtifactsOfType), len(artifacts))
	}
	if proto.Equal(gotArtifactsOfType[0], gotArtifactsOfType[1]) {
		t.Errorf("GetArtifactsByType returns duplicated artifacts: %v. want: %v, %v", gotArtifactsOfType[0], artifacts[0], artifacts[1])
	}
}

func TestGetArtifactsByNonExistentType(t *testing.T) {
	store := createStore(t)

	typeID := putAndGetArtifactTypeID(t, store, "test_type_name")

	artifacts := makeArtifactsData(typeID)
	insertArtifacts(t, store, typeID, artifacts)

	// Query artifacts of a non-exist type.
	notExistTypeName := "not_exist_type_name"
	gotArtifactsOfNotExistType, err := store.GetArtifactsByType(notExistTypeName)
	if err != nil {
		t.Fatalf("GetArtifactsByType failed: %v", err)
	}
	if len(gotArtifactsOfNotExistType) != 0 {
		t.Errorf("GetArtifactsByType number of artifacts mismatch of non-exist type, got: %v, want: 0", len(gotArtifactsOfNotExistType))
	}
}

func TestGetArtifactsByEmptyType(t *testing.T) {
	store := createStore(t)

	typeID := putAndGetArtifactTypeID(t, store, "test_type_name")

	artifacts := makeArtifactsData(typeID)
	insertArtifacts(t, store, typeID, artifacts)

	typeNameNoArtifacts := "test_type_name_no_artifacts"
	putAndGetArtifactTypeID(t, store, typeNameNoArtifacts)

	gotEmptyTypeArtifacts, err := store.GetArtifactsByType(typeNameNoArtifacts)
	if err != nil {
		t.Fatalf("GetArtifactsByType failed: %v", err)
	}
	if len(gotEmptyTypeArtifacts) != 0 {
		t.Errorf("GetArtifactsByType number of artifacts mismatch of an empty type, got: %v, want: 0", len(gotEmptyTypeArtifacts))
	}
}

func TestGetArtifactByTypeAndName(t *testing.T) {
	store := createStore(t)

	artifactName := "test_artifact"
	artifactTypeName := "test_type_name"
	typeID := putAndGetArtifactTypeID(t, store, artifactTypeName)

	uri := "/test/uri"
	wantArtifact := &mdpb.Artifact{
		TypeId: &typeID,
		Name:   &artifactName,
		Uri:    &uri,
	}

	// Insert 1 artifact.
	artifacts := []*mdpb.Artifact{wantArtifact}
	artifactIDs, err := store.PutArtifacts(artifacts)
	if err != nil {
		t.Fatalf("PutArtifacts failed: %v", err)
	}
	if len(artifactIDs) != len(artifacts) {
		t.Errorf("PutArtifacts number of artifacts mismatch, got: %v, want: %v", len(artifactIDs), len(artifacts))
	}

	// Test GetArtifactByTypeAndName functionality - query
	// artifact by both type name and artifact name.
	gotStoredArtifact, err := store.GetArtifactByTypeAndName(artifactTypeName, artifactName)
	if err != nil {
		t.Fatalf("GetArtifactByTypeAndName failed: %v", err)
	}

	waid := int64(artifactIDs[0])
	wantArtifact.Id = &waid
	wantArtifact.Type = &artifactTypeName

	if !cmp.Equal(wantArtifact, gotStoredArtifact, artifactCmpOpts...) {
		t.Errorf("store.GetArtifactByTypeAndName(%v, %v) = %v, want: %v", artifactTypeName, artifactName, gotStoredArtifact, wantArtifact)
	}

	// Query artifact with either artifactTypeName or artifactName that doesn't exist.
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
			t.Errorf("store.GetArtifactByTypeAndName(%v, %v) = %v, want: %v", tc.atname, tc.aname, gotEmptyArtifact, nil)
		}
	}
}

func insertExecutionType(s *Store, rst *mdpb.ExecutionType) (int64, error) {
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := s.PutExecutionType(rst, opts)
	if err != nil {
		return 0, err
	}
	return int64(typeID), nil
}

// putAndGetExecutionTypeID makes an execution type and returns it.
func putAndGetExecutionTypeID(t *testing.T, store *Store, exectionTypeName string) int64 {
	t.Helper()

	typeID, err := insertExecutionType(store, &mdpb.ExecutionType{
		Name:       proto.String(exectionTypeName),
		Properties: map[string]mdpb.PropertyType{"p1": mdpb.PropertyType_DOUBLE},
	})

	if err != nil {
		t.Fatalf("Cannot create execution type: %v", err)
	}

	return typeID
}

// makeAndInsertExecution inserts an Execution and sets up the environment for the Execution getter tests.
func makeAndInsertExecution(t *testing.T, store *Store, typeID int64, executionName string) *mdpb.Execution {
	t.Helper()

	wantExecution := &mdpb.Execution{
		TypeId: proto.Int64(typeID),
		Name:   proto.String(executionName),
		Properties: map[string]*mdpb.Value{
			`p1`: {Value: &mdpb.Value_DoubleValue{DoubleValue: 1.0}},
		},
		CustomProperties: map[string]*mdpb.Value{
			`p1`: {Value: &mdpb.Value_IntValue{IntValue: 1}},
		},
	}

	// Insert 1 execution.
	executions := []*mdpb.Execution{wantExecution}
	executionIDs, err := store.PutExecutions(executions)

	if err != nil {
		t.Fatalf("PutExecutions failed: %v", err)
	}
	if len(executionIDs) != len(executions) {
		t.Errorf("PutExecutions number of executions mismatch, got: %v, want: %v", len(executionIDs), len(executions))
	}

	wantExecution.Id = proto.Int64(int64(executionIDs[0]))
	wantExecution.Type = proto.String("test_type_name")
	return wantExecution
}

func TestPutAndGetExecutionsByID(t *testing.T) {
	store := createStore(t)

	typeID := putAndGetExecutionTypeID(t, store, "test_type_name")
	wantExecution := makeAndInsertExecution(t, store, typeID, "test_execution")
	executionIDs := []ExecutionID{ExecutionID(*wantExecution.Id)}

	gotExecutions, err := store.GetExecutionsByID(executionIDs)
	if err != nil {
		t.Fatalf("GetExecutionsByID failed: %v", err)
	}
	wantExecutions := []*mdpb.Execution{wantExecution}
	if !cmp.Equal(wantExecutions, gotExecutions, executionCmpOpts...) {
		t.Errorf("store.GetExecutionsByID(%v) = %v, want: %v", executionIDs, gotExecutions, wantExecutions)
	}
}

func TestGetAllExecutions(t *testing.T) {
	store := createStore(t)

	typeID := putAndGetExecutionTypeID(t, store, "test_type_name")
	wantExecution := makeAndInsertExecution(t, store, typeID, "test_execution")

	gotStoredExecutions, err := store.GetExecutions()
	if err != nil {
		t.Fatalf("GetExecutions failed: %v", err)
	}

	wantExecutions := []*mdpb.Execution{wantExecution}

	if !cmp.Equal(wantExecutions, gotStoredExecutions, executionCmpOpts...) {
		t.Errorf("store.GetExecutions() = %v, want: %v", gotStoredExecutions, wantExecutions)
	}
}

func TestGetExecutionsByType(t *testing.T) {
	store := createStore(t)

	typeName := "test_type_name"
	typeID := putAndGetExecutionTypeID(t, store, typeName)
	wantExecution := makeAndInsertExecution(t, store, typeID, "test_execution")

	gotExecutionsOfType, err := store.GetExecutionsByType(typeName)
	if err != nil {
		t.Fatalf("GetExecutionsByType failed: %v", err)
	}

	wantExecutions := []*mdpb.Execution{wantExecution}

	if !cmp.Equal(wantExecutions, gotExecutionsOfType, executionCmpOpts...) {
		t.Errorf("store.GetExecutionsByType(%v) = %v, want: %v", typeName, gotExecutionsOfType, wantExecutions)
	}
}

func TestGetExecutionsByNonExistentType(t *testing.T) {
	store := createStore(t)

	// Query executions of a non-existent type.
	notExistTypeName := "not_exist_type_name"
	gotExecutionsOfNotExistType, err := store.GetExecutionsByType(notExistTypeName)
	if err != nil {
		t.Fatalf("GetExecutionsByType failed: %v", err)
	}
	if len(gotExecutionsOfNotExistType) != 0 {
		t.Errorf("GetExecutionsByType number of executions mismatch of non-exist type, got: %v, want: 0", len(gotExecutionsOfNotExistType))
	}
}

func TestGetExecutionsByEmptyType(t *testing.T) {
	store := createStore(t)

	// Test querying executions of an empty type having no execution.
	typeNameNoExecutions := "test_type_name_no_execution"
	putAndGetExecutionTypeID(t, store, typeNameNoExecutions)

	gotEmptyTypeExecutions, err := store.GetExecutionsByType(typeNameNoExecutions)
	if err != nil {
		t.Fatalf("GetExecutionsByType failed: %v", err)
	}
	if len(gotEmptyTypeExecutions) != 0 {
		t.Errorf("GetExecutionsByType number of artifacts mismatch of an empty type, got: %v, want: 0", len(gotEmptyTypeExecutions))
	}
}

func TestPutAndGetExecutionByTypeAndName(t *testing.T) {
	store := createStore(t)

	typeID := putAndGetExecutionTypeID(t, store, "test_type_name")

	executionName := "test_execution"
	executionTypeName := "test_type_name"
	wantExecution := makeAndInsertExecution(t, store, typeID, executionName)

	// Test GetExecutionByTypeAndName functionality
	// query execution by both type name and execution name.
	gotStoredExecution, err := store.GetExecutionByTypeAndName(executionTypeName, executionName)
	if err != nil {
		t.Fatalf("GetExecutionByTypeAndName failed: %v", err)
	}

	if !cmp.Equal(wantExecution, gotStoredExecution, executionCmpOpts...) {
		t.Errorf("store.GetExecutionByTypeAndName(%v, %v) = %v, want: %v", executionTypeName, executionName, gotStoredExecution, wantExecution)
	}

	// Query execution with either executionTypeName or executionName that doesn't exist.
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

func insertContextType(s *Store, rst *mdpb.ContextType) (int64, error) {
	opts := &PutTypeOptions{AllFieldsMustMatch: true}
	typeID, err := s.PutContextType(rst, opts)
	if err != nil {
		return 0, err
	}
	return int64(typeID), nil
}

// putAndGetContextTypeID makes and returns a Context Type ID.
func putAndGetContextTypeID(t *testing.T, store *Store, contextTypeName string) int64 {
	t.Helper()

	typeID, err := insertContextType(store, &mdpb.ContextType{
		Name:       proto.String(contextTypeName),
		Properties: map[string]mdpb.PropertyType{"p1": mdpb.PropertyType_STRING},
	})

	if err != nil {
		t.Fatalf("Cannot create context type: %v", err)
	}

	return typeID
}

func makeAndInsertContext(t *testing.T, store *Store, typeID int64, contextName string) *mdpb.Context {
	t.Helper()

	wantContext := &mdpb.Context{
		TypeId: proto.Int64(typeID),
		Name:   proto.String(contextName),
		Properties: map[string]*mdpb.Value{
			`p1`: {Value: &mdpb.Value_StringValue{StringValue: `val`}},
		},
		CustomProperties: map[string]*mdpb.Value{
			`p1`: {Value: &mdpb.Value_IntValue{IntValue: 1}},
		},
	}

	// Insert 1 context.
	contexts := []*mdpb.Context{wantContext}
	contextIDs, err := store.PutContexts(contexts)

	if err != nil {
		t.Fatalf("PutContexts failed: %v", err)
	}
	if len(contextIDs) != len(contexts) {
		t.Errorf("PutContexts number of contexts mismatch, got: %v, want: %v", len(contextIDs), len(contexts))
	}

	wantedContextID := int64(contextIDs[0])
	wantContext.Id = &wantedContextID
	wantContext.Type = proto.String("test_type_name")

	return wantContext
}

func TestPutAndGetContextsByIds(t *testing.T) {
	store := createStore(t)

	typeID := putAndGetContextTypeID(t, store, "test_type_name")
	wantContext := makeAndInsertContext(t, store, typeID, "test_context")
	contextIDs := []ContextID{ContextID(wantContext.GetId())}

	// Query contexts by ids.
	gotContexts, err := store.GetContextsByID(contextIDs)
	if err != nil {
		t.Fatalf("GetContextsByID failed: %v", err)
	}

	wantContexts := []*mdpb.Context{wantContext}
	if !cmp.Equal(wantContexts, gotContexts, contextCmpOpts...) {
		t.Errorf("store.GetContextsByID(%v) = %v, want: %v", contextIDs, gotContexts, wantContexts)
	}
}

func TestGetAllContexts(t *testing.T) {
	store := createStore(t)

	typeID := putAndGetContextTypeID(t, store, "test_type_name")
	wantContext := makeAndInsertContext(t, store, typeID, "test_context")

	// Query all contexts.
	gotStoredContexts, err := store.GetContexts()
	if err != nil {
		t.Fatalf("GetContexts failed: %v", err)
	}

	wantContexts := []*mdpb.Context{wantContext}
	if !cmp.Equal(wantContexts, gotStoredContexts, contextCmpOpts...) {
		t.Errorf("store.GetContexts() = %v, want: %v", gotStoredContexts, wantContexts)
	}
}

func TestGetContextsByType(t *testing.T) {
	store := createStore(t)

	typeName := "test_type_name"
	typeID := putAndGetContextTypeID(t, store, typeName)
	wantContext := makeAndInsertContext(t, store, typeID, "test_context")

	// Query contexts of a particular type.
	gotContextsOfType, err := store.GetContextsByType(typeName)
	if err != nil {
		t.Fatalf("GetContextsByType failed: %v", err)
	}

	wantContexts := []*mdpb.Context{wantContext}

	if !cmp.Equal(wantContexts, gotContextsOfType, contextCmpOpts...) {
		t.Errorf("store.GetContextsByType(%v) = %v, want: %v", typeName, gotContextsOfType, wantContexts)
	}
}

func TestGetContextByTypeAndName(t *testing.T) {
	store := createStore(t)

	contextName := "test_context"
	contextTypeName := "test_type_name"
	typeID := putAndGetContextTypeID(t, store, contextTypeName)

	wantContext := makeAndInsertContext(t, store, typeID, contextName)

	// Test GetContextByTypeAndName functionality query
	// context by both type name and context name.
	gotStoredContext, err := store.GetContextByTypeAndName(contextTypeName, contextName)
	if err != nil {
		t.Fatalf("GetContextByTypeAndName failed: %v", err)
	}

	if !cmp.Equal(wantContext, gotStoredContext, contextCmpOpts...) {
		t.Errorf("store.GetContextByTypeAndName(%v, %v) = %v, want: %v", contextTypeName, contextName, gotStoredContext, wantContext)
	}

	// Query context with either contextTypeName or contextName that doesn't exist.
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
			t.Errorf("store.GetContextByTypeAndName(%v, %v) = %v, want: %v", tc.ctname, tc.cname, gotEmptyContext, nil)
		}
	}
}

func TestPutAndGetEvents(t *testing.T) {
	store := createStore(t)
	atid, err := insertArtifactType(store, &mdpb.ArtifactType{
		Name: proto.String("artifact_type_name"),
	})
	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}
	as := []*mdpb.Artifact{
		&mdpb.Artifact{TypeId: &atid},
		&mdpb.Artifact{TypeId: &atid},
	}

	aids, err := store.PutArtifacts(as)
	if err != nil {
		t.Fatalf("PutArtifacts failed: %v", err)
	}
	a1id, a2id := int64(aids[0]), int64(aids[1])

	etid, err := insertExecutionType(store, &mdpb.ExecutionType{
		Name: proto.String("execution_type_name"),
	})
	if err != nil {
		t.Fatalf("Cannot create execution type: %v", err)
	}
	es := []*mdpb.Execution{
		&mdpb.Execution{TypeId: &etid},
		&mdpb.Execution{TypeId: &etid},
	}

	eids, err := store.PutExecutions(es)
	if err != nil {
		t.Fatalf("PutExecutions failed: %v", err)
	}
	e1id := int64(eids[0])

	// insert events
	wantEvents := []*mdpb.Event{
		&mdpb.Event{
			ArtifactId:  &a1id,
			ExecutionId: &e1id,
			Path: &mdpb.Event_Path{
				Steps: []*mdpb.Event_Path_Step{
					{
						Value: &mdpb.Event_Path_Step_Key{Key: `param1`},
					},
				},
			},
			Type:                   mdpb.Event_INPUT.Enum(),
			MillisecondsSinceEpoch: proto.Int64(100000),
		},
		&mdpb.Event{
			ArtifactId:  &a2id,
			ExecutionId: &e1id,
			Path: &mdpb.Event_Path{
				Steps: []*mdpb.Event_Path_Step{
					{
						Value: &mdpb.Event_Path_Step_Index{Index: 1},
					},
				},
			},
			Type:                   mdpb.Event_OUTPUT.Enum(),
			MillisecondsSinceEpoch: proto.Int64(200000),
		},
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
	store := createStore(t)
	// create test types
	atid, err := insertArtifactType(store, &mdpb.ArtifactType{
		Name: proto.String("artifact_type_name"),
	})
	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}
	etid, err := insertExecutionType(store, &mdpb.ExecutionType{
		Name: proto.String("execution_type_name"),
	})
	if err != nil {
		t.Fatalf("Cannot create execution type: %v", err)
	}
	// create an stored input artifact ia
	ia := &mdpb.Artifact{TypeId: &atid}
	as := []*mdpb.Artifact{ia}
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
	aep[1] = &ArtifactAndEvent{
		Artifact: oa,
		Event: &mdpb.Event{
			Type:                   &oet,
			MillisecondsSinceEpoch: proto.Int64(100000),
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
	eids := []ExecutionID{reid}
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
	store := createStore(t)
	// create test types
	atid, err := insertArtifactType(store, &mdpb.ArtifactType{
		Name: proto.String("artifact_type_name"),
	})
	if err != nil {
		t.Fatalf("Cannot create artifact type: %v", err)
	}
	etid, err := insertExecutionType(store, &mdpb.ExecutionType{
		Name: proto.String("execution_type_name"),
	})
	if err != nil {
		t.Fatalf("Cannot create execution type: %v", err)
	}
	ctid, err := insertContextType(store, &mdpb.ContextType{
		Name: proto.String("context_type_name"),
	})
	if err != nil {
		t.Fatalf("Cannot create context type: %v", err)
	}
	// create an stored input artifact ia
	ia := &mdpb.Artifact{TypeId: &atid}
	as := []*mdpb.Artifact{ia}
	aids, err := store.PutArtifacts(as)
	if err != nil {
		t.Fatalf("PutArtifacts failed: %v", err)
	}
	aid := int64(aids[0])
	ia.Id = &aid
	// prepare an execution and an output artifact, and publish input and output together with events
	// input has no event update, output has a new event
	e := &mdpb.Execution{TypeId: &etid}
	aep := []*ArtifactAndEvent{{Artifact: ia}}
	// prepare an context.
	cname := "context_name"
	c := &mdpb.Context{TypeId: &ctid, Name: &cname}
	ic := []*mdpb.Context{c}
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
	c.Type = proto.String("context_type_name")
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
	c.Type = proto.String("context_type_name")
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
	e.Type = gotExecutions[0].Type
	e.CreateTimeSinceEpoch = gotExecutions[0].CreateTimeSinceEpoch
	e.LastUpdateTimeSinceEpoch = gotExecutions[0].LastUpdateTimeSinceEpoch
	if !proto.Equal(e, gotExecutions[0]) {
		t.Errorf("GetExecutionsByContext returned result is incorrect. want: %v, got: %v", e, gotExecutions[0])
	}
}

func insertContext(s *Store, ctid int64, textContext string) (*mdpb.Context, error) {
	c := &mdpb.Context{}
	if err := prototext.Unmarshal([]byte(textContext), c); err != nil {
		return nil, err
	}
	c.TypeId = &ctid
	contexts := []*mdpb.Context{c}
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
	if err := prototext.Unmarshal([]byte(textExecution), e); err != nil {
		return nil, err
	}
	e.TypeId = &etid
	executions := []*mdpb.Execution{e}
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
	if err := prototext.Unmarshal([]byte(textArtifact), a); err != nil {
		return nil, err
	}
	a.TypeId = &atid
	artifacts := []*mdpb.Artifact{a}
	aids, err := s.PutArtifacts(artifacts)
	if err != nil {
		return nil, err
	}
	aid := int64(aids[0])
	a.Id = &aid
	return a, nil
}

func TestPutAndUseAttributionsAndAssociations(t *testing.T) {
	store := createStore(t)
	// prepare types
	ctid, err := insertContextType(store, &mdpb.ContextType{
		Name: proto.String("context_type_name"),
	})
	if err != nil {
		t.Fatalf("Cannot create context type: %v", err)
	}
	etid, err := insertExecutionType(store, &mdpb.ExecutionType{
		Name: proto.String("execution_type_name"),
	})
	if err != nil {
		t.Fatalf("Cannot create execution type: %v", err)
	}
	atid, err := insertArtifactType(store, &mdpb.ArtifactType{
		Name:       proto.String("artifact_type_name"),
		Properties: map[string]mdpb.PropertyType{"p1": mdpb.PropertyType_STRING},
	})
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
	attributions := []*mdpb.Attribution{
		{
			ArtifactId: wantArtifact.Id,
			ContextId:  wantContext.Id,
		},
	}
	associations := []*mdpb.Association{
		{
			ExecutionId: wantExecution.Id,
			ContextId:   wantContext.Id,
		},
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
	wantContext.Type = proto.String("context_type_name")
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
	wantContext.Type = proto.String("context_type_name")
	if !proto.Equal(wantContext, gotContexts[0]) {
		t.Errorf("GetContextsByExecution returned result is incorrect. want: %v, got: %v", wantContext, gotContexts[0])
	}
	// query execution and artifact from context
	gotArtifacts, err := store.GetArtifactsByContext(ContextID(wantContext.GetId()))
	if err != nil {
		t.Fatalf("GetArtifactsByContext failed: %v", err)
	}
	if len(gotArtifacts) != 1 {
		t.Errorf("GetArtifactsByContext returned number of results is incorrect. want: %v, got: %v", 1, len(gotArtifacts))
	}
	// skip comparing create/update timestamps
	wantArtifact.CreateTimeSinceEpoch = gotArtifacts[0].CreateTimeSinceEpoch
	wantArtifact.LastUpdateTimeSinceEpoch = gotArtifacts[0].LastUpdateTimeSinceEpoch
	wantArtifact.Type = proto.String("artifact_type_name")

	if !proto.Equal(wantArtifact, gotArtifacts[0]) {
		t.Errorf("GetArtifactsByContext returned result is incorrect. want: %v, got: %v", wantArtifact, gotArtifacts[0])
	}
	gotExecutions, err := store.GetExecutionsByContext(ContextID(wantContext.GetId()))
	if err != nil {
		t.Fatalf("GetExecutionsByContext failed: %v", err)
	}
	if len(gotExecutions) != 1 {
		t.Errorf("GetExecutionsByContext returned number of results is incorrect. want: %v, got: %v", 1, len(gotArtifacts))
	}
	// skip comparing create/update timestamps
	wantExecution.CreateTimeSinceEpoch = gotExecutions[0].CreateTimeSinceEpoch
	wantExecution.LastUpdateTimeSinceEpoch = gotExecutions[0].LastUpdateTimeSinceEpoch
	wantExecution.Type = proto.String("execution_type_name")
	if !proto.Equal(wantExecution, gotExecutions[0]) {
		t.Errorf("GetExecutionsByContext returned result is incorrect. want: %v, got: %v", wantExecution, gotExecutions[0])
	}
}

func TestPutDuplicatedAttributionsAndEmptyAssociations(t *testing.T) {
	store := createStore(t)
	ctid, err := insertContextType(store, &mdpb.ContextType{
		Name: proto.String("context_type_name"),
	})
	if err != nil {
		t.Fatalf("Cannot create context type: %v", err)
	}
	atid, err := insertArtifactType(store, &mdpb.ArtifactType{
		Name:       proto.String("artifact_type_name"),
		Properties: map[string]mdpb.PropertyType{"p1": mdpb.PropertyType_STRING},
	})
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
	wantContext.Type = proto.String("context_type_name")
	if !proto.Equal(wantContext, gotContexts[0]) {
		t.Errorf("GetContextsByArtifact returned result is incorrect. want: %v, got: %v", wantContext, gotContexts[0])
	}
	// query execution and artifact from context
	gotArtifacts, err := store.GetArtifactsByContext(ContextID(wantContext.GetId()))
	if err != nil {
		t.Fatalf("GetArtifactsByContext failed: %v", err)
	}
	if len(gotArtifacts) != 1 {
		t.Errorf("GetArtifactsByContext returned number of results is incorrect. want: %v, got: %v", 1, len(gotArtifacts))
	}
	// skip comparing create/update timestamps
	wantArtifact.CreateTimeSinceEpoch = gotArtifacts[0].CreateTimeSinceEpoch
	wantArtifact.LastUpdateTimeSinceEpoch = gotArtifacts[0].LastUpdateTimeSinceEpoch
	wantArtifact.Type = proto.String("artifact_type_name")

	if !proto.Equal(wantArtifact, gotArtifacts[0]) {
		t.Errorf("GetArtifactsByContext returned result is incorrect. want: %v, got: %v", wantArtifact, gotArtifacts[0])
	}
}
