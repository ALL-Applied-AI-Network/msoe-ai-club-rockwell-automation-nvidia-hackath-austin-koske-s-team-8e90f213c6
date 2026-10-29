"""Contract fixtures are simulated; no human labels or real cause accuracy."""
from copy import deepcopy
import json

import pytest
from fastapi.testclient import TestClient

from databridge.sqlite_inspections import SQLiteInspectionBridge
from dependencies import get_inspection_service
from main import app
from models.inspection import OperatorDecision, OperatorStatus
from pipelines.base import PipelineResult
from pipelines.mock import MockInspectionPipeline
from services.artifacts import ArtifactStore
from services.event_broker import EventBroker
from services.inspection import InspectionService
from services.investigation import FIXTURES, PROFILE


@pytest.fixture
def client(tmp_path):
    bridge = SQLiteInspectionBridge(tmp_path / "investigation.sqlite3")
    service = InspectionService(bridge, MockInspectionPipeline(), EventBroker(), ArtifactStore(tmp_path / "artifacts"))
    app.dependency_overrides[get_inspection_service] = lambda: service
    with TestClient(app) as api:
        yield api, service
        api.portal.call(service.stop)
        api.portal.call(bridge.close)
    app.dependency_overrides.clear()


def create(api):
    session = api.post('/api/v1/sessions', json={'run_name':'SIMULATED UNIT CONTRACT', 'source_mode':'recording',
        'recipe':{'current_recipe_id':'B','current_recipe_name':'Unit fixture B'}}).json()
    return session, f"/api/v1/sessions/{session['session_id']}"


def fixture_document(api, service):
    """Bind simulated endpoints to a temporary model-protocol stub, not physical evidence."""
    document = json.loads((FIXTURES / 'transient.json').read_text())
    class FixturePipeline:
        mode = 'model'
        qa = True
        async def analyze(self, image_bytes, session, frame_number, simulate=None, context=None):
            return PipelineResult(decision=OperatorDecision(status=OperatorStatus.WARNING,
                headline='UNIT CONTRACT ONLY', detail='Not an actual inference or approved label',
                recommended_action='No physical action', model_state='SIMULATED UNIT FIXTURE',
                evidence={'measurement_valid':self.qa,'lab':document['vials'][frame_number-1]['lab'],
                    'models':{'unit_fixture':'f'*64},'reference':{'sha256':'a'*64,'status':'unreviewed_not_certified_empty'}}))
    service.pipeline = FixturePipeline()
    session, resource = create(api)
    clock = 'recording_seek:' + session['session_id']
    result = deepcopy(document)
    result.update(data_domain='real', session_id=session['session_id'], camera_id=session['camera_id'],
                  profile_id=PROFILE, clock_domain_id=clock)
    for reference in result['references']:
        reference.update(data_domain='real', camera_id=session['camera_id'], profile_id=PROFILE, clock_domain_id=clock)
        reference['qualification'].update(status='human_reviewed', reviewer='UNIT CONTRACT ONLY', origin='human_review')
    for index, row in enumerate(result['vials']):
        event = api.post(resource + '/frames', params={'frame_id':f'unit-{index}','width':640,'height':480,
            'source_timestamp_ms':int(row['source_time_s']*1000),'source_clock':'recording_seek'},
            content=b'contract stub, not model pixels', headers={'content-type':'image/png'}).json()
        row.update(observation_id=event['event_id'], source_evidence_id=event['image']['url'],
                   camera_id=session['camera_id'], profile_id=PROFILE, clock_domain_id=clock,
                   temporal_reference_status='human_reviewed')
    return resource, {'reviewer':'UNIT CONTRACT ONLY','evidence_id':'simulated-contract-review', 'document':result,
        'temporal_reference_reviews':[{'reference_sha256':'a'*64,'status':'human_reviewed','origin':'human_review',
            'data_domain':'real','camera_id':session['camera_id'],'profile_id':PROFILE,
            'clock_domain_id':clock,'source_time_s':0,'available_s':0,
            'condition':'empty','alignment':'confirmed','reviewer':'UNIT CONTRACT ONLY','evidence_id':'simulated-reference-review'}]}


def test_frames_do_not_become_completed_vials(client):
    api, _ = client
    _, resource = create(api)
    for index in range(3):
        api.post(resource + '/frames', params={'frame_id':str(index),'width':640,'height':480},
                 content=b'mock pixels', headers={'content-type':'image/png'})
    panel = api.get(resource + '/investigation').json()
    assert panel['frame_count'] == 3 and panel['completed_vials'] == 0
    assert panel['mode'] == 'run'
    assert panel['diagnosis']['status'] == 'unavailable'
    assert panel['diagnosis']['confirmed_root_cause'] is None


def test_synthetic_examples_are_separate_and_not_persisted(client):
    api, _ = client
    _, resource = create(api)
    index = api.get('/api/v1/investigation/examples').json()
    assert len(index) == 4
    for row in index:
        panel = api.get('/api/v1/investigation/examples/' + row['id']).json()
        assert panel['mode'] == 'synthetic_example' and panel['session_id'] is None
        assert panel['completed_vials'] == 4
        assert panel['diagnosis']['confirmed_root_cause'] is None
        assert panel['diagnosis']['quality_decision'] is None
        assert panel['diagnosis']['hardware_command'] is None
    assert api.get(resource + '/investigation').json()['completed_vials'] == 0
    camera = api.get('/api/v1/investigation/examples/camera').json()
    assert camera['diagnosis']['status'] == 'unavailable'
    dose = api.get('/api/v1/investigation/examples/dose-record').json()
    assert dose['diagnosis']['primary_hypothesis'] == 'metered_delivery_record_mismatch'
    assert api.get('/api/v1/investigation/examples/unknown').status_code == 404


def test_import_binds_exact_source_and_persists_without_changing_frame_decisions(client):
    api, service = client
    resource, payload = fixture_document(api, service)
    before = api.get(resource + '/events').json()
    response = api.post(resource + '/investigation', json=payload)
    assert response.status_code == 201, response.text
    panel = response.json()
    assert panel['diagnosis']['pattern'] == 'transient_toward_previous_recipe'
    assert panel['diagnosis']['primary_hypothesis'] == 'previous_recipe_carryover'
    assert panel['diagnosis']['confirmed_root_cause'] is None
    assert api.get(resource + '/investigation').json()['record_id'] == panel['record_id']
    assert api.get(resource + '/events').json() == before


@pytest.mark.parametrize('mutation', ['duplicate','qa','lab','clock','source','session','synthetic','completion'])
def test_import_cannot_upgrade_or_reassociate_evidence(client, mutation):
    api, service = client
    resource, payload = fixture_document(api, service)
    document = payload['document']
    if mutation == 'duplicate':document['vials'][1]['observation_id'] = document['vials'][0]['observation_id']
    if mutation == 'qa':document['vials'][0]['qa_pass'] = False
    if mutation == 'lab':document['vials'][0]['lab'][0] += .1
    if mutation == 'clock':document['clock_domain_id'] = 'invented-native-clock'
    if mutation == 'source':document['vials'][0]['source_evidence_id'] = 'other-camera-image'
    if mutation == 'session':document['session_id'] = 'other-run'
    if mutation == 'synthetic':document['data_domain'] = 'synthetic'
    if mutation == 'completion':document['vials'][1]['completion_evidence_id'] = document['vials'][0]['completion_evidence_id']
    assert api.post(resource + '/investigation', json=payload).status_code == 422
    assert api.get(resource + '/investigation').json()['completed_vials'] == 0


def test_claimed_temporal_review_without_matching_record_stays_unavailable(client):
    api, service = client
    resource, payload = fixture_document(api, service)
    payload['temporal_reference_reviews'] = []
    response = api.post(resource + '/investigation', json=payload)
    assert response.status_code == 201
    assert response.json()['diagnosis']['status'] == 'unavailable'
    assert any('temporal_reference_unqualified' in reason for reason in response.json()['diagnosis']['blocked_reasons'])
