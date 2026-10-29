"""Completed-vial hypotheses; live frames and synthetic examples stay separate."""
from copy import deepcopy
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from uuid import UUID

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "investigation"
PROFILE = "vial-fill-640x480-temporal-v1"
CHECKS = [
    "Confirm completion and physical IDs for at least three consecutive vials.",
    "Provide independent accepted endpoint references for the intended and previous recipes.",
    "Review camera alignment, temporal reference and sampling QA.",
    "Obtain recipe-command and metered-delivery records from CSI staff.",
]


def run_diagnosis(document):
    from replay.diagnostics import diagnose
    result = diagnose(document)
    if not result["next_checks"]:
        result["next_checks"] = CHECKS if result["status"] == "unavailable" else [
            "Retain the source evidence; this research color band does not approve product."]
    return result


def panel(document, result, mode, **metadata):
    return dict(schema="investigation-panel/v1", mode=mode,
                session_id=document["session_id"] if mode == "run" else None,
                completed_vials=sum(row.get("completed") is True and type(row.get("available_s")) in (int, float) and row["available_s"] <= document["as_of_s"]
                                    for row in document["vials"]),
                as_of_s=document["as_of_s"], diagnosis=result,
                limitation="Candidate explanations and attributed records do not identify a physical cause or approve product.",
                **metadata)


def example_index():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    return [dict(id=row["id"], label=row["label"]) for row in manifest["cases"]]


def synthetic_example(case_id):
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    row = next((r for r in manifest["cases"] if r["id"] == case_id), None)
    if row is None:
        raise ValueError("Unknown synthetic example")
    content = (FIXTURES / row["file"]).read_bytes()
    if sha256(content).hexdigest() != row["sha256"]:
        raise RuntimeError("Synthetic example differs from its pinned experiment input")
    document = json.loads(content)
    if document["data_domain"] != "synthetic":
        raise RuntimeError("Example must remain synthetic")
    return panel(document, run_diagnosis(document), "synthetic_example",
                 label=row["label"], example_id=case_id, input_sha256=row["sha256"],
                 note="SYNTHETIC EXAMPLE · feature-model study, not this camera/run and not real cause accuracy.")


async def current_investigation(service, session):
    saved = await service.databridge.latest_investigation(session.session_id)
    if saved:
        payload = json.loads(saved["input_json"])
        return panel(payload["document"], json.loads(saved["result_json"]), "run",
                     record_id=saved["investigation_id"], reviewer=saved["reviewer"],
                     evidence_id=saved["evidence_id"], imported_at=saved["created_at"],
                     frame_count=session.frames_processed,
                     note="Imported completed-vial evidence at the recorded analysis time. Human declarations are attributed, not authenticated.")
    events = await service.databridge.list_events(session.session_id, 1)
    frame = events[0] if events else None
    document = dict(schema="vial-diagnosis-input/v1", data_domain="real",
                    session_id=str(session.session_id), camera_id=session.camera_id,
                    profile_id=PROFILE, clock_domain_id="unqualified",
                    recipe_id=session.recipe.current_recipe_id, context_evidence_id=None,
                    context_available_s=None, as_of_s=0, references=[], vials=[])
    return panel(document, run_diagnosis(document), "run", frame_count=session.frames_processed,
                 frame_qa=frame.decision.evidence.get("quality_flags", []) if frame else [],
                 note="No qualified completed-vial records supplied. Consecutive frames of one fill are not separate vials.")


def required_text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(f"{name} requires an attributed text value")


async def import_investigation(service, session, payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("document"), dict):
        raise ValueError("Supply reviewer, evidence_id and a vial-diagnosis-input/v1 document")
    required_text(payload.get("reviewer"), "reviewer")
    required_text(payload.get("evidence_id"), "evidence_id")
    document = deepcopy(payload["document"])
    if document.get("data_domain") != "real":
        raise ValueError("Run imports require real evidence. Use the separate synthetic example mode.")
    if document.get("session_id") != str(session.session_id) or document.get("camera_id") != session.camera_id:
        raise ValueError("Completed-vial records belong to another run/camera")
    if document.get("profile_id") != PROFILE:
        raise ValueError("Completed-vial records require the released camera profile")
    if document.get("recipe_id") in (None, "", "unconfirmed"):
        raise ValueError("Intended recipe must be independently supplied")
    rows = document.get("vials")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 32 or not all(isinstance(r, dict) for r in rows):
        raise ValueError("Supply one to 32 explicitly reviewed completed-vial records")
    if not isinstance(document.get("references"), list) or len(document["references"]) > 4:
        raise ValueError("Supply at most four independent recipe references")
    reviews = payload.get("temporal_reference_reviews", [])
    if not isinstance(reviews, list) or not all(isinstance(r, dict) for r in reviews):
        raise ValueError("Temporal reference reviews must be a list of attributed review records")
    observations, completions = set(), set()
    for row in rows:
        observation_id = row.get("observation_id")
        if not isinstance(observation_id, str) or observation_id in observations:
            raise ValueError("Distinct source event IDs are required; frames cannot be repeated as vials")
        observations.add(observation_id)
        try:
            found = await service.databridge.list_events(session.session_id, 1, event_id=UUID(observation_id))
        except (ValueError, TypeError) as error:
            raise ValueError("observation_id must identify a saved source event") from error
        if not found:
            raise ValueError("Source event was not saved in this run")
        event = found[0]
        evidence = event.decision.evidence
        if event.pipeline_mode != "model" or not evidence.get("models"):
            raise ValueError("Mock/test events cannot qualify real diagnosis")
        if row.get("source_evidence_id") != event.image.url:
            raise ValueError("Source evidence must match the saved original image URL")
        if row.get("qa_pass") is not bool(evidence.get("measurement_valid")):
            raise ValueError("Imported QA cannot override the saved measurement QA")
        if row.get("lab") != evidence.get("lab"):
            raise ValueError("Use the saved exact Lab measurement; do not replace it with estimated color")
        clock = evidence.get("source_clock")
        if clock not in ("recording_seek", "browser_elapsed") or event.source_timestamp_ms is None:
            raise ValueError("Stills/unknown clocks need an explicit source mapping before RCA")
        expected_clock = f"{clock}:{session.session_id}"
        if document.get("clock_domain_id") != expected_clock or row.get("clock_domain_id") != expected_clock:
            raise ValueError("Clock domain must match the saved recording/capture clock; it is not native PTS")
        if not isinstance(row.get("source_time_s"), (float, int)) or not isclose(row["source_time_s"], event.source_timestamp_ms / 1000, abs_tol=1e-9):
            raise ValueError("Completed-vial source time differs from its saved frame")
        if row.get("recipe_id") != event.recipe.current_recipe_id or row.get("recipe_id") != document["recipe_id"]:
            raise ValueError("Intended recipe must match the source-bound recipe context")
        marker = row.get("completion_evidence_id")
        required_text(marker, "completion_evidence_id")
        if marker in completions:
            raise ValueError("Each physical vial needs its own completion evidence")
        completions.add(marker)
        reference = evidence.get("reference") or {}
        qualified = isinstance(reference.get("sha256"), str) and len(reference["sha256"]) == 64 and any(
            review.get("reference_sha256") == reference.get("sha256")
            and review.get("status") == "human_reviewed" and review.get("origin") == "human_review"
            and review.get("data_domain") == "real"
            and review.get("camera_id") == session.camera_id and review.get("profile_id") == PROFILE
            and review.get("clock_domain_id") == expected_clock
            and type(review.get("source_time_s")) in (int, float)
            and type(review.get("available_s")) in (int, float) and type(row.get("start_s")) in (int, float)
            and 0 <= review["source_time_s"] <= review["available_s"] <= row["start_s"]
            and review["source_time_s"] < row["start_s"]
            and review.get("condition") == "empty" and review.get("alignment") == "confirmed"
            and isinstance(review.get("reviewer"), str) and review["reviewer"].strip()
            and isinstance(review.get("evidence_id"), str) and review["evidence_id"].strip() and not review.get("demonstration")
            for review in reviews)
        row["temporal_reference_status"] = "human_reviewed" if qualified else "unreviewed_not_certified_empty"
    result = run_diagnosis(document)
    normalized = dict(payload, document=document, submitted_document=payload["document"])
    record_id = await service.databridge.save_investigation(session.session_id, payload["reviewer"], payload["evidence_id"], normalized, result)
    return panel(document, result, "run", record_id=record_id, reviewer=payload["reviewer"],
                 evidence_id=payload["evidence_id"], frame_count=session.frames_processed,
                 note="Imported completed-vial evidence. Human declarations are attributed, not authenticated; no physical cause is confirmed.")
