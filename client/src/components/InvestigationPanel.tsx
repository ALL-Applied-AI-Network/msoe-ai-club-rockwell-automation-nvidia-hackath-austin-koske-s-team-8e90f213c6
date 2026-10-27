import { useEffect, useRef, useState } from "react";
import type { Investigation } from "@/models/investigation";
import { artifactUrl } from "@/services/inspectionService";
import { getInvestigation, getInvestigationExample, importInvestigation, listInvestigationExamples } from "@/services/investigationService";

const causes: Record<string, string> = {
  previous_recipe_carryover: "Residue from the previous recipe",
  vial_local_optical_or_sampling_change: "Camera, reflection or sampling changes",
  recipe_or_dispensing_mismatch: "Wrong recipe or delivered amounts",
  commanded_recipe_record_mismatch: "Recipe-command record disagrees",
  metered_delivery_record_mismatch: "Delivery record disagrees",
};
const support: Record<string, string> = {
  consistent_pattern: "Pattern is consistent; cause not established",
  not_established: "Not established by this sequence",
  not_excluded_by_healthy_background: "Not excluded by a healthy background",
  not_excluded_without_process_records: "Not excluded without process records",
  attributed_process_record: "Supported by an attributed record; verify the record",
};
const missingChecks = [
  "Confirm completion and physical IDs for at least three consecutive vials.",
  "Provide independent accepted recipe endpoint references.",
  "Review camera alignment, temporal reference and sampling QA.",
  "Obtain staff-provided recipe and metered-delivery records.",
];
const words = (value: string) => value.replaceAll("_", " ");

export default function InvestigationPanel({ sessionId, frameNumber, processing = false }: {
  sessionId?: string; frameNumber?: number; processing?: boolean;
}) {
  const [mode, setMode] = useState<"run" | "synthetic_example">("run");
  const [caseId, setCaseId] = useState("transient");
  const [examples, setExamples] = useState<{ id: string; label: string }[]>([]);
  const [snapshot, setData] = useState<Investigation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const generation = useRef(0);
  const importAbort = useRef(new AbortController());
  const frameTick = mode === "run" ? frameNumber : 0;
  const data = snapshot?.mode === mode && (mode === "run" ? snapshot.session_id === sessionId : snapshot.example_id === caseId) ? snapshot : null;

  useEffect(() => {
    const controller = new AbortController();
    void listInvestigationExamples(controller.signal).then(setExamples).catch(() => undefined);
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const current = ++generation.current;
    const controller = new AbortController();
    importAbort.current.abort();
    importAbort.current = new AbortController();
    setImporting(false);
    if (mode === "run" && !sessionId) {
      setData(null); setLoading(false); setError(null);
      return () => controller.abort();
    }
    setLoading(true);
    const request = mode === "synthetic_example"
      ? getInvestigationExample(caseId, controller.signal)
      : getInvestigation(sessionId!, controller.signal);
    void request.then((result) => {
      if (current === generation.current) { setData(result); setError(null); }
    }).catch((reason) => {
      if (!controller.signal.aborted && current === generation.current) setError(reason instanceof Error ? reason.message : "Investigation unavailable");
    }).finally(() => { if (current === generation.current) setLoading(false); });
    return () => { controller.abort(); importAbort.current.abort(); };
  }, [mode, caseId, sessionId, frameTick]);

  const loadRecords = async (file?: File) => {
    if (!file || !sessionId || processing || mode !== "run") return;
    const current = generation.current;
    setImporting(true);
    try {
      const result = await importInvestigation(sessionId, file, importAbort.current.signal);
      if (current === generation.current) { setData(result); setError(null); }
    } catch (reason) {
      if (current === generation.current) setError(reason instanceof Error ? reason.message : "Could not import completed-vial evidence");
    } finally {
      if (current === generation.current) setImporting(false);
    }
  };

  const download = () => {
    if (!data) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = data.mode === "synthetic_example" ? `SYNTHETIC-investigation-${data.example_id}.json` : `investigation-${data.session_id}.json`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const synthetic = data?.mode === "synthetic_example";
  const diagnosis = error ? null : data?.diagnosis;
  const status = diagnosis?.status ?? "unavailable";
  const candidates: { cause: string; support: string; evidence_id?: string }[] = diagnosis?.hypotheses.length ? diagnosis.hypotheses : Object.keys(causes).slice(0, 3).map((cause) => ({ cause, support: status === "no_deviation_observed" ? "No deviation observed under the research band; no causal attribution" : "Not assessed: required evidence is missing or unqualified" }));

  return (
    <section id="investigation" className="panel investigation-panel" aria-label="Investigation">
      <div className="panel-heading compact"><div><span className="section-number">05</span><div><h2>Investigation</h2><p>Possible causes, supporting evidence and next checks</p></div></div><span className={`status-pill ${status === "unavailable" ? "unavailable" : "warning"}`}>{status === "hypotheses" ? "HYPOTHESES" : status === "no_deviation_observed" ? "NO DEVIATION OBSERVED" : "UNAVAILABLE"}</span></div>
      <div className="investigation-content">
        <div className="investigation-modes"><button className={`button quiet ${mode === "run" ? "selected" : ""}`} onClick={() => setMode("run")} type="button">THIS RUN</button><button className={`button quiet ${mode === "synthetic_example" ? "selected" : ""}`} onClick={() => setMode("synthetic_example")} type="button">SYNTHETIC EXAMPLE</button></div>
        {mode === "synthetic_example" && <label className="investigation-case">EXAMPLE<select value={caseId} onChange={(event) => setCaseId(event.target.value)}>{examples.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>}
        {synthetic && <div className="synthetic-banner" role="note"><strong>SYNTHETIC EXAMPLE · NOT THIS CAMERA/RUN</strong><p>Four simulated completed vials. Feature-model experiment; no physical cause or real accuracy is established.</p></div>}
        {error && <p className="history-error" role="alert">{error}</p>}
        {!sessionId && mode === "run" && <p>Start an inspection run to attach real evidence. Frame uploads alone do not establish completed vials.</p>}
        {loading && <span className="investigation-loading">Updating evidence…</span>}
        <h3>{diagnosis?.operator_message ?? "Diagnosis unavailable: completed-vial evidence is required."}</h3>
        {data && <p className="investigation-note">{data.note}</p>}
        <div className="investigation-counts"><span><strong>{data?.completed_vials ?? 0}</strong> completed-vial records</span>{!synthetic && <span><strong>{data?.frame_count ?? frameNumber ?? 0}</strong> processed frames · not vial count</span>}<span>Confirmed cause: <strong>not identified</strong></span></div>
        {data?.record_id && <p>Imported by {data.reviewer} · evidence {data.evidence_id} · analysis boundary {data.as_of_s.toFixed(3)} s in the declared source clock.</p>}
        {diagnosis && diagnosis.evidence.length > 0 && <div className="investigation-table-wrap"><table className="investigation-table"><caption>Completed-vial endpoint evidence · experimental ΔE2000 band {diagnosis.policy.deviation_de2000}, not a product tolerance</caption><thead><tr><th>Vial</th><th>Position</th><th>ΔE2000</th><th>Source evidence</th></tr></thead><tbody>{diagnosis.evidence.map((row) => <tr key={row.observation_id}><td>{row.vial_id}</td><td>{row.position_after_change}</td><td>{row.delta_e2000.toFixed(2)}</td><td>{row.source_evidence_id.startsWith("/api/v1/artifacts/") ? <a href={artifactUrl(row.source_evidence_id)} target="_blank" rel="noreferrer">Original frame</a> : row.source_evidence_id}</td></tr>)}</tbody></table></div>}
        <h4>Possible causes</h4>
        <ul className="investigation-causes">{candidates.map((item) => <li key={item.cause}><strong>{causes[item.cause] ?? words(item.cause)}</strong><span>{support[item.support] ?? item.support}</span>{"evidence_id" in item && item.evidence_id && <small>Record: {item.evidence_id}</small>}</li>)}</ul>
        <h4>Next checks</h4><ol>{(diagnosis?.next_checks ?? missingChecks).map((check) => <li key={check}>{check}</li>)}</ol>
        {diagnosis && diagnosis.blocked_reasons.length > 0 && <details><summary>Missing or unqualified evidence ({diagnosis.blocked_reasons.length})</summary><ul>{diagnosis.blocked_reasons.map((reason) => <li key={reason}>{words(reason)}</li>)}</ul></details>}
        {!synthetic && data?.frame_qa && data.frame_qa.length > 0 && <p>Latest frame sampling checks: {data.frame_qa.map(words).join(" · ")}. These do not establish a physical cause.</p>}
        <p className="investigation-limit">{data?.limitation ?? "Candidate explanations do not identify a physical cause or approve product."}</p>
        <div className="investigation-actions">{mode === "run" && <><button className="button quiet" type="button" disabled={!sessionId || processing || importing} onClick={() => fileRef.current?.click()}>{importing ? "IMPORTING…" : "IMPORT REVIEWED VIAL RECORDS"}</button><input ref={fileRef} hidden type="file" accept="application/json,.json" onChange={(event) => { void loadRecords(event.target.files?.[0]); event.target.value = ""; }} /></>}<button className="button quiet" type="button" disabled={!data || Boolean(error)} onClick={download}>DOWNLOAD EVIDENCE</button></div>
        {mode === "run" && <p className="investigation-note">Import attributed completion/identity records, exact saved model measurements and independent references. Consecutive frames of one fill cannot be relabeled as separate vials.</p>}
      </div>
    </section>
  );
}
