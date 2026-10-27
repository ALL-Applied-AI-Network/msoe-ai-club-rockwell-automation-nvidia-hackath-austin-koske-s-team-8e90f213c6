import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import PageChrome from "@/components/PageChrome";
import type { InspectionSession } from "@/models/inspection";
import { listInspectionRuns } from "@/services/inspectionService";

function dateLabel(value: string) {
  return new Intl.DateTimeFormat([], { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

export default function RunHistoryPage() {
  const [runs, setRuns] = useState<InspectionSession[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listInspectionRuns().then(setRuns).catch((reason) => setError(reason instanceof Error ? reason.message : "Could not load runs"));
  }, []);

  return (
    <PageChrome>
      <main className="history-main">
        <section className="page-intro"><div><span className="kicker">PERSISTED INSPECTION EVIDENCE</span><h1>Run history</h1><p>Review every camera session, image batch, and recording replay after an API restart.</p></div><Link className="button primary" to="/">NEW INSPECTION RUN</Link></section>
        {error && <div className="history-error">{error}</div>}
        <section className="run-table panel">
          <div className="run-table-head"><span>RUN</span><span>SOURCE</span><span>ACTIVE RECIPE</span><span>FRAMES</span><span>UPDATED</span><span /></div>
          {runs.length === 0 ? <div className="empty-history">No persisted runs yet. Start one from the monitor.</div> : runs.map((run) => (
            <div className="run-table-row" key={run.session_id}>
              <div><strong>{run.run_name}</strong><small>{run.session_id.slice(0, 8)} · {run.camera_id}</small></div>
              <div><span className="source-badge">{run.source_mode.replace("_", " ").toUpperCase()}</span><small>{run.source_name ?? "—"}</small></div>
              <div><strong>{run.recipe.current_recipe_name}</strong><small>{run.recipe.current_recipe_id}</small></div>
              <strong className="frame-total">{run.frames_processed}</strong>
              <span>{dateLabel(run.updated_at)}</span>
              <Link to={`/runs/${run.session_id}`}>REVIEW →</Link>
            </div>
          ))}
        </section>
      </main>
    </PageChrome>
  );
}
