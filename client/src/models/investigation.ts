export interface Investigation {
  schema: "investigation-panel/v1";
  mode: "run" | "synthetic_example";
  session_id: string | null;
  completed_vials: number;
  as_of_s: number;
  frame_count?: number;
  frame_qa?: string[];
  label?: string;
  example_id?: string;
  record_id?: string;
  reviewer?: string;
  evidence_id?: string;
  imported_at?: string;
  note: string;
  limitation: string;
  diagnosis: {
    status: "unavailable" | "no_deviation_observed" | "hypotheses";
    pattern: string | null;
    primary_hypothesis: string | null;
    hypotheses: { cause: string; support: string; evidence_id?: string }[];
    evidence: { vial_id: string; observation_id: string; source_evidence_id: string; position_after_change: number; source_time_s: number; delta_e2000: number }[];
    blocked_reasons: string[];
    next_checks: string[];
    cause_status: "not_identified";
    confirmed_root_cause: null;
    quality_decision: null;
    hardware_command: null;
    operator_message: string;
    policy: { deviation_de2000: number; minimum_vials: number };
  };
}
