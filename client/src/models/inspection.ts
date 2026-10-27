export type OperatorStatus =
  | "starting"
  | "processing"
  | "running_ok"
  | "warning"
  | "failure"
  | "unavailable"
  | "stopped";

export interface RecipeContext {
  current_recipe_id: string;
  current_recipe_name: string;
  previous_recipe_id?: string | null;
  vial_index: number;
  is_first_vial_after_change: boolean;
}

export interface CreateSessionRequest {
  camera_id: string;
  station_name: string;
  run_name: string;
  source_mode: SourceMode;
  source_name?: string | null;
  recipe: RecipeContext;
}

export type SourceMode = "live_camera" | "images" | "recording";

export interface RecipeSegmentCreate {
  start_ms: number;
  end_ms?: number | null;
  recipe: RecipeContext;
}

export interface RecipeSegment extends RecipeSegmentCreate {
  segment_id: string;
  session_id: string;
  created_at: string;
}

export interface InspectionSession {
  session_id: string;
  camera_id: string;
  station_name: string;
  run_name: string;
  source_mode: SourceMode;
  source_name?: string | null;
  recipe: RecipeContext;
  status: OperatorStatus;
  pipeline_mode: "mock" | "model";
  created_at: string;
  updated_at: string;
  frames_processed: number;
}

export interface PipelineMeasurement {
  label: string;
  value: string;
  state: "neutral" | "good" | "warning" | "failure";
}

export interface InspectionEvent {
  type: "inspection.frame";
  event_id: string;
  session_id: string;
  frame_id: string;
  frame_number: number;
  source_timestamp_ms?: number | null;
  original_filename?: string | null;
  recipe: RecipeContext;
  created_at: string;
  pipeline_mode: "mock" | "model";
  image: {
    mime_type: string;
    base64?: string | null;
    url?: string | null;
    width: number;
    height: number;
  };
  visuals?: {
    rendered_frame: InspectionEvent["image"] | null;
    mask: InspectionEvent["image"] | null;
    status: "available" | "not_configured";
  };
  decision: {
    status: OperatorStatus;
    headline: string;
    detail: string;
    recommended_action: string;
    recipe_progress_percent: number | null;
    model_state: string;
    confidence: number | null;
    measurements: PipelineMeasurement[];
    evidence?: {
      measurement_valid?: boolean;
      quality_flags?: string[];
      blocked_reasons?: string[];
      presence?: number;
      proposal_pixels?: number;
      sampling_pixels?: number;
      processing_ms?: number;
      reference?: { status: string; sha256: string };
      recipe_context_matched?: boolean;
      [key: string]: unknown;
    };
  };
  acknowledged_at?: string | null;
  acknowledgement?: OperatorResponse | null;
}

export interface OperatorResponse {
  reviewer: string;
  action: "reviewed" | "hold_for_review" | "request_reference" | "investigate";
  note: string;
}

export interface HealthStatus {
  status: string;
  service: string;
  pipeline_mode: "mock" | "model";
  connected_frontends: number;
  queue_depth: number;
  persistence: string;
}

export interface ArtifactRecord {
  artifact_id: string;
  session_id: string;
  kind: "source" | "rendered" | "mask" | "recording" | "reference";
  url: string;
  mime_type: string;
  original_filename?: string | null;
  size_bytes: number;
  sha256: string;
  created_at: string;
}

export type SimulationMode = "warning" | "failure" | undefined;
