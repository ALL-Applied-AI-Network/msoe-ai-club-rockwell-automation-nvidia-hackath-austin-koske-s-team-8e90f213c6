export type OperatorState =
  | "STARTING"
  | "RUNNING_OK"
  | "WARNING"
  | "FAILURE"
  | "RECOVERING"
  | "STOPPED"
  | "UNKNOWN";

export type StreamPhase =
  | "idle"
  | "starting"
  | "streaming"
  | "stopped"
  | "failed";

export type InspectionContext = {
  cameraId: string;
  currentRecipeId: string;
  previousRecipeId: string;
  vialIndex: string;
  expectedColorRgb: string;
  colorRoi: string;
  liquidPromptBox: string;
  firstVialAfterChange: boolean;
};

export type HealthResponse = {
  status: string;
  service: string;
  sam2_configured: boolean;
  connected_frontends: number;
};

export type MaskProposal = {
  detection_index: number;
  label: string;
  confidence: number | null;
  prompt_box_xyxy_exclusive: [number, number, number, number];
  mask_score: number | null;
  mask_png_base64: string;
};

export type ColorInspection = {
  status?: "alert" | "within_threshold" | "unavailable";
  reason?: string;
  delta_e_2000?: number;
  threshold?: number;
  alert?: boolean;
  current_recipe_id?: string | null;
  likely_cause?: {
    type: string;
    explanation: string;
  } | null;
};

export type CameraFrameEvent = {
  type: "camera.frame";
  camera_id: string;
  frame_id: string;
  received_at: string;
  image: {
    mime_type: string;
    base64: string;
    width: number;
    height: number;
  };
  inspection: {
    status: "ok" | "no_liquid_proposal" | "segmentation_unavailable";
    source_id: string;
    stream_id: string;
    pts_seconds: number;
    decision: {
      pool_present: boolean;
      pool_mask_area_ratio: number | null;
      sequence: {
        state: string;
        trend: string;
        frames_in_window?: number;
      };
      sample_usability: {
        usable: boolean | null;
        reasons: string[];
      };
      color_inspection: ColorInspection;
    };
    proposals: MaskProposal[];
    combined_mask_png_base64: string | null;
    unavailable_reason: string | null;
    provenance: {
      segmentation_model: string | null;
    };
  };
};

export type OperatorSummary = {
  state: OperatorState;
  title: string;
  detail: string;
  updatedAt: number | null;
};

