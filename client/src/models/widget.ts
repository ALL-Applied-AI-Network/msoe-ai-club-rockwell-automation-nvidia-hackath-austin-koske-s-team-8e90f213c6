export interface Widget {
  widget_id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateWidgetRequest {
  name: string;
  description?: string | null;
}

export interface UpdateWidgetRequest {
  name?: string;
  description?: string | null;
}
