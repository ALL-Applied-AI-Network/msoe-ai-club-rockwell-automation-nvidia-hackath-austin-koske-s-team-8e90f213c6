import { api } from "@/services/api";
import type { CreateWidgetRequest, UpdateWidgetRequest, Widget } from "@/models/widget";

const BASE_URL = "/widgets";

export async function listWidgets(): Promise<Widget[]> {
  const { data } = await api.get<Widget[]>(BASE_URL);
  return data;
}

export async function createWidget(body: CreateWidgetRequest): Promise<Widget> {
  const { data } = await api.post<Widget>(BASE_URL, body);
  return data;
}

export async function updateWidget(
  widgetId: string,
  body: UpdateWidgetRequest
): Promise<Widget> {
  const { data } = await api.put<Widget>(`${BASE_URL}/${widgetId}`, body);
  return data;
}

export async function deleteWidget(widgetId: string): Promise<void> {
  await api.delete(`${BASE_URL}/${widgetId}`);
}
