import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as widgetService from "@/services/widgetService";
import type { CreateWidgetRequest, UpdateWidgetRequest } from "@/models/widget";

const WIDGETS_KEY = ["widgets"];

export function useWidgets() {
  return useQuery({ queryKey: WIDGETS_KEY, queryFn: widgetService.listWidgets });
}

export function useCreateWidget() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateWidgetRequest) => widgetService.createWidget(body),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: WIDGETS_KEY }),
  });
}

export function useUpdateWidget() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ widgetId, body }: { widgetId: string; body: UpdateWidgetRequest }) =>
      widgetService.updateWidget(widgetId, body),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: WIDGETS_KEY }),
  });
}

export function useDeleteWidget() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (widgetId: string) => widgetService.deleteWidget(widgetId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: WIDGETS_KEY }),
  });
}
