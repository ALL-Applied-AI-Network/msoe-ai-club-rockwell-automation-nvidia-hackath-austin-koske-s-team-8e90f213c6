import type { Widget } from "@/models/widget";

interface WidgetListProps {
  widgets: Widget[];
  onDelete: (widgetId: string) => void;
}

export default function WidgetList({ widgets, onDelete }: WidgetListProps) {
  if (widgets.length === 0) {
    return <p>No widgets yet — create one above.</p>;
  }

  return (
    <ul className="widget-list">
      {widgets.map((widget) => (
        <li key={widget.widget_id}>
          <div>
            <strong>{widget.name}</strong>
            {widget.description && <div>{widget.description}</div>}
          </div>
          <button onClick={() => onDelete(widget.widget_id)}>Delete</button>
        </li>
      ))}
    </ul>
  );
}
