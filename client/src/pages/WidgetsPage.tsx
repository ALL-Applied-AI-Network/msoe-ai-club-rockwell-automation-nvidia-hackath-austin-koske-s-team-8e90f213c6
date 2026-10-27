import { useState, type FormEvent } from "react";
import { useCreateWidget, useDeleteWidget, useWidgets } from "@/hooks/useWidgets";
import WidgetList from "@/components/WidgetList";

/**
 * Example page wired end-to-end through the api layer (router -> service ->
 * in-memory adapter). Copy this shape — model + service + hook + page
 * — for your first real feature, then delete widgets.
 */
export default function WidgetsPage() {
  const { data: widgets, isLoading, isError } = useWidgets();
  const createWidget = useCreateWidget();
  const deleteWidget = useDeleteWidget();
  const [name, setName] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    createWidget.mutate({ name: name.trim() });
    setName("");
  };

  if (isLoading) return <p>Loading widgets...</p>;
  if (isError) {
    return <p>Failed to load widgets. Start the API on port 8443 and refresh.</p>;
  }

  return (
    <div>
      <h1>Widgets</h1>
      <p>
        Frontend/API example. Data resets when the API restarts. Swap this
        for your first operator feature.
      </p>

      <form className="widget-form" onSubmit={handleSubmit}>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="New widget name"
        />
        <button type="submit" disabled={createWidget.isPending}>
          Add
        </button>
      </form>

      <WidgetList widgets={widgets ?? []} onDelete={(id) => deleteWidget.mutate(id)} />
    </div>
  );
}
