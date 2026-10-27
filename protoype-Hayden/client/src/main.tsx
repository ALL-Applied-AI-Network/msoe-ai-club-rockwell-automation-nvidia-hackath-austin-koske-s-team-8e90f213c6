import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { OperatorDashboard } from "./pages/OperatorDashboard";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <OperatorDashboard />
  </StrictMode>,
);

