import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";

export default function PageChrome({ children }: { children: ReactNode }) {
  const location = useLocation();
  return (
    <div className="shell">
      <header className="topbar">
        <Link className="brand" to="/">
          <div className="brand-mark" aria-hidden="true"><span /></div>
          <div><strong>LINEWATCH</strong><small>CHANGEOVER INTELLIGENCE</small></div>
        </Link>
        <nav className="main-nav">
          <Link className={location.pathname === "/" ? "active" : ""} to="/">MONITOR</Link>
          <Link className={location.pathname === "/recipe-check" ? "active" : ""} to="/recipe-check">RECIPE CHECK</Link>
          <Link className={location.pathname.startsWith("/history") || location.pathname.startsWith("/runs") ? "active" : ""} to="/history">RUN HISTORY</Link>
        </nav>
        <div className="topbar-status"><div><strong>PERSISTED REVIEW</strong><small>SQLITE + ARTIFACT STORE</small></div></div>
      </header>
      {children}
      <footer><div><strong>LINEWATCH · OPERATOR-IN-THE-LOOP PROTOTYPE</strong><span>Rockwell Automation + NVIDIA Innovation Lab · Fall 2026</span></div><div className="team-credit"><strong>© 2026 Project Team</strong><span>Persistent inspection evidence · No PLC control</span></div></footer>
    </div>
  );
}
