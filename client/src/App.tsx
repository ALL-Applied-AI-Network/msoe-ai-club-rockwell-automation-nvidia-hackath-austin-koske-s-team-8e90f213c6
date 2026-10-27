import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import OperatorDashboard from "@/pages/OperatorDashboard";
import RunHistoryPage from "@/pages/RunHistoryPage";
import RunReviewPage from "@/pages/RunReviewPage";
import "./App.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<OperatorDashboard />} />
          <Route path="/history" element={<RunHistoryPage />} />
          <Route path="/runs/:runId" element={<RunReviewPage />} />
          <Route path="*" element={<OperatorDashboard />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
