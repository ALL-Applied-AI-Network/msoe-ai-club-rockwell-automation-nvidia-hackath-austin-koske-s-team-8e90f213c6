import axios from "axios";

// Vite proxies relative API requests during local development.
export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "/api/v1",
});
