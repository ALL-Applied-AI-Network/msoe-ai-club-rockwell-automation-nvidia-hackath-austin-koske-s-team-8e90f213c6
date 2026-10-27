import axios from "axios";

// Vite proxies relative API requests during local development.
export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "/api/v1",
});

api.interceptors.response.use(undefined, (error: unknown) => {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") error.message = detail;
    else if (Array.isArray(detail)) error.message = detail.map((item) => item.msg).join("; ");
  }
  return Promise.reject(error);
});
