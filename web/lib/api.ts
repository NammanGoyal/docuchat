import { SseParser } from "./sse";

// Inlined at build time; this is the URL the *browser* uses to reach the API.
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export interface DocumentInfo {
  id: string;
  filename: string;
  kind: "pdf" | "markdown" | "text";
  content_hash: string;
  size_bytes: number;
  num_pages: number | null;
  num_chunks: number;
  embedding_model: string;
  created_at: string;
}

export interface UploadResult extends DocumentInfo {
  duplicate: boolean;
}

export interface Citation {
  marker: number;
  document_id: string;
  filename: string;
  chunk_index: number;
  page: number | null;
  text: string;
  score: number;
  similarity: number;
}

export interface Confidence {
  similarity: number;
  term_coverage: number;
}

export interface AskMeta {
  refused: boolean;
  confidence: Confidence;
  citations: Citation[];
}

export interface Health {
  status: string;
  embedder: string;
  answerer: string;
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    // not JSON
  }
  return `${res.status} ${res.statusText}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) throw new Error(await errorMessage(res));
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const getHealth = () => request<Health>("/healthz");
export const listDocuments = () => request<DocumentInfo[]>("/documents");
export const deleteDocument = (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" });

export function uploadDocument(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  return request<UploadResult>("/documents", { method: "POST", body: form });
}

export interface AskHandlers {
  onMeta: (meta: AskMeta) => void;
  onToken: (text: string) => void;
}

/** POST /ask with stream=true and dispatch server-sent events as they arrive. */
export async function askStream(
  question: string,
  topK: number,
  handlers: AskHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, top_k: topK, stream: true }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await errorMessage(res));

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = new SseParser();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const ev of parser.push(value)) {
      const data = JSON.parse(ev.data);
      if (ev.event === "meta") handlers.onMeta(data as AskMeta);
      else if (ev.event === "token") handlers.onToken(data.text as string);
      else if (ev.event === "error") throw new Error(data.detail ?? "generation failed");
      else if (ev.event === "done") return;
    }
  }
  throw new Error("stream ended unexpectedly");
}
