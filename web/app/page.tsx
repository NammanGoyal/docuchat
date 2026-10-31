"use client";

import Link from "next/link";
import { type FormEvent, useRef, useState } from "react";

import { Answer } from "@/components/Answer";
import { type AskMeta, askStream } from "@/lib/api";

interface Turn {
  id: number;
  question: string;
  answer: string;
  meta: AskMeta | null;
  status: "streaming" | "done" | "error";
  error?: string;
}

export default function ChatPage() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [selected, setSelected] = useState<{ turn: number; marker: number } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const busy = turns.some((t) => t.status === "streaming");

  const update = (id: number, patch: (t: Turn) => Partial<Turn>) =>
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...patch(t) } : t)));

  async function ask(e: FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    const id = Date.now();
    setTurns((ts) => [...ts, { id, question: q, answer: "", meta: null, status: "streaming" }]);
    setQuestion("");
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await askStream(
        q,
        topK,
        {
          onMeta: (meta) => update(id, () => ({ meta })),
          onToken: (text) => update(id, (t) => ({ answer: t.answer + text })),
        },
        controller.signal,
      );
      update(id, () => ({ status: "done" }));
    } catch (err) {
      const aborted = controller.signal.aborted;
      update(id, () => ({ status: "error", error: aborted ? "Stopped." : (err as Error).message }));
    }
  }

  function cite(turn: number, marker: number) {
    setSelected({ turn, marker });
    document.getElementById(`src-${turn}-${marker}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  return (
    <div className="flex flex-col gap-6">
      {turns.length === 0 && (
        <div className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-slate-600">
          <h1 className="mb-2 text-xl font-semibold text-slate-900">Ask your documents</h1>
          <p>
            Answers are built only from passages retrieved from your <Link className="underline" href="/upload">uploaded
            files</Link>, and every claim links to its source. If nothing relevant is found, DocuChat says so instead of
            guessing.
          </p>
        </div>
      )}

      {turns.map((t) => {
        const markers = new Set(t.meta?.citations.map((c) => c.marker) ?? []);
        return (
          <section key={t.id} className="space-y-3" data-testid="turn">
            <div className="ml-auto max-w-[80%] rounded-lg bg-slate-900 px-4 py-2 text-white">{t.question}</div>
            <div
              className={`rounded-lg border bg-white p-4 ${t.meta?.refused ? "border-amber-300 bg-amber-50" : "border-slate-200"}`}
            >
              {t.answer ? (
                <Answer text={t.answer} validMarkers={markers} onCite={(m) => cite(t.id, m)} />
              ) : (
                t.status === "streaming" && <p className="animate-pulse text-slate-400">Searching your documents…</p>
              )}
              {t.status === "error" && <p className="mt-2 text-sm text-red-600">Error: {t.error}</p>}
              {t.meta && (
                <p className="mt-3 text-xs text-slate-500">
                  {t.meta.refused ? "Refused: " : ""}best similarity {t.meta.confidence.similarity.toFixed(3)} · term
                  coverage {(t.meta.confidence.term_coverage * 100).toFixed(0)}%
                </p>
              )}
            </div>
            {t.meta && t.meta.citations.length > 0 && (
              <ol className="grid items-start gap-2 sm:grid-cols-2">
                {t.meta.citations.map((c) => {
                  const active = selected?.turn === t.id && selected.marker === c.marker;
                  return (
                    <li
                      key={c.marker}
                      id={`src-${t.id}-${c.marker}`}
                      className={`rounded-lg border bg-white p-3 text-sm transition ${
                        active ? "border-indigo-500 ring-2 ring-indigo-200" : "border-slate-200"
                      }`}
                    >
                      <button
                        type="button"
                        className="flex w-full items-baseline gap-2 text-left"
                        onClick={() => setSelected(active ? null : { turn: t.id, marker: c.marker })}
                      >
                        <span className="rounded bg-indigo-100 px-1.5 text-xs font-semibold text-indigo-700">
                          {c.marker}
                        </span>
                        <span className="truncate font-medium">{c.filename}</span>
                        <span className="ml-auto shrink-0 text-xs text-slate-500">
                          {c.page ? `p. ${c.page} · ` : ""}chunk {c.chunk_index}
                        </span>
                      </button>
                      <p className={`mt-2 whitespace-pre-wrap text-slate-600 ${active ? "" : "line-clamp-3"}`}>
                        {c.text}
                      </p>
                    </li>
                  );
                })}
              </ol>
            )}
          </section>
        );
      })}

      <form onSubmit={ask} className="sticky bottom-4 flex gap-2 rounded-lg border border-slate-200 bg-white p-2 shadow-sm">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. How many days of annual leave do employees get?"
          className="flex-1 rounded-md px-3 py-2 outline-none"
          maxLength={2000}
          aria-label="Question"
        />
        <label className="flex items-center gap-1 text-sm text-slate-500">
          top&nbsp;k
          <select
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="rounded border border-slate-200 px-1 py-1"
          >
            {[3, 5, 8, 10].map((k) => (
              <option key={k}>{k}</option>
            ))}
          </select>
        </label>
        {busy ? (
          <button
            type="button"
            onClick={() => abortRef.current?.abort()}
            className="rounded-md bg-slate-200 px-4 py-2 text-sm font-medium"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!question.trim()}
            className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            Ask
          </button>
        )}
      </form>
    </div>
  );
}
