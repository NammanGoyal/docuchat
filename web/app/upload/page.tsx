"use client";

import Link from "next/link";
import { type DragEvent, useState } from "react";

import { uploadDocument } from "@/lib/api";

const ACCEPT = ".pdf,.md,.markdown,.txt";

interface Item {
  name: string;
  state: "uploading" | "added" | "duplicate" | "error";
  detail?: string;
}

export default function UploadPage() {
  const [items, setItems] = useState<Item[]>([]);
  const [dragging, setDragging] = useState(false);

  async function handleFiles(files: FileList | null) {
    if (!files) return;
    // Sequential on purpose: each upload is embedded synchronously by the API.
    for (const file of Array.from(files)) {
      setItems((xs) => [{ name: file.name, state: "uploading" }, ...xs]);
      const set = (patch: Partial<Item>) =>
        setItems((xs) => xs.map((x) => (x.name === file.name && x.state === "uploading" ? { ...x, ...patch } : x)));
      try {
        const doc = await uploadDocument(file);
        set(
          doc.duplicate
            ? { state: "duplicate", detail: `already indexed (same content as ${doc.filename})` }
            : { state: "added", detail: `${doc.num_chunks} chunks${doc.num_pages ? `, ${doc.num_pages} pages` : ""}` },
        );
      } catch (err) {
        set({ state: "error", detail: (err as Error).message });
      }
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    void handleFiles(e.dataTransfer.files);
  }

  const badge: Record<Item["state"], string> = {
    uploading: "bg-slate-100 text-slate-600",
    added: "bg-emerald-100 text-emerald-700",
    duplicate: "bg-amber-100 text-amber-700",
    error: "bg-red-100 text-red-700",
  };

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Upload documents</h1>
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-10 text-center ${
          dragging ? "border-indigo-500 bg-indigo-50" : "border-slate-300 bg-white"
        }`}
      >
        <span className="font-medium">Drop files here or click to choose</span>
        <span className="text-sm text-slate-500">PDF, Markdown or plain text, up to 10 MB each</span>
        <input
          type="file"
          multiple
          accept={ACCEPT}
          className="sr-only"
          onChange={(e) => {
            void handleFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </label>

      {items.length > 0 && (
        <ul className="divide-y divide-slate-200 rounded-lg border border-slate-200 bg-white">
          {items.map((item, i) => (
            <li key={`${item.name}-${i}`} className="flex items-center gap-3 px-4 py-2 text-sm">
              <span className="font-medium">{item.name}</span>
              <span className={`rounded px-2 py-0.5 text-xs ${badge[item.state]}`}>{item.state}</span>
              {item.detail && <span className="text-slate-500">{item.detail}</span>}
            </li>
          ))}
        </ul>
      )}

      <p className="text-sm text-slate-500">
        Uploaded files are listed on the <Link className="underline" href="/documents">Documents</Link> page. Then{" "}
        <Link className="underline" href="/">ask a question</Link>.
      </p>
    </div>
  );
}
