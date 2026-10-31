"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { type DocumentInfo, deleteDocument, listDocuments } from "@/lib/api";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    listDocuments()
      .then((d) => {
        setDocs(d);
        setError(null);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(refresh, [refresh]);

  async function remove(doc: DocumentInfo) {
    if (!window.confirm(`Delete ${doc.filename} and its ${doc.num_chunks} chunks?`)) return;
    try {
      await deleteDocument(doc.id);
      refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center">
        <h1 className="text-xl font-semibold">Documents</h1>
        <Link href="/upload" className="ml-auto rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white">
          Upload
        </Link>
      </div>
      {error && <p className="text-sm text-red-600">Error: {error}</p>}
      {docs === null && !error && <p className="text-slate-500">Loading…</p>}
      {docs?.length === 0 && (
        <p className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-slate-600">
          No documents yet. <Link className="underline" href="/upload">Upload one</Link> to get started.
        </p>
      )}
      {docs && docs.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-2">File</th>
                <th className="px-4 py-2">Type</th>
                <th className="px-4 py-2 text-right">Pages</th>
                <th className="px-4 py-2 text-right">Chunks</th>
                <th className="px-4 py-2 text-right">Size</th>
                <th className="px-4 py-2">Uploaded</th>
                <th className="px-4 py-2" />
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {docs.map((d) => (
                <tr key={d.id}>
                  <td className="px-4 py-2 font-medium" title={`sha256 ${d.content_hash}\n${d.embedding_model}`}>
                    {d.filename}
                  </td>
                  <td className="px-4 py-2">{d.kind}</td>
                  <td className="px-4 py-2 text-right">{d.num_pages ?? "—"}</td>
                  <td className="px-4 py-2 text-right">{d.num_chunks}</td>
                  <td className="px-4 py-2 text-right">{formatBytes(d.size_bytes)}</td>
                  <td className="px-4 py-2 text-slate-500">{new Date(d.created_at).toLocaleString()}</td>
                  <td className="px-4 py-2 text-right">
                    <button onClick={() => remove(d)} className="text-red-600 hover:underline" type="button">
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
