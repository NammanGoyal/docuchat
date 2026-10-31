"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { getHealth, type Health } from "@/lib/api";

const LINKS = [
  { href: "/", label: "Chat" },
  { href: "/upload", label: "Upload" },
  { href: "/documents", label: "Documents" },
];

export function Nav() {
  const pathname = usePathname();
  const [health, setHealth] = useState<Health | null>(null);
  const [apiDown, setApiDown] = useState(false);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setApiDown(true));
  }, []);

  return (
    <header className="border-b border-slate-200 bg-white">
      <nav className="mx-auto flex max-w-5xl items-center gap-6 px-4 py-3">
        <span className="text-lg font-semibold">DocuChat</span>
        <ul className="flex gap-1">
          {LINKS.map(({ href, label }) => (
            <li key={href}>
              <Link
                href={href}
                className={`rounded-md px-3 py-1.5 text-sm ${
                  pathname === href ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                {label}
              </Link>
            </li>
          ))}
        </ul>
        <span className="ml-auto text-xs text-slate-500" data-testid="api-status">
          {apiDown ? (
            <span className="text-red-600">API unreachable</span>
          ) : health ? (
            <>
              embedder <code>{health.embedder}</code> · answerer <code>{health.answerer}</code>
            </>
          ) : (
            "connecting…"
          )}
        </span>
      </nav>
    </header>
  );
}
