import type { ReactNode } from "react";

/** Render an answer: "> " lines become quotes, and [n] markers become buttons
 *  that select the matching citation. */
export function Answer({
  text,
  validMarkers,
  onCite,
}: {
  text: string;
  validMarkers: Set<number>;
  onCite: (marker: number) => void;
}) {
  const withCitations = (line: string, key: string): ReactNode[] =>
    line.split(/(\[\d+\])/g).map((part, i) => {
      const m = /^\[(\d+)\]$/.exec(part);
      const marker = m ? Number(m[1]) : NaN;
      if (m && validMarkers.has(marker)) {
        return (
          <button
            key={`${key}-${i}`}
            type="button"
            onClick={() => onCite(marker)}
            className="mx-0.5 rounded bg-indigo-100 px-1.5 text-xs font-semibold text-indigo-700 hover:bg-indigo-200"
            aria-label={`Show source ${marker}`}
          >
            {marker}
          </button>
        );
      }
      return <span key={`${key}-${i}`}>{part}</span>;
    });

  return (
    <div className="space-y-2 leading-relaxed">
      {text
        .split("\n")
        .filter((line) => line.trim() !== "")
        .map((line, i) =>
          line.startsWith("> ") ? (
            <blockquote key={i} className="border-l-4 border-indigo-300 pl-3 text-slate-700">
              {withCitations(line.slice(2), `q${i}`)}
            </blockquote>
          ) : (
            <p key={i}>{withCitations(line, `p${i}`)}</p>
          ),
        )}
    </div>
  );
}
