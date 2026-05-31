import { FormEvent, useState } from "react";
import { Loader2, Search, Sparkles } from "lucide-react";
import type { SearchResult } from "../types/graph";

type Props = {
  results: SearchResult[];
  loading: boolean;
  onSearch: (query: string) => void;
  onSelect: (result: SearchResult) => void;
};

export function SearchPanel({ results, loading, onSearch, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const normalizedQuery = query.trim();

  function submit(event: FormEvent) {
    event.preventDefault();
    if (normalizedQuery) onSearch(normalizedQuery);
  }

  return (
    <section className="panel search-panel" aria-label="Search">
      <div className="panel-head compact">
        <div className="panel-title">
          <Search size={16} />
          <h2>Entity Search</h2>
        </div>
        <span className="panel-meta">{results.length} hits</span>
      </div>

      <form className="search-box" onSubmit={submit}>
        <Search size={18} aria-hidden="true" />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Find startups, investors, people, topics"
          aria-label="Search entities"
        />
        <button
          className="icon-button solid"
          type="submit"
          aria-label="Run search"
          disabled={!normalizedQuery && !loading}
        >
          {loading ? <Loader2 size={18} className="spin" /> : <Search size={18} />}
        </button>
      </form>

      <div className="search-footnote">
        <Sparkles size={14} />
        <span>Tip: search investor names, startup aliases, or topic keywords.</span>
      </div>

      <div className="result-list">
        {results.map((result) => (
          <button key={result.id} className="result-row" onClick={() => onSelect(result)}>
            <span className={`type-dot ${result.type.toLowerCase()}`} />
            <span className="result-main">
              <strong>{result.name}</strong>
              <small>{result.type}</small>
            </span>
            <span className="score-pill">{formatScore(result.score)}</span>
          </button>
        ))}
        {!loading && results.length === 0 && (
          <p className="empty-text">Run ingest first, then search to jump into a focused subgraph.</p>
        )}
      </div>
    </section>
  );
}

function formatScore(score: number): string {
  if (!Number.isFinite(score)) return "-";
  if (score >= 100) return Math.round(score).toString();
  return score.toFixed(2);
}

