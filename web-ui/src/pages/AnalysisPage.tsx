import { useState } from 'react'
import { ApiSpinnerOverlay } from '../components/ApiSpinnerOverlay'
import { postAnalysis } from '../api'
import './AnalysisPage.css'

export function AnalysisPage() {
  const [seed, setSeed] = useState(0)
  const [risk, setRisk] = useState(0.5)
  const [data, setData] = useState<Awaited<ReturnType<typeof postAnalysis>> | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setErr(null)
    setBusy(true)
    try {
      setData(await postAnalysis({ seed, risk }))
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex))
      setData(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <ApiSpinnerOverlay visible={busy} message="Computing analysis…" />
      <h1 className="page-title">Spymaster analysis</h1>
      <p className="muted">Embedding-stage ranking on a board sampled from your cached game vocabulary.</p>
      {err ? <div className="error-banner">{err}</div> : null}
      <form onSubmit={onSubmit} className="form-grid" style={{ marginBottom: '1.5rem' }}>
        <label>
          Seed <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} />
        </label>
        <label className="risk-control risk-control--inline">
          <span className="risk-control__label">Risk</span>
          <div className="risk-control__row">
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={risk}
              onChange={(e) => setRisk(+e.target.value)}
              aria-valuetext={`${risk.toFixed(2)}`}
            />
            <output className="risk-control__value">{risk.toFixed(2)}</output>
          </div>
        </label>
        <button type="submit" className="btn-primary" disabled={busy} style={{ marginLeft: '0.5rem' }}>
          {busy ? 'Computing…' : 'Compute'}
        </button>
      </form>

      {data ? (
        <>
          <h2>Chosen</h2>
          {data.trace.chosen ? (
            <p>
              <strong>
                {data.trace.chosen.clue}
              </strong>{' '}
              (N={data.trace.chosen.n}) targets: {data.trace.chosen.targets.join(', ')}
            </p>
          ) : (
            <p>No legal clue (passed).</p>
          )}

          <h2>Top candidates</h2>
          <div className="table-scroll">
            <table className="analysis-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>clue</th>
                  <th>N</th>
                  <th>score</th>
                  <th>margin</th>
                  <th>embedding</th>
                  <th>LLM</th>
                  <th>reason</th>
                </tr>
              </thead>
              <tbody>
                {data.trace.top_candidates.map((c, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{c.clue}</td>
                    <td>{c.n}</td>
                    <td>{c.score.toFixed(3)}</td>
                    <td>{c.margin.toFixed(3)}</td>
                    <td>{c.embedding_score.toFixed(3)}</td>
                    <td>{c.llm_score != null ? c.llm_score.toFixed(3) : '—'}</td>
                    <td>{c.llm_reason ?? ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2>Board (first team {data.first_team})</h2>
          <div className="analysis-grid">
            {data.board.map((c) => (
              <div key={c.word} className={`analysis-card ${c.color}`}>
                {c.word}
              </div>
            ))}
          </div>
        </>
      ) : null}
    </div>
  )
}
