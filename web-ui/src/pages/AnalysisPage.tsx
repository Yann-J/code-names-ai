import { useState } from 'react'
import { ApiSpinnerOverlay } from '../components/ApiSpinnerOverlay'
import { postAnalysis } from '../api'
import './AnalysisPage.css'

export function AnalysisPage() {
  const [seed, setSeed] = useState(0)
  const [risk, setRisk] = useState(0.5)
  const [data, setData] = useState<Awaited<ReturnType<typeof postAnalysis>> | null>(null)
  const [activeTeam, setActiveTeam] = useState<'red' | 'blue'>('red')
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const currentTrace = data ? data.traces[activeTeam.toUpperCase()] ?? null : null

  function randomizeSeed() {
    setSeed(Math.floor(Math.random() * 2_147_483_647))
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setErr(null)
    setBusy(true)
    try {
      const nextData = await postAnalysis({ seed, risk })
      setData(nextData)
      if (nextData.traces[nextData.first_team]) {
        setActiveTeam(nextData.first_team === 'BLUE' ? 'blue' : 'red')
      }
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
        <button type="button" className="btn-secondary" onClick={randomizeSeed} disabled={busy}>
          Randomize seed
        </button>
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
          <h2>Board (first team {data.first_team})</h2>
          <div className="analysis-grid">
            {data.board.map((c) => (
              <div key={c.word} className={`analysis-card ${c.color}`}>
                {c.word}
              </div>
            ))}
          </div>

          <div className="analysis-tabs" role="tablist" aria-label="Recommendation team">
            <button
              type="button"
              role="tab"
              aria-selected={activeTeam === 'red'}
              className={activeTeam === 'red' ? 'analysis-tab active' : 'analysis-tab'}
              onClick={() => setActiveTeam('red')}
            >
              Red team
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTeam === 'blue'}
              className={activeTeam === 'blue' ? 'analysis-tab active' : 'analysis-tab'}
              onClick={() => setActiveTeam('blue')}
            >
              Blue team
            </button>
          </div>

          {currentTrace ? (
            <>
              <h2>Chosen</h2>
              {currentTrace.chosen ? (
                <p>
                  <strong>
                    {currentTrace.chosen.clue}
                  </strong>{' '}
                  (N={currentTrace.chosen.n}) targets: {currentTrace.chosen.targets.join(', ')}
                </p>
              ) : (
                <p>No legal clue (passed).</p>
              )}

              <h2>Top candidates</h2>
              <div className="table-scroll">
                <table className="analysis-table">
                  <thead>
                    <tr>
                      <th title="Rank among candidates for this team and board">#</th>
                      <th title="Single-word clue proposal">clue</th>
                      <th title="Number of intended target cards. Hover each value to see the target words.">N</th>
                      <th title="Final weighted score used for candidate ranking (after any reranking).">final score</th>
                      <th title="Raw expected reward estimate before weighting. Higher is generally better.">exp. reward</th>
                      <th title="Safety margin versus dangerous cards; higher means safer separation.">margin</th>
                      <th title="Embedding-only score before optional LLM reranking.">embedding</th>
                      <th title="LLM reranker score when enabled (otherwise empty).">LLM</th>
                      <th title="Short rationale returned by the LLM reranker.">reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentTrace.top_candidates.map((c, i) => (
                      <tr key={i}>
                        <td>{i + 1}</td>
                        <td>{c.clue}</td>
                        <td title={c.targets.join(', ')}>{c.n}</td>
                        <td>{c.score.toFixed(3)}</td>
                        <td>{c.components.expected_reward_raw.toFixed(3)}</td>
                        <td>{c.margin.toFixed(3)}</td>
                        <td>{c.embedding_score.toFixed(3)}</td>
                        <td>{c.llm_score != null ? c.llm_score.toFixed(3) : '—'}</td>
                        <td>{c.llm_reason ?? ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p className="muted">No recommendation available for this team.</p>
          )}
        </>
      ) : null}
    </div>
  )
}
