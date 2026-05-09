import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ApiSpinnerOverlay } from '../components/ApiSpinnerOverlay'
import { postAnalysis } from '../api'
import './AnalysisPage.css'

const MAX_ANALYSIS_SEED = 2_147_483_647

function nextRandomSeed(): number {
  return Math.floor(Math.random() * MAX_ANALYSIS_SEED)
}

function parseSeedParam(searchParams: URLSearchParams): number | undefined {
  const raw = searchParams.get('seed')
  if (raw == null || raw === '') return undefined
  const n = Number.parseInt(raw, 10)
  return Number.isFinite(n) ? n : undefined
}

function parseRiskParam(searchParams: URLSearchParams): number | undefined {
  const raw = searchParams.get('risk')
  if (raw == null || raw === '') return undefined
  const n = Number.parseFloat(raw)
  if (!Number.isFinite(n)) return undefined
  return Math.min(1, Math.max(0, n))
}

export function AnalysisPage() {
  const [searchParams] = useSearchParams()
  const [seed, setSeed] = useState(() => parseSeedParam(searchParams) ?? nextRandomSeed())
  const [risk, setRisk] = useState(() => parseRiskParam(searchParams) ?? 0.5)
  const [data, setData] = useState<Awaited<ReturnType<typeof postAnalysis>> | null>(null)
  const [activeTeam, setActiveTeam] = useState<'red' | 'blue'>('red')
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const currentTrace = data ? data.traces[activeTeam.toUpperCase()] ?? null : null

  useEffect(() => {
    const ps = parseSeedParam(searchParams)
    const pr = parseRiskParam(searchParams)
    if (ps !== undefined) setSeed(ps)
    if (pr !== undefined) setRisk(pr)
  }, [searchParams])

  function randomizeSeed() {
    setSeed(nextRandomSeed())
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
              {currentTrace.risk_snapshot != null ? (
                <p className="muted risk-snapshot" style={{ marginBottom: '0.75rem' }}>
                  {currentTrace.risk_snapshot.dynamic_enabled ? 'Dynamic risk' : 'Risk context'}: base{' '}
                  {currentTrace.risk_snapshot.base_risk.toFixed(2)} → effective{' '}
                  {currentTrace.risk_snapshot.effective_risk.toFixed(2)} (Δ objectives{' '}
                  {currentTrace.risk_snapshot.delta_objectives >= 0 ? '+' : ''}
                  {currentTrace.risk_snapshot.delta_objectives.toFixed(0)}: ours{' '}
                  {currentTrace.risk_snapshot.ours_unrevealed}, theirs{' '}
                  {currentTrace.risk_snapshot.theirs_unrevealed})
                </p>
              ) : null}
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
                      <th title="Number of intended target cards.">N</th>
                      <th title="Intended target words for this candidate clue.">target words</th>
                      <th title="Final ranking score after optional LLM rerank (blend of MC EV and LLM value).">final score</th>
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
                        <td>{c.n}</td>
                        <td>{c.targets.length > 0 ? c.targets.join(', ') : '—'}</td>
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
