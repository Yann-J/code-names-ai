import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ApiSpinnerOverlay } from '../components/ApiSpinnerOverlay'
import { createGame } from '../api'
import './PlayNewPage.css'

function randomSeed(): number {
  return Math.floor(Math.random() * 0x7fffffff)
}

type Role = 'human' | 'ai'

function RoleSegment({
  value,
  onChange,
  disabled,
}: {
  value: Role
  onChange: (v: Role) => void
  disabled?: boolean
}) {
  return (
    <div className="role-segment" role="group" aria-label="Human or AI">
      <button
        type="button"
        className={`role-segment__btn${value === 'human' ? ' role-segment__btn--on' : ''}`}
        disabled={disabled}
        onClick={() => onChange('human')}
      >
        Human
      </button>
      <button
        type="button"
        className={`role-segment__btn${value === 'ai' ? ' role-segment__btn--on' : ''}`}
        disabled={disabled}
        onClick={() => onChange('ai')}
      >
        AI
      </button>
    </div>
  )
}

export function PlayNewPage() {
  const nav = useNavigate()
  const [seed, setSeed] = useState(randomSeed)
  const [risk, setRisk] = useState(0.5)
  const [redSpy, setRedSpy] = useState<Role>('ai')
  const [redGuess, setRedGuess] = useState<Role>('ai')
  const [blueSpy, setBlueSpy] = useState<Role>('ai')
  const [blueGuess, setBlueGuess] = useState<Role>('ai')
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setErr(null)
    setBusy(true)
    try {
      const { id } = await createGame({
        seed,
        risk,
        red_spy: redSpy,
        red_guess: redGuess,
        blue_spy: blueSpy,
        blue_guess: blueGuess,
      })
      nav(`/play/${id}`)
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="play-new">
      <ApiSpinnerOverlay visible={busy} message="Starting game…" />
      <h1 className="page-title">New game</h1>
      <p className="muted play-new__lead">Pick who plays for each team. A random board seed is used unless you change it below.</p>
      {err ? <div className="error-banner">{err}</div> : null}

      <form onSubmit={onSubmit}>
        <div className="role-matrix" aria-label="Team roles">
          <div className="role-matrix__corner" aria-hidden />
          <div className="role-matrix__colhead">Spymaster</div>
          <div className="role-matrix__colhead">Guesser</div>

          <div className="role-matrix__rowhead role-matrix__rowhead--red">Red</div>
          <div className="role-matrix__cell role-matrix__cell--red">
            <RoleSegment value={redSpy} onChange={setRedSpy} disabled={busy} />
          </div>
          <div className="role-matrix__cell role-matrix__cell--red">
            <RoleSegment value={redGuess} onChange={setRedGuess} disabled={busy} />
          </div>

          <div className="role-matrix__rowhead role-matrix__rowhead--blue">Blue</div>
          <div className="role-matrix__cell role-matrix__cell--blue">
            <RoleSegment value={blueSpy} onChange={setBlueSpy} disabled={busy} />
          </div>
          <div className="role-matrix__cell role-matrix__cell--blue">
            <RoleSegment value={blueGuess} onChange={setBlueGuess} disabled={busy} />
          </div>
        </div>

        <details className="play-new__advanced">
          <summary>Advanced — seed &amp; AI risk</summary>
          <div className="play-new__advanced-inner">
            <label className="play-new__field">
              <span>Board seed</span>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(+e.target.value)}
                disabled={busy}
              />
              <button
                type="button"
                className="play-new__shuffle"
                disabled={busy}
                onClick={() => setSeed(randomSeed())}
              >
                Randomize
              </button>
            </label>
            <label className="play-new__field play-new__field--risk risk-control">
              <span>Risk (0–1)</span>
              <div className="risk-control__row">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={risk}
                  onChange={(e) => setRisk(+e.target.value)}
                  disabled={busy}
                  aria-valuetext={`${risk.toFixed(2)}`}
                />
                <output className="risk-control__value">{risk.toFixed(2)}</output>
              </div>
            </label>
          </div>
        </details>

        <p className="form-actions">
          <button type="submit" className="btn-primary btn-primary--lg" disabled={busy}>
            Start game
          </button>
        </p>
      </form>
    </div>
  )
}
