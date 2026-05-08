import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ApiSpinnerOverlay } from '../components/ApiSpinnerOverlay'
import * as api from '../api'
import './PlayGamePage.css'

const SPY_KEY = 'codenamesSpyView'

function colorClass(c: string): string {
  const x = c.toLowerCase()
  if (x === 'red') return 'c-red'
  if (x === 'blue') return 'c-blue'
  if (x === 'neutral') return 'c-neutral'
  if (x === 'assassin') return 'c-assassin'
  return 'c-neutral'
}

function secretAttr(card: api.BoardCard, spyOn: boolean): string | undefined {
  if (!spyOn || card.revealed) return undefined
  const s = card.secret_color
  if (!s) return undefined
  return s.toLowerCase()
}

function guessOutcomeEmoji(guesserTeam: string, outcome: string | null): string {
  if (!outcome) return '❔'
  const o = outcome.toLowerCase()
  if (o === 'assassin') return '💀'
  if (o === 'neutral') return '⬜'
  const gt = guesserTeam.toUpperCase()
  const own = gt === 'RED' ? 'red' : gt === 'BLUE' ? 'blue' : ''
  if (o === own) return '✅'
  return '❌'
}

interface HistoryRound {
  team: string
  clue: api.CluePayload | null
  guesses: Array<{ word: string; outcome_color: string | null; team: string }>
}

function buildHistoryRounds(events: api.TurnEventPayload[]): HistoryRound[] {
  const rounds: HistoryRound[] = []
  let clueTeam = ''
  let clue: api.CluePayload | null = null
  const guesses: HistoryRound['guesses'] = []

  const pushRound = () => {
    if (clue || guesses.length > 0) {
      rounds.push({ team: clueTeam, clue, guesses: [...guesses] })
      guesses.length = 0
    }
  }

  for (const ev of events) {
    if (ev.kind === 'CLUE') {
      pushRound()
      clueTeam = ev.team
      clue = ev.clue
    } else if (ev.guess) {
      guesses.push({
        word: ev.guess,
        outcome_color: ev.outcome_color,
        team: ev.team,
      })
    }
  }
  pushRound()
  return rounds
}

function winnerReasonText(winReason: api.GameSnapshot['win_reason']): string {
  if (winReason === 'assassin') return 'by forcing the other team to reveal the assassin.'
  if (winReason === 'all_words') return 'by revealing all of their team words.'
  return 'when the game ended.'
}

export function PlayGamePage() {
  const { gameId } = useParams<{ gameId: string }>()
  const [spyOn, setSpyOn] = useState(() => localStorage.getItem(SPY_KEY) === '1')
  const [state, setState] = useState<api.GameSnapshot | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [clueWord, setClueWord] = useState('')
  const [clueCount, setClueCount] = useState(1)
  const [dismissedEndModalFor, setDismissedEndModalFor] = useState<string | null>(null)

  const load = useCallback(async () => {
    if (!gameId) return
    const s = await api.getGame(gameId, { includeSecretColors: spyOn })
    setState(s)
  }, [gameId, spyOn])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      if (!gameId) return
      setLoading(true)
      setErr(null)
      try {
        await load()
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : String(e))
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [gameId, spyOn, load])

  useEffect(() => {
    if (!gameId || !state?.ui.waiting_on_ai) return
    const id = setInterval(() => {
      api.getGame(gameId, { includeSecretColors: spyOn }).then(setState).catch(() => {})
    }, 1200)
    return () => clearInterval(id)
  }, [gameId, spyOn, state?.ui.waiting_on_ai])

  const historyRounds = useMemo(
    () => buildHistoryRounds(state?.turn_history ?? []),
    [state?.turn_history],
  )

  async function run<T>(fn: () => Promise<T>): Promise<void> {
    setBusy(true)
    setErr(null)
    try {
      const next = await fn()
      if (typeof next !== 'undefined') setState(next as api.GameSnapshot)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function onSpymaster(e: React.FormEvent) {
    e.preventDefault()
    if (!gameId) return
    await run(() => api.postSpymaster(gameId, { word: clueWord.trim(), count: clueCount }, spyOn))
    setClueWord('')
  }

  async function onCardGuess(word: string) {
    if (!gameId) return
    await run(() => api.postGuesses(gameId, [word], spyOn))
  }

  async function onEndTurn() {
    if (!gameId) return
    await run(() => api.postEndGuessTurn(gameId, spyOn))
  }

  if (!gameId) return <p>Missing game id.</p>

  const showSpinner = loading || busy

  if (err && !state && !loading) return <div className="error-banner">{err}</div>
  if (!state && loading) {
    return <ApiSpinnerOverlay visible message="Loading game…" />
  }
  if (!state) return null

  const phase = state.current_phase
  const clue = state.latest_clue
  const guesserIdle = phase !== 'GUESSER' || !clue || (clue.word === '' && clue.count === 0)
  const activeClue =
    phase === 'GUESSER' && clue && !(clue.word === '' && clue.count === 0)
  const teamKey = state.current_team === 'BLUE' ? 'blue' : 'red'

  const fx = state.guess_flash
  const showEndModal = state.is_over && dismissedEndModalFor !== state.id
  const winnerTeam = state.winner ? state.winner.toUpperCase() : 'Unknown'

  return (
    <div className={`play-root play-page${spyOn ? ' spy-on' : ''}`}>
      <ApiSpinnerOverlay visible={showSpinner} message={loading ? 'Loading game…' : 'Updating…'} />
      {fx?.kind === 'team' ? <div className="fx-celebrate" aria-hidden /> : null}
      {fx?.kind === 'other' || fx?.kind === 'assassin' ? <div className="fx-oops" aria-hidden /> : null}

      <h1 className="page-title">
        Game <code>{state.id}</code>
      </h1>
      <p className="session-meta muted">
        Session: <code>{state.id}</code> · Seed: <code>{state.seed}</code>
      </p>
      {err ? <div className="error-banner">{err}</div> : null}

      <div className="play-toolbar">
        <label className="spy-toggle">
          <input
            type="checkbox"
            checked={spyOn}
            onChange={(e) => {
              localStorage.setItem(SPY_KEY, e.target.checked ? '1' : '0')
              setSpyOn(e.target.checked)
            }}
          />
          Spymaster view (show hidden colors)
        </label>
        <span className="status-line">
          Team: <strong>{state.current_team}</strong> · Phase: <strong>{state.current_phase}</strong>
          {state.winner ? (
            <>
              {' '}
              · Winner: <strong>{state.winner}</strong>
            </>
          ) : null}
        </span>
      </div>

      <div
        className={`clue-panel${guesserIdle ? ' idle' : ''}${activeClue ? ` clue-panel--active clue-panel--team-${teamKey}` : ''}`}
      >
        <div className="clue-panel-inner">
          <div>
            <div className="clue-label">
              {phase === 'GUESSER' && clue && !(clue.word === '' && clue.count === 0)
                ? 'Active clue'
                : 'Clue'}
            </div>
            {phase === 'GUESSER' && clue && !(clue.word === '' && clue.count === 0) ? (
              <>
                <div className="clue-main">
                  {clue.word}
                  <span className="clue-count">{clue.count}</span>
                </div>
                {state.guesser_attempts_remaining != null ? (
                  <div className="attempts-badge">
                    Guesses remaining this clue: {state.guesser_attempts_remaining}
                  </div>
                ) : null}
              </>
            ) : clue && !(clue.word === '' && clue.count === 0) ? (
              <div className="clue-main muted" style={{ fontSize: '1.25rem', color: 'var(--muted)' }}>
                Last: {clue.word}
                <span className="clue-count">{clue.count}</span>
              </div>
            ) : (
              <div className="clue-main muted" style={{ fontSize: '1.1rem' }}>
                —
              </div>
            )}
          </div>
          {state.ui.show_end_turn ? (
            <button type="button" className="end-turn-btn" disabled={busy} onClick={() => onEndTurn()}>
              End turn
            </button>
          ) : null}
        </div>
      </div>

      <div className="board-wrap">
        {state.cards.map((c) => {
          const back = c.revealed && c.revealed_as ? colorClass(c.revealed_as) : colorClass('neutral')
          const sec = secretAttr(c, spyOn)
          const can = state.ui.can_click_guess && !c.revealed
          return (
            <div
              key={c.word}
              className={`card-slot${c.revealed ? ' is-revealed' : ''}`}
              data-secret={sec}
            >
              {can ? (
                <button
                  type="button"
                  className="card-hit"
                  disabled={busy}
                  aria-label={`Guess ${c.word}`}
                  onClick={() => onCardGuess(c.word)}
                />
              ) : null}
              <div className="card-inner">
                <div className="card-face card-front">
                  {c.word}
                  {c.revealed ? ' ✓' : ''}
                </div>
                <div className={`card-face card-back ${back}`}>{c.word}</div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="history-block">
        <h2>History</h2>
        {historyRounds.length === 0 ? (
          <p className="history-empty muted">No turns yet.</p>
        ) : (
          <div className="history-table-wrap">
            <table className="history-table">
              <thead>
                <tr>
                  <th scope="col">Clue</th>
                  <th scope="col">Guesses</th>
                </tr>
              </thead>
              <tbody>
                {historyRounds.map((round, i) => {
                  const side = round.team.toUpperCase() === 'BLUE' ? 'blue' : 'red'
                  return (
                    <tr key={i} className={`history-row history-row--${side}`}>
                      <td className="history-cell history-cell--clue">
                        {round.clue ? (
                          <div className="history-clue-inner">
                            <span className={`history-team-tag history-team-tag--${side}`}>{round.team}</span>
                            <span className="history-clue-word">{round.clue.word}</span>
                            <span className="history-clue-count">{round.clue.count}</span>
                          </div>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                      <td className="history-cell history-cell--guesses">
                        {round.guesses.length === 0 ? (
                          <span className="muted">—</span>
                        ) : (
                          <ul className="history-guess-list">
                            {round.guesses.map((g, j) => {
                              const oc = (g.outcome_color ?? '').toLowerCase()
                              const emoji = guessOutcomeEmoji(g.team, g.outcome_color)
                              return (
                                <li
                                  key={j}
                                  className={`history-guess-line history-guess-line--${oc || 'unknown'}`}
                                >
                                  <span className="history-guess-emoji" title={g.outcome_color ?? ''}>
                                    {emoji}
                                  </span>
                                  <span className="history-guess-word">{g.word}</span>
                                  {g.outcome_color ? (
                                    <span className="history-guess-color">{g.outcome_color}</span>
                                  ) : null}
                                </li>
                              )
                            })}
                          </ul>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {state.is_over ? (
        <p>
          <strong>Game over.</strong> <Link to="/play">New game</Link>
        </p>
      ) : state.ui.waiting_on_ai ? (
        <p className="status-line muted">Waiting on AI…</p>
      ) : state.ui.show_spymaster_form ? (
        <div className="action-panel">
          <h2>Your clue ({state.current_team} spymaster)</h2>
          <p className="muted" style={{ marginTop: 0 }}>
            If the guesser is AI, the clue word must exist in the embedding matrix (use common English words).
          </p>
          <form onSubmit={onSpymaster}>
            <label>
              Word{' '}
              <input
                type="text"
                value={clueWord}
                onChange={(e) => setClueWord(e.target.value)}
                pattern="[a-zA-Z\-]+"
                required
              />
            </label>
            <label>
              Count{' '}
              <input
                type="number"
                min={0}
                max={9}
                value={clueCount}
                onChange={(e) => setClueCount(+e.target.value)}
                required
              />
            </label>
            <button type="submit" className="btn-primary" disabled={busy} style={{ marginLeft: '0.5rem' }}>
              Give clue
            </button>
          </form>
        </div>
      ) : null}

      {showEndModal ? (
        <div className="endgame-modal-backdrop" role="presentation">
          <section className="endgame-modal" role="dialog" aria-modal="true" aria-labelledby="endgame-modal-title">
            <h2 id="endgame-modal-title">Game over</h2>
            <p className="endgame-modal__winner">
              <strong>{winnerTeam}</strong> wins {winnerReasonText(state.win_reason)}
            </p>
            <p className="muted endgame-modal__seed">
              Replay or analyze this same game with seed <code>{state.seed}</code>.
            </p>
            <p className="muted endgame-modal__hint">
              Use this seed in the New Game advanced section, or in Spymaster Analysis to inspect suggestions on this
              board.
            </p>
            <div className="endgame-modal__actions">
              <Link to="/play" className="btn-primary">
                Start a new game
              </Link>
              <Link to="/analysis" className="btn-secondary">
                Open analysis
              </Link>
              <button type="button" className="btn-secondary" onClick={() => setDismissedEndModalFor(state.id)}>
                Close
              </button>
            </div>
          </section>
        </div>
      ) : null}
    </div>
  )
}
