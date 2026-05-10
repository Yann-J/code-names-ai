import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ApiSpinnerOverlay } from '../components/ApiSpinnerOverlay'
import * as api from '../api'
import './PlayGamePage.css'

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

function teamSide(team: string): 'red' | 'blue' {
  return team.toUpperCase() === 'BLUE' ? 'blue' : 'red'
}

function teamRoles(snapshot: api.GameSnapshot, team: string): api.TeamRoles {
  return teamSide(team) === 'blue' ? snapshot.roles.blue : snapshot.roles.red
}

interface TeamScore {
  side: 'red' | 'blue'
  revealed: number
  total: number
}

/** Standard board: ``first_team`` holds 9 cards, opponent 8. Reveals come from ``revealed_as``. */
function computeScores(s: api.GameSnapshot): { red: TeamScore; blue: TeamScore } {
  const first = s.first_team?.toUpperCase() === 'BLUE' ? 'blue' : 'red'
  const totals = { red: first === 'red' ? 9 : 8, blue: first === 'blue' ? 9 : 8 }
  let red = 0
  let blue = 0
  for (const c of s.cards) {
    if (!c.revealed_as) continue
    const r = c.revealed_as.toLowerCase()
    if (r === 'red') red++
    else if (r === 'blue') blue++
  }
  return {
    red: { side: 'red', revealed: red, total: totals.red },
    blue: { side: 'blue', revealed: blue, total: totals.blue },
  }
}

/** Fill missing ``secret_color`` on unrevealed cards from ``donor`` (same words). */
function afterDoublePaint(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
  })
}

function mergeUnrevealedSecrets(surface: api.GameSnapshot, donor: api.GameSnapshot | null): api.GameSnapshot {
  if (!donor) return surface
  const byWord = new Map(donor.cards.map((c) => [c.word.toLowerCase(), c.secret_color]))
  let changed = false
  const cards = surface.cards.map((c) => {
    if (c.revealed || c.secret_color != null) return c
    const sec = byWord.get(c.word.toLowerCase())
    if (sec == null) return c
    changed = true
    return { ...c, secret_color: sec }
  })
  return changed ? { ...surface, cards } : surface
}

function snapshotFreshness(s: api.GameSnapshot): [number, number, number] {
  let revealed = 0
  for (const c of s.cards) {
    if (c.revealed) revealed++
  }
  return [s.turn_history.length, revealed, s.live_mutation_seq ?? 0]
}

function freshnessCmp(
  a: [number, number, number],
  b: [number, number, number],
): number {
  for (let i = 0; i < 3; i++) {
    if (a[i] !== b[i]) return a[i] - b[i]
  }
  return 0
}

/** Prefer REST ``rest`` when it is strictly fresher than the role websocket snapshot (avoids stale WS masking a new reveal). */
function fresherSyncedSurface(rest: api.GameSnapshot, ws: api.GameSnapshot | null): api.GameSnapshot {
  if (!ws) return rest
  const r = snapshotFreshness(rest)
  const w = snapshotFreshness(ws)
  const c = freshnessCmp(r, w)
  if (c > 0) return rest
  if (c < 0) return ws
  return rest
}

function connectLiveRoleWs(
  kind: 'guess' | 'spy',
  token: string,
  setSnap: (s: api.GameSnapshot) => void,
): () => void {
  let cancelled = false
  let ws: WebSocket | null = null
  const startId = window.setTimeout(() => {
    if (cancelled) return
    try {
      ws = new WebSocket(api.liveWsUrl(kind, token))
    } catch {
      return
    }
    ws.onmessage = (ev) => {
      if (cancelled) return
      try {
        const msg = JSON.parse(ev.data) as { state?: api.GameSnapshot; snapshot?: { state: api.GameSnapshot } }
        const st = msg.state ?? msg.snapshot?.state
        if (st) setSnap(st)
      } catch {
        /* ignore malformed */
      }
    }
  }, 0)
  return () => {
    cancelled = true
    window.clearTimeout(startId)
    ws?.close()
  }
}

const BOOTSTRAP_ICONS = 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons'

function selectNodeText(el: HTMLElement) {
  const range = document.createRange()
  range.selectNodeContents(el)
  const sel = window.getSelection()
  sel?.removeAllRanges()
  sel?.addRange(range)
}

function RemoteLinksModal({
  open,
  onClose,
  sessionId,
  onLivePrepared,
}: {
  open: boolean
  onClose: () => void
  sessionId: string
  onLivePrepared: (r: api.CreateLiveRoomResponse) => void
}) {
  const [live, setLive] = useState<api.CreateLiveRoomResponse | null>(null)
  const [sharing, setSharing] = useState(false)
  const [shareErr, setShareErr] = useState<string | null>(null)
  const [copiedWhich, setCopiedWhich] = useState<'guesser' | 'spymaster' | null>(null)
  const copyFeedbackTimeoutRef = useRef<ReturnType<typeof window.setTimeout> | null>(null)

  useEffect(() => {
    if (!open) {
      if (copyFeedbackTimeoutRef.current) {
        window.clearTimeout(copyFeedbackTimeoutRef.current)
        copyFeedbackTimeoutRef.current = null
      }
      return
    }
    queueMicrotask(() => {
      setCopiedWhich(null)
    })
  }, [open])

  useEffect(
    () => () => {
      if (copyFeedbackTimeoutRef.current) window.clearTimeout(copyFeedbackTimeoutRef.current)
    },
    [],
  )

  async function refresh() {
    setSharing(true)
    setShareErr(null)
    try {
      const r = await api.createLiveRoom({ session_id: sessionId })
      setLive(r)
      onLivePrepared(r)
    } catch (e) {
      setShareErr(e instanceof Error ? e.message : String(e))
    } finally {
      setSharing(false)
    }
  }

  async function copyLine(which: 'guesser' | 'spymaster', text: string | null) {
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
      if (copyFeedbackTimeoutRef.current) window.clearTimeout(copyFeedbackTimeoutRef.current)
      setCopiedWhich(which)
      copyFeedbackTimeoutRef.current = window.setTimeout(() => {
        setCopiedWhich(null)
        copyFeedbackTimeoutRef.current = null
      }, 1800)
    } catch {
      /* clipboard denied or unavailable */
    }
  }

  if (!open) return null

  return (
    <div className="remote-links-modal-root" role="presentation">
      <button type="button" className="remote-links-modal-scrim" aria-label="Dismiss" onClick={onClose} />
      <div className="remote-links-modal-dialog" role="dialog" aria-modal="true" aria-labelledby="remote-links-title">
        <div className="remote-links-modal-header">
          <h2 id="remote-links-title" className="remote-links-modal-title">
            Remote links
          </h2>
          <button type="button" className="remote-links-modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <p className="muted remote-links-modal-intro">Play this game with remote human players. You can share either a spymaster link or a guesser link</p>
        <div className="remote-links-modal-actions">
          <button type="button" className="btn-secondary" disabled={sharing} onClick={() => refresh()}>
            {live ? 'Refresh links' : 'Generate links'}
          </button>
        </div>
        {shareErr ? <p className="error-banner">{shareErr}</p> : null}
        {live ? (
          <ul className="remote-links-list muted">
            {live.guesser_url ? (
              <li>
                Guesser:{' '}
                <code
                  className="remote-url"
                  title="Click to select"
                  onClick={(e) => selectNodeText(e.currentTarget)}
                >
                  {live.guesser_url}
                </code>{' '}
                <button
                  type="button"
                  className={`btn-copy-url${copiedWhich === 'guesser' ? ' btn-copy-url--copied' : ''}`}
                  onClick={() => copyLine('guesser', live.guesser_url)}
                  title="Copy link"
                  aria-label="Copy operative link"
                >
                  <img
                    src={`${BOOTSTRAP_ICONS}/${copiedWhich === 'guesser' ? 'check-lg.svg' : 'clipboard.svg'}`}
                    className="btn-copy-url-icon"
                    alt=""
                    aria-hidden
                  />
                </button>
              </li>
            ) : (
              <li>No guesser link — add a human guesser to enable it.</li>
            )}
            {live.spymaster_url ? (
              <li>
                Spymaster:{' '}
                <code
                  className="remote-url"
                  title="Click to select"
                  onClick={(e) => selectNodeText(e.currentTarget)}
                >
                  {live.spymaster_url}
                </code>{' '}
                <button
                  type="button"
                  className={`btn-copy-url${copiedWhich === 'spymaster' ? ' btn-copy-url--copied' : ''}`}
                  onClick={() => copyLine('spymaster', live.spymaster_url)}
                  title="Copy link"
                  aria-label="Copy captain link"
                >
                  <img
                    src={`${BOOTSTRAP_ICONS}/${copiedWhich === 'spymaster' ? 'check-lg.svg' : 'clipboard.svg'}`}
                    className="btn-copy-url-icon"
                    alt=""
                    aria-hidden
                  />
                </button>
              </li>
            ) : (
              <li>No spymaster link — add a human spymaster to enable it.</li>
            )}
          </ul>
        ) : null}
      </div>
    </div>
  )
}

function LastAiSpymasterBody({ trace }: { trace: api.AnalysisTracePayload }) {
  return (
    <>
      {trace.risk_snapshot != null ? (
        <p className="muted last-ai-modal__risk">
          {trace.risk_snapshot.dynamic_enabled ? 'Dynamic risk' : 'Risk context'}: base{' '}
          {trace.risk_snapshot.base_risk.toFixed(2)} → effective {trace.risk_snapshot.effective_risk.toFixed(2)} (Δ
          objectives {trace.risk_snapshot.delta_objectives >= 0 ? '+' : ''}
          {trace.risk_snapshot.delta_objectives.toFixed(0)}: ours {trace.risk_snapshot.ours_unrevealed}, theirs{' '}
          {trace.risk_snapshot.theirs_unrevealed})
        </p>
      ) : null}
      <h3 className="last-ai-modal__h3">Chosen</h3>
      {trace.chosen ? (
        <p>
          <strong>{trace.chosen.clue}</strong> (N={trace.chosen.n}) targets: {trace.chosen.targets.join(', ')}
        </p>
      ) : (
        <p>No legal clue (passed).</p>
      )}
      <h3 className="last-ai-modal__h3">Top candidates</h3>
      <div className="table-scroll">
        <table className="analysis-table">
          <thead>
            <tr>
              <th>#</th>
              <th>clue</th>
              <th>N</th>
              <th>target words</th>
              <th>final score</th>
              <th>exp. reward</th>
              <th>margin</th>
              <th>embedding</th>
              <th>LLM</th>
              <th>reason</th>
            </tr>
          </thead>
          <tbody>
            {trace.top_candidates.map((c, i) => (
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
  )
}

function LastAiGuesserBody({ trace }: { trace: api.GuesserTracePayload }) {
  const { stop_policy: sp } = trace
  return (
    <>
      <p className="last-ai-modal__clue-line">
        Clue <strong>{trace.clue_word}</strong> (N={trace.clue_count}) · picks:{' '}
        <strong>{trace.guesses.length > 0 ? trace.guesses.join(', ') : '—'}</strong>
      </p>
      <p className="muted">
        Stop: {trace.stop_reason}
        {trace.bonus_attempted ? ' · bonus pick considered' : ''}
      </p>
      {trace.risk_snapshot != null ? (
        <p className="muted last-ai-modal__risk">
          {trace.risk_snapshot.dynamic_enabled ? 'Dynamic risk' : 'Risk context'}: base{' '}
          {trace.risk_snapshot.base_risk.toFixed(2)} → effective {trace.risk_snapshot.effective_risk.toFixed(2)}
        </p>
      ) : null}
      <h3 className="last-ai-modal__h3">Stop policy</h3>
      <p className="muted last-ai-modal__policy">
        confidence floor {sp.confidence_floor.toFixed(2)}, bonus gap threshold {sp.bonus_gap_threshold.toFixed(2)},
        risk knob {sp.risk.toFixed(2)}
      </p>
      <h3 className="last-ai-modal__h3">Scored cards</h3>
      <div className="table-scroll">
        <table className="analysis-table">
          <thead>
            <tr>
              <th>rank</th>
              <th>word</th>
              <th>similarity</th>
              <th>score</th>
              <th>picked</th>
              <th>bonus</th>
              <th>LLM</th>
              <th>reason</th>
            </tr>
          </thead>
          <tbody>
            {trace.candidates.map((c) => (
              <tr key={c.word} className={c.committed ? 'last-ai-modal__row--committed' : undefined}>
                <td>{c.rank + 1}</td>
                <td>{c.word}</td>
                <td>{c.similarity.toFixed(3)}</td>
                <td>{c.score.toFixed(3)}</td>
                <td>{c.committed ? 'yes' : ''}</td>
                <td>{c.is_bonus ? 'yes' : ''}</td>
                <td>{c.llm_score != null ? c.llm_score.toFixed(3) : '—'}</td>
                <td>{c.llm_reason ?? ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

export function PlayGamePage() {
  const params = useParams<{ gameId?: string; band?: string; token?: string }>()
  const gameId = params.gameId
  const liveToken = params.token
  const remoteBand = params.band === 'guess' || params.band === 'spy' ? params.band : null
  const isRemote = !!(liveToken && remoteBand)

  const [spyCaptainView, setSpyCaptainView] = useState(false)

  const [state, setState] = useState<api.GameSnapshot | null>(null)
  const [liveRoomInfo, setLiveRoomInfo] = useState<api.CreateLiveRoomResponse | null>(null)
  const [liveGuessSnap, setLiveGuessSnap] = useState<api.GameSnapshot | null>(null)
  const [liveSpySnap, setLiveSpySnap] = useState<api.GameSnapshot | null>(null)
  const [remoteModalOpen, setRemoteModalOpen] = useState(false)
  const [lastAiModalOpen, setLastAiModalOpen] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [clueWord, setClueWord] = useState('')
  const [clueCount, setClueCount] = useState(1)
  const [dismissedEndModalFor, setDismissedEndModalFor] = useState<string | null>(null)
  /** Remote captain link only: show key-card tint on unrevealed words; off = same board as operatives see. */
  const [remoteCaptainKeyView, setRemoteCaptainKeyView] = useState(true)
  /** Defer card transform transitions until after mount so already-revealed cells do not animate. */
  const [boardMotionReady, setBoardMotionReady] = useState(false)

  const pollStateRef = useRef(state)
  useEffect(() => {
    pollStateRef.current = state
  }, [state])

  useEffect(() => {
    const id = requestAnimationFrame(() => setBoardMotionReady(true))
    return () => cancelAnimationFrame(id)
  }, [])

  const prevPhaseSyncRef = useRef<string | null>(null)

  useEffect(() => {
    prevPhaseSyncRef.current = null
  }, [gameId, isRemote])

  useEffect(() => {
    if (!isRemote || remoteBand !== 'spy') return
    const id = window.setTimeout(() => setRemoteCaptainKeyView(true), 0)
    return () => window.clearTimeout(id)
  }, [isRemote, remoteBand, liveToken])

  /** Ensure live room + WS tokens for this session so the host stays in sync without opening the share modal. */
  useEffect(() => {
    if (isRemote || !gameId) return
    let cancelled = false
    const bootId = window.setTimeout(() => {
      setLiveRoomInfo(null)
      setLiveGuessSnap(null)
      setLiveSpySnap(null)
      ;(async () => {
        try {
          const r = await api.createLiveRoom({ session_id: gameId })
          if (!cancelled) setLiveRoomInfo(r)
        } catch {
          /* offline / no vocab — local play still works without live sync */
        }
      })()
    }, 0)
    return () => {
      cancelled = true
      window.clearTimeout(bootId)
    }
  }, [gameId, isRemote])

  useEffect(() => {
    const ph = state?.current_phase
    if (isRemote || ph == null || !state) return
    if (prevPhaseSyncRef.current === ph) return
    prevPhaseSyncRef.current = ph
    const snap = state
    queueMicrotask(() => {
      if (ph === 'SPYMASTER') {
        const spymaster = teamRoles(snap, snap.current_team).spymaster
        if (spymaster === 'human') setSpyCaptainView(true)
      } else if (ph === 'GUESSER') {
        const guesser = teamRoles(snap, snap.current_team).guesser
        if (guesser === 'human') setSpyCaptainView(false)
      } else {
        setSpyCaptainView(false)
      }
    })
  }, [isRemote, state])

  const displayState = useMemo((): api.GameSnapshot | null => {
    if (!state) return null
    if (isRemote) return state
    if (!spyCaptainView) return fresherSyncedSurface(state, liveGuessSnap)
    const primary = fresherSyncedSurface(state, liveSpySnap)
    // REST ``state`` and spy WS can diverge; operative WS never carries secrets — merge so key card tint works.
    return mergeUnrevealedSecrets(mergeUnrevealedSecrets(primary, state), liveSpySnap)
  }, [state, liveGuessSnap, liveSpySnap, isRemote, spyCaptainView])

  const guessWsToken = useMemo(
    () => api.parseLiveWsToken(liveRoomInfo?.guesser_websocket_url, 'guess'),
    [liveRoomInfo?.guesser_websocket_url],
  )

  const spyWsToken = useMemo(
    () => api.parseLiveWsToken(liveRoomInfo?.spymaster_websocket_url, 'spy'),
    [liveRoomInfo?.spymaster_websocket_url],
  )

  const effectiveSpyOn = useMemo(() => {
    if (isRemote) return remoteBand === 'spy' && remoteCaptainKeyView
    return spyCaptainView
  }, [isRemote, remoteBand, remoteCaptainKeyView, spyCaptainView])

  /** True if REST or live role WS shows AI in progress (host ``state`` can lag behind WS-only updates). */
  const anyWaitingPoll = useMemo(() => {
    if (isRemote) return !!state?.ui?.waiting_on_ai
    return !!(
      state?.ui?.waiting_on_ai ||
      liveGuessSnap?.ui?.waiting_on_ai ||
      liveSpySnap?.ui?.waiting_on_ai
    )
  }, [isRemote, state?.ui?.waiting_on_ai, liveGuessSnap?.ui?.waiting_on_ai, liveSpySnap?.ui?.waiting_on_ai])

  const load = useCallback(async () => {
    if (!gameId || isRemote) return
    const lite = await api.getGame(gameId, { includeSecretColors: false })
    if (lite.current_phase === 'SPYMASTER') {
      setState(await api.getGame(gameId, { includeSecretColors: true }))
    } else {
      setState(lite)
    }
  }, [gameId, isRemote])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      if (isRemote || !gameId) return
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
  }, [gameId, load, isRemote])

  useEffect(() => {
    if (!isRemote || !liveToken || !remoteBand) return
    let cancelled = false
    let ws: WebSocket | null = null
    const startId = window.setTimeout(() => {
      if (cancelled) return
      setLoading(true)
      setErr(null)
      /** Only the first snapshot ends the initial connect spinner; later messages must not clear ``loading`` during HTTP advance-ai / guess flows. */
      let endedInitialRemoteLoad = false
      const url = api.liveWsUrl(remoteBand, liveToken)
      try {
        ws = new WebSocket(url)
      } catch {
        setErr('Could not open live connection')
        setLoading(false)
        return
      }
      ws.onmessage = (ev) => {
        if (cancelled) return
        try {
          const msg = JSON.parse(ev.data) as { state?: api.GameSnapshot; snapshot?: { state: api.GameSnapshot } }
          const st = msg.state ?? msg.snapshot?.state
          if (st) {
            setState(st)
            if (!endedInitialRemoteLoad) {
              endedInitialRemoteLoad = true
              setLoading(false)
            }
          }
        } catch {
          if (!cancelled) setErr('Invalid update from server')
        }
      }
      ws.onerror = () => {
        if (!cancelled) {
          setErr('Live connection error')
          setLoading(false)
        }
      }
      ws.onclose = () => {
        if (!cancelled) setLoading(false)
      }
    }, 0)
    return () => {
      cancelled = true
      window.clearTimeout(startId)
      ws?.close()
    }
  }, [isRemote, remoteBand, liveToken])

  useEffect(() => {
    if (isRemote || !guessWsToken) {
      const clearId = window.setTimeout(() => setLiveGuessSnap(null), 0)
      return () => window.clearTimeout(clearId)
    }
    return connectLiveRoleWs('guess', guessWsToken, setLiveGuessSnap)
  }, [isRemote, guessWsToken])

  useEffect(() => {
    if (isRemote || !spyWsToken) {
      const clearId = window.setTimeout(() => setLiveSpySnap(null), 0)
      return () => window.clearTimeout(clearId)
    }
    return connectLiveRoleWs('spy', spyWsToken, setLiveSpySnap)
  }, [isRemote, spyWsToken])

  useEffect(() => {
    if (isRemote || !gameId || !anyWaitingPoll) return
    const id = window.setInterval(() => {
      if (!gameId) return
      const cur = pollStateRef.current
      const sec = cur?.current_phase === 'SPYMASTER'
      api.getGame(gameId, { includeSecretColors: sec }).then(setState).catch(() => {})
    }, 1200)
    return () => window.clearInterval(id)
  }, [gameId, isRemote, anyWaitingPoll])

  /** Key card view needs ``secret_color`` on unrevealed cells; REST omits them outside optional fetches / spy WS. */
  useEffect(() => {
    if (isRemote || !gameId || !state || state.is_over || !spyCaptainView) return
    const missing = state.cards.some((c) => !c.revealed && c.secret_color == null)
    if (!missing) return
    let cancelled = false
    const t = window.setTimeout(() => {
      api
        .getGame(gameId, { includeSecretColors: true })
        .then((s) => {
          if (!cancelled) setState(s)
        })
        .catch(() => {})
    }, 0)
    return () => {
      cancelled = true
      window.clearTimeout(t)
    }
  }, [isRemote, gameId, spyCaptainView, state])

  /** Captain overlay needs REST ``include_secret_colors`` (and ``last_ai_analysis``); operative loads may omit both. */
  useEffect(() => {
    if (isRemote || !gameId || !spyCaptainView) return
    let cancelled = false
    void api.getGame(gameId, { includeSecretColors: true }).then((full) => {
      if (cancelled) return
      setState((prev) => (prev ? fresherSyncedSurface(full, prev) : full))
    })
    return () => {
      cancelled = true
    }
  }, [isRemote, gameId, spyCaptainView])

  const historyRounds = useMemo(
    () => buildHistoryRounds(displayState?.turn_history ?? []),
    [displayState?.turn_history],
  )

  const analysisHref = useMemo(() => {
    if (!displayState) return '/analysis'
    return `/analysis?${new URLSearchParams({ seed: String(displayState.seed), risk: String(displayState.risk) }).toString()}`
  }, [displayState])

  function includeSecretsForApi(): boolean {
    const s = pollStateRef.current
    return !!(s && !s.is_over && s.current_phase === 'SPYMASTER')
  }

  async function run<T>(fn: () => Promise<T>): Promise<void> {
    setLoading(true)
    setErr(null)
    try {
      const next = await fn()
      if (typeof next !== 'undefined') setState(next as api.GameSnapshot)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      queueMicrotask(() => setLoading(false))
    }
  }

  async function onSpymaster(e: React.FormEvent) {
    e.preventDefault()
    const body = { word: clueWord.trim(), count: clueCount }
    if (isRemote) {
      if (!liveToken || remoteBand !== 'spy') return
      await run(() => api.postLiveSpymaster(liveToken, body))
    } else {
      if (!gameId) return
      await run(() => api.postSpymaster(gameId, body, includeSecretsForApi()))
    }
    setClueWord('')
  }

  async function onCardGuess(word: string) {
    setErr(null)
    try {
      if (isRemote) {
        if (!liveToken) return
        const afterGuess = await api.postLiveGuesses(liveToken, [word])
        setState(afterGuess)
        await afterDoublePaint()
        setLoading(true)
        const finalSnap = await api.postLiveAdvanceAi(liveToken)
        setState(finalSnap)
      } else {
        if (!gameId) return
        const incGuess = includeSecretsForApi()
        const afterGuess = await api.postGuesses(gameId, [word], incGuess)
        setState(afterGuess)
        await afterDoublePaint()
        setLoading(true)
        const finalSnap = await api.postAdvanceAi(gameId, spyCaptainView)
        setState(finalSnap)
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  async function onEndTurn() {
    if (isRemote) {
      if (!liveToken) return
      setErr(null)
      try {
        const afterEnd = await api.postLiveEndGuessTurn(liveToken)
        setState(afterEnd)
        await afterDoublePaint()
        setLoading(true)
        const finalSnap = await api.postLiveAdvanceAi(liveToken)
        setState(finalSnap)
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e))
      } finally {
        setLoading(false)
      }
      return
    }
    if (!gameId) return
    setErr(null)
    try {
      const incGuess = includeSecretsForApi()
      const afterEnd = await api.postEndGuessTurn(gameId, incGuess)
      setState(afterEnd)
      await afterDoublePaint()
      setLoading(true)
      const finalSnap = await api.postAdvanceAi(gameId, spyCaptainView)
      setState(finalSnap)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  if (!isRemote && !gameId) return <p>Missing game id.</p>
  if (isRemote && (!liveToken || !remoteBand)) return <p>Invalid remote link.</p>

  if (err && !state && !loading) return <div className="error-banner">{err}</div>
  if (!state && loading) {
    return <ApiSpinnerOverlay visible message="Loading game…" />
  }
  if (!state || !displayState) return null

  const phase = displayState.current_phase
  const clue = displayState.latest_clue
  const guesserIdle = phase !== 'GUESSER' || !clue || (clue.word === '' && clue.count === 0)
  const activeClue =
    phase === 'GUESSER' && clue && !(clue.word === '' && clue.count === 0)
  const teamKey = teamSide(displayState.current_team)
  const showSpymasterForm = displayState.ui.show_spymaster_form
  /** Remote operative link never submits clues; only the captain link shows the form. */
  const showSpymasterFormUi = showSpymasterForm && (!isRemote || remoteBand === 'spy')
  const cluePanelIdle = guesserIdle && !showSpymasterFormUi
  const cluePanelActive = activeClue || showSpymasterFormUi

  const fx = displayState.guess_flash
  const showEndModal = displayState.is_over && dismissedEndModalFor !== displayState.id
  const winnerTeam = displayState.winner ? displayState.winner.toUpperCase() : 'Unknown'
  const aiBusy = !!(
    displayState?.ui?.waiting_on_ai ||
    (!isRemote && (liveGuessSnap?.ui?.waiting_on_ai || liveSpySnap?.ui?.waiting_on_ai))
  )
  const blockingUi = loading || aiBusy

  return (
    <div
      className={`play-root play-page${effectiveSpyOn ? ' spy-on' : ''}`}
      aria-busy={blockingUi || undefined}
    >
      <RemoteLinksModal
        open={remoteModalOpen && !isRemote}
        onClose={() => setRemoteModalOpen(false)}
        sessionId={displayState.id}
        onLivePrepared={setLiveRoomInfo}
      />
      <ApiSpinnerOverlay
        visible={loading || aiBusy}
        message="Waiting for AI…"
      />
      {fx?.kind === 'team' ? <div className="fx-celebrate" aria-hidden /> : null}
      {fx?.kind === 'other' || fx?.kind === 'assassin' ? <div className="fx-oops" aria-hidden /> : null}

      <div className="play-title-row">
        <h1 className="page-title">
          {isRemote ? (
            remoteBand === 'guess' ? (
              <>Remote · operative</>
            ) : (
              <>Remote · captain · session <code>{displayState.id}</code></>
            )
          ) : (
            <>
              Game <code>{displayState.id}</code>
            </>
          )}
        </h1>
        <div className="play-title-right">
          {(() => {
            const { red, blue } = computeScores(displayState)
            return (
              <div className="score-chips" aria-label="Team scores">
                <span className="score-chip score-chip--blue">
                  <span className="score-chip__label">BLUE</span>
                  <span className="score-chip__value">
                    {blue.revealed}/{blue.total}
                  </span>
                </span>
                <span className="score-chip score-chip--red">
                  <span className="score-chip__label">RED</span>
                  <span className="score-chip__value">
                    {red.revealed}/{red.total}
                  </span>
                </span>
              </div>
            )
          })()}
          {!isRemote ? (
            <button
              type="button"
              className="btn-remote-links"
              onClick={() => setRemoteModalOpen(true)}
              title="Remote play links"
              aria-label="Remote play links"
            >
              <img
                src={`${BOOTSTRAP_ICONS}/cloud.svg`}
                className="btn-remote-links-icon"
                alt=""
                aria-hidden
              />
            </button>
          ) : null}
        </div>
      </div>
      {err ? <div className="error-banner">{err}</div> : null}

      <div className="play-toolbar">
        {isRemote ? (
          <>
            <span className="muted status-line remote-role-hint">
              {remoteBand === 'guess' ? 'Operative view' : 'Captain view'}
            </span>
            {remoteBand === 'spy' ? (
              <>
                <label className="spy-toggle">
                  <input
                    type="checkbox"
                    checked={remoteCaptainKeyView}
                    onChange={(e) => setRemoteCaptainKeyView(e.target.checked)}
                  />
                  Reveal Cards
                </label>
              </>
            ) : null}
          </>
        ) : (
          <>
            <label className="spy-toggle">
              <input
                type="checkbox"
                checked={spyCaptainView}
                onChange={(e) => setSpyCaptainView(e.target.checked)}
              />
              Reveal Cards
            </label>
          </>
        )}
        <span className="status-line">
          Team: <span className={`history-team-tag history-team-tag--${teamKey}`}>{displayState.current_team}</span> ·
          Phase: <strong>{displayState.current_phase}</strong>
          {displayState.winner ? (
            <>
              {' '}
              · Winner:{' '}
              <span className={`history-team-tag history-team-tag--${teamSide(displayState.winner)}`}>{displayState.winner}</span>
            </>
          ) : null}
        </span>
      </div>

      <div
        className={`clue-panel${cluePanelIdle ? ' idle' : ''}${cluePanelActive ? ` clue-panel--active clue-panel--team-${teamKey}` : ''}`}
      >
        <div className={`clue-panel-inner${showSpymasterFormUi ? ' clue-panel-inner--spymaster' : ''}`}>
          {showSpymasterFormUi ? (
            <div className="clue-spymaster-block">
              <div className="clue-label clue-label--row">
                Your clue
                <span className={`history-team-tag history-team-tag--${teamKey}`}>{displayState.current_team} spymaster</span>
              </div>
              <form className="clue-spymaster-form" onSubmit={onSpymaster}>
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
                <button type="submit" className="btn-primary" disabled={blockingUi}>
                  Give clue
                </button>
              </form>
            </div>
          ) : (
            <>
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
                    {displayState.guesser_attempts_remaining != null ? (
                      <div className="attempts-badge">
                        Guesses remaining this clue: {displayState.guesser_attempts_remaining}
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
              {displayState.ui.show_end_turn ? (
                <button type="button" className="end-turn-btn" disabled={blockingUi} onClick={() => onEndTurn()}>
                  End turn
                </button>
              ) : null}
            </>
          )}
        </div>
      </div>

      <div className="board-outer">
        <div className={`board-wrap${boardMotionReady ? ' board-wrap--motion' : ''}`}>
          {displayState.cards.map((c) => {
            const back = c.revealed && c.revealed_as ? colorClass(c.revealed_as) : colorClass('neutral')
            const sec = secretAttr(c, effectiveSpyOn)
            const can =
              displayState.ui.can_click_guess &&
              !c.revealed &&
              (!isRemote || remoteBand === 'guess')
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
                    disabled={blockingUi}
                    aria-label={`Guess ${c.word}`}
                    onClick={() => onCardGuess(c.word)}
                  />
                ) : null}
                <div className="card-inner">
                  <div className="card-face card-front">
                    <span className="card-word">
                      {c.word}
                      {c.revealed ? ' ✓' : ''}
                    </span>
                  </div>
                  <div className={`card-face card-back ${back}`}>
                    <span className="card-word">{c.word}</span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
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
                  const side = teamSide(round.team)
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

      {displayState.ui.waiting_on_ai ? <p className="status-line muted">Waiting on AI…</p> : null}

      {isRemote && remoteBand === 'guess' ? null : (
        <footer className="play-footer">
          <p className="session-meta muted">
            Session: <code>{displayState.id}</code> · Seed: <code>{displayState.seed}</code>
            {isRemote ? <span> · Updates sync live while this tab is open.</span> : null}
            {!isRemote && (guessWsToken || spyWsToken) ? <span> · Live sync enabled for remote players.</span> : null}
          </p>
          <div className="play-footer-links">
            {!isRemote && spyCaptainView ? (
              <button
                type="button"
                className="play-footer-link-button"
                disabled={!displayState.last_ai_analysis}
                title={
                  displayState.last_ai_analysis
                    ? 'Open scoring and rationale for the latest AI spymaster or guesser step'
                    : 'Run a turn with an AI role to capture analysis (captain view with secret colors)'
                }
                onClick={() => setLastAiModalOpen(true)}
              >
                Last AI move
              </button>
            ) : null}
            <Link to={analysisHref}>Analyze this board</Link>
            {displayState.is_over ? <Link to="/play">New game</Link> : null}
          </div>
        </footer>
      )}

      {lastAiModalOpen && displayState.last_ai_analysis ? (
        <div
          className="endgame-modal-backdrop"
          role="presentation"
          onClick={() => setLastAiModalOpen(false)}
        >
          <section
            className="endgame-modal endgame-modal--wide last-ai-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="last-ai-modal-title"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 id="last-ai-modal-title">Last AI move</h2>
            <p className="muted last-ai-modal__subtitle">
              <span className={`history-team-tag history-team-tag--${teamSide(displayState.last_ai_analysis.team)}`}>
                {displayState.last_ai_analysis.team}
              </span>{' '}
              {displayState.last_ai_analysis.kind === 'spymaster' ? 'spymaster' : 'guesser'}
            </p>
            {displayState.last_ai_analysis.kind === 'spymaster' ? (
              <LastAiSpymasterBody trace={displayState.last_ai_analysis.trace} />
            ) : (
              <LastAiGuesserBody trace={displayState.last_ai_analysis.trace} />
            )}
            <div className="endgame-modal__actions">
              <button type="button" className="btn-secondary" onClick={() => setLastAiModalOpen(false)}>
                Close
              </button>
            </div>
          </section>
        </div>
      ) : null}

      {showEndModal ? (
        <div className="endgame-modal-backdrop" role="presentation">
          <section className="endgame-modal" role="dialog" aria-modal="true" aria-labelledby="endgame-modal-title">
            <h2 id="endgame-modal-title">Game over</h2>
            <p className="endgame-modal__winner">
              <span className={`history-team-tag history-team-tag--${teamSide(winnerTeam)}`}>{winnerTeam}</span> wins{' '}
              {winnerReasonText(displayState.win_reason)}
            </p>
            <p className="muted endgame-modal__seed">
              Replay or analyze this same game with seed <code>{displayState.seed}</code>.
            </p>
            <p className="muted endgame-modal__hint">
              The Open analysis button pre-fills seed and risk so Spymaster Analysis samples the same board layout.
            </p>
            <div className="endgame-modal__actions">
              <Link to="/play" className="btn-primary">
                Start a new game
              </Link>
              <Link to={analysisHref} className="btn-secondary">
                Open analysis
              </Link>
              {isRemote && remoteBand === 'spy' && liveToken ? (
                <button
                  type="button"
                  className="btn-primary"
                  disabled={blockingUi}
                  onClick={() =>
                    run(() => api.postLiveRematch(liveToken))
                  }
                >
                  Rematch · same captain link
                </button>
              ) : null}
              <button type="button" className="btn-secondary" onClick={() => setDismissedEndModalFor(displayState.id)}>
                Close
              </button>
            </div>
          </section>
        </div>
      ) : null}
    </div>
  )
}
