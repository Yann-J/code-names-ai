export interface TeamRoles {
  spymaster: 'human' | 'ai'
  guesser: 'human' | 'ai'
}

export interface BoardCard {
  word: string
  revealed: boolean
  revealed_as: string | null
  secret_color: string | null
}

export interface GameUi {
  show_end_turn: boolean
  can_click_guess: boolean
  waiting_on_ai: boolean
  show_spymaster_form: boolean
  show_guesser_form: boolean
}

export interface CluePayload {
  word: string
  count: number
}

export interface TurnEventPayload {
  team: string
  kind: 'CLUE' | 'GUESS'
  clue: CluePayload | null
  guess: string | null
  outcome_color: string | null
}

export interface GuessFlash {
  kind: 'team' | 'other' | 'assassin'
  word: string
}

export interface GameSnapshot {
  id: string
  seed: number
  risk: number
  roles: { red: TeamRoles; blue: TeamRoles }
  current_team: string
  current_phase: string
  winner: string | null
  win_reason: 'assassin' | 'all_words' | null
  is_over: boolean
  guesser_attempts_remaining: number | null
  latest_clue: CluePayload | null
  guess_count_after_latest_clue: number
  cards: BoardCard[]
  turn_history: TurnEventPayload[]
  ui: GameUi
  guess_flash: GuessFlash | null
}

export interface CreateGameBody {
  seed: number
  risk: number
  red_spy: TeamRoles['spymaster']
  red_guess: TeamRoles['guesser']
  blue_spy: TeamRoles['spymaster']
  blue_guess: TeamRoles['guesser']
}

async function readError(r: Response): Promise<string> {
  const t = await r.text()
  try {
    const j = JSON.parse(t) as { detail?: string }
    return j.detail ?? t
  } catch {
    return t
  }
}

export async function createGame(body: CreateGameBody): Promise<{ id: string; state: GameSnapshot }> {
  const r = await fetch('/api/games', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(await readError(r))
  return r.json()
}

export async function getGame(
  id: string,
  opts?: { includeSecretColors?: boolean; flashFx?: string; flashGw?: string },
): Promise<GameSnapshot> {
  const p = new URLSearchParams()
  if (opts?.includeSecretColors) p.set('include_secret_colors', 'true')
  if (opts?.flashFx) p.set('fx', opts.flashFx)
  if (opts?.flashGw) p.set('gw', opts.flashGw)
  const q = p.toString()
  const r = await fetch(`/api/games/${id}${q ? `?${q}` : ''}`)
  if (!r.ok) throw new Error(await readError(r))
  return r.json()
}

export async function postSpymaster(
  id: string,
  body: { word: string; count: number },
  includeSecretColors?: boolean,
): Promise<GameSnapshot> {
  const p = includeSecretColors ? '?include_secret_colors=true' : ''
  const r = await fetch(`/api/games/${id}/spymaster${p}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(await readError(r))
  return r.json()
}

export async function postGuesses(
  id: string,
  words: string[],
  includeSecretColors?: boolean,
): Promise<GameSnapshot> {
  const p = includeSecretColors ? '?include_secret_colors=true' : ''
  const r = await fetch(`/api/games/${id}/guesses${p}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ words }),
  })
  if (!r.ok) throw new Error(await readError(r))
  return r.json()
}

export async function postEndGuessTurn(
  id: string,
  includeSecretColors?: boolean,
): Promise<GameSnapshot> {
  const p = includeSecretColors ? '?include_secret_colors=true' : ''
  const r = await fetch(`/api/games/${id}/end-guess-turn${p}`, { method: 'POST' })
  if (!r.ok) throw new Error(await readError(r))
  return r.json()
}

export interface RiskSnapshotPayload {
  base_risk: number
  effective_risk: number
  delta_objectives: number
  ours_unrevealed: number
  theirs_unrevealed: number
  dynamic_enabled: boolean
}

export interface AnalysisTracePayload {
  chosen: { clue: string; n: number; targets: string[] } | null
  top_candidates: Array<{
    clue: string
    targets: string[]
    n: number
    score: number
    embedding_score: number
    components: {
      expected_reward_raw: number
      friendly_min_sim: number
      total: number
    }
    margin: number
    zipf: number
    llm_score: number | null
    llm_reason: string | null
  }>
  veto_count: number
  illegal_count: number
  risk_snapshot?: RiskSnapshotPayload | null
}

export interface AnalysisResponse {
  seed: number
  risk: number
  traces: Record<string, AnalysisTracePayload>
  board: Array<{ word: string; color: string }>
  first_team: string
}

export async function postAnalysis(body: { seed: number; risk: number }): Promise<AnalysisResponse> {
  const r = await fetch('/api/analysis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(await readError(r))
  return r.json()
}
