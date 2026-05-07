import { Link } from 'react-router-dom'

export function HomePage() {
  return (
    <div>
      <h1 className="page-title">Code Names AI (PWA)</h1>
      <p className="muted">
        Client-side UI with JSON API. Game state stays on the server; this shell caches the latest snapshot
        and works offline for assets only — you need the server running to play.
      </p>
      <p>
        <Link to="/play">Start a new game</Link> · <Link to="/analysis">Spymaster analysis</Link>
      </p>
    </div>
  )
}
