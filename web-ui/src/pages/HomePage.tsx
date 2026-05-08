import { Link } from 'react-router-dom'

export function HomePage() {
  return (
    <div>
      <h1 className="page-title">Code Names AI</h1>
      <p className="muted">
        This is my experiment to create an AI agent that plays word guessing games, based on both word embedding proximity + LLM scoring of candidate clues (and a very small RL routine to fine-tune all the candidate clue scoring parameters). For more information, see the <a href="https://github.com/Yann-J/code-names-ai">GitHub repository</a>.
      </p>
      <p>
        <Link to="/play">Start a new game</Link> · <Link to="/analysis">Clue ranking analysis</Link>
      </p>
      <p className="muted">
        Disclaimer: this is a hobby experiment made purely for learning and educational purposes, with no
        commercial objective. Use it as is, with no guarantees.
      </p>
    </div>
  )
}
