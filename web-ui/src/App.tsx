import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import './App.css'
import { AnalysisPage } from './pages/AnalysisPage'
import { HomePage } from './pages/HomePage'
import { PlayGamePage } from './pages/PlayGamePage'
import { PlayNewPage } from './pages/PlayNewPage'

export default function App() {
  return (
    <BrowserRouter basename="/app">
      <a
        className="github-corner"
        href="https://github.com/Yann-J/code-names-ai"
        target="_blank"
        rel="noreferrer"
        aria-label="View on GitHub"
      >
        <img
          src="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/github.svg"
          alt=""
          aria-hidden="true"
        />
        <span>View on GitHub</span>
      </a>
      <header className="app-header">
        <strong>Code Names AI</strong>
        <nav className="app-nav">
          <NavLink to="/" end>Home</NavLink>
          <NavLink to="/play">New game</NavLink>
          <NavLink to="/analysis">Analysis</NavLink>
        </nav>
      </header>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/play" element={<PlayNewPage />} />
          <Route path="/play/:gameId" element={<PlayGamePage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
