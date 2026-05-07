import './ApiSpinnerOverlay.css'

type Props = {
  visible: boolean
  message?: string
}

export function ApiSpinnerOverlay({ visible, message = 'Loading…' }: Props) {
  if (!visible) return null
  return (
    <div className="api-spinner-overlay" role="status" aria-live="polite" aria-busy="true">
      <div className="api-spinner-panel">
        <div className="api-spinner-ring" aria-hidden />
        <p className="api-spinner-msg">{message}</p>
      </div>
    </div>
  )
}
