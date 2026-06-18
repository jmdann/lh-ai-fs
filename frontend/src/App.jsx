import { useState } from 'react'
import './styles.css'

const KIND_LABELS = {
  fact_discrepancy: 'fact discrepancy',
  quote_altered: 'altered quote',
  quote_fabricated: 'fabricated quote',
  authority_unsupported: 'unsupported authority',
  authority_unverifiable: 'unverifiable authority',
}

const KIND_ORDER = [
  'fact_discrepancy',
  'quote_altered',
  'quote_fabricated',
  'authority_unsupported',
  'authority_unverifiable',
]

function confBucket(c) {
  if (c == null) return 'unknown'
  if (c >= 0.8) return 'good'
  if (c >= 0.5) return 'warn'
  return 'bad'
}

function ReportSummary({ report }) {
  const findingsByKind = {}
  for (const f of report.findings) {
    findingsByKind[f.kind] = (findingsByKind[f.kind] || 0) + 1
  }
  return (
    <section className="card">
      <h2>{report.case_name}</h2>
      <div className="summary-grid">
        <div className="stat">
          <div className="stat-num">{report.citations.length}</div>
          <div className="stat-label">citations</div>
        </div>
        <div className="stat">
          <div className="stat-num">{report.findings.length}</div>
          <div className="stat-label">findings</div>
        </div>
        {KIND_ORDER.filter(k => findingsByKind[k]).map(k => (
          <div className="stat" key={k}>
            <div className="stat-num">{findingsByKind[k]}</div>
            <div className="stat-label">{KIND_LABELS[k]}</div>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 10, fontSize: 12, color: 'var(--muted)' }}>
        Generated {new Date(report.generated_at).toLocaleString()}
      </div>
    </section>
  )
}

function MemoCard({ memo }) {
  if (!memo) return null
  // Render [find-N] tokens as inline pills.
  const parts = memo.split(/(\[find-\d+\])/g)
  return (
    <section className="card">
      <h2>Judicial memo</h2>
      <p className="memo">
        {parts.map((p, i) =>
          /^\[find-\d+\]$/.test(p) ? (
            <span className="ref" key={i}>{p}</span>
          ) : (
            <span key={i}>{p}</span>
          )
        )}
      </p>
    </section>
  )
}

function FindingCard({ finding }) {
  const bucket = confBucket(finding.confidence)
  return (
    <div className="finding">
      <div className="finding-head">
        <strong>{finding.id}</strong>
        <span className="kind-tag">{KIND_LABELS[finding.kind] || finding.kind}</span>
        {finding.confidence != null && (
          <span className={`conf-bar`}>
            <span className={`conf-fill conf-${bucket}`}>
              <div style={{ width: `${Math.round(finding.confidence * 100)}%` }} />
            </span>
            <span>{(finding.confidence * 100).toFixed(0)}%</span>
          </span>
        )}
      </div>
      <div>{finding.summary}</div>
      {finding.confidence_reasoning && (
        <div style={{ marginTop: 6, fontSize: 12, color: 'var(--muted)', fontStyle: 'italic' }}>
          confidence reasoning: {finding.confidence_reasoning}
        </div>
      )}
      <ul className="evidence">
        {finding.evidence.map((ev, i) => (
          <li key={i}>
            <span className="role">{ev.role}</span>
            <strong>{ev.span.doc_id}</strong>
            {' — '}
            <span>{ev.span.quote.length > 200 ? ev.span.quote.slice(0, 200) + '…' : ev.span.quote}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function FindingsList({ findings }) {
  if (!findings.length) {
    return (
      <section className="card">
        <h2>Findings</h2>
        <p style={{ color: 'var(--muted)' }}>No findings emitted.</p>
      </section>
    )
  }
  const grouped = {}
  for (const f of findings) {
    grouped[f.kind] = grouped[f.kind] || []
    grouped[f.kind].push(f)
  }
  return (
    <section className="card">
      <h2>Findings ({findings.length})</h2>
      {KIND_ORDER.filter(k => grouped[k]).map(k => (
        <div key={k}>
          <h3>{KIND_LABELS[k]} ({grouped[k].length})</h3>
          {grouped[k].map(f => <FindingCard key={f.id} finding={f} />)}
        </div>
      ))}
    </section>
  )
}

function CitationsTable({ citations, agentResults }) {
  if (!citations.length) return null
  const quoteByCite = {}
  const authorityByCite = {}
  for (const ar of agentResults) {
    if (ar.kind === 'quote_check') {
      for (const q of ar.data || []) quoteByCite[q.citation_id] = q
    } else if (ar.kind === 'authority_check') {
      for (const a of ar.data || []) authorityByCite[a.citation_id] = a
    }
  }
  return (
    <section className="card">
      <h2>Citations ({citations.length})</h2>
      <table className="cites">
        <thead>
          <tr>
            <th>id</th>
            <th>Authority</th>
            <th>Has quote</th>
            <th>Quote verdict</th>
            <th>Authority verdict</th>
          </tr>
        </thead>
        <tbody>
          {citations.map(c => (
            <tr key={c.id}>
              <td><code>{c.id}</code></td>
              <td>{c.cited_authority}</td>
              <td>{c.quoted_text ? 'yes' : '—'}</td>
              <td>{quoteByCite[c.id]?.verdict || '—'}</td>
              <td>{authorityByCite[c.id]?.verdict || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

function AgentTrace({ agentResults }) {
  return (
    <section className="card">
      <h2>Agent trace</h2>
      <details className="trace">
        <summary>{agentResults.length} agent results — click to expand</summary>
        <div className="trace-list">
          {agentResults.map((ar, i) => (
            <div key={i}>
              <pre>
                {`${ar.kind.padEnd(16)} ${ar.agent.padEnd(28)} `}
                <span className={`outcome-${ar.outcome}`}>{ar.outcome}</span>
                {` ${ar.latency_ms}ms`}
                {ar.error ? `\n  error: ${ar.error}` : ''}
              </pre>
            </div>
          ))}
        </div>
      </details>
    </section>
  )
}

function CopyJsonButton({ report }) {
  const [copied, setCopied] = useState(false)
  const onCopy = async () => {
    await navigator.clipboard.writeText(JSON.stringify(report, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  return (
    <div className="copy-row">
      <button onClick={onCopy}>{copied ? 'copied ✓' : 'copy raw JSON'}</button>
    </div>
  )
}

function App() {
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const runAnalysis = async () => {
    setLoading(true)
    setError(null)
    setReport(null)
    try {
      const response = await fetch('http://localhost:8002/analyze', { method: 'POST' })
      if (!response.ok) throw new Error(`Server responded with ${response.status}`)
      const data = await response.json()
      // /analyze now returns the VerificationReport at the root, not nested
      // under a `report` key — match the FastAPI response_model.
      setReport(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <h1>BS Detector</h1>
      <p className="subtitle">Legal brief verification pipeline</p>

      <button className="run-button" onClick={runAnalysis} disabled={loading}>
        {loading ? 'Analyzing…' : 'Run Analysis'}
      </button>

      {error && (
        <div className="card" style={{ borderColor: 'var(--bad)', color: 'var(--bad)' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {!report && !loading && !error && (
        <p style={{ marginTop: 20, color: 'var(--muted)' }}>
          Click "Run Analysis" to evaluate the Rivera v. Harmon Construction Group MSJ.
        </p>
      )}

      {report && (
        <>
          <ReportSummary report={report} />
          <MemoCard memo={report.judicial_memo} />
          <FindingsList findings={report.findings} />
          <CitationsTable citations={report.citations} agentResults={report.agent_results} />
          <AgentTrace agentResults={report.agent_results} />
          <CopyJsonButton report={report} />
        </>
      )}
    </div>
  )
}

export default App
