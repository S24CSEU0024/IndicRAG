import { useState, useEffect } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";

function App() {
  const [activeTab, setActiveTab] = useState("chat"); // "chat" | "benchmarks" | "documents"

  // Chat state
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  // Pipeline configuration
  const [retrievalMethod, setRetrievalMethod] = useState("hybrid");
  const [hybridAlpha, setHybridAlpha] = useState(0.5);
  const [threshold, setThreshold] = useState(0.35);
  const [directLlm, setDirectLlm] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  // Benchmark & Evaluation state
  const [evaluationData, setEvaluationData] = useState(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [difficultFilter, setDifficultFilter] = useState("all");

  // Document store state
  const [documentsData, setDocumentsData] = useState(null);
  const [passagesData, setPassagesData] = useState([]);
  const [passageSearch, setPassageSearch] = useState("");
  const [systemHealth, setSystemHealth] = useState(null);

  // Fetch initial health, docs, and evaluation on mount
  useEffect(() => {
    fetchHealth();
    fetchDocuments();
    fetchEvaluation();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) setSystemHealth(await res.json());
    } catch (err) {
      console.error("Health fetch error:", err);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/documents`);
      if (res.ok) setDocumentsData(await res.json());

      const passRes = await fetch(`${API_BASE}/api/documents/passages?page=1&page_size=50`);
      if (passRes.ok) {
        const pData = await passRes.json();
        setPassagesData(pData.passages || []);
      }
    } catch (err) {
      console.error("Documents fetch error:", err);
    }
  };

  const fetchEvaluation = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/evaluate`);
      if (res.ok) setEvaluationData(await res.json());
    } catch (err) {
      console.error("Evaluation fetch error:", err);
    }
  };

  const runEvaluation = async () => {
    setEvalLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/evaluate/run?sample_size=100`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        setEvaluationData(data.results);
      }
    } catch (err) {
      console.error("Error running evaluation:", err);
    } finally {
      setEvalLoading(false);
    }
  };

  const askQuestion = async (queryOverride) => {
    const textToAsk = (queryOverride || question).trim();
    if (!textToAsk || loading) return;

    // Append user message
    setMessages((prev) => [
      ...prev,
      {
        type: "user",
        text: textToAsk,
      },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: textToAsk,
          retrieval_method: retrievalMethod,
          alpha: parseFloat(hybridAlpha),
          threshold: parseFloat(threshold),
          direct_llm: directLlm,
          top_k: 5,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          text: data.answer || data.response || "No answer generated.",
          answerable: data.answerable,
          answerability: data.answerability,
          confidence: data.confidence,
          detected_language: data.detected_language,
          is_code_mixed: data.is_code_mixed,
          explanation: data.explanation,
          evidence_quality: data.evidence_quality,
          retrieval_method: data.retrieval_method,
          verification: data.verification,
          hallucination_check: data.hallucination_check,
          sources: data.supporting_passages || data.sources || [],
        },
      ]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          text: "Could not connect to the IndicRAG FastAPI backend. Please ensure the server is running on port 8000.",
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const filteredPassages = passagesData.filter((p) => {
    if (!passageSearch.trim()) return true;
    const term = passageSearch.toLowerCase();
    return (
      (p.text && p.text.toLowerCase().includes(term)) ||
      (p.source && p.source.toLowerCase().includes(term)) ||
      (p.section_title && p.section_title.toLowerCase().includes(term))
    );
  });

  return (
    <div className="app">
      {/* ================= HEADER ================= */}
      <header className="header">
        <div className="brand">
          <div className="logo">IR</div>
          <div>
            <h1>IndicRAG-QA</h1>
            <p>Evidence-Grounded Cross-Lingual Question Answering for Indic & Code-Mixed Languages</p>
          </div>
        </div>

        {/* NAVIGATION TABS */}
        <nav className="header-nav">
          <button
            className={`nav-tab ${activeTab === "chat" ? "active" : ""}`}
            onClick={() => setActiveTab("chat")}
          >
            💬 Interactive QA
          </button>
          <button
            className={`nav-tab ${activeTab === "benchmarks" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("benchmarks");
              if (!evaluationData) fetchEvaluation();
            }}
          >
            📊 Benchmarks (Modules 1-6)
          </button>
          <button
            className={`nav-tab ${activeTab === "documents" ? "active" : ""}`}
            onClick={() => setActiveTab("documents")}
          >
            📚 Document Store ({documentsData ? documentsData.total_passages : 425} Passages)
          </button>
        </nav>

        <div className="header-actions">
          <div className="status">
            <span className="status-dot"></span>
            {systemHealth?.hardware_acceleration ? (
              <span>GPU: {systemHealth.hardware_acceleration.toUpperCase()}</span>
            ) : (
              <span>System Online</span>
            )}
          </div>

          {activeTab === "chat" && (
            <button
              className={`config-toggle ${showSettings ? "active" : ""}`}
              onClick={() => setShowSettings(!showSettings)}
              title="Pipeline Configuration"
            >
              ⚙️ Pipeline Settings
            </button>
          )}

          {activeTab === "chat" && messages.length > 0 && (
            <button className="clear-button" onClick={clearChat}>
              Clear Chat
            </button>
          )}
        </div>
      </header>

      {/* PIPELINE SETTINGS DRAWER (When open in Chat) */}
      {showSettings && activeTab === "chat" && (
        <section className="settings-drawer">
          <div className="settings-content">
            <div className="setting-item">
              <label>Retrieval Method (Module 1 & 3):</label>
              <select
                value={retrievalMethod}
                onChange={(e) => setRetrievalMethod(e.target.value)}
              >
                <option value="hybrid">Balanced Hybrid (BM25 + FAISS)</option>
                <option value="dense">Multilingual Dense (FAISS Semantic)</option>
                <option value="bm25">Lexical BM25 (Okapi)</option>
                <option value="tfidf">Lexical TF-IDF</option>
              </select>
            </div>

            {retrievalMethod === "hybrid" && (
              <div className="setting-item">
                <label>
                  Hybrid Alpha (α): <strong>{hybridAlpha}</strong> (0.0 = Dense, 1.0 = Lexical)
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={hybridAlpha}
                  onChange={(e) => setHybridAlpha(e.target.value)}
                />
              </div>
            )}

            <div className="setting-item">
              <label>
                Answerability Cutoff (τ): <strong>{threshold}</strong> (Rejection Threshold)
              </label>
              <input
                type="range"
                min="0.20"
                max="0.60"
                step="0.02"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
              />
            </div>

            <div className="setting-item toggle-item">
              <label>
                <input
                  type="checkbox"
                  checked={directLlm}
                  onChange={(e) => setDirectLlm(e.target.checked)}
                />
                Direct LLM Mode (Module 4: No RAG Context / Hallucination Baseline)
              </label>
            </div>
          </div>
        </section>
      )}

      {/* ================= TAB 1: CHAT INTERFACE ================= */}
      {activeTab === "chat" && (
        <main className="main">
          {messages.length === 0 ? (
            <section className="welcome">
              <div className="welcome-icon">✦</div>
              <h2>
                Evidence-Grounded Cross-Lingual
                <span> Question Answering</span>
              </h2>
              <p className="welcome-text">
                Ask questions across English, Indic (Hindi), and Romanized Code-Mixed (Hinglish) formats.
                IndicRAG retrieves verified passages from university policy documents and rejects unanswerable
                questions to prevent hallucinations.
              </p>

              {/* FEATURE HIGHLIGHTS */}
              <div className="features">
                <div className="feature-card">
                  <div className="feature-icon">🌐</div>
                  <h3>Multilingual & Code-Mixed</h3>
                  <p>Handles English, Devanagari Hindi, and Hinglish queries with automatic language routing.</p>
                </div>

                <div className="feature-card">
                  <div className="feature-icon">⚖️</div>
                  <h3>Hybrid Retrieval</h3>
                  <p>Combines Lexical (BM25) and Dense (Multilingual Sentence-BERT + FAISS) with calibrated α.</p>
                </div>

                <div className="feature-card">
                  <div className="feature-icon">🛡️</div>
                  <h3>Hallucination Control</h3>
                  <p>Automatically flags unsupported questions as UNANSWERABLE with zero hallucination.</p>
                </div>
              </div>

              {/* CATEGORIZED TEST PROMPTS */}
              <div className="examples">
                <p className="examples-title">Select a query type to test live:</p>
                <div className="example-buttons">
                  <button
                    className="example-btn en-btn"
                    onClick={() =>
                      askQuestion("What is the minimum passing marks in the end semester examination?")
                    }
                  >
                    <span className="badge-tag">EN</span>
                    What is the minimum passing marks in the end semester examination?
                  </button>

                  <button
                    className="example-btn indic-btn"
                    onClick={() =>
                      askQuestion("एंड सेमेस्टर परीक्षा पास करने के लिए न्यूनतम कितने प्रतिशत अंक चाहिए?")
                    }
                  >
                    <span className="badge-tag">Indic</span>
                    एंड सेमेस्टर परीक्षा पास करने के लिए न्यूनतम कितने प्रतिशत अंक चाहिए?
                  </button>

                  <button
                    className="example-btn codemix-btn"
                    onClick={() =>
                      askQuestion("Scholarship ke liye minimum eligibility criteria kya hai?")
                    }
                  >
                    <span className="badge-tag">Hinglish</span>
                    Scholarship ke liye minimum eligibility criteria kya hai?
                  </button>

                  <button
                    className="example-btn unans-btn"
                    onClick={() =>
                      askQuestion("Can students keep pet dogs in the university hostel rooms?")
                    }
                  >
                    <span className="badge-tag unans-tag">Unanswerable Trap</span>
                    Can students keep pet dogs in the university hostel rooms?
                  </button>

                  <button
                    className="example-btn diff-btn"
                    onClick={() =>
                      askQuestion("Kitne rupees ke stamp paper par single girl child ka affidavit banta hai?")
                    }
                  >
                    <span className="badge-tag diff-tag">Module 6 Difficult</span>
                    Kitne rupees ke stamp paper par single girl child ka affidavit banta hai?
                  </button>

                  <button
                    className="example-btn diff-btn"
                    onClick={() =>
                      askQuestion("Who are the members of the Scholarship Committee that reviews merit scholarships?")
                    }
                  >
                    <span className="badge-tag">Named Entity</span>
                    Who are the members of the Scholarship Committee?
                  </button>
                </div>
              </div>
            </section>
          ) : (
            <section className="chat-container">
              <div className="chat-header">
                <div>
                  <h2>Conversation</h2>
                  <p>Evidence-grounded multilingual responses with passage citations</p>
                </div>
                <div className="rag-badges">
                  <span className="pipeline-pill">Method: {retrievalMethod.toUpperCase()}</span>
                  <span className="pipeline-pill">α: {hybridAlpha}</span>
                  <span className="pipeline-pill">τ: {threshold}</span>
                  {directLlm && <span className="warning-pill">DIRECT LLM (NO RAG)</span>}
                </div>
              </div>

              <div className="messages">
                {messages.map((message, index) => (
                  <div className={`message-row ${message.type}`} key={index}>
                    <div className="avatar">{message.type === "user" ? "You" : "IR"}</div>

                    <div className="message-content">
                      <div className="message-header-meta">
                        <span className="message-label">
                          {message.type === "user" ? "User Query" : "IndicRAG Answer"}
                        </span>

                        {message.type === "assistant" && !message.error && (
                          <div className="meta-badges">
                            {message.detected_language && (
                              <span className={`lang-badge ${message.detected_language.toLowerCase()}`}>
                                {message.detected_language}
                                {message.is_code_mixed ? " (Code-Mixed)" : ""}
                              </span>
                            )}

                            {message.answerability && (
                              <span
                                className={`answerable-badge ${
                                  message.answerable ? "badge-yes" : "badge-no"
                                }`}
                              >
                                {message.answerable ? "✓ ANSWERABLE" : "⚠ UNANSWERABLE"}
                              </span>
                            )}

                            {message.confidence !== undefined && (
                              <span className="confidence-badge">
                                Confidence: {Math.round(message.confidence * 100)}%
                              </span>
                            )}
                          </div>
                        )}
                      </div>

                      <div
                        className={`message-bubble ${
                          message.error
                            ? "error-bubble"
                            : message.answerable === false
                            ? "unanswerable-bubble"
                            : ""
                        }`}
                      >
                        {message.text}
                      </div>

                      {/* EXPLANATION & VERIFICATION */}
                      {message.type === "assistant" && !message.error && (
                        <div className="explanation-card">
                          <div className="explanation-title">
                            💡 Evidence Grounding & Verification
                          </div>
                          <p>{message.explanation}</p>

                          {message.verification && (
                            <div className="verification-pills">
                              <span
                                className={`verif-pill ${
                                  message.verification.all_citations_valid ? "valid" : "warning"
                                }`}
                              >
                                📑 Citations:{" "}
                                {message.verification.has_citations
                                  ? `Validated [${message.verification.cited_ranks.join(", ")}]`
                                  : "Direct / Rejection"}
                              </span>

                              {message.hallucination_check && (
                                <span
                                  className={`verif-pill ${
                                    message.hallucination_check.is_grounded ? "grounded" : "hallucinated"
                                  }`}
                                >
                                  🛡️ Groundedness:{" "}
                                  {Math.round(message.hallucination_check.groundedness_score * 100)}%
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      )}

                      {/* SUPPORTING EVIDENCE PASSAGES */}
                      {message.sources && message.sources.length > 0 && message.answerable && (
                        <div className="sources">
                          <div className="sources-title">
                            📚 Supporting Evidence Passages ({message.sources.length} Retrieved)
                          </div>

                          {message.sources.map((source, sourceIndex) => (
                            <div className="source-card" key={sourceIndex}>
                              <div className="source-top">
                                <span className="source-number">[Passage {sourceIndex + 1}]</span>
                                <div className="source-meta">
                                  <strong>{source.source || source.document || "Document"}</strong>
                                  <span className="page-tag">Page {source.page || source.page_number}</span>
                                  {source.score !== undefined && (
                                    <span className="score-tag">Score: {source.score}</span>
                                  )}
                                </div>
                              </div>

                              {source.section_title && (
                                <div className="source-section-title">
                                  📌 {source.section_title}
                                </div>
                              )}

                              <p className="source-body">{source.text}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="message-row assistant">
                    <div className="avatar">IR</div>
                    <div className="message-content">
                      <div className="message-label">IndicRAG Processing</div>
                      <div className="message-bubble loading-bubble">
                        <span></span>
                        <span></span>
                        <span></span>
                        <small>Retrieving evidence across documents & generating grounded response...</small>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </section>
          )}

          {/* INPUT SECTION */}
          <div className="input-section">
            <div className="input-box">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a university regulation question in English, Hindi, or Hinglish..."
                rows="1"
              />
              <button
                className="send-button"
                onClick={() => askQuestion()}
                disabled={!question.trim() || loading}
              >
                {loading ? "..." : "↑"}
              </button>
            </div>
            <p className="input-hint">
              IndicRAG evaluates question answerability and generates strictly grounded answers with citations.
            </p>
          </div>
        </main>
      )}

      {/* ================= TAB 2: BENCHMARKS & EVALUATION ================= */}
      {activeTab === "benchmarks" && (
        <main className="benchmarks-view">
          <div className="benchmarks-header">
            <div>
              <h2>Evaluation Suite & Module Benchmarks</h2>
              <p>Comprehensive comparative results across all 6 NLP course project modules</p>
            </div>
            <button
              className="run-bench-btn"
              onClick={runEvaluation}
              disabled={evalLoading}
            >
              {evalLoading ? "⏳ Running Benchmark Suite..." : "⚡ Run Live Benchmark Suite"}
            </button>
          </div>

          {evaluationData ? (
            <div className="modules-grid">
              {/* MODULE 1 */}
              <div className="module-card">
                <div className="card-badge">Module 1</div>
                <h3>TF-IDF vs BM25 vs Multilingual Dense Retrieval</h3>
                <p className="card-subtitle">Retrieval metrics comparison on university document passages</p>

                <table className="metric-table">
                  <thead>
                    <tr>
                      <th>Retriever</th>
                      <th>Recall@1</th>
                      <th>Recall@3</th>
                      <th>Recall@5</th>
                      <th>Precision@5</th>
                      <th>HitRate@5</th>
                      <th>MRR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {["TF-IDF", "BM25", "Multilingual_Dense"].map((key) => {
                      const m = evaluationData.module_1?.[key] || {};
                      return (
                        <tr key={key} className={key === "BM25" ? "highlight-row" : ""}>
                          <td><strong>{key.replace("_", " ")}</strong></td>
                          <td>{m["Recall@1"]}</td>
                          <td>{m["Recall@3"]}</td>
                          <td>{m["Recall@5"]}</td>
                          <td>{m["Precision@5"]}</td>
                          <td>{m["HitRate@5"]}</td>
                          <td><strong>{m["MRR"]}</strong></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* MODULE 2 */}
              <div className="module-card">
                <div className="card-badge">Module 2</div>
                <h3>Cross-Lingual & Code-Mixed Performance</h3>
                <p className="card-subtitle">Performance breakdown across English, Indic (Hindi), and Hinglish queries</p>

                <table className="metric-table">
                  <thead>
                    <tr>
                      <th>Language Type</th>
                      <th>Queries Evaluated</th>
                      <th>Recall@5</th>
                      <th>HitRate@5</th>
                      <th>MRR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(evaluationData.module_2?.results_by_language || {}).map(([lang, d]) => (
                      <tr key={lang}>
                        <td>
                          <span className={`lang-pill ${lang.toLowerCase()}`}>{lang}</span>
                        </td>
                        <td>{d.query_count}</td>
                        <td>{d["Recall@5"]}</td>
                        <td>{d["HitRate@5"]}</td>
                        <td><strong>{d["MRR"]}</strong></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* MODULE 3 */}
              <div className="module-card">
                <div className="card-badge">Module 3</div>
                <h3>Hybrid Retrieval Sensitivity (Alpha Sweep)</h3>
                <p className="card-subtitle">Score(p) = α·Score_lexical + (1−α)·Score_semantic</p>

                <table className="metric-table">
                  <thead>
                    <tr>
                      <th>Mode</th>
                      <th>Alpha (α)</th>
                      <th>Recall@5</th>
                      <th>HitRate@5</th>
                      <th>MRR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(evaluationData.module_3?.alpha_sweep || {}).map(([mode, d]) => (
                      <tr key={mode} className={d.alpha === 0.5 ? "highlight-row" : ""}>
                        <td>{mode}</td>
                        <td>{d.alpha}</td>
                        <td>{d["Recall@5"]}</td>
                        <td>{d["HitRate@5"]}</td>
                        <td><strong>{d["MRR"]}</strong></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* MODULE 4 */}
              <div className="module-card">
                <div className="card-badge">Module 4</div>
                <h3>Direct LLM QA vs Retrieval-Augmented Generation (RAG)</h3>
                <p className="card-subtitle">Demonstrating hallucination reduction with passage grounding</p>

                <table className="metric-table">
                  <thead>
                    <tr>
                      <th>Pipeline</th>
                      <th>Exact Match</th>
                      <th>Token F1</th>
                      <th>Correctness</th>
                      <th>Groundedness</th>
                      <th>Hallucination Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Direct LLM (No RAG)</td>
                      <td>{evaluationData.module_4?.Direct_LLM?.ExactMatch}</td>
                      <td>{evaluationData.module_4?.Direct_LLM?.TokenF1}</td>
                      <td>{evaluationData.module_4?.Direct_LLM?.AnswerCorrectness}</td>
                      <td><span className="bad-metric">{Math.round((evaluationData.module_4?.Direct_LLM?.GroundednessScore || 0) * 100)}%</span></td>
                      <td><span className="bad-metric">{Math.round((evaluationData.module_4?.Direct_LLM?.HallucinationRate || 0) * 100)}%</span></td>
                    </tr>
                    <tr className="highlight-row">
                      <td><strong>IndicRAG (Hybrid + Citations)</strong></td>
                      <td>{evaluationData.module_4?.IndicRAG?.ExactMatch}</td>
                      <td>{evaluationData.module_4?.IndicRAG?.TokenF1}</td>
                      <td>{evaluationData.module_4?.IndicRAG?.AnswerCorrectness}</td>
                      <td><span className="good-metric">{Math.round((evaluationData.module_4?.IndicRAG?.GroundednessScore || 0) * 100)}%</span></td>
                      <td><span className="good-metric">{Math.round((evaluationData.module_4?.IndicRAG?.HallucinationRate || 0) * 100)}%</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* MODULE 5 */}
              <div className="module-card full-width">
                <div className="card-badge">Module 5</div>
                <h3>Answerability Detection & Hallucination Prevention</h3>
                <p className="card-subtitle">Evaluating rejection of intentionally unanswerable questions</p>

                <div className="metrics-summary-row">
                  <div className="stat-box">
                    <span className="stat-label">Accuracy</span>
                    <span className="stat-val">{Math.round((evaluationData.module_5?.metrics?.Accuracy || 0) * 100)}%</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Precision</span>
                    <span className="stat-val">{Math.round((evaluationData.module_5?.metrics?.Precision || 0) * 100)}%</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Recall</span>
                    <span className="stat-val">{Math.round((evaluationData.module_5?.metrics?.Recall || 0) * 100)}%</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">F1-Score</span>
                    <span className="stat-val">{Math.round((evaluationData.module_5?.metrics?.F1 || 0) * 100)}%</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Rejection Rate</span>
                    <span className="stat-val highlight">{Math.round((evaluationData.module_5?.metrics?.RejectionRate || 0) * 100)}%</span>
                  </div>
                </div>

                {evaluationData.module_5?.metrics?.ConfusionMatrix && (
                  <div className="confusion-matrix-box">
                    <h4>Confusion Matrix</h4>
                    <div className="cm-grid">
                      <div className="cm-cell tp">
                        <span className="cm-num">{evaluationData.module_5.metrics.ConfusionMatrix.TruePositive}</span>
                        <span className="cm-desc">True Positive (Answered Correctly)</span>
                      </div>
                      <div className="cm-cell fn">
                        <span className="cm-num">{evaluationData.module_5.metrics.ConfusionMatrix.FalseNegative}</span>
                        <span className="cm-desc">False Negative (False Rejection)</span>
                      </div>
                      <div className="cm-cell fp">
                        <span className="cm-num">{evaluationData.module_5.metrics.ConfusionMatrix.FalsePositive}</span>
                        <span className="cm-desc">False Positive (Hallucination Risk)</span>
                      </div>
                      <div className="cm-cell tn">
                        <span className="cm-num">{evaluationData.module_5.metrics.ConfusionMatrix.TrueNegative}</span>
                        <span className="cm-desc">True Negative (Correctly Rejected)</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* MODULE 6 */}
              <div className="module-card full-width">
                <div className="card-badge">Module 6</div>
                <div className="card-header-flex">
                  <div>
                    <h3>Qualitative Analysis: 16 Difficult Cases</h3>
                    <p className="card-subtitle">
                      Challenging scenarios involving code-mixing, slang, typos, exact numbers, named entities, and ambiguity
                    </p>
                  </div>
                  <div className="accuracy-pill">
                    Accuracy: {Math.round((evaluationData.module_6?.correct_handling_accuracy || 0) * 100)}%
                  </div>
                </div>

                <div className="table-filters">
                  <button
                    className={`filter-btn ${difficultFilter === "all" ? "active" : ""}`}
                    onClick={() => setDifficultFilter("all")}
                  >
                    All (16)
                  </button>
                  <button
                    className={`filter-btn ${difficultFilter === "Code-Mixed" ? "active" : ""}`}
                    onClick={() => setDifficultFilter("Code-Mixed")}
                  >
                    Code-Mixed / Slang
                  </button>
                  <button
                    className={`filter-btn ${difficultFilter === "Indic" ? "active" : ""}`}
                    onClick={() => setDifficultFilter("Indic")}
                  >
                    Devanagari Hindi
                  </button>
                  <button
                    className={`filter-btn ${difficultFilter === "UNANSWERABLE" ? "active" : ""}`}
                    onClick={() => setDifficultFilter("UNANSWERABLE")}
                  >
                    Hallucination Traps
                  </button>
                </div>

                <table className="metric-table cases-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Type</th>
                      <th>Query</th>
                      <th>Detected Lang</th>
                      <th>Expected</th>
                      <th>Predicted</th>
                      <th>Confidence</th>
                      <th>Top Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(evaluationData.module_6?.cases || [])
                      .filter((c) => {
                        if (difficultFilter === "all") return true;
                        if (difficultFilter === "UNANSWERABLE") return c.expected_answerable === "UNANSWERABLE";
                        return c.detected_language === difficultFilter || c.type.includes(difficultFilter);
                      })
                      .map((c) => (
                        <tr key={c.case_id}>
                          <td><strong>{c.case_id}</strong></td>
                          <td><span className="case-type">{c.type}</span></td>
                          <td className="case-query">"{c.query}"</td>
                          <td><span className="lang-pill">{c.detected_language}</span></td>
                          <td><span className={`status-pill ${c.expected_answerable.toLowerCase()}`}>{c.expected_answerable}</span></td>
                          <td><span className={`status-pill ${c.predicted_answerable.toLowerCase()}`}>{c.predicted_answerable}</span></td>
                          <td>{Math.round(c.confidence * 100)}%</td>
                          <td className="case-source">{c.top_source}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="loading-state">
              <p>Loading benchmark evaluation results...</p>
            </div>
          )}
        </main>
      )}

      {/* ================= TAB 3: DOCUMENT STORE ================= */}
      {activeTab === "documents" && (
        <main className="documents-view">
          <div className="documents-header">
            <h2>Indexed Document Knowledge Base</h2>
            <p>Institutional policies, examination rules, and admission regulations</p>
          </div>

          <div className="doc-cards-row">
            {(documentsData?.documents || []).map((doc, idx) => (
              <div className="doc-summary-card" key={idx}>
                <div className="doc-icon">📄</div>
                <div className="doc-details">
                  <h3>{doc.filename}</h3>
                  <div className="doc-stats">
                    <span>{doc.total_pages} Pages</span>
                    <span>•</span>
                    <span>{doc.total_passages} Passages</span>
                    <span>•</span>
                    <span>{(doc.size_bytes / 1024).toFixed(1)} KB</span>
                  </div>
                </div>
                <span className="doc-status-badge">Indexed</span>
              </div>
            ))}
          </div>

          <div className="passages-section">
            <div className="passages-header-flex">
              <h3>Retrievable Passages ({passagesData.length} Loaded)</h3>
              <input
                type="text"
                className="search-passages-input"
                placeholder="Search passages by keyword..."
                value={passageSearch}
                onChange={(e) => setPassageSearch(e.target.value)}
              />
            </div>

            <div className="passages-list">
              {filteredPassages.slice(0, 30).map((p, idx) => (
                <div className="passage-item" key={idx}>
                  <div className="passage-item-top">
                    <span className="passage-id">Chunk #{p.chunk_id}</span>
                    <span className="passage-source">{p.source} (Page {p.page})</span>
                  </div>
                  {p.section_title && (
                    <div className="passage-title">📌 {p.section_title}</div>
                  )}
                  <p className="passage-text">{p.text}</p>
                </div>
              ))}
            </div>
          </div>
        </main>
      )}

      {/* ================= FOOTER ================= */}
      <footer className="footer">
        <div>
          <strong>IndicRAG-QA</strong> • CSET 346 Natural Language Processing Project
        </div>
        <div className="footer-sub">
          Hybrid Lexical-Dense Multilingual Retrieval • Hallucination Rejection • 425 Passages Indexed
        </div>
      </footer>
    </div>
  );
}

export default App;