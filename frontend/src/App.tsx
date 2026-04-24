import { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";
import Chat from "./Chat";
import mountain from "./assets/mountain2.png";
import rabbitImg from "./assets/rabbithole.png";
// IMPORT from our new Scroll Journey component
import { BranchScrollJourney, TunnelCanvas } from "./ArticleScrollJourney";

/* ─── Types ──────────────────────────────────────────────────────────────── */
interface ArticleNode {
  id: number;
  title: string;
  score: number;
  branch: number;
  description?: string;
  dimensions?: string[];
  dimensionScores?: number[];
}

type ScoringMode = "tfidf" | "svd" | "combined";

const ALGO_LABELS: Record<ScoringMode, string> = {
  tfidf: "TF-IDF",
  svd: "SVD",
  combined: "Combined",
};

/* ─── Sparkle Cursor ─────────────────────────────────────────────────────── */
interface Sparkle {
  id: number;
  x: number;
  y: number;
  size: number;
  opacity: number;
  rotation: number;
  born: number;
}

function SparkleCursor() {
  const [sparkles, setSparkles] = useState<Sparkle[]>([]);
  const [cursorPos, setCursorPos] = useState({ x: -999, y: -999 });
  const nextId = useRef(0);
  const lastPos = useRef({ x: -999, y: -999 });

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      setCursorPos({ x: e.clientX, y: e.clientY });
      const dx = e.clientX - lastPos.current.x;
      const dy = e.clientY - lastPos.current.y;
      if (Math.hypot(dx, dy) < 6) return;
      lastPos.current = { x: e.clientX, y: e.clientY };

      const newSparkle: Sparkle = {
        id: nextId.current++,
        x: e.clientX,
        y: e.clientY,
        size: 18 + Math.random() * 28,
        opacity: 0.7 + Math.random() * 0.3,
        rotation: Math.random() * 360,
        born: Date.now(),
      };
      setSparkles((prev) => [...prev.slice(-28), newSparkle]);
    };
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  // Fade out for sparkles
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      setSparkles((prev) => prev.filter((s) => now - s.born < 650));
    }, 50);
    return () => clearInterval(interval);
  }, []);

  // Pastel dark-blue palette for our beautiful sparkles yay
  const colors = [
    "rgba(147,197,253,VAL)", // sky-300
    "rgba(165,180,252,VAL)", // indigo-300
    "rgba(196,181,253,VAL)", // violet-300
    "rgba(125,211,252,VAL)", // cyan-300
    "rgba(99,179,237,VAL)", // blue-400
  ];

  return (
    <div className="sparkle-layer" style={{ pointerEvents: "none" }}>
      <div
        style={{
          position: "fixed",
          left: cursorPos.x,
          top: cursorPos.y,
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: "rgba(147, 197, 253, 0.6)",
          boxShadow: "0 0 12px rgba(147, 197, 253, 0.8)",
          transform: "translate(-50%, -50%)",
          pointerEvents: "none",
          zIndex: 9999, // Ensure cursor is always on top
        }}
      />
      {sparkles.map((s, i) => {
        const age = (Date.now() - s.born) / 650;
        const alpha = s.opacity * (1 - age * 0.8);
        const colorTemplate = colors[i % colors.length];
        const color = colorTemplate.replace("VAL", alpha.toFixed(2));
        const scale = 1 - age * 0.4;
        return (
          <svg
            key={s.id}
            className="sparkle-svg"
            style={{
              left: s.x,
              top: s.y,
              width: s.size * 1.3,
              height: s.size * 1.4,
              transform: `translate(-50%,-50%) rotate(${s.rotation}deg) scale(${scale})`,
              opacity: alpha,
              position: "fixed",
              zIndex: 9999,
            }}
            viewBox="0 0 24 24"
          >
            {/* 4-point star */}
            <path
              d="M12 2 L13.5 10.5 L22 12 L13.5 13.5 L12 22 L10.5 13.5 L2 12 L10.5 10.5 Z"
              fill={color}
            />
          </svg>
        );
      })}
    </div>
  );
}

/* ─── Star Field ─────────────────────────────────────────────────────────── */
function StarField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight * 0.55;
    };
    resize();
    window.addEventListener("resize", resize);

    const stars = Array.from({ length: 120 }, () => ({
      x: Math.random(),
      y: Math.random() * 0.7,
      r: 0.5 + Math.random() * 1.8,
      phase: Math.random() * Math.PI * 2,
      speed: 0.006 + Math.random() * 0.01,
      cross: Math.random() > 0.7,
    }));

    let t = 0;
    let animId: number;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      t++;
      stars.forEach((s) => {
        const alpha = 0.3 + 0.7 * Math.abs(Math.sin(t * s.speed + s.phase));
        const px = s.x * canvas.width;
        const py = s.y * canvas.height;
        if (s.cross) {
          ctx.save();
          ctx.translate(px, py);
          ctx.rotate(Math.PI / 4);
          ctx.fillStyle = `rgba(255,255,255,${alpha})`;
          const arm = s.r * 3.5;
          ctx.fillRect(-0.8, -arm, 1.6, arm * 2);
          ctx.fillRect(-arm, -0.8, arm * 2, 1.6);
          ctx.restore();
        } else {
          ctx.beginPath();
          ctx.arc(px, py, s.r, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(220,235,255,${alpha})`;
          ctx.fill();
        }
      });
      animId = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
    };
  }, []);
  return <canvas ref={canvasRef} className="star-canvas" />;
}

/* ─── Algorithm Toggle ───────────────────────────────────────────────────── */
function AlgoToggle({
  value,
  onChange,
}: {
  value: ScoringMode;
  onChange: (m: ScoringMode) => void;
}) {
  return (
    <div className="algo-toggle">
      {(["tfidf", "svd", "combined"] as ScoringMode[]).map((m) => (
        <button
          key={m}
          type="button"
          className={`algo-btn ${value === m ? "active" : ""}`}
          onClick={() => onChange(m)}
        >
          {ALGO_LABELS[m]}
        </button>
      ))}
    </div>
  );
}

/* ─── Root App ───────────────────────────────────────────────────────────── */
export default function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null);
  const [article, setArticle] = useState("");
  const [branches, setBranches] = useState<ArticleNode[][]>([]);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [scoringMode, setScoringMode] = useState<ScoringMode>("tfidf");
  const [numArticles, setNumArticles] = useState(5);
  const [underground, setUnderground] = useState(false);

  // Custom scrolling state
  const [currentArticleIndex, setCurrentArticleIndex] = useState(0);

  // New RAG state from main branch
  const [modifiedQuery, setModifiedQuery] = useState<string | null>(null);
  const [ragSummary, setRagSummary] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/config")
      .then((r) => r.json())
      .then((d) => setUseLlm(d.use_llm));
  }, []);

  const handleSearch = useCallback(
    async (e?: React.FormEvent) => {
      if (e) e.preventDefault();
      if (!article.trim()) return;

      setLoading(true);
      setHasSearched(false);
      setUnderground(true);
      setCurrentArticleIndex(0); // Reset scroll index
      setModifiedQuery(null);
      setRagSummary(null);

      try {
        // Using the new API endpoint and JSON body from main branch
        const res = await fetch("/api/rag_query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            article,
            scoring_mode: scoringMode,
            path_length: numArticles,
          }),
        });

        const data = await res.json();
        if (!res.ok) {
          console.error("rag_query error:", data);
        } else {
          setBranches(data.results ?? []);
          setModifiedQuery(data.modified_query ?? null);
          setRagSummary(data.summary ?? null);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
        setHasSearched(true);
      }
    },
    [article, scoringMode, numArticles],
  );

  const handleSurface = () => {
    setUnderground(false);
    setBranches([]);
    setHasSearched(false);
    setCurrentArticleIndex(0);
    setModifiedQuery(null);
    setRagSummary(null);
  };

  if (useLlm === null) return <></>;

  return (
    <>
      <SparkleCursor />

      {/* ── Sky world (search UI) ── */}
      <div className={`sky-world ${underground ? "sky-exit" : ""}`}>
        <div
          className="mountain-bg"
          style={{ backgroundImage: `url(${mountain})` }}
        />
        <StarField />

        <div className="hero-content">
          <p className="hero-eyebrow">
            Wikipedia Discovery Engine for Famous People!
          </p>
          <h1 className="hero-title">
            <span className="title-rabbit">🐇</span> Rabbit Hole
          </h1>
          <p className="hero-sub">
            Search something about someone. Fall down the rabbit hole.
          </p>

          <form className="search-form" onSubmit={handleSearch}>
            <input
              className="search-input"
              placeholder="Enter a phrase describing a category of articles (ex. 'American participants in the Winter Olympics')"
              value={article}
              onChange={(e) => setArticle(e.target.value)}
              required
            />
            <AlgoToggle value={scoringMode} onChange={setScoringMode} />
            <div className="slider-row">
              <label htmlFor="num-articles-slider" className="slider-label">
                Max articles per tunnel: <strong>{numArticles}</strong>
              </label>
              <input
                id="num-articles-slider"
                type="range"
                min={2}
                max={10}
                value={numArticles}
                onChange={(e) => setNumArticles(Number(e.target.value))}
                className="slider-input"
              />
              <div className="slider-range-labels">
                <span>2</span>
                <span>10</span>
              </div>
            </div>
            <button className="dig-btn" type="submit" disabled={loading}>
              {loading ? (
                <span className="dig-dots">
                  <span />
                  <span />
                  <span />
                </span>
              ) : (
                "✦ Dig the Hole"
              )}
            </button>
          </form>
        </div>
      </div>

      {/* ── FIXED SIDEBAR RENDERED AT THE ROOT LEVEL ── */}
      {underground && branches.length > 0 && branches[0]?.length > 0 && (
        <div
          className="journey-tunnel-wrap"
          style={{
            position: "fixed",
            top: 0,
            right: 0,
            width: "300px",
            height: "100vh",
            zIndex: 100, // Safe high Z-index to sit above the dirt background
            pointerEvents: "none",
            display: "flex",
            flexDirection: "column",
            justifyContent: "flex-end",
            alignItems: "center",
            paddingBottom: "40px",
            boxSizing: "border-box",
          }}
        >
          <TunnelCanvas />
          <div className="tunnel-bottom-fade" />

          <div
            style={{
              position: "absolute",
              top: "50%", // Adjust this percentage to move it higher or lower in the tunnel
              left: "55%",
              transform: "translate(-50%, -50%)", // Perfectly centers the wrapper
              zIndex: 200, // Keeps it above the tunnel, but safely behind the text
            }}
          >
            <style>
              {`
                @keyframes rabbitBob {
                  0%, 100% { transform: translateY(0); }
                  50% { transform: translateY(-15px); }
                }
              `}
            </style>
            <img
              src={rabbitImg}
              alt="Flying Rabbit"
              style={{
                width: "120px",
                height: "120px",
                objectFit: "contain",
                flexShrink: 0,
                animation: `rabbitBob 4s ease-in-out infinite`,
                // Use drop-shadow instead of box-shadow so it hugs the transparent PNG shape
                filter: "drop-shadow(0 12px 16px rgba(0,0,0,0.6))",
              }}
            />
          </div>

          <div
            className="tunnel-overlay-text"
            style={{
              textAlign: "center",
              zIndex: 10,
              width: "100%",
              paddingBottom: "20px",
              paddingLeft: "20px",
              paddingRight: "20px",
              boxSizing: "border-box",
            }}
          >
            <strong
              style={{
                display: "block",
                fontSize: "1.8rem",
                fontWeight: 900,
                textShadow: "0 0 8px rgba(0,0,0,0.8)",
                color: "white",
                marginBottom: "4px",
              }}
            >
              {currentArticleIndex + 1} / {numArticles}
            </strong>
            <span
              style={{
                display: "block",
                fontSize: "0.85rem",
                textTransform: "uppercase",
                letterSpacing: "1px",
                color: "#93c5fd",
                textShadow: "0 0 4px rgba(0,0,0,0.8)",
              }}
            >
              article{branches[0].length !== 1 ? "s" : ""} discovered
            </span>
          </div>
        </div>
      )}

      {/* ── Underground world (results) ── */}
      <div
        className={`underground-world ${underground ? "underground-enter" : ""}`}
      >
        <div className="dirt-bg" />

        <svg
          className="root-svg"
          viewBox="0 0 1400 900"
          preserveAspectRatio="xMidYMid slice"
        >
          <g stroke="#4a3f6b" strokeWidth="1.2" fill="none" opacity="0.4">
            <path d="M0,60 Q180,140 140,300 Q100,460 280,620" />
            <path d="M1400,40 Q1220,120 1260,310 Q1300,500 1100,660" />
            <path d="M700,0 Q740,160 670,340 Q600,520 700,700" />
            <path d="M350,0 Q330,90 400,220 Q470,350 340,480" />
            <path d="M1050,0 Q1070,100 1000,230 Q930,360 1060,490" />
            <path d="M150,0 Q120,70 160,190 Q200,310 120,430" />
            <path d="M0,300 Q90,350 70,480 Q50,610 160,700" />
            <path d="M1400,350 Q1320,390 1340,520" />
          </g>
        </svg>

        {Array.from({ length: 25 }).map((_, i) => (
          <div
            key={i}
            className="pebble"
            style={{
              left: `${(i * 41 + 7) % 97}%`,
              top: `${(i * 53 + 12) % 90}%`,
              width: `${4 + (i % 5) * 3}px`,
              height: `${3 + (i % 4) * 2}px`,
            }}
          />
        ))}

        <div className="dirt-seam" />

        <div className="ug-nav-btns">
          <button className="surface-btn" onClick={handleSurface}>
            ↑ Back to Surface
          </button>
        </div>

        {loading && (
          <div className="ug-loading">
            <div className="shovel">⛏</div>
            <p>Digging your rabbit holes…</p>
            <div className="ug-dots">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}

        {hasSearched && !branches.length && !loading && (
          <div className="ug-empty">
            <p>🕳️ No tunnels found — try different keywords.</p>
          </div>
        )}

        {/* ── NEW RAG Context Summary ── */}
        {(modifiedQuery || ragSummary) && (
          <div
            className="rag-context"
            style={{ zIndex: 1, position: "relative" }}
          >
            {modifiedQuery && (
              <p className="rag-modified-query">
                <span className="rag-label">Reformatted Query:</span>{" "}
                {modifiedQuery}
              </p>
            )}
            {ragSummary && <p className="rag-summary">{ragSummary}</p>}
          </div>
        )}

        {/* ── Our custom vertical scrolling journey ── */}
        {branches.length > 0 && branches[0]?.length > 0 && (
          <BranchScrollJourney
            branch={branches[0]}
            onSurface={handleSurface}
            onArticleChange={setCurrentArticleIndex}
            scoringMode={scoringMode}
          />
        )}

        {/* ── Chatbot from main ── */}
        {useLlm && (
          <Chat
            onSearchTerm={(val) => {
              setArticle(val);
              handleSearch();
            }}
          />
        )}
      </div>
    </>
  );
}
