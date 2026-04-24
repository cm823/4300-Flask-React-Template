/**
 * ArticleScrollJourney.tsx  — revised
 */

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

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

/* ─── Branch colours ────────────────────────────────────────────────────── */
const BRANCH_COLORS = ["#93c5fd", "#a5b4fc", "#c4b5fd"];

const ALGO_SCORE_LABELS: Record<string, string> = {
  tfidf: "TF-IDF Score",
  svd: "SVD Score",
  combined: "Combined Score",
};

/* ─── Core SVG radar (reused by per-doc, query, and aggregate charts) ─────── */
function DimensionRadar({
  title,
  dimensions,
  scores,
  color,
}: {
  title: string;
  dimensions: string[];
  scores: number[];
  color: string;
}) {
  if (dimensions.length < 3) return null;
  const rawScores = scores.map(Math.abs);
  const maxScore = Math.max(...rawScores);
  if (maxScore === 0) return null;
  const normalized = rawScores.map((s) => s / maxScore);

  const SIZE = 400;
  const cx = SIZE / 2;
  const cy = SIZE / 2;
  const r = SIZE * 0.3;
  // Labels sit further out so long strings don't overlap the spokes
  const labelR = SIZE * 0.52;
  const n = dimensions.length;

  const angle = (i: number) => (i / n) * 2 * Math.PI - Math.PI / 2;
  const ptx = (i: number, s: number) => cx + Math.cos(angle(i)) * r * s;
  const pty = (i: number, s: number) => cy + Math.sin(angle(i)) * r * s;

  const gridLevels = [0.25, 0.5, 0.75, 1.0];
  const dataPoints = normalized.map((s, i) => `${ptx(i, s)},${pty(i, s)}`).join(" ");

  // Extra horizontal padding for left/right labels, vertical for top/bottom
  const PAD_H = 130;
  const PAD_V = 80;
  const vbX = -PAD_H;
  const vbY = -PAD_V;
  const vbW = SIZE + PAD_H * 2;
  const vbH = SIZE + PAD_V * 2;

  return (
    <div className="node-radar-wrap">
      <p className="node-radar-title">{title}</p>
      <svg
        viewBox={`${vbX} ${vbY} ${vbW} ${vbH}`}
        className="node-radar-svg"
        aria-hidden="true"
      >
        {gridLevels.map((level) => (
          <polygon
            key={level}
            points={dimensions.map((_, i) => `${ptx(i, level)},${pty(i, level)}`).join(" ")}
            fill="none"
            stroke="rgba(255,255,255,0.07)"
            strokeWidth={1.2}
          />
        ))}
        {dimensions.map((_, i) => (
          <line
            key={i}
            x1={cx} y1={cy}
            x2={ptx(i, 1)} y2={pty(i, 1)}
            stroke="rgba(255,255,255,0.12)"
            strokeWidth={1}
          />
        ))}
        <polygon
          points={dataPoints}
          fill={color}
          fillOpacity={0.18}
          stroke={color}
          strokeWidth={2}
          strokeOpacity={0.85}
          strokeLinejoin="round"
        />
        {normalized.map((s, i) => (
          <circle
            key={i}
            cx={ptx(i, s)}
            cy={pty(i, s)}
            r={5}
            fill={color}
            fillOpacity={0.9}
          />
        ))}
        {dimensions.map((theme, i) => {
          const lx = cx + Math.cos(angle(i)) * labelR;
          const ly = cy + Math.sin(angle(i)) * labelR;
          const ta = lx < cx - 10 ? "end" : lx > cx + 10 ? "start" : "middle";
          return (
            <text
              key={i}
              x={lx}
              y={ly}
              textAnchor={ta}
              dominantBaseline="middle"
              fontSize="13"
              fontFamily="Lato, sans-serif"
              fill="rgba(210,225,255,0.75)"
              letterSpacing="0.02em"
            >
              {theme}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

/* ─── Per-document radar (uses the node's own top dimensions) ─────────────── */
function NodeRadar({ node, color }: { node: ArticleNode; color: string }) {
  if (!node.dimensions || !node.dimensionScores) return null;
  return (
    <DimensionRadar
      title={node.title}
      dimensions={node.dimensions}
      scores={node.dimensionScores}
      color={color}
    />
  );
}

/* ─── Aggregate radar (union of all article dimensions, summed) ───────────── */
function AggregateRadar({ nodes, color }: { nodes: ArticleNode[]; color: string }) {
  const totals = new Map<string, number>();
  for (const node of nodes) {
    if (!node.dimensions || !node.dimensionScores) continue;
    node.dimensions.forEach((dim, i) => {
      totals.set(dim, (totals.get(dim) ?? 0) + (node.dimensionScores![i] ?? 0));
    });
  }
  if (totals.size < 3) return null;
  const top6 = [...totals.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6);
  return (
    <DimensionRadar
      title="All Articles"
      dimensions={top6.map(([name]) => name)}
      scores={top6.map(([, score]) => score)}
      color={color}
    />
  );
}

/* ─── Asset imports ──────────────────────────────────────────────────────── */
import eggImg from "./assets/egg.png";
import cardImg from "./assets/card.png";
import roseImg from "./assets/rose.png";
import bunnyImg from "./assets/bunny.png";

const ALL_IMGS = [eggImg, cardImg, roseImg, bunnyImg];

/* ─── Load anime.js from CDN once ───────────────────────────────────────── */
function useAnimeJS() {
  const [ready, setReady] = useState(!!(window as any).anime);
  useEffect(() => {
    if ((window as any).anime) {
      setReady(true);
      return;
    }
    const s = document.createElement("script");
    s.src = "https://cdnjs.cloudflare.com/ajax/libs/animejs/2.0.2/anime.min.js";
    s.onload = () => setReady(true);
    document.head.appendChild(s);
  }, []);
  return ready;
}

/* ─── Three.js Tunnel ────────────────────────────────────────────────────── */
export function TunnelCanvas() {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const W = mount.clientWidth;
    const H = mount.clientHeight || window.innerHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(window.devicePixelRatio);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);

    const camera = new THREE.PerspectiveCamera(75, W / H, 0.1, 100);
    camera.position.z = 2;

    const col = new THREE.Color(0x93c5fd);
    const rings: THREE.Mesh[] = [];

    for (let i = 0; i < 18; i++) {
      const mat = new THREE.MeshBasicMaterial({
        color: col,
        transparent: true,
        opacity: 0.6 - i * 0.022,
      });

      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.65, 0.012, 8, 56),
        mat,
      );
      ring.position.z = -i * 0.6;
      scene.add(ring);
      rings.push(ring);
    }

    const pCount = 90;
    const pPos = new Float32Array(pCount * 3);
    for (let i = 0; i < pCount; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = 0.2 + Math.random() * 0.4;
      pPos[i * 3] = Math.cos(a) * r;
      pPos[i * 3 + 1] = Math.sin(a) * r;
      pPos[i * 3 + 2] = -Math.random() * 11;
    }

    const pgeo = new THREE.BufferGeometry();
    pgeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
    scene.add(
      new THREE.Points(
        pgeo,
        new THREE.PointsMaterial({
          color: col,
          size: 0.032,
          transparent: true,
          opacity: 0.85,
        }),
      ),
    );

    let frame = 8;
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      frame += 0.01;
      rings.forEach((r, i) => {
        r.position.z = ((-i * 0.6 + frame * 1.1) % 11) - 11;
        r.rotation.z += 0.002 * (i % 2 === 0 ? 1 : -1);
      });
      const pos = pgeo.attributes.position as THREE.BufferAttribute;
      const arr = pos.array as Float32Array;
      for (let i = 0; i < pCount; i++) {
        arr[i * 3 + 2] += 0.05;
        if (arr[i * 3 + 2] > 2) arr[i * 3 + 2] = -11;
      }
      pos.needsUpdate = true;
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      if (mount.contains(renderer.domElement))
        mount.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <div
      ref={mountRef}
      className="journey-tunnel"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        width: "100%",
        height: "100%",
        zIndex: -1,
      }}
    />
  );
}

/* ─── Dirt divider — full viewport width ─────────────────────────────────── */
function DirtDivider() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.classList.add("divider-rumble");
          document.body.classList.add("screen-shake");
          setTimeout(() => {
            el.classList.remove("divider-rumble");
            document.body.classList.remove("screen-shake");
          }, 650);
        }
      },
      { threshold: 0.4 },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className="dirt-divider"
      aria-hidden="true"
      style={{
        width: "100vw",
        position: "relative",
        left: "50%",
        right: "50%",
        marginLeft: "-50vw",
        marginRight: "-50vw",
      }}
    >
      <svg
        className="divider-roots"
        viewBox="0 0 1400 90"
        preserveAspectRatio="none"
      >
        <g stroke="#7a5c30" strokeWidth="1.5" fill="none" opacity="0.6">
          <path d="M0,35 Q180,8 350,45 Q520,78 700,38 Q880,4 1050,48 Q1220,82 1400,35" />
          <path d="M0,60 Q200,38 400,65 Q600,88 800,58 Q1000,28 1200,62 Q1320,78 1400,58" />
          <path d="M0,18 Q140,5 300,28 Q480,52 650,22 Q840,0 1000,30 Q1180,55 1400,18" />
        </g>
      </svg>
      <div className="divider-stones">
        {Array.from({ length: 16 }).map((_, i) => (
          <span
            key={i}
            className="div-stone"
            style={{
              left: `${(i * 6.4 + 1.5) % 97}%`,
              width: `${5 + (i % 4) * 3}px`,
              height: `${3 + (i % 3) * 2}px`,
              position: "absolute",
            }}
          />
        ))}
      </div>
      <p
        className="divider-label"
        style={{ textAlign: "center", width: "100%", color: "white" }}
      >
        digging deeper…
      </p>
    </div>
  );
}

/* ─── Single article panel ───────────────────────────────────────────────── */
function ArticlePanel({
  node,
  index,
  animeReady,
  onVisible,
  scoringMode = "tfidf",
}: {
  node: ArticleNode;
  index: number;
  animeReady: boolean;
  onVisible: (idx: number) => void;
  scoringMode?: string;
}) {
  const titleRef = useRef<HTMLHeadingElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const hasAnimated = useRef(false);

  useEffect(() => {
    const panel = panelRef.current;
    const title = titleRef.current;
    if (!panel || !title) return;

    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          onVisible(index);

          if (!hasAnimated.current && animeReady) {
            hasAnimated.current = true;
            const anime = (window as any).anime;
            const words = title.querySelectorAll(".word");
            anime.timeline({ loop: false }).add({
              targets: words,
              scale: [14, 1],
              opacity: [0, 1],
              easing: "easeOutCirc",
              duration: 800,
              delay: (_: Element, i: number) => 700 * i,
            });
          }
        }
      },
      { threshold: 0.5 },
    );
    obs.observe(panel);
    return () => obs.disconnect();
  }, [animeReady, index, onVisible]);

  const words = node.title.split(" ");
  const imageSrc = ALL_IMGS[index % ALL_IMGS.length];

  return (
    <div
      ref={panelRef}
      className="article-panel"
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "3rem",
        margin: "0 auto",
        maxWidth: "900px",
        paddingRight: "150px", // Leaves space so the fixed right-tunnel doesn't overlap text
      }}
    >
      <style>
        {`
          @keyframes articleImageBob {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-15px); }
          }
        `}
      </style>

      <img
        src={imageSrc}
        alt=""
        style={{
          width: "120px", // Reduced from 180px to make it smaller
          height: "120px", // Give it a fixed height to match (makes a perfect square)
          objectFit: "contain", // Ensures the image fills the box without distortion
          borderRadius: "16px", // (Optional) Rounds the corners for a cleaner look
          flexShrink: 0,
          animation: `articleImageBob 4s ease-in-out infinite`,
          boxShadow: "0 8px 24px rgba(0,0,0,0.4)", // (Optional) Adds a nice floaty shadow
        }}
      />

      <div
        className="article-panel-inner"
        style={{ flex: 1, zIndex: 1, position: "relative" }}
      >
        <span className="article-score-badge">
          {ALGO_SCORE_LABELS[scoringMode] ?? "Score"}: {node.score.toFixed(3)}
        </span>
        <h2 ref={titleRef} className="ml15">
          {words.map((w, i) => (
            <span key={i} className="word">
              {w}
              {i < words.length - 1 ? "\u00a0" : ""}
            </span>
          ))}
        </h2>
        {node.description && <p className="article-desc">{node.description}</p>}
        <a
          href={`https://en.wikipedia.org/wiki/${encodeURIComponent(node.title)}`}
          target="_blank"
          rel="noopener noreferrer"
          className="article-wiki-link"
        >
          Read on Wikipedia ↗
        </a>
      </div>
    </div>
  );
}

/* ─── Root export ────────────────────────────────────────────────────────── */
/* ─── Root export ────────────────────────────────────────────────────────── */
export function BranchScrollJourney({
  branch,
  onSurface,
  onArticleChange,
  scoringMode = "tfidf",
}: {
  branch: ArticleNode[];
  onSurface: () => void;
  onArticleChange: (idx: number) => void;
  scoringMode?: string;
}) {
  const animeReady = useAnimeJS();
  const [showScrollTop, setShowScrollTop] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 300);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="journey-wrapper">
      {/* TunnelCanvas and sidebar wrapper have been moved to App.tsx */}

      {/* Scrollable article sections */}
      <div
        className="journey-scroll"
        style={{ position: "relative", zIndex: 1 }}
      >
        {(() => {
          const showRadar = scoringMode === "svd" || scoringMode === "combined";
          const queryNode = showRadar ? branch.find((n) => n.id === -1) : undefined;
          const articleNodes = branch.filter((n) => n.id !== -1);

          return (
            <>
              {/* ── Query SVD profile ── */}
              {showRadar && queryNode?.dimensions && queryNode.dimensionScores && (
                <div className="svd-radar-section">
                  <p className="svd-radar-section-label">Query SVD Profile</p>
                  <div className="node-radar-panel node-radar-panel--row">
                    <NodeRadar node={queryNode} color="#fbbf24" />
                  </div>
                </div>
              )}

              {/* ── Per-article segments ── */}
              {articleNodes.map((node, i) => (
                <div key={node.id} className="journey-segment">
                  <ArticlePanel
                    node={node}
                    index={i}
                    animeReady={animeReady}
                    onVisible={onArticleChange}
                    scoringMode={scoringMode}
                  />
                  {showRadar && node.dimensions && node.dimensionScores && (
                    <div className="node-radar-panel node-radar-panel--row">
                      <NodeRadar
                        node={node}
                        color={BRANCH_COLORS[i % BRANCH_COLORS.length]}
                      />
                    </div>
                  )}
                  {i < articleNodes.length - 1 && <DirtDivider />}
                </div>
              ))}

              {/* ── Aggregate SVD profile ── */}
              {showRadar && articleNodes.some((n) => n.dimensions) && (
                <div className="svd-radar-section">
                  <p className="svd-radar-section-label">Aggregate SVD Profile</p>
                  <div className="node-radar-panel node-radar-panel--row">
                    <AggregateRadar nodes={articleNodes} color="#34d399" />
                  </div>
                </div>
              )}
            </>
          );
        })()}

        <div
          className="journey-end-cap"
          style={{ textAlign: "center", paddingBottom: "4rem" }}
        >
          <p className="end-cap-text" style={{ marginBottom: "2rem" }}>
            — bottom of the rabbit hole —
          </p>

          <button
            className="surface-btn"
            onClick={onSurface}
            style={{
              padding: "12px 28px",
              background: "rgba(15, 23, 42, 0.8)",
              color: "#93c5fd",
              border: "1px solid rgba(147, 197, 253, 0.4)",
              borderRadius: "30px",
              cursor: "pointer",
              backdropFilter: "blur(4px)",
              fontWeight: "bold",
              fontSize: "1.1rem",
              transition: "all 0.2s ease",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(15, 23, 42, 1)";
              e.currentTarget.style.transform = "translateY(-2px)";
              e.currentTarget.style.boxShadow =
                "0 4px 12px rgba(147, 197, 253, 0.2)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "rgba(15, 23, 42, 0.8)";
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "none";
            }}
          >
            ↑ Back to Surface
          </button>
        </div>
      </div>

      {/* Scroll to Top Button */}
      {showScrollTop && (
        <button
          onClick={scrollToTop}
          style={{
            position: "fixed",
            top: "24px",
            right: "24px",
            zIndex: 9999,
            padding: "10px 18px",
            background: "rgba(15, 23, 42, 0.7)",
            color: "#93c5fd",
            border: "1px solid rgba(147, 197, 253, 0.3)",
            borderRadius: "24px",
            cursor: "pointer",
            backdropFilter: "blur(8px)",
            fontWeight: "bold",
            transition: "all 0.2s ease",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "rgba(15, 23, 42, 0.9)";
            e.currentTarget.style.transform = "translateY(-2px)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "rgba(15, 23, 42, 0.7)";
            e.currentTarget.style.transform = "translateY(0)";
          }}
        >
          ↑ Top
        </button>
      )}
    </div>
  );
}
