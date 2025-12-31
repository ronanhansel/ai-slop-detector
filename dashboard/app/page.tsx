"use client";

import React, {
  useState,
  useEffect,
  useRef,
  useMemo,
  useCallback,
} from "react";
import {
  BarChart,
  Activity,
  Search,
  Terminal,
  ShieldAlert,
  Brain,
  Users,
  MessageSquare,
  TrendingUp,
  AlertTriangle,
  Database,
  RefreshCw,
  Cpu,
  Flame,
  Eye,
  Hash,
  AtSign,
  Zap,
} from "lucide-react";
import * as THREE from "three";

// --- TYPES ---
interface Comment {
  id: number;
  commenter_id: string;
  comment_id: string;
  post_id: string;
  comment_content: string;
  cleaned_content: string;
  num_emojis: number;
  num_caps_words: number;
  num_unicode_chars: number;
  contains_media: boolean;
  contains_link: boolean;
  num_tagged_people: number;
  tagged_grok: boolean;
  used_slang: boolean;
  sentiment_label: string;
  sentiment_prob: number;
  irony_label: string;
  irony_prob: number;
  hate_label: string;
  hate_prob: number;
  offensive_label: string;
  offensive_prob: number;
  hate: number;
  aggression: number;
  violence: number;
  anger: number;
  rage: number;
  negative_emotion: number;
  positive_emotion: number;
  politics: number;
  government: number;
  swearing_terms: number;
}

interface DataStats {
  totalComments: number;
  uniqueUsers: number;
  hateComments: number;
  offensiveComments: number;
  grokResponses: number;
  sentimentBreakdown: {
    positive: number;
    negative: number;
    neutral: number;
  };
  topUsers: { user: string; count: number }[];
}

interface ProcessedData {
  comments: Comment[];
  stats: DataStats;
}

// --- 3D SCENE COMPONENT ---
const Scene3D = ({ data }: { data: Comment[] }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const animationRef = useRef<number | null>(null);
  const [hoverInfo, setHoverInfo] = useState<
    (Comment & { x: number; y: number }) | null
  >(null);

  useEffect(() => {
    if (!mountRef.current || data.length === 0) return;

    // Clean up any existing renderer first (handles StrictMode double-mount)
    if (rendererRef.current) {
      rendererRef.current.dispose();
      if (mountRef.current.contains(rendererRef.current.domElement)) {
        mountRef.current.removeChild(rendererRef.current.domElement);
      }
      rendererRef.current = null;
    }
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    const currentMount = mountRef.current;
    const width = currentMount.clientWidth;
    const height = currentMount.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#0f172a");
    scene.fog = new THREE.Fog("#0f172a", 10, 50);

    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(15, 10, 20);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    currentMount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const gridHelper = new THREE.GridHelper(30, 30, "#334155", "#1e293b");
    scene.add(gridHelper);

    const axesHelper = new THREE.AxesHelper(10);
    scene.add(axesHelper);

    const geometry = new THREE.BufferGeometry();
    const positions: number[] = [];
    const colors: number[] = [];

    // Sample data for performance (max 5000 points)
    const sampleSize = Math.min(data.length, 5000);
    const sampleData = data.slice(0, sampleSize);

    sampleData.forEach((d) => {
      // X-axis: Toxicity (hate_prob + offensive_prob) / 2
      const toxicity = (d.hate_prob + d.offensive_prob) / 2;

      // Y-axis: Rage-Bait Index (aggression + rage + anger + hate)
      const rbi = Math.min(1, (d.aggression + d.rage + d.anger + d.hate) * 2);

      // Z-axis: Sentiment (-1 to 1 mapped from label)
      let sentimentVal = 0;
      if (d.sentiment_label === "negative") sentimentVal = -d.sentiment_prob;
      else if (d.sentiment_label === "positive")
        sentimentVal = d.sentiment_prob;

      const x = toxicity * 20 - 10;
      const y = rbi * 15;
      const z = sentimentVal * 10;

      positions.push(x, y, z);

      const color = new THREE.Color();
      if (d.hate_label === "HATE") {
        color.setHSL(0.0, 1.0, 0.5); // Red for hate
      } else if (d.offensive_label === "offensive") {
        color.setHSL(0.08, 1.0, 0.5); // Orange for offensive
      } else if (d.sentiment_label === "negative") {
        color.setHSL(0.6, 0.8, 0.5); // Blue for negative
      } else if (d.sentiment_label === "positive") {
        color.setHSL(0.35, 0.8, 0.5); // Green for positive
      } else {
        color.setHSL(0.0, 0.0, 0.5); // Gray for neutral
      }

      colors.push(color.r, color.g, color.b);
    });

    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3)
    );
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.25,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      sizeAttenuation: true,
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    raycaster.params.Points.threshold = 0.5;

    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };
    let theta = 0;
    let phi = Math.PI / 3;
    const radius = 25;

    const onMouseMove = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      if (isDragging) {
        const deltaMove = {
          x: event.clientX - previousMousePosition.x,
          y: event.clientY - previousMousePosition.y,
        };

        theta -= deltaMove.x * 0.01;
        phi = Math.max(0.1, Math.min(Math.PI - 0.1, phi - deltaMove.y * 0.01));

        camera.position.x = radius * Math.sin(phi) * Math.sin(theta);
        camera.position.y = radius * Math.cos(phi);
        camera.position.z = radius * Math.sin(phi) * Math.cos(theta);
        camera.lookAt(0, 0, 0);

        previousMousePosition = { x: event.clientX, y: event.clientY };
      }

      raycaster.setFromCamera(mouse, camera);
      const intersections = raycaster.intersectObject(points);

      if (intersections.length > 0) {
        const index = intersections[0].index;
        if (index !== undefined && sampleData[index]) {
          setHoverInfo({
            ...sampleData[index],
            x: event.clientX,
            y: event.clientY,
          });
        }
        document.body.style.cursor = "pointer";
      } else {
        setHoverInfo(null);
        document.body.style.cursor = isDragging ? "grabbing" : "default";
      }
    };

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    renderer.domElement.addEventListener("mousemove", onMouseMove);
    renderer.domElement.addEventListener("mousedown", onMouseDown);
    renderer.domElement.addEventListener("mouseup", onMouseUp);

    let animationId: number;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      animationRef.current = animationId;
      if (!isDragging) {
        theta += 0.001;
        camera.position.x = radius * Math.sin(phi) * Math.sin(theta);
        camera.position.z = radius * Math.sin(phi) * Math.cos(theta);
        camera.lookAt(0, 0, 0);
      }
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationId);
      animationRef.current = null;
      renderer.domElement.removeEventListener("mousemove", onMouseMove);
      renderer.domElement.removeEventListener("mousedown", onMouseDown);
      renderer.domElement.removeEventListener("mouseup", onMouseUp);
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      if (currentMount.contains(renderer.domElement)) {
        currentMount.removeChild(renderer.domElement);
      }
      rendererRef.current = null;
    };
  }, [data]);

  return (
    <div className="relative w-full h-full bg-slate-900 overflow-hidden rounded-xl border border-slate-700 shadow-2xl">
      <div ref={mountRef} className="w-full h-full" />

      <div className="absolute bottom-4 left-4 text-slate-400 text-xs font-mono pointer-events-none">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500 rounded-full"></div> Hate Speech
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-orange-500 rounded-full"></div> Offensive
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-blue-500 rounded-full"></div> Negative
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded-full"></div> Positive
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-gray-500 rounded-full"></div> Neutral
        </div>
        <div className="mt-2">Drag to Rotate | Hover to Inspect</div>
      </div>

      <div className="absolute top-4 right-4 pointer-events-none">
        <div className="bg-slate-800/80 backdrop-blur p-3 rounded-lg border border-slate-700">
          <h3 className="text-white font-bold text-sm mb-2">Axes Legend</h3>
          <div className="text-xs text-slate-300 space-y-1">
            <p>
              <span className="text-green-400">Y-Axis (Vertical):</span>{" "}
              Rage-Bait Index
            </p>
            <p>
              <span className="text-red-400">X-Axis (Horizontal):</span>{" "}
              Toxicity Score
            </p>
            <p>
              <span className="text-blue-400">Z-Axis (Depth):</span> Sentiment
            </p>
          </div>
        </div>
      </div>

      {hoverInfo && (
        <div
          className="absolute z-50 bg-slate-800/90 backdrop-blur-md border border-slate-600 p-4 rounded-lg shadow-xl w-72 pointer-events-none transform -translate-x-1/2 -translate-y-full"
          style={{
            left: Math.max(150, hoverInfo.x - 200),
            top: Math.max(150, hoverInfo.y - 100),
          }}
        >
          <div className="flex justify-between items-start mb-2">
            <span
              className={`text-xs px-2 py-0.5 rounded font-bold ${
                hoverInfo.hate_label === "HATE"
                  ? "bg-red-500/20 text-red-400"
                  : hoverInfo.offensive_label === "offensive"
                  ? "bg-orange-500/20 text-orange-400"
                  : "bg-blue-500/20 text-blue-400"
              }`}
            >
              {hoverInfo.hate_label === "HATE"
                ? "HATE SPEECH"
                : hoverInfo.offensive_label === "offensive"
                ? "OFFENSIVE"
                : "NORMAL"}
            </span>
            <span className="text-xs text-slate-400">
              @{hoverInfo.commenter_id}
            </span>
          </div>
          <p className="text-white text-sm italic mb-3 line-clamp-3">
            &quot;
            {hoverInfo.comment_content?.slice(0, 150) ||
              hoverInfo.cleaned_content?.slice(0, 150)}
            ...&quot;
          </p>
          <div className="space-y-1 text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-slate-400">Sentiment:</span>
              <span
                className={
                  hoverInfo.sentiment_label === "negative"
                    ? "text-red-400"
                    : hoverInfo.sentiment_label === "positive"
                    ? "text-emerald-400"
                    : "text-slate-300"
                }
              >
                {hoverInfo.sentiment_label} (
                {(hoverInfo.sentiment_prob * 100).toFixed(0)}%)
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Hate Prob:</span>
              <span className="text-white">
                {(hoverInfo.hate_prob * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Offensive Prob:</span>
              <span className="text-white">
                {(hoverInfo.offensive_prob * 100).toFixed(1)}%
              </span>
            </div>
            {hoverInfo.tagged_grok && (
              <div className="flex justify-between">
                <span className="text-slate-400">Tagged Grok:</span>
                <span className="text-purple-400">Yes</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// --- STAT CARD COMPONENT ---
const StatCard = ({
  label,
  value,
  subtext,
  icon: Icon,
  color,
}: {
  label: string;
  value: string | number;
  subtext?: string;
  icon: React.ElementType;
  color: string;
}) => (
  <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
    <div className="flex justify-between items-start">
      <div>
        <p className="text-slate-400 text-sm font-medium">{label}</p>
        <h3 className="text-2xl font-bold text-white mt-1">{value}</h3>
      </div>
      <div className={`p-2 rounded-lg bg-${color}-500/10`}>
        <Icon className={`w-6 h-6 text-${color}-500`} />
      </div>
    </div>
    {subtext && (
      <div className="mt-4 flex items-center text-sm">
        <span className="text-slate-500">{subtext}</span>
      </div>
    )}
  </div>
);

// --- USER TABLE COMPONENT ---
const UserTable = ({ data, stats }: { data: Comment[]; stats: DataStats }) => {
  const [filter, setFilter] = useState("");
  const [sortBy, setSortBy] = useState<"count" | "hate" | "offensive">("count");

  const userStats = useMemo(() => {
    const userMap: Record<
      string,
      {
        count: number;
        hateCount: number;
        offensiveCount: number;
        negativeCount: number;
        grokTags: number;
        totalHateProb: number;
        totalOffensiveProb: number;
      }
    > = {};

    data.forEach((d) => {
      if (!d.commenter_id || d.commenter_id.length > 30) return;

      if (!userMap[d.commenter_id]) {
        userMap[d.commenter_id] = {
          count: 0,
          hateCount: 0,
          offensiveCount: 0,
          negativeCount: 0,
          grokTags: 0,
          totalHateProb: 0,
          totalOffensiveProb: 0,
        };
      }

      const u = userMap[d.commenter_id];
      u.count++;
      if (d.hate_label === "HATE") u.hateCount++;
      if (d.offensive_label === "offensive") u.offensiveCount++;
      if (d.sentiment_label === "negative") u.negativeCount++;
      if (d.tagged_grok) u.grokTags++;
      u.totalHateProb += d.hate_prob || 0;
      u.totalOffensiveProb += d.offensive_prob || 0;
    });

    return Object.entries(userMap)
      .map(([user, stat]) => ({
        user,
        ...stat,
        avgHateProb: stat.totalHateProb / stat.count,
        avgOffensiveProb: stat.totalOffensiveProb / stat.count,
        toxicityScore: (stat.hateCount + stat.offensiveCount) / stat.count,
      }))
      .sort((a, b) => {
        if (sortBy === "hate") return b.avgHateProb - a.avgHateProb;
        if (sortBy === "offensive")
          return b.avgOffensiveProb - a.avgOffensiveProb;
        return b.count - a.count;
      });
  }, [data, sortBy]);

  const filtered = userStats
    .filter((u) => u.user.toLowerCase().includes(filter.toLowerCase()))
    .slice(0, 100);

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
      <div className="p-4 border-b border-slate-700 flex flex-wrap justify-between items-center gap-4">
        <h3 className="text-white font-semibold">
          User Analysis ({userStats.length} users)
        </h3>
        <div className="flex gap-2 items-center">
          <select
            className="bg-slate-900 border border-slate-700 text-white px-3 py-2 rounded-lg text-sm"
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
          >
            <option value="count">Sort by Activity</option>
            <option value="hate">Sort by Hate Score</option>
            <option value="offensive">Sort by Offensive Score</option>
          </select>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search users..."
              className="bg-slate-900 border border-slate-700 text-white pl-10 pr-4 py-2 rounded-lg text-sm focus:outline-none focus:border-blue-500"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          </div>
        </div>
      </div>
      <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-900/50 text-slate-400 font-medium sticky top-0">
            <tr>
              <th className="px-6 py-3">User</th>
              <th className="px-6 py-3">Comments</th>
              <th className="px-6 py-3">Hate Rate</th>
              <th className="px-6 py-3">Offensive Rate</th>
              <th className="px-6 py-3">Grok Tags</th>
              <th className="px-6 py-3">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700">
            {filtered.map((u, i) => (
              <tr key={i} className="hover:bg-slate-700/30 transition-colors">
                <td className="px-6 py-4 font-medium text-white">@{u.user}</td>
                <td className="px-6 py-4 text-slate-300">{u.count}</td>
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          u.avgHateProb > 0.5 ? "bg-red-500" : "bg-emerald-500"
                        }`}
                        style={{ width: `${u.avgHateProb * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-slate-400">
                      {(u.avgHateProb * 100).toFixed(0)}%
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          u.avgOffensiveProb > 0.5
                            ? "bg-orange-500"
                            : "bg-emerald-500"
                        }`}
                        style={{ width: `${u.avgOffensiveProb * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-slate-400">
                      {(u.avgOffensiveProb * 100).toFixed(0)}%
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 text-slate-300">
                  {u.grokTags > 0 && (
                    <span className="text-purple-400">{u.grokTags}</span>
                  )}
                </td>
                <td className="px-6 py-4">
                  {u.hateCount > 0 ? (
                    <span className="inline-flex items-center gap-1 text-red-400 border border-red-500/30 bg-red-500/10 px-2 py-1 rounded text-xs">
                      <AlertTriangle className="w-3 h-3" /> FLAGGED
                    </span>
                  ) : u.toxicityScore > 0.3 ? (
                    <span className="inline-flex items-center gap-1 text-yellow-400 border border-yellow-500/30 bg-yellow-500/10 px-2 py-1 rounded text-xs">
                      <Eye className="w-3 h-3" /> MONITOR
                    </span>
                  ) : (
                    <span className="text-emerald-400 text-xs">NORMAL</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// --- SLOP DETECTOR (DETERMINISTIC MOCKUP) ---
const SlopDetector = () => {
  const [text, setText] = useState("");
  const [result, setResult] = useState<{
    sentiment: string;
    sentimentProb: number;
    hateProb: number;
    offensiveProb: number;
    aiProb: number;
    rbi: number;
    features: { emoji: number; caps: number; links: boolean };
  } | null>(null);
  const [loading, setLoading] = useState(false);

  const analyze = useCallback(() => {
    setLoading(true);
    setTimeout(() => {
      // Deterministic analysis based on text features
      const lowerText = text.toLowerCase();

      // Check for hate/offensive indicators
      const hateWords = [
        "hate",
        "kill",
        "die",
        "stupid",
        "idiot",
        "dumb",
        "racist",
        "nazi",
      ];
      const offensiveWords = [
        "damn",
        "hell",
        "crap",
        "ass",
        "suck",
        "wtf",
        "stfu",
      ];
      const positiveWords = [
        "love",
        "great",
        "amazing",
        "good",
        "happy",
        "thank",
        "please",
      ];
      const aiIndicators = [
        "as an ai",
        "language model",
        "i apologize",
        "i cannot",
        "furthermore",
        "moreover",
        "in conclusion",
        "it is important to note",
        "delve into",
        "utilize",
        "commence",
        "facilitate",
        "comprehensive",
      ];

      let hateScore = 0;
      let offensiveScore = 0;
      let positiveScore = 0;
      let aiScore = 0.7;

      hateWords.forEach((w) => {
        if (lowerText.includes(w)) hateScore += 0.15;
      });
      offensiveWords.forEach((w) => {
        if (lowerText.includes(w)) offensiveScore += 0.1;
      });
      positiveWords.forEach((w) => {
        if (lowerText.includes(w)) positiveScore += 0.1;
      });
      aiIndicators.forEach((w) => {
        if (lowerText.includes(w)) aiScore += 0.15;
      });

      // Additional AI indicators
      const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 0);
      const avgSentenceLength =
        sentences.reduce((sum, s) => sum + s.split(" ").length, 0) /
        Math.max(sentences.length, 1);

      // Very long, formal sentences suggest AI
      if (avgSentenceLength > 25) aiScore += 0.1;
      if (avgSentenceLength > 35) aiScore += 0.15;

      // Overly formal/robotic patterns
      const formalityMarkers = (
        lowerText.match(
          /\b(however|nevertheless|therefore|thus|hence|consequently)\b/g
        ) || []
      ).length;
      aiScore += Math.min(0.2, formalityMarkers * 0.05);

      // Features
      const emojiCount = (
        text.match(
          /[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{2600}-\u{26FF}]/gu
        ) || []
      ).length;
      const capsCount = (text.match(/[A-Z]{2,}/g) || []).length;
      const hasLinks = /https?:\/\//.test(text);
      const hasExclamation = (text.match(/!/g) || []).length;

      // Adjust scores based on features
      if (capsCount > 2) {
        hateScore += 0.1;
        offensiveScore += 0.1;
      }
      if (hasExclamation > 2) {
        offensiveScore += 0.05 * hasExclamation;
      }

      // Determine sentiment
      let sentiment = "neutral";
      let sentimentProb = 0.6;

      if (hateScore > 0.2 || offensiveScore > 0.2) {
        sentiment = "negative";
        sentimentProb = 0.7 + Math.min(0.25, hateScore + offensiveScore);
      } else if (positiveScore > 0.1) {
        sentiment = "positive";
        sentimentProb = 0.6 + positiveScore;
      }

      // Rage-Bait Index
      const rbi = Math.min(
        1,
        hateScore + offensiveScore + capsCount * 0.05 + hasExclamation * 0.02
      );

      setResult({
        sentiment,
        sentimentProb: Math.min(0.99, sentimentProb),
        hateProb: Math.min(0.95, hateScore),
        offensiveProb: Math.min(0.95, offensiveScore),
        aiProb: Math.min(0.95, aiScore),
        rbi,
        features: {
          emoji: emojiCount,
          caps: capsCount,
          links: hasLinks,
        },
      });
      setLoading(false);
    }, 800);
  }, [text]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
        <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
          <Brain className="w-5 h-5 text-blue-400" />
          Text Analyzer (Demo)
        </h3>
        <p className="text-slate-400 text-sm mb-4">
          This is a deterministic mockup that analyzes text for hate speech,
          offensive content, and sentiment based on keyword matching and text
          features.
        </p>
        <textarea
          className="w-full h-40 bg-slate-900 border border-slate-700 rounded-lg p-4 text-white focus:border-blue-500 focus:outline-none resize-none font-mono text-sm"
          placeholder="Enter text to analyze for toxicity indicators..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <div className="mt-4 flex justify-end">
          <button
            onClick={analyze}
            disabled={!text || loading}
            className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Activity className="w-4 h-4" />
            )}
            Analyze Text
          </button>
        </div>
      </div>

      {result && (
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-700 pb-4">
            <h3 className="text-white font-semibold">Analysis Results</h3>
            <span className="text-xs text-slate-400 font-mono">
              DETERMINISTIC MODEL
            </span>
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400">Hate Speech Probability</span>
                <span
                  className={
                    result.hateProb > 0.3 ? "text-red-400" : "text-emerald-400"
                  }
                >
                  {(result.hateProb * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-1000 ${
                    result.hateProb > 0.3 ? "bg-red-500" : "bg-emerald-500"
                  }`}
                  style={{ width: `${result.hateProb * 100}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400">Offensive Probability</span>
                <span
                  className={
                    result.offensiveProb > 0.3
                      ? "text-orange-400"
                      : "text-emerald-400"
                  }
                >
                  {(result.offensiveProb * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-1000 ${
                    result.offensiveProb > 0.3
                      ? "bg-orange-500"
                      : "bg-emerald-500"
                  }`}
                  style={{ width: `${result.offensiveProb * 100}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400">AI-Generated Probability</span>
                <span
                  className={
                    result.aiProb > 0.3 ? "text-purple-400" : "text-emerald-400"
                  }
                >
                  {(result.aiProb * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-1000 ${
                    result.aiProb > 0.3 ? "bg-purple-500" : "bg-emerald-500"
                  }`}
                  style={{ width: `${result.aiProb * 100}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400">Rage-Bait Index</span>
                <span className="text-yellow-400">
                  {result.rbi.toFixed(2)} / 1.0
                </span>
              </div>
              <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-yellow-500 to-red-600 transition-all duration-1000"
                  style={{ width: `${result.rbi * 100}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-4">
              <div className="bg-slate-900 p-4 rounded-lg">
                <p className="text-xs text-slate-400 uppercase tracking-wider">
                  Sentiment
                </p>
                <p
                  className={`text-lg font-bold ${
                    result.sentiment === "negative"
                      ? "text-red-400"
                      : result.sentiment === "positive"
                      ? "text-emerald-400"
                      : "text-slate-300"
                  }`}
                >
                  {result.sentiment.toUpperCase()}
                </p>
                <p className="text-xs text-slate-500">
                  {(result.sentimentProb * 100).toFixed(0)}% confidence
                </p>
              </div>
              <div className="bg-slate-900 p-4 rounded-lg">
                <p className="text-xs text-slate-400 uppercase tracking-wider">
                  Features Detected
                </p>
                <div className="text-xs text-slate-300 mt-2 space-y-1">
                  <p>Emojis: {result.features.emoji}</p>
                  <p>ALL CAPS words: {result.features.caps}</p>
                  <p>Links: {result.features.links ? "Yes" : "No"}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// --- CRAWLER CONSOLE (MOCKUP) ---
const CrawlerConsole = () => {
  const [logs, setLogs] = useState<
    { time: string; msg: string; type: string }[]
  >([]);

  useEffect(() => {
    const messages = [
      { msg: "Initializing data pipeline...", type: "info" },
      { msg: "Loading CSV dataset: final_merged_data_nlp.csv", type: "info" },
      { msg: "Parsing 181,076 rows...", type: "info" },
      { msg: "Extracting NLP features...", type: "process" },
      { msg: "Computing sentiment analysis scores...", type: "process" },
      { msg: "Calculating hate speech probabilities...", type: "process" },
      { msg: "Generating rage-bait indices...", type: "process" },
      { msg: "Dataset loaded successfully!", type: "success" },
      { msg: "Ready for visualization.", type: "success" },
    ];

    let i = 0;
    const interval = setInterval(() => {
      if (i < messages.length) {
        setLogs((prev) => [
          ...prev,
          { time: new Date().toLocaleTimeString(), ...messages[i] },
        ]);
        i++;
      } else {
        clearInterval(interval);
      }
    }, 600);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-black font-mono text-xs p-4 rounded-xl border border-slate-800 h-64 overflow-hidden flex flex-col">
      <div className="flex items-center gap-2 text-slate-500 border-b border-slate-800 pb-2 mb-2">
        <Terminal className="w-4 h-4" />
        <span>Data Pipeline Console</span>
      </div>
      <div className="flex-1 overflow-y-auto space-y-1">
        {logs.map((log, idx) => (
          <div key={idx} className="flex gap-4">
            <span className="text-slate-500">[{log.time}]</span>
            <span
              className={
                log.type === "success"
                  ? "text-emerald-400"
                  : log.type === "process"
                  ? "text-blue-400"
                  : "text-slate-400"
              }
            >
              {log.msg}
            </span>
          </div>
        ))}
        <div className="animate-pulse text-emerald-500">_</div>
      </div>
    </div>
  );
};

// --- SENTIMENT DISTRIBUTION CHART ---
const SentimentChart = ({ stats }: { stats: DataStats }) => {
  const total =
    stats.sentimentBreakdown.positive +
    stats.sentimentBreakdown.negative +
    stats.sentimentBreakdown.neutral;
  const posPercent = (stats.sentimentBreakdown.positive / total) * 100;
  const negPercent = (stats.sentimentBreakdown.negative / total) * 100;
  const neuPercent = (stats.sentimentBreakdown.neutral / total) * 100;

  return (
    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
      <h3 className="text-white font-semibold mb-4">Sentiment Distribution</h3>
      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-emerald-400">Positive</span>
            <span className="text-slate-400">
              {stats.sentimentBreakdown.positive.toLocaleString()} (
              {posPercent.toFixed(1)}%)
            </span>
          </div>
          <div className="w-full h-4 bg-slate-900 rounded-full overflow-hidden">
            <div
              className="h-full bg-emerald-500"
              style={{ width: `${posPercent}%` }}
            />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-red-400">Negative</span>
            <span className="text-slate-400">
              {stats.sentimentBreakdown.negative.toLocaleString()} (
              {negPercent.toFixed(1)}%)
            </span>
          </div>
          <div className="w-full h-4 bg-slate-900 rounded-full overflow-hidden">
            <div
              className="h-full bg-red-500"
              style={{ width: `${negPercent}%` }}
            />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-slate-300">Neutral</span>
            <span className="text-slate-400">
              {stats.sentimentBreakdown.neutral.toLocaleString()} (
              {neuPercent.toFixed(1)}%)
            </span>
          </div>
          <div className="w-full h-4 bg-slate-900 rounded-full overflow-hidden">
            <div
              className="h-full bg-slate-500"
              style={{ width: `${neuPercent}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

// --- TOP USERS CHART ---
const TopUsersChart = ({ stats }: { stats: DataStats }) => {
  const maxCount = stats.topUsers[0]?.count || 1;

  return (
    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
      <h3 className="text-white font-semibold mb-4">Most Active Users</h3>
      <div className="space-y-3">
        {stats.topUsers.slice(0, 10).map((user, i) => (
          <div key={i}>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-300 truncate max-w-[150px]">
                @{user.user}
              </span>
              <span className="text-slate-400">{user.count}</span>
            </div>
            <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500"
                style={{ width: `${(user.count / maxCount) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// --- MAIN APP COMPONENT ---
export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [data, setData] = useState<ProcessedData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/data")
      .then((res) => res.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError("Failed to load data");
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 text-blue-500 animate-spin mx-auto mb-4" />
          <p className="text-white text-lg">Loading dataset...</p>
          <p className="text-slate-400 text-sm">Processing 180,000+ comments</p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-white text-lg">Error loading data</p>
          <p className="text-slate-400 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-blue-500/30">
      {/* Sidebar */}
      <nav className="fixed top-0 left-0 h-full w-20 bg-slate-900 border-r border-slate-800 flex flex-col items-center py-6 z-50">
        <div className="mb-8">
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-900/20">
            <ShieldAlert className="text-white w-6 h-6" />
          </div>
        </div>

        <div className="flex flex-col gap-6 w-full">
          {[
            { id: "dashboard", icon: BarChart, label: "Viz" },
            { id: "users", icon: Users, label: "Users" },
            { id: "detector", icon: Brain, label: "Analyze" },
            { id: "data", icon: Database, label: "Data" },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex flex-col items-center gap-1 py-3 transition-all relative ${
                activeTab === item.id
                  ? "text-blue-400"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              <item.icon className="w-6 h-6" />
              <span className="text-[10px] font-medium">{item.label}</span>
              {activeTab === item.id && (
                <div className="absolute right-0 top-0 h-full w-0.5 bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.5)]" />
              )}
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content */}
      <main className="pl-20 min-h-screen">
        {/* Header */}
        <header className="px-8 py-6 border-b border-slate-800 flex justify-between items-center bg-slate-950/80 backdrop-blur sticky top-0 z-40">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              AI Slop Detector Dashboard
            </h1>
            <p className="text-slate-400 text-sm">
              Political Discourse Analysis &amp; Toxicity Detection
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="bg-slate-900 px-3 py-1 rounded border border-slate-800 text-xs text-slate-400 font-mono">
              Dataset: {data.stats.totalComments.toLocaleString()} Comments
            </div>
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600" />
          </div>
        </header>

        <div className="p-8 max-w-7xl mx-auto space-y-8">
          {/* Dashboard View */}
          {activeTab === "dashboard" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard
                  label="Total Comments"
                  value={data.stats.totalComments.toLocaleString()}
                  subtext={`${data.stats.uniqueUsers.toLocaleString()} unique users`}
                  icon={MessageSquare}
                  color="blue"
                />
                <StatCard
                  label="Hate Speech Detected"
                  value={data.stats.hateComments.toLocaleString()}
                  subtext={`${(
                    (data.stats.hateComments / data.stats.totalComments) *
                    100
                  ).toFixed(1)}% of total`}
                  icon={Flame}
                  color="red"
                />
                <StatCard
                  label="Offensive Content"
                  value={data.stats.offensiveComments.toLocaleString()}
                  subtext={`${(
                    (data.stats.offensiveComments / data.stats.totalComments) *
                    100
                  ).toFixed(1)}% of total`}
                  icon={AlertTriangle}
                  color="orange"
                />
                <StatCard
                  label="Grok Interactions"
                  value={data.stats.grokResponses.toLocaleString()}
                  subtext="AI assistant mentions"
                  icon={Zap}
                  color="purple"
                />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[600px]">
                <div className="lg:col-span-2 h-full">
                  <Scene3D data={data.comments} />
                </div>
                <div className="space-y-6 overflow-y-auto">
                  <SentimentChart stats={data.stats} />
                  <TopUsersChart stats={data.stats} />
                </div>
              </div>
            </div>
          )}

          {/* User Audit View */}
          {activeTab === "users" && (
            <UserTable data={data.comments} stats={data.stats} />
          )}

          {/* Detector Tool */}
          {activeTab === "detector" && <SlopDetector />}

          {/* Data View */}
          {activeTab === "data" && (
            <div className="grid gap-6">
              <CrawlerConsole />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                  <h3 className="text-white font-semibold mb-4">
                    Dataset Information
                  </h3>
                  <div className="space-y-3">
                    <div className="flex justify-between p-3 bg-slate-900/50 rounded-lg">
                      <span className="text-slate-400">Source File</span>
                      <span className="text-white font-mono text-sm">
                        final_merged_data_nlp.csv
                      </span>
                    </div>
                    <div className="flex justify-between p-3 bg-slate-900/50 rounded-lg">
                      <span className="text-slate-400">Total Rows</span>
                      <span className="text-white">181,076</span>
                    </div>
                    <div className="flex justify-between p-3 bg-slate-900/50 rounded-lg">
                      <span className="text-slate-400">Loaded Comments</span>
                      <span className="text-emerald-400">
                        {data.stats.totalComments.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between p-3 bg-slate-900/50 rounded-lg">
                      <span className="text-slate-400">Unique Users</span>
                      <span className="text-white">
                        {data.stats.uniqueUsers.toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                  <h3 className="text-white font-semibold mb-4">
                    Feature Columns
                  </h3>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {[
                      "sentiment_label",
                      "sentiment_prob",
                      "hate_label",
                      "hate_prob",
                      "offensive_label",
                      "offensive_prob",
                      "irony_label",
                      "irony_prob",
                      "num_emojis",
                      "num_caps_words",
                      "tagged_grok",
                      "used_slang",
                      "aggression",
                      "anger",
                      "rage",
                      "violence",
                    ].map((col) => (
                      <div
                        key={col}
                        className="bg-slate-900/50 px-2 py-1.5 rounded text-slate-300 font-mono"
                      >
                        {col}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                <h3 className="text-white font-semibold mb-4">NLP Pipeline</h3>
                <p className="text-slate-400 text-sm mb-4">
                  This dataset was processed using multiple NLP models to
                  extract features:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-slate-900/50 p-4 rounded-lg">
                    <h4 className="text-blue-400 font-semibold mb-2">
                      Sentiment Analysis
                    </h4>
                    <p className="text-slate-400 text-xs">
                      Twitter-RoBERTa based model for political sentiment
                      classification (positive/negative/neutral)
                    </p>
                  </div>
                  <div className="bg-slate-900/50 p-4 rounded-lg">
                    <h4 className="text-red-400 font-semibold mb-2">
                      Hate Speech Detection
                    </h4>
                    <p className="text-slate-400 text-xs">
                      Fine-tuned transformer model for detecting hate speech and
                      offensive language
                    </p>
                  </div>
                  <div className="bg-slate-900/50 p-4 rounded-lg">
                    <h4 className="text-purple-400 font-semibold mb-2">
                      Empath Features
                    </h4>
                    <p className="text-slate-400 text-xs">
                      Lexical analysis for 194 emotional and topical categories
                      including rage, politics, violence
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
