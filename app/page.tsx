"use client";
/* eslint-disable @next/next/no-img-element */

import { useMemo, useState } from "react";
import {
  AlertTriangle,
  Archive,
  ArrowRight,
  BadgeHelp,
  Biohazard,
  BookOpen,
  Box,
  Check,
  ChevronRight,
  CircleDot,
  Compass,
  ExternalLink,
  Eye,
  Footprints,
  Gauge,
  Layers3,
  Map as MapIcon,
  PackageOpen,
  Play,
  Radar,
  Route as RouteIcon,
  Search,
  ShieldAlert,
  Skull,
  Sparkles,
  Swords,
  Target,
  Trees,
  WandSparkles,
  Zap,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  NativeSelect,
  NativeSelectOption,
} from "@/components/ui/native-select";
import { Progress } from "@/components/ui/progress";
import {
  Tabs,
  TabsContent,
} from "@/components/ui/tabs";
import {
  BOSS_GUIDES,
  DEVICE_GUIDES,
  ELITE_GUIDES,
  ENDINGS,
  EVENT_NOTES,
  EVENT_POOLS,
  LAYER_NODE_POOLS,
  NODE_DATA,
  PARTS,
  SOURCES,
  UNKNOWN_POOLS,
  type NodeRecord,
} from "./game-data";
import { STAGE_DATA, type StageRecord } from "./stage-data";

type PlannerResult = {
  route: string[];
  score: number;
  actionCost: number;
  reasons: string[];
  originalIndex: number;
};

type EnemyType = "普通" | "精英" | "领袖" | "装置";

type EnemyEntry = {
  name: string;
  type: EnemyType;
  codes: string[];
  icon: string;
  stages: string[];
  mechanic: string;
  tip: string;
};

const ROMAN = ["0", "I", "II", "III", "IV", "V", "VI"];
const NODE_BY_NAME = new Map(NODE_DATA.map((node) => [node.name, node]));
const BOSS_NAMES = new Set(Object.keys(BOSS_GUIDES));
const ELITE_NAMES = new Set(Object.keys(ELITE_GUIDES));

function prtsUrl(name: string) {
  return "https://prts.wiki/w/" + encodeURIComponent(name);
}

function bilibiliUrl(name: string) {
  return (
    "https://search.bilibili.com/all?keyword=" +
    encodeURIComponent("明日方舟 黑流树海 " + name + " 攻略")
  );
}

function classifyEnemy(name: string): EnemyType {
  if (BOSS_NAMES.has(name)) return "领袖";
  if (ELITE_NAMES.has(name)) return "精英";
  return "普通";
}

function mechanismFor(name: string, type: EnemyType) {
  if (BOSS_GUIDES[name]) {
    return BOSS_GUIDES[name].mechanic[0];
  }
  if (ELITE_GUIDES[name]) return ELITE_GUIDES[name].mechanic[0];
  if (name.includes("空值体")) {
    return "伪装成既有敌人并继承其能力；首次受到伤害后显露真身。";
  }
  if (name.includes("猎犬proto")) {
    return "追猎相关强化单位；通过的追猎越多，本局后续个体越危险。";
  }
  if (name.includes("火种")) {
    return "沿地面移动并在接触近战单位后爆炸，造成法术与元素压力。";
  }
  if (name.includes("痛楚化身")) {
    return "卡德霍阶段收尾单位；通常需要在本体倒下后连续处理。";
  }
  if (name.includes("术师") || name.includes("法杖")) {
    return "远程法术威胁；优先观察索敌范围与站位连锁。";
  }
  if (name.includes("猎手") || name.includes("弩手") || name.includes("炮")) {
    return "远程火力单位；先用承伤位稳定索敌，再安排速杀。";
  }
  if (name.includes("虫")) {
    return "数量与路线变化较大，容易形成漏怪或爆炸连锁。";
  }
  return "PRTS未标注需要单独展开的特殊机制；具体属性以对应关卡为准。";
}

function tipFor(name: string, type: EnemyType) {
  if (BOSS_GUIDES[name]) return BOSS_GUIDES[name].plan[0];
  if (ELITE_GUIDES[name]) return ELITE_GUIDES[name].plan[0];
  if (name.includes("空值体")) return "先用低成本伤害识破，再按显露后的本体机制处理。";
  if (name.includes("虫")) return "避免在己方密集站位附近触发死亡效果。";
  if (type === "领袖") return "先看路线与阶段节点，把主爆发留给不可回避的接敌窗口。";
  if (type === "精英") return "先阅读完整机制，再决定承伤位和输出时机。";
  return "无额外处理建议；结合关卡路线和出场顺序处理。";
}

const ENEMY_INDEX: EnemyEntry[] = (() => {
  const map = new Map<
    string,
    { codes: Set<string>; icons: Set<string>; stages: Set<string>; type: EnemyType }
  >();

  for (const stage of STAGE_DATA) {
    for (const enemy of stage.enemies) {
      const found = map.get(enemy.name) ?? {
        codes: new Set<string>(),
        icons: new Set<string>(),
        stages: new Set<string>(),
        type: classifyEnemy(enemy.name),
      };
      if (enemy.code) found.codes.add(enemy.code);
      if (enemy.icon) found.icons.add(enemy.icon);
      found.stages.add(stage.name);
      if (BOSS_NAMES.has(enemy.name)) found.type = "领袖";
      map.set(enemy.name, found);
    }
  }

  const enemies = Array.from(map.entries()).map(([name, data]) => ({
    name,
    type: data.type,
    codes: Array.from(data.codes),
    icon: Array.from(data.icons)[0] ?? "",
    stages: Array.from(data.stages),
    mechanic: mechanismFor(name, data.type),
    tip: tipFor(name, data.type),
  }));

  const devices: EnemyEntry[] = DEVICE_GUIDES.map((device) => ({
    name: device.name,
    type: "装置",
    codes: [],
    icon: "",
    stages: STAGE_DATA.filter(
      (stage) =>
        stage.intro.includes(device.name) ||
        stage.terrain.some((terrain) => terrain.name.includes(device.name)),
    ).map((stage) => stage.name),
    mechanic: device.mechanic,
    tip: device.tip,
  }));

  const rank: Record<EnemyType, number> = {
    领袖: 0,
    精英: 1,
    普通: 2,
    装置: 3,
  };

  return [...enemies, ...devices].sort(
    (a, b) => rank[a.type] - rank[b.type] || a.name.localeCompare(b.name, "zh"),
  );
})();

const EVENT_COUNT = new Set(Object.values(EVENT_POOLS).flat()).size;

function selectFromPool(floor: number, preferred: string[]) {
  const pool = LAYER_NODE_POOLS[floor] ?? [];
  const selected = preferred.filter((name) => pool.includes(name));
  for (const node of pool) {
    if (selected.length >= 5) break;
    if (!selected.includes(node)) selected.push(node);
  }
  return selected.slice(0, 5);
}

function makeRoutes(floor: number) {
  return [
    selectFromPool(floor, [
      "不期而遇",
      "先行一步",
      "诡意行商",
      "命运所指",
      "险路恶敌",
      "羽瞰点",
    ]),
    selectFromPool(floor, [
      "秘境行商",
      "误入奇境",
      "狭路相逢",
      "失与得",
      "紧急作战",
    ]),
    selectFromPool(floor, [
      "紧急作战",
      "得偿所愿",
      "不期而遇",
      "应急助力",
      "险路小径",
      "险路尽头",
    ]),
  ];
}

function nodeExpectedScore(name: string, floor: number) {
  if (name === "未知·诡秘" || name === "未知·凶戾") {
    const pool =
      name === "未知·诡秘"
        ? UNKNOWN_POOLS[floor].mystery
        : UNKNOWN_POOLS[floor].ferocity;
    if (!pool.length) return 0;
    return (
      pool.reduce(
        (sum, candidate) => sum + (NODE_BY_NAME.get(candidate)?.baseScore ?? 0),
        0,
      ) / pool.length
    );
  }
  return NODE_BY_NAME.get(name)?.baseScore ?? 0;
}

function scoreRoute(
  route: string[],
  floor: number,
  endingId: string,
  parts: Set<string>,
  actions: number,
  originalIndex: number,
): PlannerResult {
  const ending = ENDINGS.find((item) => item.id === endingId) ?? ENDINGS[0];
  let score = 0;
  let actionCost = 0;
  const reasons: string[] = [];
  const thirdGoal = ["3", "normal13", "hunt13", "23"].includes(endingId);
  const secondGoal = ["2", "23"].includes(endingId);

  for (const name of route) {
    score += nodeExpectedScore(name, floor);
    actionCost += name === "羽瞰点" ? 0 : 1;

    if (ending.priority.includes(name)) {
      score += 3;
      reasons.push(name + " 命中 " + ending.short + " 目标优先级");
    }

    if (secondGoal && floor <= 4 && !parts.has("alpha") && name === "不期而遇") {
      score += 7;
      reasons.push("缺沙盘α：提高“线人”事件覆盖");
    }
    if (secondGoal && floor <= 3 && !parts.has("beta") && name === "诡意行商") {
      score += 6;
      reasons.push("缺沙盘β：优先检查坎诺特库存");
    }
    if (secondGoal && floor === 5 && name === "命运所指") {
      score += 12;
      reasons.push("V层二结局抉择为硬门槛");
    }

    if (
      thirdGoal &&
      floor >= 2 &&
      floor <= 4 &&
      !parts.has("beacon") &&
      name === "先行一步"
    ) {
      score += 11;
      reasons.push("缺怦然信标：先行一步是三结局硬门槛");
    }
    if (thirdGoal && !parts.has("key") && name === "失与得") {
      score += 4;
      reasons.push("为三结局代价链保留交换窗口");
    }
    if (thirdGoal && !parts.has("key") && name === "不期而遇") {
      score += 3;
      reasons.push("寻找泪之聚落等三结局代价事件");
    }

    if (parts.has("seed") && name === "秘境行商") {
      score += 6;
      reasons.push("种子可在园圃培育");
    }
    if (parts.has("natural") && name === "秘境行商") {
      score += 3;
      reasons.push("高估价自然物可在机械师处变现");
    }
    if (parts.has("natural") && name === "失与得") {
      score += 2;
    }
    if (parts.has("processed") && name === "误入奇境") {
      score += 6;
      reasons.push("有加工品，可支付黑潭入口");
    }
    if (parts.has("relic") && name === "失与得") {
      score += 5;
      reasons.push("可交换收藏品转化为有效收益");
    }
    if (parts.has("cage") && name === "不期而遇") {
      score += 2.5;
      reasons.push("笼控器提高“黑诞”事件链价值");
    }
    if (parts.has("ticket") && name === "险路尽头") {
      score += 4;
      reasons.push("险路尽头可取出留存招募券");
    }
    if (parts.has("ingots") && ["诡意行商", "秘境行商", "得偿所愿"].includes(name)) {
      score += 2;
      reasons.push("当前锭量支持付费刷新或采购");
    }
    if (endingId === "hunt13" && name === "追猎") {
      score += 8;
    }
  }

  if (route.includes("追猎")) {
    actionCost = actions;
  }

  if (parts.has("processed")) {
    actionCost = Math.max(0, actionCost - 1);
  }

  if (actionCost > actions) {
    score -= (actionCost - actions) * 9;
    reasons.push("行动力不足，存在追猎风险");
  } else {
    score += Math.min(3, actions - actionCost) * 0.8;
  }

  return {
    route,
    score: Math.round(score * 10) / 10,
    actionCost,
    reasons: Array.from(new Set(reasons)).slice(0, 4),
    originalIndex,
  };
}

function nodeTone(name: string) {
  if (name.includes("紧急") || name.includes("恶敌") || name.includes("凶戾")) {
    return "danger";
  }
  if (
    name.includes("行商") ||
    name.includes("得偿") ||
    name.includes("秘") ||
    name.includes("诡秘")
  ) {
    return "gold";
  }
  if (name.includes("先行") || name.includes("命运") || name.includes("奇境")) {
    return "violet";
  }
  return "green";
}

function EnemyAvatar({
  name,
  type,
  onClick,
  size = "md",
  icon,
}: {
  name: string;
  type: EnemyType;
  onClick?: () => void;
  size?: "sm" | "md" | "lg";
  icon?: string;
}) {
  const Icon = type === "领袖" ? Skull : type === "装置" ? Box : Biohazard;
  const iconUrl = icon
    ? icon.startsWith("/")
      ? "https://tomimi.dev" + icon
      : icon
    : "";
  const content = (
    <>
      <span className="enemy-scan-line" />
      {iconUrl ? <img src={iconUrl} alt="" loading="lazy" /> : <Icon />}
      <small>{type === "领袖" ? "BOSS" : type === "装置" ? "DEV" : type === "精英" ? "ELITE" : "UNIT"}</small>
    </>
  );
  if (!onClick) {
    return (
      <div
        className={"enemy-avatar enemy-avatar-" + size + " enemy-avatar-" + type}
        aria-hidden="true"
      >
        {content}
      </div>
    );
  }
  return (
    <button
      type="button"
      className={"enemy-avatar enemy-avatar-" + size + " enemy-avatar-" + type}
      onClick={onClick}
      aria-label={"打开 " + name + " 档案"}
    >
      {content}
    </button>
  );
}

function SectionHeader({
  kicker,
  title,
  copy,
}: {
  kicker: string;
  title: string;
  copy: string;
}) {
  return (
    <div className="section-heading">
      <div>
        <span className="eyebrow">{kicker}</span>
        <h2>{title}</h2>
      </div>
      <p>{copy}</p>
    </div>
  );
}

function NodeGlyph({ node }: { node: string }) {
  const tone = nodeTone(node);
  const Icon =
    tone === "danger"
      ? Swords
      : tone === "gold"
        ? Sparkles
        : tone === "violet"
          ? WandSparkles
          : CircleDot;
  return (
    <span className={"node-glyph tone-" + tone}>
      <Icon />
    </span>
  );
}

function stageAttention(stage: StageRecord) {
  const notes: string[] = [];
  if (stage.emergency) notes.push("紧急作战：" + stage.emergency);
  if (stage.total.includes("~")) notes.push("敌人总数存在随机区间，路线/替换会改变实际数量。");
  if (stage.enemies.some((enemy) => BOSS_NAMES.has(enemy.name))) {
    notes.push("领袖关：把爆发按阶段拆分，不要一次性清空技能。");
  }
  if (stage.terrain.some((terrain) => terrain.name.includes("地穴"))) {
    notes.push("存在地穴：位移或诱导可以显著压低正面压力。");
  }
  if (stage.intro.includes("草丛")) {
    notes.push("草丛提供隐匿；高台落点与远程索敌顺序要一起规划。");
  }
  if (!notes.length) notes.push("先核对出入口和高压敌人的首次接敌时间，再决定开局站位。");
  return notes.slice(0, 4);
}

function basicPlan(stage: StageRecord) {
  const boss = stage.enemies.find((enemy) => BOSS_GUIDES[enemy.name]);
  if (boss) return BOSS_GUIDES[boss.name].plan;
  const plan = [
    "开局先建立主路线阻挡与持续输出，保留一个机动位应对支路或随机精英。",
    "中段按敌人数量最高的一组准备群攻；远程与高机动敌人优先点杀。",
  ];
  if (stage.terrain.length) {
    plan.push("把“" + stage.terrain.map((item) => item.name).join(" / ") + "”作为阵地设计的一部分。");
  }
  return plan;
}

export default function Home() {
  const [activeTab, setActiveTab] = useState("planner");
  const [floor, setFloor] = useState(3);
  const [ending, setEnding] = useState("3");
  const [actions, setActions] = useState(6);
  const [ownedParts, setOwnedParts] = useState<Set<string>>(
    new Set(["processed"]),
  );
  const [routes, setRoutes] = useState<string[][]>(() => makeRoutes(3));
  const [plannerRun, setPlannerRun] = useState(1);
  const [nodeQuery, setNodeQuery] = useState("");
  const [nodeKind, setNodeKind] = useState("全部");
  const [nodeFloor, setNodeFloor] = useState(0);
  const [selectedNode, setSelectedNode] = useState<NodeRecord | null>(null);
  const [unknownFloor, setUnknownFloor] = useState(4);
  const [unknownType, setUnknownType] = useState<"mystery" | "ferocity">(
    "mystery",
  );
  const [stageQuery, setStageQuery] = useState("");
  const [stageKind, setStageKind] = useState("全部");
  const [selectedStage, setSelectedStage] = useState<StageRecord | null>(null);
  const [enemyQuery, setEnemyQuery] = useState("");
  const [enemyType, setEnemyType] = useState("全部");
  const [enemyLimit, setEnemyLimit] = useState(48);
  const [selectedEnemy, setSelectedEnemy] = useState<EnemyEntry>(
    ENEMY_INDEX.find((enemy) => enemy.name === "卡德霍，黑流之源") ??
      ENEMY_INDEX[0],
  );

  const endingData = ENDINGS.find((item) => item.id === ending) ?? ENDINGS[0];

  const plannerResults = useMemo(
    () =>
      routes
        .map((route, index) =>
          scoreRoute(route, floor, ending, ownedParts, actions, index),
        )
        .sort((a, b) => b.score - a.score),
    [routes, floor, ending, ownedParts, actions],
  );

  const filteredNodes = useMemo(
    () =>
      NODE_DATA.filter((node) => {
        const matchesQuery =
          !nodeQuery ||
          node.name.includes(nodeQuery) ||
          node.group.includes(nodeQuery) ||
          node.possible.some((item) => item.includes(nodeQuery));
        const matchesKind = nodeKind === "全部" || node.kind === nodeKind;
        const matchesFloor = nodeFloor === 0 || node.layers.includes(nodeFloor);
        return matchesQuery && matchesKind && matchesFloor;
      }),
    [nodeQuery, nodeKind, nodeFloor],
  );

  const unknownCandidates =
    UNKNOWN_POOLS[unknownFloor]?.[unknownType] ?? [];

  const filteredStages = useMemo(
    () =>
      STAGE_DATA.filter((stage) => {
        const queryMatch =
          !stageQuery ||
          stage.name.includes(stageQuery) ||
          stage.enemies.some((enemy) => enemy.name.includes(stageQuery)) ||
          stage.intro.includes(stageQuery);
        const kindMatch = stageKind === "全部" || stage.kind === stageKind;
        return queryMatch && kindMatch;
      }),
    [stageQuery, stageKind],
  );

  const filteredEnemies = useMemo(
    () =>
      ENEMY_INDEX.filter((enemy) => {
        const queryMatch =
          !enemyQuery ||
          enemy.name.includes(enemyQuery) ||
          enemy.codes.some((code) => code.includes(enemyQuery)) ||
          enemy.stages.some((stage) => stage.includes(enemyQuery));
        return queryMatch && (enemyType === "全部" || enemy.type === enemyType);
      }),
    [enemyQuery, enemyType],
  );

  function togglePart(id: string) {
    setOwnedParts((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function updateRoute(routeIndex: number, stepIndex: number, value: string) {
    setRoutes((current) =>
      current.map((route, index) =>
        index === routeIndex
          ? route.map((node, step) => (step === stepIndex ? value : node))
          : route,
      ),
    );
  }

  function openEnemyFromStage(name: string) {
    const enemy = ENEMY_INDEX.find((item) => item.name === name);
    if (enemy) setSelectedEnemy(enemy);
    setEnemyQuery(name);
    setActiveTab("archive");
    setSelectedStage(null);
    window.setTimeout(
      () =>
        document
          .getElementById("workspace-tabs")
          ?.scrollIntoView({ behavior: "smooth", block: "start" }),
      50,
    );
  }

  return (
    <main className="site-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="黑流树海路线参谋首页">
          <span className="brand-mark">
            <Trees />
          </span>
          <span>
            <strong>BLACKFLOW // ROUTE LAB</strong>
            <small>黑流树海路线参谋</small>
          </span>
        </a>
        <nav className="top-links" aria-label="功能索引">
          <button className={activeTab === "planner" ? "active" : ""} aria-current={activeTab === "planner" ? "page" : undefined} onClick={() => setActiveTab("planner")}><RouteIcon />路线决策</button>
          <button className={activeTab === "nodes" ? "active" : ""} aria-current={activeTab === "nodes" ? "page" : undefined} onClick={() => setActiveTab("nodes")}><Layers3 />节点图鉴</button>
          <button className={activeTab === "unknown" ? "active" : ""} aria-current={activeTab === "unknown" ? "page" : undefined} onClick={() => setActiveTab("unknown")}><Radar />未知反查</button>
          <button className={activeTab === "stages" ? "active" : ""} aria-current={activeTab === "stages" ? "page" : undefined} onClick={() => setActiveTab("stages")}><Swords />作战检索</button>
          <button className={activeTab === "archive" ? "active" : ""} aria-current={activeTab === "archive" ? "page" : undefined} onClick={() => setActiveTab("archive")}><Archive />敌人档案馆</button>
        </nav>
        <div className="live-pill">
          <span />
          数据快照 2026.08
        </div>
      </header>

      {activeTab === "planner" && <section id="top" className="hero">
        <div className="hero-grid" />
        <div className="hero-orbit orbit-one" />
        <div className="hero-orbit orbit-two" />
        <div className="hero-copy">
          <span className="eyebrow">IS-6 // 沉沦者的黑流树海</span>
          <h1>
            把未知节点
            <br />
            变成<span>可计算路线</span>
          </h1>
          <p>
            输入结局目标、层数、行动力与背包零件，比较地图上的候选线路。
            同时反查诡秘/凶戾节点，搜索全关卡编成与敌人机制。
          </p>
          <div className="hero-actions">
            <Button
              size="lg"
              className="primary-cta"
              onClick={() => {
                setActiveTab("planner");
                document
                  .getElementById("workspace-tabs")
                  ?.scrollIntoView({ behavior: "smooth" });
              }}
            >
              <RouteIcon />
              开始规划
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="ghost-cta"
              onClick={() => {
                setActiveTab("unknown");
                document
                  .getElementById("workspace-tabs")
                  ?.scrollIntoView({ behavior: "smooth" });
              }}
            >
              <BadgeHelp />
              反查未知节点
            </Button>
          </div>
        </div>

        <div className="hero-console">
          <div className="console-top">
            <span>ACTIVE RECOMMENDATION</span>
            <span className="console-signal">SIGNAL 98%</span>
          </div>
          <div className="console-target">
            <span>当前目标</span>
            <strong>{endingData.name}</strong>
          </div>
          <div className="mini-route">
            {plannerResults[0]?.route.slice(0, 4).map((node, index) => (
              <div className="mini-route-step" key={node + index}>
                <NodeGlyph node={node} />
                <small>{node}</small>
                {index < 3 && <ChevronRight className="route-chevron" />}
              </div>
            ))}
          </div>
          <div className="console-score">
            <div>
              <span>期望收益</span>
              <strong>{plannerResults[0]?.score.toFixed(1)}</strong>
            </div>
            <div>
              <span>行动消耗</span>
              <strong>{plannerResults[0]?.actionCost}</strong>
            </div>
            <div>
              <span>当前层</span>
              <strong>{ROMAN[floor]}</strong>
            </div>
          </div>
          <Progress
            value={Math.min(100, (plannerResults[0]?.score ?? 0) * 3.4)}
            className="console-progress"
          />
          <p className="console-note">
            基于“所有作战均可通过”的收益模型；随机事件按当前层事件池估值。
          </p>
        </div>
      </section>}

      <section className="metric-strip">
        <div>
          <strong>{NODE_DATA.length}</strong>
          <span>节点类别</span>
        </div>
        <div>
          <strong>{EVENT_COUNT}</strong>
          <span>不期而遇事件</span>
        </div>
        <div>
          <strong>{STAGE_DATA.length}</strong>
          <span>作战档案</span>
        </div>
        <div>
          <strong>{ENEMY_INDEX.filter((item) => item.type !== "装置").length}</strong>
          <span>敌方单位索引</span>
        </div>
        <div>
          <strong>{DEVICE_GUIDES.length}</strong>
          <span>核心装置机制</span>
        </div>
      </section>

      <section id="workspace-tabs" className="workspace">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsContent value="planner" className="tab-panel">
            <SectionHeader
              kicker="ROUTE OPTIMIZER"
              title="按结局与背包，重排三条地图路线"
              copy="把你在游戏地图上看到的节点依次填入。评分包含结局硬门槛、零件变现、行动力与未知节点期望。"
            />

            <div className="planner-layout">
              <aside className="control-panel">
                <div className="panel-title">
                  <Gauge />
                  <div>
                    <span>INPUT MATRIX</span>
                    <strong>探索状态</strong>
                  </div>
                </div>

                <label className="control-label">
                  <span>目标结局</span>
                  <NativeSelect
                    value={ending}
                    onChange={(event) => setEnding(event.target.value)}
                    className="control-select"
                  >
                    {ENDINGS.map((item) => (
                      <NativeSelectOption value={item.id} key={item.id}>
                        {item.name}
                      </NativeSelectOption>
                    ))}
                  </NativeSelect>
                </label>

                <div className="control-row">
                  <label className="control-label">
                    <span>当前层</span>
                    <NativeSelect
                      value={String(floor)}
                      onChange={(event) => {
                        const nextFloor = Number(event.target.value);
                        setFloor(nextFloor);
                        setRoutes(makeRoutes(nextFloor));
                      }}
                      className="control-select"
                    >
                      {[1, 2, 3, 4, 5, 6].map((value) => (
                        <NativeSelectOption value={String(value)} key={value}>
                          {ROMAN[value]} 层
                        </NativeSelectOption>
                      ))}
                    </NativeSelect>
                  </label>
                  <label className="control-label">
                    <span>剩余行动力</span>
                    <Input
                      type="number"
                      min={0}
                      max={20}
                      value={actions}
                      onChange={(event) =>
                        setActions(Math.max(0, Number(event.target.value)))
                      }
                      className="control-input"
                    />
                  </label>
                </div>

                <div className="hard-gates">
                  <span className="control-caption">当前目标硬门槛</span>
                  {endingData.must.map((item) => (
                    <div key={item}>
                      <Check />
                      <span>{item}</span>
                    </div>
                  ))}
                </div>

                <div className="part-control">
                  <span className="control-caption">背包 / 关键状态</span>
                  <div className="part-grid">
                    {PARTS.map((part) => {
                      const active = ownedParts.has(part.id);
                      return (
                        <button
                          type="button"
                          key={part.id}
                          className={active ? "part-chip active" : "part-chip"}
                          onClick={() => togglePart(part.id)}
                          title={part.helps}
                        >
                          {active ? <Check /> : <PackageOpen />}
                          {part.label}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <Button
                  className="planner-button"
                  onClick={() => setPlannerRun((value) => value + 1)}
                >
                  <Zap />
                  重算收益排序
                </Button>
                <p className="run-stamp">
                  SCORE PASS #{String(plannerRun).padStart(3, "0")}
                </p>
              </aside>

              <div className="route-workbench">
                <div className="route-editor-head">
                  <div>
                    <span className="control-caption">地图候选线路</span>
                    <p>每一格选择实际可见节点；未知节点可直接保留为占位。</p>
                  </div>
                  <Badge variant="outline">作战胜率假设：100%</Badge>
                </div>

                <div className="route-editors">
                  {routes.map((route, routeIndex) => (
                    <div className="route-editor" key={routeIndex}>
                      <div className="route-label">
                        <strong>路线 {String.fromCharCode(65 + routeIndex)}</strong>
                        <span>
                          当前 #{plannerResults.findIndex(
                            (item) => item.originalIndex === routeIndex,
                          ) + 1}
                        </span>
                      </div>
                      <div className="route-node-row">
                        {route.map((node, stepIndex) => (
                          <div className="route-node-select" key={stepIndex}>
                            <NodeGlyph node={node} />
                            <NativeSelect
                              value={node}
                              onChange={(event) =>
                                updateRoute(
                                  routeIndex,
                                  stepIndex,
                                  event.target.value,
                                )
                              }
                              className="node-select"
                            >
                              <NativeSelectOption value="未知·诡秘">
                                未知·诡秘
                              </NativeSelectOption>
                              <NativeSelectOption value="未知·凶戾">
                                未知·凶戾
                              </NativeSelectOption>
                              {floor <= 5 && (
                                <NativeSelectOption value="追猎">
                                  追猎（节点外）
                                </NativeSelectOption>
                              )}
                              {(LAYER_NODE_POOLS[floor] ?? []).map((option) => (
                                <NativeSelectOption value={option} key={option}>
                                  {option}
                                </NativeSelectOption>
                              ))}
                            </NativeSelect>
                            {stepIndex < route.length - 1 && (
                              <ArrowRight className="editor-arrow" />
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="result-stack">
                  {plannerResults.map((result, rank) => (
                    <article
                      className={rank === 0 ? "result-card winner" : "result-card"}
                      key={result.originalIndex}
                    >
                      <div className="result-rank">
                        <span>{rank === 0 ? "推荐" : "备选"}</span>
                        <strong>0{rank + 1}</strong>
                      </div>
                      <div className="result-main">
                        <div className="result-title">
                          <div>
                            <strong>
                              路线 {String.fromCharCode(65 + result.originalIndex)}
                            </strong>
                            {rank === 0 && (
                              <Badge className="best-badge">最高期望</Badge>
                            )}
                          </div>
                          <span>行动 {result.actionCost} / {actions}</span>
                        </div>
                        <div className="result-path">
                          {result.route.map((node, index) => (
                            <span key={node + index}>
                              {node}
                              {index < result.route.length - 1 && (
                                <ChevronRight />
                              )}
                            </span>
                          ))}
                        </div>
                        <div className="reason-list">
                          {result.reasons.length ? (
                            result.reasons.map((reason) => (
                              <span key={reason}>
                                <CircleDot />
                                {reason}
                              </span>
                            ))
                          ) : (
                            <span>
                              <CircleDot />
                              以基础节点收益与行动力余量排序
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="result-score">
                        <span>EV SCORE</span>
                        <strong>{result.score.toFixed(1)}</strong>
                        <small>
                          {result.actionCost <= actions ? "可达" : "追猎风险"}
                        </small>
                      </div>
                    </article>
                  ))}
                </div>
              </div>
            </div>

            <div className="model-note">
              <AlertTriangle />
              <div>
                <strong>模型边界</strong>
                <p>
                  这是路线选择器，不是地图识别器。节点连通性由你录入的候选线路保证；
                  分数是可解释的收益权重，不会虚构游戏内精确概率。未知节点按该层合法候选的平均值计算。
                </p>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="nodes" className="tab-panel">
            <SectionHeader
              kicker="NODE ENCYCLOPEDIA"
              title="作战 / 非作战节点全分类"
              copy="按层数过滤出现范围。点击节点查看关卡池、事件池、三档奖励、商店分支与出现限制。"
            />

            <div className="filter-bar">
              <div className="search-box">
                <Search />
                <Input
                  value={nodeQuery}
                  onChange={(event) => setNodeQuery(event.target.value)}
                  placeholder="搜节点、事件或关卡，例如：线人 / 纵怒"
                />
              </div>
              <NativeSelect
                value={nodeKind}
                onChange={(event) => setNodeKind(event.target.value)}
                className="filter-select"
              >
                {["全部", "作战", "非作战"].map((item) => (
                  <NativeSelectOption value={item} key={item}>
                    {item}节点
                  </NativeSelectOption>
                ))}
              </NativeSelect>
              <NativeSelect
                value={String(nodeFloor)}
                onChange={(event) => setNodeFloor(Number(event.target.value))}
                className="filter-select"
              >
                <NativeSelectOption value="0">全部层数</NativeSelectOption>
                {[1, 2, 3, 4, 5, 6].map((value) => (
                  <NativeSelectOption value={String(value)} key={value}>
                    {ROMAN[value]} 层
                  </NativeSelectOption>
                ))}
              </NativeSelect>
              <span className="result-count">{filteredNodes.length} 项</span>
            </div>

            <div className="node-category-label">
              <span className="combat-dot" />
              作战节点包含普通、紧急、居民、据点、恶敌与追猎
              <span className="utility-dot" />
              其余节点按事件 / 商店 / 传送 / 出口归入非作战
            </div>

            <div className="node-grid">
              {filteredNodes.map((node) => (
                <button
                  type="button"
                  className={"node-card " + (node.kind === "作战" ? "combat" : "utility")}
                  key={node.name}
                  onClick={() => setSelectedNode(node)}
                >
                  <div className="node-card-top">
                    <NodeGlyph node={node.name} />
                    <div>
                      <span>{node.group}</span>
                      <h3>{node.name}</h3>
                    </div>
                    <ChevronRight />
                  </div>
                  <p>{node.summary}</p>
                  <div className="layer-badges">
                    {node.layers.map((layer) => (
                      <span key={layer}>
                        {ROMAN[layer]}
                        {node.count[layer] ? " · " + node.count[layer] : ""}
                      </span>
                    ))}
                  </div>
                  <div className="node-preview">
                    {node.possible.slice(0, 2).map((item) => (
                      <span key={item}>{item}</span>
                    ))}
                  </div>
                </button>
              ))}
            </div>

            <div className="event-matrix">
              <div className="event-matrix-head">
                <div>
                  <span className="eyebrow">ENCOUNTER POOL</span>
                  <h3>不期而遇 · 分层事件池</h3>
                </div>
                <span>{EVENT_COUNT} 个独立事件名</span>
              </div>
              <div className="event-columns">
                {Object.entries(EVENT_POOLS).map(([eventFloor, events]) => (
                  <div className="event-column" key={eventFloor}>
                    <div className="event-floor">
                      <strong>{ROMAN[Number(eventFloor)]}</strong>
                      <span>{events.length} possibilities</span>
                    </div>
                    {events.map((event) => (
                      <div className="event-row" key={event}>
                        <span>{event}</span>
                        <small>
                          {EVENT_NOTES[event] ?? "事件选项随前置、资源与已持有物变化。"}
                        </small>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="unknown" className="tab-panel">
            <SectionHeader
              kicker="FOG REVERSE LOOKUP"
              title="未知诡秘 / 未知凶戾候选反查"
              copy="先按层数排除不合法节点，再结合立即揭示、固定出口与居民据点状态做人工二次缩小。"
            />

            <div className="unknown-console">
              <aside className="unknown-controls">
                <div className="panel-title">
                  <Eye />
                  <div>
                    <span>OBSERVATION</span>
                    <strong>观测条件</strong>
                  </div>
                </div>
                <label className="control-label">
                  <span>节点所在层</span>
                  <NativeSelect
                    value={String(unknownFloor)}
                    onChange={(event) =>
                      setUnknownFloor(Number(event.target.value))
                    }
                    className="control-select"
                  >
                    {[1, 2, 3, 4, 5, 6].map((value) => (
                      <NativeSelectOption value={String(value)} key={value}>
                        {ROMAN[value]} 层
                      </NativeSelectOption>
                    ))}
                  </NativeSelect>
                </label>
                <div className="unknown-toggle">
                  <button
                    type="button"
                    className={unknownType === "mystery" ? "active" : ""}
                    onClick={() => setUnknownType("mystery")}
                  >
                    <Sparkles />
                    <span>未知的诡秘</span>
                    <small>事件 / 商店 / 视野</small>
                  </button>
                  <button
                    type="button"
                    className={unknownType === "ferocity" ? "active" : ""}
                    onClick={() => setUnknownType("ferocity")}
                  >
                    <ShieldAlert />
                    <span>未知的凶戾</span>
                    <small>作战 / 居民 / 恶敌</small>
                  </button>
                </div>
                <div className="deduction-note">
                  <Radar />
                  <p>
                    固定揭示的出口、曲折密道等通常不应继续保留为未知候选；
                    据点和流窜居民还受保密等级与本层据点生成状态约束。
                  </p>
                </div>
              </aside>

              <div className="candidate-panel">
                <div className="candidate-head">
                  <div>
                    <span>CANDIDATE SET</span>
                    <h3>
                      {ROMAN[unknownFloor]} 层 ·
                      {unknownType === "mystery" ? " 未知的诡秘" : " 未知的凶戾"}
                    </h3>
                  </div>
                  <strong>{unknownCandidates.length}</strong>
                </div>

                {unknownCandidates.length ? (
                  <div className="candidate-grid">
                    {unknownCandidates.map((name, index) => {
                      const node = NODE_BY_NAME.get(name);
                      return (
                        <button
                          type="button"
                          key={name}
                          className="candidate-card"
                          onClick={() => node && setSelectedNode(node)}
                        >
                          <span className="candidate-index">
                            {String(index + 1).padStart(2, "0")}
                          </span>
                          <NodeGlyph node={name} />
                          <div>
                            <h4>{name}</h4>
                            <p>{node?.summary ?? "受特殊标记影响的节点"}</p>
                          </div>
                          <span className="candidate-score">
                            EV {(node?.baseScore ?? 0).toFixed(1)}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                ) : (
                  <div className="empty-state">
                    <BadgeHelp />
                    <strong>本层没有可归入该观测类型的常规候选</strong>
                    <p>检查是否为剧情节点、固定出口或特殊层标记。</p>
                  </div>
                )}

                <div className="deduction-rules">
                  <div>
                    <strong>01 · 层数排除</strong>
                    <p>狭路相逢与误入奇境仅III—V层；先行一步仅II—IV层。</p>
                  </div>
                  <div>
                    <strong>02 · 据点条件</strong>
                    <p>“居民”据点仅保密等级4+；流窜居民必须由据点生成。</p>
                  </div>
                  <div>
                    <strong>03 · 结局条件</strong>
                    <p>命运所指只在V层二结局链出现；险路恶敌集中于III / V / VI层。</p>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="stages" className="tab-panel">
            <SectionHeader
              kicker="COMBAT DATABASE"
              title="输入作战名，查看编成与打法入口"
              copy="普通、紧急、追猎、居民、商店战、狭路和全部Boss关统一检索；也可以反向搜索敌人名。"
            />

            <div className="stage-search-hero">
              <div className="search-box large">
                <Search />
                <Input
                  value={stageQuery}
                  onChange={(event) => setStageQuery(event.target.value)}
                  placeholder="例如：猎犬病原 / 纵怒 / 卡德霍 / 源阶方"
                />
                {stageQuery && (
                  <button type="button" onClick={() => setStageQuery("")}>
                    清空
                  </button>
                )}
              </div>
              <div className="stage-kind-tabs">
                {["全部", "普通作战", "险路恶敌", "追猎", "特殊作战"].map(
                  (kind) => (
                    <button
                      type="button"
                      key={kind}
                      className={stageKind === kind ? "active" : ""}
                      onClick={() => setStageKind(kind)}
                    >
                      {kind}
                    </button>
                  ),
                )}
              </div>
            </div>

            <div className="stage-results-head">
              <span>SEARCH RESULTS</span>
              <strong>{filteredStages.length} / {STAGE_DATA.length}</strong>
            </div>

            <div className="stage-grid">
              {filteredStages.map((stage) => {
                const boss = stage.enemies.find((enemy) =>
                  BOSS_NAMES.has(enemy.name),
                );
                return (
                  <button
                    type="button"
                    className={"stage-card " + (stage.kind === "险路恶敌" ? "boss-stage" : "")}
                    key={stage.name}
                    onClick={() => setSelectedStage(stage)}
                  >
                    <div className="stage-card-index">
                      <span>{stage.floor ? ROMAN[stage.floor] : "SP"}</span>
                      <small>{stage.map || "SPECIAL"}</small>
                    </div>
                    <div className="stage-card-body">
                      <div className="stage-card-title">
                        <span>{stage.kind}</span>
                        <h3>{stage.name}</h3>
                      </div>
                      <p>{stage.intro || "特殊作战资料已收录，打开查看敌人编成与路线入口。"}</p>
                      <div className="stage-facts">
                        <span>
                          <Target /> {stage.total || "动态"} 敌人
                        </span>
                        <span>
                          <Footprints /> {stage.enemies.length} 种编成
                        </span>
                        <span>
                          <MapIcon /> {stage.map ? "路线图" : "特殊规则"}
                        </span>
                      </div>
                      <div className="enemy-preview-row">
                        {stage.enemies.slice(0, 5).map((enemy) => (
                          <span key={enemy.name + enemy.code}>
                            {enemy.name}
                            <small>×{enemy.count}</small>
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="stage-card-action">
                      {boss ? <Skull /> : <BookOpen />}
                      <span>{boss ? boss.name : "展开档案"}</span>
                      <ChevronRight />
                    </div>
                  </button>
                );
              })}
            </div>
          </TabsContent>

          <TabsContent value="archive" className="tab-panel">
            <SectionHeader
              kicker="ENEMY ARCHIVE"
              title="黑流树海敌人档案馆"
              copy="从全部作战编成中汇总敌方单位；关卡内点击敌人图像会直接定位到这里。Boss档案包含机制拆解与打法摘要。"
            />

            <div className="archive-layout">
              <aside className="enemy-detail">
                <div className="enemy-detail-art">
                  <EnemyAvatar
                    name={selectedEnemy.name}
                    type={selectedEnemy.type}
                    size="lg"
                    icon={selectedEnemy.icon}
                  />
                  <div className="scan-grid" />
                  <span className="scan-label">IDENT // {selectedEnemy.codes[0] || "DEVICE"}</span>
                </div>
                <div className="enemy-detail-title">
                  <Badge
                    className={"type-badge type-" + selectedEnemy.type}
                    variant="outline"
                  >
                    {selectedEnemy.type}
                  </Badge>
                  <h3>{selectedEnemy.name}</h3>
                  <p>
                    {selectedEnemy.codes.length
                      ? selectedEnemy.codes.join(" · ")
                      : "BLACKFLOW DEVICE"}
                  </p>
                </div>

                <div className="detail-section">
                  <span>机制摘要</span>
                  <p>{selectedEnemy.mechanic}</p>
                </div>
                <div className="detail-section accent">
                  <span>{selectedEnemy.type === "领袖" || selectedEnemy.type === "精英" ? "实战处理" : "必要提醒"}</span>
                  {BOSS_GUIDES[selectedEnemy.name] || ELITE_GUIDES[selectedEnemy.name] ? (
                    (BOSS_GUIDES[selectedEnemy.name] || ELITE_GUIDES[selectedEnemy.name]).plan.map((plan) => (
                      <p key={plan}>• {plan}</p>
                    ))
                  ) : (
                    <p>{selectedEnemy.tip}</p>
                  )}
                </div>

                {(BOSS_GUIDES[selectedEnemy.name] || ELITE_GUIDES[selectedEnemy.name]) && (
                  <div className="detail-section mechanics">
                    <span>完整机制（通俗整理）</span>
                    {(BOSS_GUIDES[selectedEnemy.name] || ELITE_GUIDES[selectedEnemy.name]).mechanic.map((item) => (
                      <p key={item}>— {item}</p>
                    ))}
                  </div>
                )}

                <div className="detail-section appearances">
                  <span>出现关卡 · {selectedEnemy.stages.length}</span>
                  <div>
                    {selectedEnemy.stages.slice(0, 12).map((stageName) => (
                      <button
                        type="button"
                        key={stageName}
                        onClick={() => {
                          const stage = STAGE_DATA.find(
                            (item) => item.name === stageName,
                          );
                          if (stage) setSelectedStage(stage);
                        }}
                      >
                        {stageName}
                      </button>
                    ))}
                  </div>
                </div>

                <Button asChild className="prts-button">
                  <a
                    href={prtsUrl(selectedEnemy.name)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    在 PRTS 查看完整数值
                    <ExternalLink />
                  </a>
                </Button>
              </aside>

              <div className="enemy-browser">
                <div className="filter-bar archive-filters">
                  <div className="search-box">
                    <Search />
                    <Input
                      value={enemyQuery}
                      onChange={(event) => {
                        setEnemyQuery(event.target.value);
                        setEnemyLimit(48);
                      }}
                      placeholder="搜敌人、编号或出现关卡"
                    />
                  </div>
                  <NativeSelect
                    value={enemyType}
                    onChange={(event) => {
                      setEnemyType(event.target.value);
                      setEnemyLimit(48);
                    }}
                    className="filter-select"
                  >
                    {["全部", "普通", "精英", "领袖", "装置"].map((item) => (
                      <NativeSelectOption value={item} key={item}>
                        {item}
                      </NativeSelectOption>
                    ))}
                  </NativeSelect>
                  <span className="result-count">
                    {filteredEnemies.length} 项
                  </span>
                </div>

                <div className="archive-summary">
                  {(["领袖", "精英", "普通", "装置"] as EnemyType[]).map(
                    (type) => (
                      <button
                        type="button"
                        key={type}
                        onClick={() => setEnemyType(type)}
                      >
                        <strong>
                          {ENEMY_INDEX.filter((item) => item.type === type).length}
                        </strong>
                        <span>{type}</span>
                      </button>
                    ),
                  )}
                </div>

                <div className="enemy-grid">
                  {filteredEnemies.slice(0, enemyLimit).map((enemy) => (
                    <button
                      type="button"
                      className={
                        selectedEnemy.name === enemy.name
                          ? "enemy-card active"
                          : "enemy-card"
                      }
                      key={enemy.name}
                      onClick={() => setSelectedEnemy(enemy)}
                    >
                      <EnemyAvatar
                        name={enemy.name}
                        type={enemy.type}
                        size="sm"
                        icon={enemy.icon}
                      />
                      <div>
                        <span>{enemy.type}</span>
                        <h4>{enemy.name}</h4>
                        <p>
                          {enemy.stages.length
                            ? "出现于 " + enemy.stages.slice(0, 2).join(" / ")
                            : enemy.mechanic}
                        </p>
                      </div>
                      <ChevronRight />
                    </button>
                  ))}
                </div>

                {enemyLimit < filteredEnemies.length && (
                  <Button
                    variant="outline"
                    className="load-more"
                    onClick={() => setEnemyLimit((value) => value + 48)}
                  >
                    再显示 48 项
                    <ChevronRight />
                  </Button>
                )}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </section>

      <footer className="site-footer">
        <div>
          <span className="brand-mark">
            <Trees />
          </span>
          <div>
            <strong>黑流树海路线参谋</strong>
            <p>玩家向决策工具 · 非官方资料整理</p>
          </div>
        </div>
        <div className="source-links">
          {SOURCES.map((source) => (
            <a href={source.href} target="_blank" rel="noreferrer" key={source.href}>
              {source.label}
              <ExternalLink />
            </a>
          ))}
        </div>
        <p>
          《明日方舟》相关名称与素材版权归鹰角网络所有。本站只整理公开玩法数据，
          关卡与敌人精确数值以游戏当前版本及 PRTS 页面为准。
        </p>
      </footer>

      <Dialog
        open={Boolean(selectedNode)}
        onOpenChange={(open) => !open && setSelectedNode(null)}
      >
        <DialogContent className="data-dialog node-dialog">
          {selectedNode && (
            <>
              <DialogHeader>
                <div className="dialog-kicker">
                  <NodeGlyph node={selectedNode.name} />
                  <span>
                    {selectedNode.kind} / {selectedNode.group}
                  </span>
                </div>
                <DialogTitle>{selectedNode.name}</DialogTitle>
                <DialogDescription>{selectedNode.summary}</DialogDescription>
              </DialogHeader>
              <div className="dialog-layer-row">
                {selectedNode.layers.map((layer) => (
                  <span key={layer}>
                    {ROMAN[layer]} 层
                    {selectedNode.count[layer]
                      ? " · " + selectedNode.count[layer]
                      : ""}
                  </span>
                ))}
              </div>
              <div className="possibility-list">
                <span>所有已收录可能</span>
                {selectedNode.possible.map((item, index) => (
                  <div key={item}>
                    <strong>{String(index + 1).padStart(2, "0")}</strong>
                    <p>{item}</p>
                  </div>
                ))}
              </div>
              {selectedNode.caution && (
                <div className="dialog-caution">
                  <AlertTriangle />
                  <p>{selectedNode.caution}</p>
                </div>
              )}
            </>
          )}
        </DialogContent>
      </Dialog>

      <Dialog
        open={Boolean(selectedStage)}
        onOpenChange={(open) => !open && setSelectedStage(null)}
      >
        <DialogContent className="data-dialog stage-dialog">
          {selectedStage && (
            <>
              <DialogHeader>
                <div className="dialog-kicker">
                  <Swords />
                  <span>
                    {selectedStage.kind} /{" "}
                    {selectedStage.floor
                      ? ROMAN[selectedStage.floor] + " 层"
                      : "特殊"}
                  </span>
                </div>
                <DialogTitle>{selectedStage.name}</DialogTitle>
                <DialogDescription>
                  {selectedStage.intro || "特殊作战档案"}
                </DialogDescription>
              </DialogHeader>

              <div className="stage-dialog-metrics">
                <div>
                  <span>敌人数量</span>
                  <strong>{selectedStage.total || "动态"}</strong>
                </div>
                <div>
                  <span>敌人种类</span>
                  <strong>{selectedStage.enemies.length}</strong>
                </div>
                <div>
                  <span>部署上限</span>
                  <strong>{selectedStage.deploy ?? "—"}</strong>
                </div>
                <div>
                  <span>初始费用</span>
                  <strong>{selectedStage.cost ?? "—"}</strong>
                </div>
              </div>

              <div className="stage-dialog-grid">
                <section className="stage-map-panel">
                  <div className="subhead">
                    <MapIcon />
                    <div>
                      <span>波次 / 路线</span>
                      <strong>{selectedStage.map || "特殊战规则"}</strong>
                    </div>
                  </div>
                  <div className="route-schematic">
                    {selectedStage.map ? (
                      <img
                        className="stage-map-image"
                        src={
                          "https://tomimi.dev/images/stages/level_" +
                          selectedStage.map +
                          ".webp"
                        }
                        alt={selectedStage.name + " 关卡地图"}
                        loading="lazy"
                      />
                    ) : (
                      <>
                        <span className="spawn-point">IN</span>
                        <div className="route-line route-line-a" />
                        <div className="route-line route-line-b" />
                        <span className="checkpoint one" />
                        <span className="checkpoint two" />
                        <span className="goal-point">OUT</span>
                      </>
                    )}
                    <small>关卡地图 · 精确出场波次与路线请打开动态地图</small>
                  </div>
                  <div className="stage-link-row">
                    {selectedStage.map && (
                      <Button asChild>
                        <a
                          href={"https://map.ark-nights.com/map/" + selectedStage.map}
                          target="_blank"
                          rel="noreferrer"
                        >
                          <MapIcon /> PRTS.Map 动态路线
                          <ExternalLink />
                        </a>
                      </Button>
                    )}
                    <Button variant="outline" asChild>
                      <a
                        href={prtsUrl(selectedStage.name)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        PRTS 关卡页
                        <ExternalLink />
                      </a>
                    </Button>
                  </div>

                  {selectedStage.terrain.length > 0 && (
                    <div className="terrain-list">
                      <span>地图装置 / 特殊地形</span>
                      {selectedStage.terrain.map((terrain) => (
                        <div key={terrain.name + terrain.detail}>
                          <strong>{terrain.name}</strong>
                          <p>{terrain.detail}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </section>

                <section className="stage-enemy-panel">
                  <div className="subhead">
                    <Biohazard />
                    <div>
                      <span>敌方编成</span>
                      <strong>点击图像进入敌人档案</strong>
                    </div>
                  </div>
                  <div className="stage-enemy-list">
                    {selectedStage.enemies.map((enemy, index) => {
                      const type = classifyEnemy(enemy.name);
                      return (
                        <div key={enemy.name + enemy.code + index}>
                          <EnemyAvatar
                            name={enemy.name}
                            type={type}
                            size="sm"
                            icon={enemy.icon}
                            onClick={() => openEnemyFromStage(enemy.name)}
                          />
                          <button
                            type="button"
                            onClick={() => openEnemyFromStage(enemy.name)}
                          >
                            <strong>{enemy.name}</strong>
                            <span>{mechanismFor(enemy.name, type)}</span>
                          </button>
                          <b>× {enemy.count}</b>
                        </div>
                      );
                    })}
                  </div>
                </section>
              </div>

              <div className="combat-brief">
                <section>
                  <span>
                    <AlertTriangle /> 特别注意
                  </span>
                  {stageAttention(selectedStage).map((item) => (
                    <p key={item}>• {item}</p>
                  ))}
                </section>
                <section>
                  <span>
                    <Compass /> 基本打法
                  </span>
                  {basicPlan(selectedStage).map((item) => (
                    <p key={item}>• {item}</p>
                  ))}
                </section>
              </div>

              <div className="video-row">
                <div>
                  <Play />
                  <span>
                    <strong>相关攻略视频</strong>
                    <small>搜索链接保留不同难度、分队和干员配置</small>
                  </span>
                </div>
                <Button asChild className="video-button">
                  <a
                    href={bilibiliUrl(selectedStage.name)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    搜索 {selectedStage.name} 攻略
                    <ExternalLink />
                  </a>
                </Button>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </main>
  );
}
