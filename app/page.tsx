"use client";
/* eslint-disable @next/next/no-img-element */

import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Archive,
  BadgeHelp,
  Biohazard,
  BookOpen,
  Box,
  ChevronRight,
  CircleDot,
  ExternalLink,
  Layers3,
  Map as MapIcon,
  PackageOpen,
  Play,
  Route as RouteIcon,
  Search,
  Skull,
  Sparkles,
  Swords,
  Target,
  Trees,
  WandSparkles,
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
import {
  Tabs,
  TabsContent,
} from "@/components/ui/tabs";
import {
  BOSS_GUIDES,
  DEVICE_GUIDES,
  ELITE_GUIDES,
  EVENT_NOTES,
  EVENT_POOLS,
  NODE_DATA,
  SOURCES,
  UNKNOWN_POOLS,
  type NodeRecord,
} from "./game-data";
import { STAGE_DATA, type StageRecord } from "./stage-data";
import LiveRouteControl from "./live-route-control";
import {
  BLACKFLOW_POOLS,
  BLACKFLOW_POOL_ITEMS,
  BLACKFLOW_POOL_SOURCE,
  BLACKFLOW_POOL_SYNC_DATE,
  type BlackflowPool,
} from "./item-pool-data";

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
const BOSS_NAMES = new Set(Object.keys(BOSS_GUIDES));
const ELITE_NAMES = new Set(Object.keys(ELITE_GUIDES));
const EXCLUDED_ENEMY_NAMES = new Set(["便符", "受符"]);
const POOL_ITEM_BY_ID = new Map(
  BLACKFLOW_POOL_ITEMS.map((item) => [item.id, item]),
);
const SITE_TABS = [
  "planner",
  "nodes",
  "items",
  "stages",
  "archive",
] as const;
type SiteTab = (typeof SITE_TABS)[number];

function isSiteTab(value: string): value is SiteTab {
  return SITE_TABS.includes(value as SiteTab);
}

function prtsUrl(name: string) {
  return "https://prts.wiki/w/" + encodeURIComponent(name);
}

function bilibiliUrl(name: string) {
  return (
    "https://search.bilibili.com/all?keyword=" +
    encodeURIComponent("明日方舟 黑流树海 " + name + " 攻略")
  );
}

function formatPoolProbability(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function normalizedPoolType(pool: { name: string; poolType: string }) {
  if (
    pool.poolType === "其他" &&
    /(零件|自然物|加工品|概念体)/.test(pool.name)
  ) {
    return "零件池";
  }
  return pool.poolType;
}

function poolItemSummary(pool: BlackflowPool) {
  let collectibles = 0;
  let parts = 0;
  for (const occurrence of pool.items) {
    const item = POOL_ITEM_BY_ID.get(occurrence.id);
    if (item?.type === "藏品") collectibles += 1;
    if (item?.type === "零件") parts += 1;
  }
  const labels = [];
  if (collectibles || normalizedPoolType(pool) === "藏品池") {
    labels.push(`${collectibles} 个已记录藏品`);
  }
  if (parts || normalizedPoolType(pool) === "零件池") {
    labels.push(`${parts} 个已记录零件`);
  }
  return labels.join(" · ");
}

function poolSourcePaths(pool: BlackflowPool) {
  return pool.sources.map((source) => source.path.join(" → "));
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
      if (EXCLUDED_ENEMY_NAMES.has(enemy.name)) continue;
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
const EVENT_CATALOG = Array.from(new Set(Object.values(EVENT_POOLS).flat())).map((name) => ({
  name,
  layers: Object.entries(EVENT_POOLS)
    .filter(([, events]) => events.includes(name))
    .map(([layer]) => Number(layer)),
}));

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

export default function Home() {
  const [activeTab, setActiveTab] = useState<SiteTab>("planner");
  const [nodeQuery, setNodeQuery] = useState("");
  const [nodeKind, setNodeKind] = useState("全部");
  const [nodeFloor, setNodeFloor] = useState(0);
  const [selectedNode, setSelectedNode] = useState<NodeRecord | null>(null);
  const [stageQuery, setStageQuery] = useState("");
  const [stageKind, setStageKind] = useState("全部");
  const [selectedStage, setSelectedStage] = useState<StageRecord | null>(null);
  const [enemyQuery, setEnemyQuery] = useState("");
  const [enemyType, setEnemyType] = useState("全部");
  const [enemyLimit, setEnemyLimit] = useState(48);
  const [itemPoolQuery, setItemPoolQuery] = useState("");
  const [poolTypeFilter, setPoolTypeFilter] = useState("全部");
  const [selectedPoolId, setSelectedPoolId] = useState<string | null>(null);
  const [selectedEnemy, setSelectedEnemy] = useState<EnemyEntry>(
    ENEMY_INDEX.find((enemy) => enemy.name === "卡德霍，黑流之源") ??
      ENEMY_INDEX[0],
  );

  useEffect(() => {
    const tabFromLocation = () => {
      const historyTab = window.history.state?.blackflowTab;
      const hashTab = window.location.hash.replace(/^#/, "");
      if (typeof historyTab === "string" && isSiteTab(historyTab)) {
        return historyTab;
      }
      return isSiteTab(hashTab) ? hashTab : "planner";
    };
    const initialTab = tabFromLocation();
    setActiveTab(initialTab);
    window.history.replaceState(
      { ...window.history.state, blackflowTab: initialTab },
      "",
      window.location.href,
    );
    const handlePopState = () => setActiveTab(tabFromLocation());
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  function navigateTab(nextTab: SiteTab) {
    if (nextTab === activeTab) return;
    window.history.pushState(
      { ...window.history.state, blackflowTab: nextTab },
      "",
      `#${nextTab}`,
    );
    setActiveTab(nextTab);
  }

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

  const itemPoolResults = useMemo(() => {
    const query = itemPoolQuery.trim().toLocaleLowerCase();
    if (!query) return [];
    return BLACKFLOW_POOL_ITEMS.map((item) => {
        const nameMatched = item.name.toLocaleLowerCase().includes(query);
        const pools = BLACKFLOW_POOLS.flatMap((pool) => {
          const occurrence = pool.items.find((entry) => entry.id === item.id);
          return occurrence ? [{ pool, occurrence }] : [];
        }).sort((a, b) => {
          if (
            a.occurrence.probability === null &&
            b.occurrence.probability === null
          ) {
            return a.pool.name.localeCompare(b.pool.name, "zh-CN");
          }
          if (a.occurrence.probability === null) return 1;
          if (b.occurrence.probability === null) return -1;
          return b.occurrence.probability - a.occurrence.probability;
        });
        const matchedSourcePools = pools.filter(({ pool }) =>
          pool.sources.some((source) =>
            source.path.some((part) => part.toLocaleLowerCase().includes(query)),
          ),
        );
        const generatedPools = BLACKFLOW_POOLS.filter((pool) =>
          pool.sources.some((source) => source.sourceItemId === item.id),
        );
        return {
          item,
          pools,
          generatedPools,
          nameMatched,
          matchedBySource: !nameMatched && matchedSourcePools.length > 0,
        };
      })
      .filter(({ nameMatched, matchedBySource }) => nameMatched || matchedBySource)
      .sort((a, b) =>
        Number(b.nameMatched) - Number(a.nameMatched) ||
        a.item.name.localeCompare(b.item.name, "zh-CN"),
      )
      .map((result) => ({
        ...result,
        generatedPools: result.generatedPools.map((pool) => ({
          pool,
          items: pool.items
            .map((occurrence) => POOL_ITEM_BY_ID.get(occurrence.id))
            .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry)),
        })),
      }))
      .slice(0, 24);
  }, [itemPoolQuery]);

  const visiblePools = useMemo(
    () =>
      BLACKFLOW_POOLS.filter(
        (pool) =>
          poolTypeFilter === "全部" ||
          normalizedPoolType(pool) === poolTypeFilter,
      ),
    [poolTypeFilter],
  );

  const selectedPool = selectedPoolId
    ? BLACKFLOW_POOLS.find((pool) => pool.id === selectedPoolId) ?? null
    : null;

  const selectedPoolItems = useMemo(() => {
    if (!selectedPool) return [];
    return selectedPool.items
      .flatMap((occurrence) => {
        const item = POOL_ITEM_BY_ID.get(occurrence.id);
        return item ? [{ item, occurrence }] : [];
      })
      .sort((a, b) => {
        if (
          a.occurrence.probability !== null &&
          b.occurrence.probability !== null
        ) {
          return b.occurrence.probability - a.occurrence.probability;
        }
        if (a.occurrence.probability !== null) return -1;
        if (b.occurrence.probability !== null) return 1;
        return a.item.type.localeCompare(b.item.type, "zh-CN") ||
          a.item.name.localeCompare(b.item.name, "zh-CN");
      });
  }, [selectedPool]);

  function openEnemyFromStage(name: string) {
    const enemy = ENEMY_INDEX.find((item) => item.name === name);
    if (enemy) setSelectedEnemy(enemy);
    setEnemyQuery(name);
    navigateTab("archive");
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
        <a
          className="brand"
          href="#planner"
          aria-label="黑流树海路线参谋首页"
          onClick={(event) => {
            event.preventDefault();
            navigateTab("planner");
          }}
        >
          <span className="brand-mark">
            <Trees />
          </span>
          <span>
            <strong>BLACKFLOW // ROUTE LAB</strong>
            <small>黑流树海路线参谋</small>
          </span>
        </a>
        <nav className="top-links" aria-label="功能索引">
          <button className={activeTab === "planner" ? "active" : ""} aria-current={activeTab === "planner" ? "page" : undefined} onClick={() => navigateTab("planner")}><RouteIcon />路线决策</button>
          <button className={activeTab === "nodes" ? "active" : ""} aria-current={activeTab === "nodes" ? "page" : undefined} onClick={() => navigateTab("nodes")}><Layers3 />节点图鉴</button>
          <button className={activeTab === "items" ? "active" : ""} aria-current={activeTab === "items" ? "page" : undefined} onClick={() => navigateTab("items")}><PackageOpen />藏品检索</button>
          <button className={activeTab === "stages" ? "active" : ""} aria-current={activeTab === "stages" ? "page" : undefined} onClick={() => navigateTab("stages")}><Swords />作战检索</button>
          <button className={activeTab === "archive" ? "active" : ""} aria-current={activeTab === "archive" ? "page" : undefined} onClick={() => navigateTab("archive")}><Archive />敌人档案馆</button>
        </nav>
        <div className="live-pill">
          <span />
          数据快照 2026.08
        </div>
      </header>

      <section id="workspace-tabs" className="workspace">
        <Tabs
          value={activeTab}
          onValueChange={(value) => isSiteTab(value) && navigateTab(value)}
        >
          <TabsContent value="planner" className="tab-panel" forceMount hidden={activeTab !== "planner"}>
            <LiveRouteControl />
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
              <div className="event-catalog">
                {EVENT_CATALOG.map((event) => (
                  <div className="event-catalog-row" key={event.name}>
                    <div className="event-catalog-title">
                      <strong>{event.name}</strong>
                      <span>
                        {event.layers.map((layer) => (
                          <b key={layer}>{ROMAN[layer]}</b>
                        ))}
                      </span>
                    </div>
                    <p>{EVENT_NOTES[event.name]}</p>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="items" className="tab-panel">
            <SectionHeader
              kicker="ITEM & POOL INDEX"
              title="藏品与零件来源检索"
              copy="输入藏品、零件或上游来源名称，检索直接池与间接产出链。已公开的经验概率按高到低排列，未公开概率统一放在最后。"
            />

            <div className="pool-source-note">
              <div>
                <PackageOpen />
                <p>
                  当前收录 <strong>{BLACKFLOW_POOL_ITEMS.length}</strong> 个可检索条目、
                  <strong>{BLACKFLOW_POOLS.length}</strong> 个池子。概率来自路标档案馆公开样本，
                  是经验频率，不等同于游戏源码权重。池表同步于 {BLACKFLOW_POOL_SYNC_DATE}。
                </p>
              </div>
              <a href={BLACKFLOW_POOL_SOURCE} target="_blank" rel="noreferrer">
                查看路标原始池表 <ExternalLink />
              </a>
            </div>

            <section className="item-pool-search" aria-labelledby="item-pool-search-title">
              <div className="pool-block-heading">
                <div>
                  <span>01 / REVERSE LOOKUP</span>
                  <h3 id="item-pool-search-title">按名称或上游来源检索</h3>
                </div>
                <span>{itemPoolResults.length} 个匹配条目</span>
              </div>
              <div className="pool-search-box search-box">
                <Search />
                <Input
                  value={itemPoolQuery}
                  onChange={(event) => setItemPoolQuery(event.target.value)}
                  placeholder="输入名称，例如：板藤、迷藏、复得之轮"
                />
              </div>

              {!itemPoolQuery.trim() ? (
                <div className="pool-search-placeholder">
                  <Search />
                  <strong>输入名称后开始检索</strong>
                  <p>支持部分名称；结果会列出全部池子、上游藏品/零件、完整产出链及公开概率。</p>
                </div>
              ) : itemPoolResults.length ? (
                <div className="item-pool-results">
                  {itemPoolResults.map(({ item, pools, generatedPools, matchedBySource }) => (
                    <article className="item-pool-result" key={item.id}>
                      <header>
                        <Badge variant="outline">{item.type}</Badge>
                        <div>
                          <h4>{item.name}</h4>
                          <p>
                            {pools.length} 个可能来源池
                            {generatedPools.length > 0 && ` · 可产出 ${generatedPools.length} 个池`}
                            {matchedBySource && " · 由上游来源命中"}
                          </p>
                        </div>
                      </header>
                      {pools.length ? (
                        <div className="pool-occurrence-list">
                          {pools.map(({ pool, occurrence }) => (
                            <button
                              type="button"
                              className="pool-occurrence-row"
                              key={pool.id}
                              onClick={() => setSelectedPoolId(pool.id)}
                              aria-label={`查看${pool.name}的具体内容`}
                            >
                              <div>
                                <strong>{pool.name}</strong>
                                <p>
                                  {occurrence.subpool && <span>{occurrence.subpool}</span>}
                                  {pool.officialCode || pool.poolType}
                                </p>
                                <p className="pool-source-path">
                                  {pool.sources.length > 0
                                    ? `完整来源：${poolSourcePaths(pool).join(" / ")}`
                                    : `直接来源：${pool.name}`}
                                </p>
                              </div>
                              {occurrence.probability !== null ? (
                                <div className="known-probability">
                                  <strong>{formatPoolProbability(occurrence.probability)}</strong>
                                  <span>
                                    经验概率
                                    {occurrence.sampleCount
                                      ? ` · 样本 ${occurrence.sampleCount}`
                                      : ""}
                                  </span>
                                </div>
                              ) : (
                                <div className="unknown-probability">
                                  <strong>概率不详</strong>
                                  <span>该池仅确认成员范围</span>
                                </div>
                              )}
                            </button>
                          ))}
                        </div>
                      ) : (
                        <p className="pool-no-membership">当前池表未记录可用来源。</p>
                      )}
                      {generatedPools.length > 0 && (
                        <div className="generated-pool-list">
                          <div className="generated-pool-heading">
                            <Sparkles />
                            <strong>{item.name} 能带来的藏品/零件收益</strong>
                          </div>
                          {generatedPools.map(({ pool, items }) => (
                            <button
                              type="button"
                              key={pool.id}
                              onClick={() => setSelectedPoolId(pool.id)}
                              aria-label={`查看${item.name}产出的${pool.name}`}
                            >
                              <span>{poolSourcePaths(pool).join(" / ")}</span>
                              <p>{items.map((entry) => entry.name).join("、")}</p>
                            </button>
                          ))}
                        </div>
                      )}
                    </article>
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  <BadgeHelp />
                  <strong>没有找到对应的藏品或零件</strong>
                  <p>请尝试缩短关键词，或在下方池子目录中浏览全部来源。</p>
                </div>
              )}
            </section>

            <section className="pool-directory" aria-labelledby="pool-directory-title">
              <div className="pool-block-heading">
                <div>
                  <span>02 / POOL DIRECTORY</span>
                  <h3 id="pool-directory-title">黑流树海池子总目录</h3>
                </div>
                <span>{visiblePools.length} / {BLACKFLOW_POOLS.length}</span>
              </div>
              <div className="pool-filter-tabs" aria-label="池子类型筛选">
                {["全部", "藏品池", "零件池", "混合池"].map((type) => (
                  <button
                    type="button"
                    key={type}
                    className={poolTypeFilter === type ? "active" : ""}
                    onClick={() => setPoolTypeFilter(type)}
                  >
                    {type}
                  </button>
                ))}
              </div>
              <div className="pool-directory-grid">
                {visiblePools.map((pool, index) => (
                  <button
                    type="button"
                    className="pool-directory-card"
                    key={pool.id}
                    onClick={() => setSelectedPoolId(pool.id)}
                    aria-label={`查看${pool.name}的具体内容`}
                  >
                    <span className="pool-directory-index">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <div>
                      <div className="pool-directory-title">
                        <h4>{pool.name}</h4>
                        <Badge variant="outline">{normalizedPoolType(pool)}</Badge>
                      </div>
                      <p>{pool.officialCode || "暂无公开代码"}</p>
                      <div className="pool-directory-meta">
                        <span>{poolItemSummary(pool)}</span>
                        <span>{pool.displayCategory === "node" ? "节点来源" : "独立池"}</span>
                      </div>
                      {pool.usage.length > 0 && (
                        <p className="pool-usage">触发：{pool.usage.join(" / ")}</p>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </section>
          </TabsContent>

          <TabsContent value="stages" className="tab-panel">
            <SectionHeader
              kicker="COMBAT DATABASE"
              title="输入作战名，查看敌人信息、动态路线和攻略"
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
                      <p>{stage.intro || "特殊作战资料已收录，打开查看敌人信息与动态路线。"}</p>
                      <div className="stage-facts">
                        <span>
                          <Target /> {stage.total || "动态"} 敌人
                        </span>
                        <span>
                          <Biohazard /> {stage.enemies.length} 种敌人
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
              copy="从全部作战数据中汇总敌方单位；关卡内点击敌人图像会直接定位到这里。Boss档案包含机制拆解与打法摘要。"
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
                {(BOSS_GUIDES[selectedEnemy.name] || ELITE_GUIDES[selectedEnemy.name]) && (
                  <div className="detail-section mechanics">
                    <span>完整机制</span>
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
        open={Boolean(selectedPool)}
        onOpenChange={(open) => !open && setSelectedPoolId(null)}
      >
        <DialogContent className="data-dialog pool-detail-dialog">
          {selectedPool && (
            <>
              <DialogHeader>
                <div className="dialog-kicker">
                  <PackageOpen />
                  <span>
                    {normalizedPoolType(selectedPool)} / {selectedPool.displayCategory === "node" ? "节点来源" : "独立池"}
                  </span>
                </div>
                <DialogTitle>{selectedPool.name}</DialogTitle>
                <DialogDescription>
                  {poolItemSummary(selectedPool)}。有公开经验概率的条目按概率降序排列，
                  其余条目仅确认属于该池。
                </DialogDescription>
              </DialogHeader>

              <div className="pool-detail-meta">
                <span>{selectedPool.officialCode || "暂无公开代码"}</span>
                {selectedPool.usage.length > 0 && (
                  <span>上游：{selectedPool.usage.join(" / ")}</span>
                )}
                <Button variant="outline" asChild>
                  <a
                    href={`${BLACKFLOW_POOL_SOURCE}?pool=${encodeURIComponent(selectedPool.id)}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    在路标查看原始池表 <ExternalLink />
                  </a>
                </Button>
              </div>

              {selectedPool.sources.length > 0 && (
                <div className="pool-source-chain-list">
                  {selectedPool.sources.map((source) => (
                    <div key={`${source.id}-${source.path.join("-")}`}>
                      <strong>{source.path.join(" → ")}</strong>
                      <span>{source.sourceType}</span>
                      {source.effect && <p>{source.effect}</p>}
                    </div>
                  ))}
                </div>
              )}

              {selectedPoolItems.length ? (
                <div className="pool-detail-list">
                  {selectedPoolItems.map(({ item, occurrence }, index) => (
                    <div className="pool-detail-item" key={occurrence.id}>
                      <span className="pool-detail-index">
                        {String(index + 1).padStart(3, "0")}
                      </span>
                      <Badge variant="outline">{item.type}</Badge>
                      <div>
                        <strong>{item.name}</strong>
                        <p>
                          {occurrence.subpool ||
                            (item.type === "藏品" ? "收藏品池" : "零件池")}
                        </p>
                      </div>
                      {occurrence.probability !== null ? (
                        <div className="pool-detail-probability known">
                          <strong>{formatPoolProbability(occurrence.probability)}</strong>
                          <span>
                            经验概率
                            {occurrence.sampleCount
                              ? ` · 样本 ${occurrence.sampleCount}`
                              : ""}
                          </span>
                        </div>
                      ) : (
                        <div className="pool-detail-probability">
                          <strong>概率不详</strong>
                          <span>已确认属于该池</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-state pool-detail-empty">
                  <BadgeHelp />
                  <strong>当前没有可列出的藏品或零件</strong>
                  <p>路标池表仅保留了池名称，尚未公开具体条目。</p>
                </div>
              )}
            </>
          )}
        </DialogContent>
      </Dialog>

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
                  <strong>{selectedStage.enemies.filter((enemy) => !EXCLUDED_ENEMY_NAMES.has(enemy.name)).length}</strong>
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
                      <span>敌人信息</span>
                      <strong>点击图像进入敌人档案</strong>
                    </div>
                  </div>
                  <div className="stage-enemy-list">
                    {selectedStage.enemies.filter((enemy) => !EXCLUDED_ENEMY_NAMES.has(enemy.name)).map((enemy, index) => {
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

              <div className="combat-brief single">
                <section>
                  <span>
                    <AlertTriangle /> 特别注意
                  </span>
                  {stageAttention(selectedStage).map((item) => (
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
