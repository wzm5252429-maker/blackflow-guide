"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, BrainCircuit, Download, Link2, Monitor, Pause, Play, RefreshCw, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const BRIDGE = "http://127.0.0.1:19761";
const TOKEN_KEY = "blackflow-live-token";
type WindowGeometry = { hwnd: number; title: string; dpi: number; client_rect: { width: number; height: number } };
type GameWindow = Omit<WindowGeometry, "dpi" | "client_rect"> & Partial<Pick<WindowGeometry, "dpi" | "client_rect">> & { state?: "visible" | "minimized" | "hidden" };
type Action = { action_id: string; label: string; bbox: number[] };
type Status = {
  state: string; message: string; clicks: number; has_preview: boolean; observe_only?: boolean;
  window?: WindowGeometry;
  observation?: { scene: string; floor: number | null; confidence: number; captured_at: number;
    resources: Record<string, number | null>; nodes: unknown[]; edges: unknown[];
    metadata: { image_width?: number; image_height?: number }; diagnostics: string[] };
  decision?: { action: Action | null; neural: boolean; reason: string; policy_name: string };
  events: { time: number; state: string; message: string }[];
};
const STATES: Record<string, string> = { idle: "已连接", starting: "正在准备", running: "接管中", observing: "识别预览",
  waiting_battle: "等待手动作战", waiting_observation: "等待可靠识别", paused: "已暂停", stopped: "已停止", completed: "一结局已完成", error: "需要处理" };
const SCENES: Record<string, string> = { map: "路线地图", map_zoom: "地图视野", battle_start: "作战准备", battle: "战斗中", event: "事件", shop: "商店", recruitment: "招募", reward: "战后奖励", dialog: "交互界面", ending: "结局结算", ending_complete: "一结局结算", failed: "探索失败", unknown: "尚未确认" };
const RESOURCES: Record<string, string> = { action_points: "行动力", hp: "生命", max_hp: "生命上限", gold: "源石锭", hope: "希望", parts: "零件", relics: "藏品" };

function friendly(message: string) {
  if (message === "Arknights game window not found") return "未找到明日方舟游戏窗口。请先打开游戏并取消最小化，再点击「重新检测窗口」。";
  if (message === "Game window is hidden or minimized") return "游戏窗口已隐藏或最小化，请先恢复游戏窗口，再启动自动执行。";
  if (message === "Select one game window") return "检测到多个游戏窗口，请选择要识别的窗口。";
  if (message.startsWith("route_missing_observed_resources:")) return "等待识别完整探索状态：" + message.split(":").slice(1).join(":").trim().split(",").map(k => RESOURCES[k] || k).join("、");
  if (message.startsWith("neural_weights_unavailable:")) return "本机神经网络权重尚未就绪，请检查本机接管器配置。";
  const reasons: Record<string, string> = { route_requires_observed_floor_and_current_node: "等待确认当前层和所在节点，请展开完整地图。", no_grounded_scope_legal_action: "当前画面尚未识别到可靠的可操作目标。", no_recognized_action_semantics: "此界面的操作尚未建立可靠对应，已保持等待。", unrecognized_or_terminal_scene: "当前界面尚未确认，等待下一张截图。", manual_battle_required: "请手动开始并完成战斗，战后自动继续。", observation_confidence_too_low: "画面识别把握不足，正在重新识别。" };
  return reasons[message] || (message.startsWith("observation_encoding_failed:") ? "当前观测与模型不匹配，接管已等待。" : message);
}

export default function LiveRouteControl() {
  const [token, setToken] = useState("");
  const tokenRef = useRef("");
  const [status, setStatus] = useState<Status | null>(null);
  const [windows, setWindows] = useState<GameWindow[]>([]);
  const [windowsLoaded, setWindowsLoaded] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [hwnd, setHwnd] = useState("");
  const [pairing, setPairing] = useState(false);
  const [showSetup, setShowSetup] = useState(false);
  const [connectionError, setConnectionError] = useState("");
  const pendingStart = useRef(false);
  const commandLock = useRef(false);
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [connectionLost, setConnectionLost] = useState(false);
  const [preview, setPreview] = useState("");
  const [frameError, setFrameError] = useState("");
  const mounted = useRef(true);
  const imageUrl = useRef("");
  const active = !!status && !["idle", "error", "stopped", "completed"].includes(status.state);

  const saveToken = useCallback((value: string) => {
    tokenRef.current = value;
    setToken(value);
    if (value) sessionStorage.setItem(TOKEN_KEY, value); else sessionStorage.removeItem(TOKEN_KEY);
  }, []);

  useEffect(() => {
    mounted.current = true;
    // The launcher passes this secret only in the fragment; never in a request URL.
    const fragment = window.location.hash;
    const params = new URLSearchParams(fragment.includes("?") ? fragment.split("?")[1] : "");
    const incoming = params.get("bridge_token");
    if (incoming) window.history.replaceState(window.history.state, "", window.location.pathname + window.location.search + "#planner");
    saveToken(incoming || sessionStorage.getItem(TOKEN_KEY) || "");
    return () => { mounted.current = false; if (imageUrl.current) URL.revokeObjectURL(imageUrl.current); };
  }, [saveToken]);

  const request = useCallback(async (path: string, body?: object, overrideToken?: string) => {
    const auth = overrideToken ?? tokenRef.current;
    const response = await fetch(BRIDGE + path, {
      method: body === undefined ? "GET" : "POST",
      headers: { ...(auth ? { Authorization: "Bearer " + auth } : {}), ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      cache: "no-store", credentials: "omit", signal: AbortSignal.timeout(7000),
    });
    const result = await response.json();
    if (!response.ok) {
      if (response.status === 401) { saveToken(""); setPairing(true); setStatus(null); }
      throw new Error(result.error || "本机接管器未响应");
    }
    return result;
  }, [saveToken]);

  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    let lastFrame = 0;
    const poll = async () => {
      try {
        const next: Status = await request("/v1/status");
        if (cancelled) return;
        setStatus(next); setConnectionError(""); setConnectionLost(false);
        if (next.has_preview && (next.observation?.captured_at || 0) !== lastFrame) {
          try {
          const response = await fetch(BRIDGE + "/v1/frame", { headers: { Authorization: "Bearer " + token }, cache: "no-store", signal: AbortSignal.timeout(5000) });
          if (response.ok) {
            const blob = await response.blob();
            if (!cancelled) {
              const url = URL.createObjectURL(blob);
              if (imageUrl.current) URL.revokeObjectURL(imageUrl.current);
              imageUrl.current = url; setPreview(url); setFrameError("");
              lastFrame = next.observation?.captured_at || 0;
            }
          } else if (!cancelled) setFrameError("画面暂时无法加载，正在重试；执行状态仍正常连接。");
          } catch { if (!cancelled) setFrameError("画面暂时无法加载，正在重试；执行状态仍正常连接。"); }
        }
      } catch (err) {
        if (!cancelled) { setConnectionLost(true); setConnectionError(err instanceof TypeError ? "无法连接本机接管器。请确认启动器正在运行，并允许浏览器连接本机。连接丢失后会自动暂停点击。" : String((err as Error).message)); }
      } finally { if (!cancelled) timer = setTimeout(poll, 1500); }
    };
    void poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [token, request]);

  // Games can be opened or restarted after the page connects. Refresh client
  // identities while idle; this endpoint reads metadata and never captures or
  // activates a window. Forget handles that belonged to a previous game run.
  useEffect(() => {
    if (!token || active) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const scan = async () => {
      try {
        const result = await request("/v1/windows");
        if (!cancelled) {
          const found: GameWindow[] = result.windows;
          setWindows(found); setWindowsLoaded(true);
          setHwnd(current => found.some(w => String(w.hwnd) === current) ? current : "");
        }
      } catch { /* The status request reports connection errors. */ }
      finally { if (!cancelled) timer = setTimeout(scan, 5000); }
    };
    void scan();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [token, active, request]);

  async function rescan() {
    setScanning(true); setError("");
    try {
      const result = await request("/v1/windows");
      const found: GameWindow[] = result.windows;
      setWindows(found); setWindowsLoaded(true);
      setHwnd(current => found.some(w => String(w.hwnd) === current) ? current : "");
    } catch (err) { setError((err as Error).message); }
    finally { setScanning(false); }
  }

  async function executeStart(auth?: string) {
    // A read-only session from an older page must not turn "continue" into a
    // no-op. Replace it with an explicitly requested execution session.
    if (status?.observe_only && active) await request("/v1/stop", {} , auth);
    // The bridge acknowledges stop before its OCR thread has fully exited.
    // Retry only that explicit rejection; never retry an ambiguous transport error.
    for (let attempt = 0; mounted.current; attempt += 1) {
      try {
        const next = await request("/v1/start", { ending: "first", ...(hwnd ? { hwnd: Number(hwnd) } : {}) }, auth);
        setStatus(next);
        return;
      } catch (err) {
        if ((err as Error).message !== "已有接管会话；请先停止后重试" || attempt >= 15) throw err;
        await new Promise(resolve => setTimeout(resolve, 700));
      }
    }
  }

  function reportCommandError(err: unknown) {
    const failure = err as Error;
    if (err instanceof TypeError || failure.name === "TimeoutError") {
      setShowSetup(true);
      setConnectionLost(true);
      setError("无法连接本机接管器。请先运行下载包内的 Start-BlackflowLive.cmd；如果浏览器询问是否允许访问本机网络，请允许后重试。");
    } else {
      setError(friendly(failure.message));
    }
  }

  async function command(name: string) {
    if (commandLock.current) return;
    commandLock.current = true;
    setBusy(true); setError("");
    try {
      if (!tokenRef.current) {
        await request("/v1/health");
        pendingStart.current = name === "start";
        setPairing(true);
        return;
      }
      pendingStart.current = name === "start";
      if (name === "start") await executeStart();
      else setStatus(await request("/v1/" + name, {}));
      pendingStart.current = false;
    } catch (err) { reportCommandError(err); }
    finally { commandLock.current = false; setBusy(false); }
  }

  async function pair() {
    if (commandLock.current) return;
    commandLock.current = true;
    setBusy(true); setError("");
    try {
      const result = await request("/v1/pair", { code }, "");
      saveToken(result.token); setPairing(false); setCode("");
      const startRequested = pendingStart.current;
      pendingStart.current = false;
      if (startRequested) await executeStart(result.token);
      else setStatus(await request("/v1/status", undefined, result.token));
    } catch (err) { reportCommandError(err); }
    finally { commandLock.current = false; setBusy(false); }
  }

  const obs = status?.observation;
  const target = status?.decision?.action;
  const paused = status?.state === "paused" && !status.observe_only;
  const executing = active && !status?.observe_only;
  const needsWindow = !active && windows.length > 1 && !hwnd;
  const startLabel = busy ? "正在处理…" : paused ? "继续自动执行" : executing ? "自动执行已启动" : "一键启动自动执行";
  const visibleError = error || connectionError;
  return <section className="live-route-panel" aria-label="神经网络自动执行">
    <div className="live-route-heading">
      <div><span className="eyebrow">NEURAL ROUTE / 一结局</span><h1><BrainCircuit aria-hidden="true" />神经网络自动执行</h1>
        <p>识别游戏画面，选择下一步，执行后继续观察。</p></div>
      <span role="status" className={"live-state live-state-" + (connectionLost ? "error" : status?.state || "offline")}>{connectionLost ? "连接已中断" : status ? STATES[status.state] || status.state : "等待连接本机"}</span>
    </div>
    <div className="live-readiness"><strong>实机试用阶段</strong><p>仅支持一结局的非战斗操作，尚未完成整局实机验收。识别不全或遇到未支持的界面时，会等待你处理。</p></div>
    <div className="live-route-actions">
      <Button className="live-start" disabled={busy || (executing && !paused) || needsWindow} onClick={() => void command(paused ? "resume" : "start")}><Play />{startLabel}</Button>
      {executing && !paused && <Button variant="outline" disabled={busy} onClick={() => void command("pause")}><Pause />暂停</Button>}
      {active && <Button variant="outline" disabled={busy} onClick={() => void command("stop")}><Square />停止执行</Button>}
      {!active && <button className="live-setup-toggle" onClick={() => setShowSetup(current => !current)} aria-expanded={showSetup} aria-controls="live-setup">{showSetup ? "收起连接说明" : "首次使用 / 连接帮助"}</button>}
    </div>
    <p className="live-boundary">首次配置并启动本机接管器后，一键开始。<strong>开始战斗与战斗过程由你操作，战后自动继续。</strong></p>
    {token && windowsLoaded && !active && <div className="live-window-status">
      <p role="status">{windows.length ? `已检测到 ${windows.length} 个游戏窗口：${windows.map(w => w.title + (w.state === "minimized" ? "（已最小化，请恢复）" : w.state === "hidden" ? "（已隐藏，请打开）" : w.client_rect ? `（${w.client_rect.width} × ${w.client_rect.height}）` : "")).join("、")}` : "尚未检测到游戏窗口。请进入游戏、取消最小化后重新检测。"}</p>
      <button onClick={() => void rescan()} disabled={busy || scanning}><RefreshCw size={16} />{scanning ? "正在检测" : "重新检测窗口"}</button>
    </div>}
    {windows.length > 1 && !active && <label className="live-window-label">选择要接管的游戏窗口<select aria-label="选择游戏窗口" value={hwnd} onChange={event => setHwnd(event.target.value)}>
      <option value="">请选择窗口</option>{windows.map(w => <option key={w.hwnd} value={w.hwnd}>{w.title} · 窗口 {w.hwnd}{w.client_rect ? ` · ${w.client_rect.width} × ${w.client_rect.height}` : ""}{w.state === "minimized" ? " · 已最小化" : w.state === "hidden" ? " · 已隐藏" : ""}</option>)}
    </select></label>}
    {visibleError && <p className="live-error" role="alert"><AlertTriangle />{visibleError}</p>}
    {pairing && <form className="live-pairing" onSubmit={event => { event.preventDefault(); void pair(); }}>
      <label htmlFor="bridge-code">输入本机启动器显示的 6 位连接码</label><div><Input id="bridge-code" aria-label="6 位连接码" autoComplete="off" inputMode="numeric" pattern="[0-9]{6}" maxLength={6} value={code} onChange={event => setCode(event.target.value.replace(/\D/g, ""))} /><Button disabled={busy || code.length !== 6}><Link2 />{pendingStart.current ? "连接并启动" : "连接"}</Button></div>
    </form>}
    {showSetup && <div className="live-setup" id="live-setup">
      <div><h2>首次在游戏电脑上连接</h2><a className="live-download" href="/downloads/blackflow-live-connector.zip" download><Download size={18} />下载 Windows 本机接管器</a></div>
      <ol><li><strong>准备本机环境。</strong>解压下载包，按包内说明安装 Python 3.13 和依赖，并准备 BFMapRecognizer / MAA 资源。</li><li><strong>双击启动。</strong>运行包内 <code>tools/Start-BlackflowLive.cmd</code>，会自动打开本站并连接。手动打开本站时，可输入启动器的连接码。</li><li><strong>一键开始。</strong>进入游戏的完整地图，保持窗口可见且未最小化，点击「一键启动自动执行」。浏览器询问本机网络访问时，请允许。</li></ol>
      <p>网页需要本机接管器才能读取和操作游戏，无法直接启动电脑程序。下载包包含神经网络权重和推理代码，MAA 资源需在本机另行准备。</p>
    </div>}
    <div className="live-workspace">
      <div className="live-screenshot">
        <div className="live-frame-heading"><Monitor size={16} /><span>实时游戏画面</span><small>{preview ? "本机截图" : "等待启动"}</small></div>
        {frameError && <p className="live-frame-error" role="status">{frameError}</p>}
        <div className="live-frame-content">{preview ? <><img src={preview} alt="本机接管器读取到的真实游戏窗口" />
          {target && obs?.metadata.image_width && obs.metadata.image_height && <svg className="live-target" viewBox={`0 0 ${obs.metadata.image_width} ${obs.metadata.image_height}`} aria-label={`下一步目标：${target.label}`}><rect x={target.bbox[0]} y={target.bbox[1]} width={target.bbox[2]} height={target.bbox[3]} /></svg>}</>
          : <div className="live-empty"><Monitor /><strong>启动后，这里显示识别到的游戏画面</strong><span>自动读取真实窗口和探索状态，无需手工填写路线。</span></div>}</div>
      </div>
      <div className="live-observation" aria-live="polite">
        <h2>执行状态</h2><strong>{status ? friendly(status.message) : "等待本机连接，尚未开始操作游戏。"}</strong>
        {status?.window && <p>{status.window.title} · {status.window.client_rect.width} × {status.window.client_rect.height} · {Math.round(status.window.dpi / 96 * 100)}% 缩放</p>}
        <dl><div><dt>当前界面</dt><dd>{obs ? SCENES[obs.scene] || obs.scene : "—"}</dd></div><div><dt>已确认层数</dt><dd>{obs?.floor ? `${obs.floor} 层` : "—"}</dd></div><div><dt>已执行点击</dt><dd>{status?.clicks ?? 0}</dd></div></dl>
        {obs && <div className="live-resource-grid">{Object.entries(RESOURCES).map(([key, label]) => <div key={key}><span>{label}</span><strong>{obs.resources[key] ?? "未识别"}</strong></div>)}</div>}
        {status?.decision?.neural && <p className="live-choice">神经网络选择：{target?.label || "等待下一帧"}</p>}
        {!!status?.events?.length && <ol className="live-events">{status.events.slice(-3).reverse().map((e, i) => <li key={`${e.time}-${i}`}><time>{new Date(e.time * 1000).toLocaleTimeString("zh-CN", { hour12: false })}</time>{friendly(e.message)}</li>)}</ol>}
        <p className="live-stop-hint">按 Esc 或将鼠标移至桌面左上角可暂停。关闭页面或断连超过 20 秒也会暂停。</p>
      </div>
    </div>
    <section className="live-capabilities" aria-labelledby="live-capability-title">
      <h2 id="live-capability-title">目前能做到哪一步？</h2>
      <div className="live-capability-grid">
        <div><span className="live-capability-label">已接通</span><h3>看图 → 决策 → 点击</h3><p>在可靠识别当前层、节点与必要资源后，由冻结神经网络选择合法路线和已支持的菜单选项；每步重新截图核对。</p></div>
        <div><span className="live-capability-label">仍有限制</span><h3>部分界面需要接手</h3><p>商店、招募、多步骤事件及完整背包扫描仍有覆盖缺口。未识别的资源不会自行补全，未知选项会等待你处理。</p></div>
        <div><span className="live-capability-label">需要你完成</span><h3>手动作战，战后继续</h3><p>不会点击「开始战斗」，不负责部署干员或释放技能。目前以一结局为目标，不能承诺全自动通关或真实通关率。</p></div>
      </div>
      <p className="live-evidence">验证进度：地图识别已通过保存的真实截图回放；模型训练与模拟评估不能替代整局实机验收。</p>
    </section>
  </section>;
}
