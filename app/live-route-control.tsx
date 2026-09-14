"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, Download, Link2, Monitor, Pause, Play, ScanLine, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const BRIDGE = "http://127.0.0.1:19761";
const TOKEN_KEY = "blackflow-live-token";
type GameWindow = { hwnd: number; title: string; dpi: number; client_rect: { width: number; height: number } };
type Action = { action_id: string; label: string; bbox: number[] };
type Status = {
  state: string; message: string; clicks: number; has_preview: boolean; observe_only?: boolean;
  window?: GameWindow;
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
  const [hwnd, setHwnd] = useState("");
  const [pairing, setPairing] = useState(false);
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [connectionLost, setConnectionLost] = useState(false);
  const [preview, setPreview] = useState("");
  const imageUrl = useRef("");
  const active = !!status && !["idle", "error", "stopped", "completed"].includes(status.state);

  const saveToken = useCallback((value: string) => {
    tokenRef.current = value;
    setToken(value);
    if (value) sessionStorage.setItem(TOKEN_KEY, value); else sessionStorage.removeItem(TOKEN_KEY);
  }, []);

  useEffect(() => {
    // The launcher passes this secret only in the fragment; never in a request URL.
    const fragment = window.location.hash;
    const params = new URLSearchParams(fragment.includes("?") ? fragment.split("?")[1] : "");
    const incoming = params.get("bridge_token");
    if (incoming) window.history.replaceState(window.history.state, "", window.location.pathname + window.location.search + "#planner");
    saveToken(incoming || sessionStorage.getItem(TOKEN_KEY) || "");
    return () => { if (imageUrl.current) URL.revokeObjectURL(imageUrl.current); };
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
        setStatus(next); setError(""); setConnectionLost(false);
        if (next.has_preview && (next.observation?.captured_at || 0) !== lastFrame) {
          lastFrame = next.observation?.captured_at || 0;
          const response = await fetch(BRIDGE + "/v1/frame", { headers: { Authorization: "Bearer " + token }, cache: "no-store", signal: AbortSignal.timeout(5000) });
          if (response.ok) {
            const blob = await response.blob();
            if (!cancelled) {
              const url = URL.createObjectURL(blob);
              if (imageUrl.current) URL.revokeObjectURL(imageUrl.current);
              imageUrl.current = url; setPreview(url);
            }
          }
        }
      } catch (err) {
        if (!cancelled) { setConnectionLost(true); setError(err instanceof TypeError ? "无法连接本机接管器。请确认启动器正在运行，并允许浏览器连接本机。连接丢失后会自动暂停点击。" : String((err as Error).message)); }
      } finally { if (!cancelled) timer = setTimeout(poll, 1500); }
    };
    void poll();
    void request("/v1/windows").then(result => { if (!cancelled) setWindows(result.windows); }).catch(() => {});
    return () => { cancelled = true; clearTimeout(timer); };
  }, [token, request]);

  async function command(name: string) {
    setBusy(true); setError("");
    try {
      if (!tokenRef.current) {
        await request("/v1/health");
        setPairing(true);
        return;
      }
      const next = await request("/v1/" + name, { ending: "first", ...(hwnd ? { hwnd: Number(hwnd) } : {}) });
      setStatus(next);
    } catch (err) {
      setError(err instanceof TypeError || (err as Error).name === "TimeoutError" ? "未连接到本机接管器。请先在游戏电脑运行「Start-BlackflowLive.cmd」；如浏览器询问本机网络访问，请允许后重试。" : (err as Error).message);
    } finally { setBusy(false); }
  }

  async function pair() {
    setBusy(true); setError("");
    try {
      const result = await request("/v1/pair", { code }, "");
      saveToken(result.token); setPairing(false); setCode("");
      setStatus(await request("/v1/status", undefined, result.token));
    } catch (err) { setError((err as Error).message); } finally { setBusy(false); }
  }

  const obs = status?.observation;
  const target = status?.decision?.action;
  return <section className="live-route-panel" aria-label="自动路线决策">
    <div className="live-route-heading">
      <div><span className="eyebrow">LIVE ROUTE CONTROL</span><h2><ScanLine aria-hidden="true" />自动路线决策</h2>
        <p>读取真实游戏画面，由本机神经网络选择下一步。目标：一结局。</p></div>
      <span className={"live-state live-state-" + (connectionLost ? "error" : status?.state || "offline")}>{connectionLost ? "连接已中断" : status ? STATES[status.state] || status.state : "等待连接本机"}</span>
    </div>
    <div className="live-route-actions">
      <Button className="live-start" disabled={busy || active} onClick={() => void command("start")}><Play />自动路线决策</Button>
      <Button variant="outline" disabled={busy || active} onClick={() => void command("observe")}><Monitor />识别预览</Button>
      {active && status?.state !== "paused" && <Button variant="outline" disabled={busy} onClick={() => void command("pause")}><Pause />暂停</Button>}
      {status?.state === "paused" && <Button variant="outline" disabled={busy} onClick={() => void command("resume")}><Play />继续</Button>}
      {active && <Button variant="outline" disabled={busy} onClick={() => void command("stop")}><Square />停止接管</Button>}
      {windows.length > 1 && <label className="live-window-label">游戏窗口<select aria-label="选择游戏窗口" value={hwnd} onChange={event => setHwnd(event.target.value)} disabled={active}>
        <option value="">请选择窗口</option>{windows.map(w => <option key={w.hwnd} value={w.hwnd}>{w.title} · {w.client_rect.width} × {w.client_rect.height}</option>)}
      </select></label>}
    </div>
    <p className="live-boundary">开始战斗和战斗过程由你操作，战后自动继续。按 Esc 或将鼠标移至桌面左上角可暂停。</p>
    {error && <p className="live-error" role="alert"><AlertTriangle />{error}</p>}
    {pairing && <form className="live-pairing" onSubmit={event => { event.preventDefault(); void pair(); }}>
      <label htmlFor="bridge-code">输入本机启动器显示的 6 位连接码</label><div><Input id="bridge-code" autoComplete="off" inputMode="numeric" pattern="[0-9]{6}" maxLength={6} value={code} onChange={event => setCode(event.target.value.replace(/\D/g, ""))} /><Button disabled={busy || code.length !== 6}><Link2 />连接</Button></div>
    </form>}
    <div className="live-workspace">
      <div className="live-screenshot">
        {preview ? <><img src={preview} alt="本机接管器读取到的真实游戏窗口" />
          {target && obs?.metadata.image_width && <svg className="live-target" viewBox={`0 0 ${obs.metadata.image_width} ${obs.metadata.image_height}`} aria-label={`下一步目标：${target.label}`}><rect x={target.bbox[0]} y={target.bbox[1]} width={target.bbox[2]} height={target.bbox[3]} /></svg>}</>
          : <div className="live-empty"><Monitor /><strong>游戏画面将在连接后显示</strong><span>支持窗口全屏、无边框和不同宽高比；每次操作重新校准坐标。</span></div>}
      </div>
      <div className="live-observation" aria-live="polite">
        <strong>{status ? friendly(status.message) : "先启动本机接管器，再点击自动路线决策。"}</strong>
        {status?.window && <p>{status.window.title} · {status.window.client_rect.width} × {status.window.client_rect.height} · {Math.round(status.window.dpi / 96 * 100)}% 缩放</p>}
        <dl><div><dt>当前界面</dt><dd>{obs ? SCENES[obs.scene] || obs.scene : "—"}</dd></div><div><dt>已确认层数</dt><dd>{obs?.floor ? `${obs.floor} 层` : "—"}</dd></div><div><dt>已执行点击</dt><dd>{status?.clicks ?? 0}</dd></div></dl>
        {obs && <div className="live-resource-grid">{Object.entries(RESOURCES).map(([key, label]) => <div key={key}><span>{label}</span><strong>{obs.resources[key] ?? "未识别"}</strong></div>)}</div>}
        {status?.decision?.neural && <p className="live-choice">神经网络选择：{target?.label || "等待下一帧"}</p>}
        {!!status?.events.length && <ol className="live-events">{status.events.slice(-4).reverse().map((e, i) => <li key={`${e.time}-${i}`}><time>{new Date(e.time * 1000).toLocaleTimeString("zh-CN", { hour12: false })}</time>{friendly(e.message)}</li>)}</ol>}
      </div>
    </div>
    <details className="live-setup"><summary>首次连接 / 使用说明</summary>
      <ol><li>在游戏电脑下载并解压本机模块，按其中说明安装 Python 依赖；地图识别使用电脑中已有的 BFMapRecognizer / MAA 资源。</li><li>运行模块中的 <code>tools/Start-BlackflowLive.cmd</code>，启动器会打开本站并连接本机。网页无法独立启动或控制 Windows 程序。</li><li>点击「自动路线决策」。若提示配对，输入启动器的连接码；本机网络访问由浏览器授权。</li><li>保持完整地图可见。无法识别的状态会等待，不会用模拟数据代替截图；关闭页面或连接中断会暂停接管。</li></ol>
      <a href="/downloads/blackflow-live-connector.zip" download><Download size={16} />下载本机接管模块</a><p>此版本已接通实时识别和控制接口；尚未完成一结局全流程实机验收。下载包包含当前神经网络权重和推理代码，MAA 运行资源使用本机已有安装。</p>
    </details>
  </section>;
}
