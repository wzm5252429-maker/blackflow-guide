"""Read-only screenshot perception using the installed MAA assets.

Unlike MAA's task runner this module never attaches an input controller. OCR,
template matches and CorridorNet all consume the *same supplied BGR image*.
Boxes are image pixels; the capture layer owns the inverse desktop transform.
No old map_result.json, simulator state or invented resource default is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable

import numpy as np

from .models import LiveObservation, ObservedAction, ObservedNode


DEFAULT_MAA_ROOT = Path("D:/ArknightsAuto/BFMapRecognizer_v1.0.2_Windows/BFMapRecognizer")


@dataclass(frozen=True)
class OCRSpan:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


def _cv():
    import cv2
    return cv2


def _session(path: Path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _read_image(path: Path):
    return _cv().imdecode(np.fromfile(str(path), dtype=np.uint8), _cv().IMREAD_COLOR)


def _clean(text: str) -> str:
    return re.sub(r"[\s“”\"‘’·•:：，,。.!！?？<>/\\]", "", re.sub(r"<[^>]+>", "", text))


def _box(points: np.ndarray) -> tuple[float, float, float, float]:
    low, high = points.min(axis=0), points.max(axis=0)
    return tuple(float(x) for x in (*low, *(high - low)))


class MaaPaddleOCR:
    """DB text detection and CTC recognition with MAA's Chinese Paddle models.

    Detection runs over the whole image, so there are no 16:9 resource ROIs.
    The expansion of each detected rectangle protects character ascenders.
    """

    def __init__(self, maa_root: str | Path = DEFAULT_MAA_ROOT, minimum_confidence: float = 0.65):
        root = Path(maa_root) / "resource" / "PaddleOCR"
        self.det = _session(root / "det" / "inference.onnx")
        self.rec = _session(root / "rec" / "inference.onnx")
        self.characters = [""] + (root / "rec" / "keys.txt").read_text(encoding="utf-8").splitlines() + [" "]
        self.minimum_confidence = minimum_confidence
        self.det_input = self.det.get_inputs()[0].name
        self.rec_input = self.rec.get_inputs()[0].name

    def recognize_crop(self, image: np.ndarray) -> tuple[str, float]:
        if not image.size or min(image.shape[:2]) < 2:
            return "", 0.0
        cv = _cv()
        height, width = image.shape[:2]
        target_width = min(2048, max(16, math.ceil(width * 48 / height)))
        resized = cv.resize(image, (target_width, 48)).astype(np.float32) / 127.5 - 1.0
        # Paddle recognition is BGR; channel swapping would reduce accuracy.
        prediction = self.rec.run(None, {self.rec_input: resized.transpose(2, 0, 1)[None]})[0][0]
        indexes = prediction.argmax(axis=-1)
        scores = prediction.max(axis=-1)
        keep = (indexes != 0) & np.concatenate(([True], indexes[1:] != indexes[:-1]))
        values = indexes[keep]
        if not len(values) or int(values.max()) >= len(self.characters):
            return "", 0.0
        return "".join(self.characters[int(i)] for i in values), float(scores[keep].mean())

    def recognize(self, image: np.ndarray) -> tuple[OCRSpan, ...]:
        cv = _cv()
        height, width = image.shape[:2]
        scale = min(1.0, 1280.0 / max(height, width))
        dw, dh = max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)
        resized = cv.resize(image, (dw, dh)).astype(np.float32) / 255.0
        normalized = (resized - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        probability = self.det.run(None, {self.det_input: normalized.transpose(2, 0, 1)[None]})[0][0, 0]
        binary = (probability > 0.3).astype(np.uint8) * 255
        contours, _ = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        spans: list[OCRSpan] = []
        for contour in contours[:1200]:
            x, y, w, h = cv.boundingRect(contour)
            if min(w, h) < 3 or float(probability[y:y+h, x:x+w].mean()) < 0.45:
                continue
            (cx, cy), (bw, bh), angle = cv.minAreaRect(contour)
            if min(bw, bh) < 2:
                continue
            # DB unclip's area/perimeter expansion, represented by a rectangle.
            expansion = bw * bh * 1.5 / (2 * (bw + bh))
            points = cv.boxPoints(((cx, cy), (bw + 2 * expansion, bh + 2 * expansion), angle))
            points[:, 0] *= width / probability.shape[1]
            points[:, 1] *= height / probability.shape[0]
            bx, by, bwidth, bheight = _box(points)
            left, top = max(0, math.floor(bx)), max(0, math.floor(by))
            right, bottom = min(width, math.ceil(bx+bwidth)), min(height, math.ceil(by+bheight))
            if right <= left or bottom <= top:
                continue
            text, score = self.recognize_crop(image[top:bottom, left:right])
            if text.strip() and score >= self.minimum_confidence:
                spans.append(OCRSpan(text.strip(), score, (left, top, right-left, bottom-top)))
        return tuple(sorted(spans, key=lambda span: (span.bbox[1], span.bbox[0])))


@dataclass(frozen=True)
class TemplateHit:
    name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    scale: float


class MaaTemplates:
    """Scale-search MAA templates over the supplied content, without fixed ROIs."""

    def __init__(self, maa_root: str | Path = DEFAULT_MAA_ROOT):
        root = Path(maa_root) / "resource"
        self.root = root
        self.directory = root / "template" / "Roguelike" / "BlackFlow"
        self._cache: dict[str, Any] = {}
        manifest_path = root / "roguelike" / "BlackFlow" / "map_perception" / "templates" / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.node_specs = self.manifest["templates"]
        self.labels = {_clean(x["display_name"]): x["node_type"] for x in self.node_specs if x.get("display_name") and x.get("node_type")}
        self.labels.update({_clean(x["text"]): x["node_type"] for x in self.manifest.get("ocr_labels", [])})

    def template(self, name: str):
        if name not in self._cache:
            candidate = next((p for p in (Path(__file__).parent / "assets" / name, self.directory / name, self.root / "template" / "Roguelike" / "base" / name, self.root / "template" / name) if p.is_file()), None)
            if candidate is None:
                # Some shared assets live one directory deeper. Search only once.
                candidate = next(self.root.joinpath("template").rglob(name), None)
            self._cache[name] = _read_image(candidate) if candidate else None
        return self._cache[name]

    def match(self, image: np.ndarray, name: str, *, threshold: float = 0.84, maximum: int = 1, scales: Iterable[float] | None = None) -> list[TemplateHit]:
        cv = _cv()
        template = self.template(name)
        if template is None:
            return []
        base = min(image.shape[1]/1280, image.shape[0]/720)
        scales = tuple(scales) if scales is not None else sorted({1.0, *(base * x for x in (0.65, 0.8, 0.9, 1.0, 1.15, 1.35))})
        hits = []
        for scale in scales:
            tw, th = max(3, round(template.shape[1]*scale)), max(3, round(template.shape[0]*scale))
            if tw > image.shape[1] or th > image.shape[0]:
                continue
            scaled = cv.resize(template, (tw, th), interpolation=cv.INTER_AREA if scale < 1 else cv.INTER_CUBIC)
            if float(scaled.std()) < 2:
                continue
            scores = cv.matchTemplate(image, scaled, cv.TM_CCOEFF_NORMED)
            for _ in range(maximum):
                _, confidence, _, (x, y) = cv.minMaxLoc(scores)
                if not math.isfinite(confidence) or confidence < threshold:
                    break
                hits.append(TemplateHit(name, confidence, (x, y, tw, th), float(scale)))
                scores[max(0,y-th//2):min(scores.shape[0],y+th//2+1), max(0,x-tw//2):min(scores.shape[1],x+tw//2+1)] = -1
        return _suppress(hits, maximum)


def _suppress(hits: list[TemplateHit], maximum: int) -> list[TemplateHit]:
    accepted = []
    for hit in sorted(hits, key=lambda h: h.confidence, reverse=True):
        x,y,w,h = hit.bbox
        if any(abs(x+w/2-(a.bbox[0]+a.bbox[2]/2)) < max(w,a.bbox[2])*0.6 and abs(y+h/2-(a.bbox[1]+a.bbox[3]/2)) < max(h,a.bbox[3])*0.6 for a in accepted):
            continue
        accepted.append(hit)
        if len(accepted) >= maximum:
            break
    return accepted


class CorridorNet:
    def __init__(self, maa_root: str | Path = DEFAULT_MAA_ROOT):
        resource = Path(maa_root) / "resource"
        self.config = json.loads((resource / "roguelike/BlackFlow/map_perception/corridor_net.runtime.json").read_text(encoding="utf-8"))
        self.session = _session(resource / "onnx/BlackFlow_corridor_net.onnx")

    def score(self, image: np.ndarray, pairs: list[tuple[tuple[float,float],tuple[float,float]]]) -> list[float]:
        cv, crops = _cv(), []
        settings = self.config["corridor"]
        for a,b in pairs:
            a,b = np.array(a,np.float32),np.array(b,np.float32)
            length = float(np.linalg.norm(b-a))
            if length < 1:
                raise ValueError("Corridor endpoints coincide")
            direction = (b-a)/length
            normal = np.array([-direction[1], direction[0]])
            start = a + direction*length*settings["endpoint_margin_ratio"]
            end = b - direction*length*settings["endpoint_margin_ratio"]
            half = normal*np.clip(length*settings["width_ratio_of_edge_length"], settings["minimum_width_pixels"], settings["maximum_width_pixels"])/2
            source = np.array([start-half,start+half,end+half,end-half],np.float32)
            dest = np.array([[0,0],[0,39],[159,39],[159,0]],np.float32)
            crops.append(cv.warpPerspective(image,cv.getPerspectiveTransform(source,dest),(160,40),flags=cv.INTER_CUBIC))
        if not crops:
            return []
        pre = self.config["preprocessing"]
        tensor = np.stack(crops)[:,:,:,::-1].astype(np.float32)*pre["scale"]
        tensor = ((tensor-np.array(pre["mean"],np.float32))/np.array(pre["std"],np.float32)).transpose(0,3,1,2)
        logits = self.session.run(None,{self.session.get_inputs()[0].name:tensor})[0].reshape(-1)
        temperature = self.config["decision"]["temperature"]
        return [float(1/(1+math.exp(-float(np.clip(logit/temperature,-60,60))))) for logit in logits]


def extract_resources(spans: Iterable[OCRSpan]) -> dict[str, int]:
    """Read labeled current quantities only, never costs/effects or bare numbers."""
    labels = {"行动力":"action_points", "行动点":"action_points", "目标生命值":"hp", "目标生命":"hp", "生命值":"hp", "生命":"hp", "源石锭":"gold", "希望":"hope", "零件":"parts", "零件箱":"parts", "收藏品":"relics", "藏品":"relics"}
    result: dict[str,int] = {}
    for span in spans:
        text = re.sub(r"\s", "", span.text)
        if span.confidence < 0.85 or re.search(r"获得|失去|消耗|花费|需要|兑换|上限提升|[+＋\-−]", text):
            continue
        for label, field in labels.items():
            match = re.fullmatch(re.escape(label)+r"[:：]?(\d{1,3})(?:[/／](\d{1,3}))?", text)
            if match:
                result[field] = int(match[1])
                if match[2] and field == "hp":
                    result["max_hp"] = int(match[2])
    return result


# Buttons need positive text/template evidence. Arbitrary OCR prose is not a target.
_OPERATIONS = {"离开":"leave", "返回":"leave", "继续":"advance", "下一步":"advance", "确认":"event_advance", "确定":"event_advance", "领取":"take", "收下":"take", "全部领取":"take", "购买":"purchase", "招募":"recruit_reserve", "临时招募":"recruit_temporary", "暂不招募":"decline_recruitment", "放弃招募":"decline_recruitment", "刷新":"refresh", "出售":"sell", "装备":"equip", "丢弃":"discard", "进入":"advance", "出发":"advance", "启程":"advance", "开始探索":"starting_reward", "继续探索":"advance", "进入下一层":"advance", "完成招募":"advance", "完成":"event_advance", "跳过":"event_advance"}
_OPERATIONS.update({"确认招募":"recruit_reserve","继续招募":"advance","确认雇佣":"emergency_hire","雇佣":"emergency_hire","确认购买":"purchase","确认出售":"sell","确认装备":"equip","确认丢弃":"discard","确认选择":"event_advance","确认收下":"take","带走":"take","放弃奖励":"leave","进入探索":"advance","继续前进":"advance"})
_BATTLE_WORDS = ("开始行动", "开始战斗", "进入战斗", "开始作战", "确认出击", "出击")
_FLOORS = ("玻利瓦尔肤层", "甜美的伤口", "血色空脉", "受害者腐殖", "卡德霍之颅", "未萌生的摇篮")


class VisionPipeline:
    """Produce grounded observations; unsupported/ambiguous screens fail closed."""

    def __init__(self, maa_root: str | Path = DEFAULT_MAA_ROOT, *, ocr=None, templates=None, corridor=None, evidence_root: str | Path | None = None):
        self.maa_root = Path(maa_root)
        self.ocr = ocr if ocr is not None else MaaPaddleOCR(maa_root)
        self.templates = templates if templates is not None else MaaTemplates(maa_root)
        self.corridor = corridor if corridor is not None else CorridorNet(maa_root)
        evidence = Path(evidence_root) if evidence_root else Path(__file__).resolve().parents[1]/"data/evidence"
        self.choices: dict[str, list[dict]] = {}
        self.item_names: dict[str,str] = {}
        self.operators: dict[str,dict] = {}
        choices_path = evidence/"rogue6_client_choice_snapshot_v1.json"
        if choices_path.is_file():
            for choice in json.loads(choices_path.read_text(encoding="utf-8")).get("choices",[]):
                title = _clean(choice.get("title", ""))
                if title:
                    self.choices.setdefault(title,[]).append(choice)
        try:
            from blackflow_rl.catalog import load_catalog
            for item in load_catalog().items.values():
                self.item_names[_clean(item.name)] = item.name
        except (ImportError, FileNotFoundError):
            pass
        recruitment = self.maa_root/"resource/recruitment.json"
        if recruitment.is_file():
            for operator in json.loads(recruitment.read_text(encoding="utf-8")).get("operators",[]):
                self.operators[_clean(operator["name"])]=operator

    def observe(self, frame, *, frame_id: str | None = None, captured_at: float | None = None) -> LiveObservation:
        image = frame.image if hasattr(frame,"image") else frame
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32:
            raise ValueError("Vision requires a nonempty BGR8 screenshot")
        frame_id = frame_id or getattr(frame,"frame_id",None) or sha256(image.tobytes()).hexdigest()[:24]
        captured_at = captured_at if captured_at is not None else getattr(getattr(frame,"geometry",None),"captured_at",time.time())
        spans = self.ocr.recognize(image)
        return self.analyze(image,spans,frame_id=frame_id,captured_at=captured_at)

    recognize = observe

    def analyze(self, image: np.ndarray, spans: Iterable[OCRSpan], *, frame_id: str, captured_at: float) -> LiveObservation:
        spans = tuple(spans)
        diagnostics = []
        text = " ".join(span.text for span in spans if span.confidence >= 0.8)
        clean_text = _clean(text)
        resources = extract_resources(spans)
        floor = next((i+1 for i,label in enumerate(_FLOORS) if label in clean_text),None)
        markers: dict[str,TemplateHit] = {}
        # These are recognition-only assets. Their task actions are never run.
        marker_names = {
            "battle_start":"Roguelike@StartAction.png",
            "battle":"BattleOfficiallyBegin.png",
            "ending":"BlackFlow@Roguelike@GamePassTheEndConfirm.png",
            "game_pass":"BlackFlow@Roguelike@GamePass.png",
            "map":"BlackFlow@Roguelike@MapZoomIn.png",
            "map_zoom":"BlackFlow@Roguelike@MapZoomOut.png",
            "event":"BlackFlow@Roguelike@CloseEvent.png",
            "reward":"BlackFlow@Roguelike@GetDrop.png",
            "recruitment":"BlackFlow@Roguelike@ChooseOperConfirm.png",
            "shop":"BlackFlow@Roguelike@CultivateConfirm.png",
            "movement_preview":"BlackFlow@Roguelike@MovePreviewEnter.png",
            "inventory":"BlackFlow@Roguelike@MovementInventoryCollapse.png",
        }
        for kind,name in marker_names.items():
            hits = self.templates.match(image,name,threshold=0.88)
            if hits:
                markers[kind] = hits[0]
        scene, confidence = "unknown", 0.0
        first_ending = False
        if any(word in clean_text for word in _BATTLE_WORDS) or "battle_start" in markers:
            scene,confidence = "battle_start", max([markers.get("battle_start",TemplateHit("",0,(),1)).confidence,0.97 if any(word in clean_text for word in _BATTLE_WORDS) else 0])
        elif "battle" in markers or ("撤退" in clean_text and ("费用" in clean_text or "敌方" in clean_text)):
            scene,confidence = "battle",0.97
        elif "ending" in markers or "game_pass" in markers or "探索完成" in clean_text or "旅途结束" in clean_text:
            scene,confidence = "ending",max([0.9]+[h.confidence for k,h in markers.items() if k in {"ending","game_pass"}])
            # Boss/ending text alone is never sufficient. Require result UI and
            # an explicit first-ending identity visible in this same screenshot.
            # ro6_ending_1.name is pinned as 强制重启 in
            # data/evidence/rogue6_ending_starting_rules_v1.json.
            first_identity=[s for s in spans if s.confidence>=.95 and any(word in _clean(s.text) for word in ("结局一","结局1","强制重启"))]
            result_confidence=max([0.0]+[s.confidence for s in spans if any(word in _clean(s.text) for word in ("探索完成","旅途结束","通关","结局达成"))]+[h.confidence for k,h in markers.items() if k in {"ending","game_pass"}])
            first_ending = bool(first_identity and result_confidence>=.95)
            if first_ending:
                scene = "ending_complete"
                confidence=min(result_confidence,*(s.confidence for s in first_identity))
        elif any(word in clean_text for word in ("行动失败", "探索失败")):
            scene,confidence = "failed",0.96
        elif "event" in markers:
            scene,confidence = "event",markers["event"].confidence
        elif "recruitment" in markers or any(word in clean_text for word in ("招募干员", "选择干员", "放弃招募", "暂不招募")):
            scene,confidence = "recruitment",0.94
        elif "reward" in markers or any(word in clean_text for word in ("战利品", "领取奖励", "获得收藏品")):
            scene,confidence = "reward",0.92
        elif "shop" in markers or any(word in clean_text for word in ("商品售价", "坎诺特", "培育区", "购买商品")):
            scene,confidence = "shop",0.92
        elif "movement_preview" in markers:
            scene,confidence = "movement_preview",markers["movement_preview"].confidence
        elif "inventory" in markers:
            scene,confidence = "inventory",markers["inventory"].confidence
        elif any(len(_clean(span.text))>=3 and _clean(span.text) in self.choices and span.bbox[0]>image.shape[1]*.55 for span in spans if span.confidence>=.9):
            scene,confidence = "event",.91
        elif "map" in markers or "map_zoom" in markers or floor is not None:
            scene,confidence = "map",max([0.87]+[h.confidence for k,h in markers.items() if k in {"map","map_zoom"}])
        elif any(_clean(span.text) in self.choices for span in spans if span.confidence >= 0.9):
            scene,confidence = "event",0.91
        elif ("黑流树海" in clean_text and any(_clean(span.text) in _OPERATIONS for span in spans if span.confidence >= 0.9)):
            scene,confidence = "dialog",0.9
        nodes,edges,current = (),(),None
        actions: list[ObservedAction] = []
        if scene in {"map","shop","recruitment","inventory","reward","movement_preview"}:
            resources.update(self._hud_resources(image,spans))
        if scene == "map":
            nodes,edges,current,map_notes = self._map(image,spans)
            diagnostics.extend(map_notes)
            if current is not None:
                adjacent = {b for a,b in edges if a == current} | {a for a,b in edges if b == current}
                for node in nodes:
                    if node.node_id in adjacent and node.node_id != current:
                        actions.append(ObservedAction("node:"+node.node_id,node.node_type,"map_node",node.bbox,node.confidence,target_node_id=node.node_id,metadata={"operation":"move","source":"maa_node_and_corridor","grounded":True,"preview_only":True}))
            if "map_zoom" in markers:
                hit = markers["map_zoom"]
                actions = [ObservedAction("map:zoom_out","缩小地图","ui",hit.bbox,hit.confidence,metadata={"operation":"event_advance","source":"maa_template","grounded":True})]
        if scene not in {"battle", "battle_start", "ending", "ending_complete", "unknown", "failed"}:
            actions.extend(self._actions(spans,scene,image.shape[1],image.shape[0]))
            if scene=="movement_preview":
                hit=markers["movement_preview"]
                actions.append(ObservedAction("map:enter_preview","进入节点","ui",hit.bbox,hit.confidence,metadata={"operation":"event_advance","grounded":True,"source":"maa_template"}))
            actions.extend(self._catalog_actions(spans,scene,image.shape[1],image.shape[0]))
            actions=self._unique_actions(actions)
        for action in actions:
            action.metadata["source_frame_id"]=frame_id
        if scene == "unknown":
            diagnostics.append("unrecognized_scene_no_input")
        if not actions and scene not in {"battle","battle_start","ending","ending_complete"}:
            diagnostics.append("no_grounded_action")
        return LiveObservation(frame_id,captured_at,scene,confidence,tuple(actions),tuple(nodes),tuple(edges),resources,current,floor,first_ending,tuple(diagnostics),{"image_width":int(image.shape[1]),"image_height":int(image.shape[0]),"ocr":[{"text":s.text,"confidence":s.confidence,"bbox":s.bbox} for s in spans],"markers":{k:{"bbox":v.bbox,"confidence":v.confidence} for k,v in markers.items()},"perception":"maa_paddle_ocr_templates_corridor_v1","first_ending_evidence":first_ending})

    def _actions(self, spans: tuple[OCRSpan,...], scene: str, width: int, height: int) -> list[ObservedAction]:
        actions = []
        for span in spans:
            clean = _clean(span.text)
            if span.confidence < 0.88 or any(word in clean for word in _BATTLE_WORDS):
                continue
            if any(word in clean for word in ("放弃探索", "结束探索", "重新开始", "结局二", "结局三", "结局四")):
                continue
            choices = self.choices.get(clean,[])
            operation = _OPERATIONS.get(clean)
            # Choices in the right-side event option column are safe to ground
            # by an exact client-data title. Narrative mentions elsewhere are not.
            is_choice = bool(choices and scene == "event" and span.bbox[0] > width*0.48)
            if not operation and not is_choice:
                continue
            # Short common words in page titles are not sufficient button evidence.
            if operation and span.bbox[1] < height*0.2:
                continue
            metadata: dict[str,Any] = {"source":"ocr","grounded":True,"operation":operation or "event","observed_text":span.text}
            if scene == "movement_preview" and operation == "advance":
                # Merely entering the selected node is separate from StartAction;
                # a battle label anywhere on the screen still blocks all input.
                metadata["operation"] = "event_advance"
            if is_choice and len(choices) == 1:
                metadata["choice_id"] = choices[0]["id"]
            if operation == "leave":
                metadata["ends_node"] = True
            local=self._nearby_texts(span,spans,width,height)
            metadata["visible_texts"]=local
            if operation=="purchase":
                prices=[]
                for nearby in local:
                    price=re.search(r"(?:售价|价格|消耗|花费)\s*[:：]?\s*(\d{1,3})\s*(?:源石锭)?",nearby)
                    if price: prices.append(int(price[1]))
                if len(set(prices))==1: metadata["price"]=prices[0]
            enabled=not any(any(word in part for word in ("条件未满足","无法选择","不可选择","余额不足","希望不足")) for part in local)
            key = sha256((scene+"|"+clean+"|"+str(tuple(round(x) for x in span.bbox))).encode()).hexdigest()[:14]
            actions.append(ObservedAction("ocr:"+key,span.text,"event_choice" if is_choice else "ui",span.bbox,span.confidence,enabled=enabled,metadata=metadata))
        return actions

    @staticmethod
    def _nearby_texts(span,spans,width,height):
        x,y,w,h=span.bbox
        cx,cy=x+w/2,y+h/2
        return [other.text for other in spans if other.confidence>=.85 and abs((other.bbox[0]+other.bbox[2]/2)-cx)<min(width*.13,max(w*1.2,width*.06)) and abs((other.bbox[1]+other.bbox[3]/2)-cy)<height*.14]

    def _catalog_actions(self,spans,scene,width,height):
        if scene not in {"shop","reward","inventory","recruitment"}: return []
        actions=[]
        for span in spans:
            x,y,w,h=span.bbox
            text=_clean(span.text)
            if span.confidence<.9 or y<height*.2 or y>height*.85: continue
            metadata={"grounded":True,"source":"ocr_exact_catalog","preview_only":True,"operation":"event","visible_texts":self._nearby_texts(span,spans,width,height)}
            if scene=="recruitment" and text in self.operators:
                operator=self.operators[text]
                metadata.update(operator_name=operator["name"],operator_id=operator.get("id"),selection_stage="operator_preview")
                kind="operator_preview"
            elif scene in {"shop","reward","inventory"} and text in self.item_names:
                metadata.update(item_name=self.item_names[text],selection_stage="item_preview")
                kind="item_preview"
            else: continue
            key=sha256((kind+text+str(span.bbox)).encode()).hexdigest()[:14]
            actions.append(ObservedAction("catalog:"+key,span.text,kind,span.bbox,span.confidence,metadata=metadata))
        return actions

    @staticmethod
    def _unique_actions(actions):
        accepted=[]
        for action in sorted(actions,key=lambda a:a.confidence,reverse=True):
            x,y,w,h=action.bbox
            if any(abs(x+w/2-(a.bbox[0]+a.bbox[2]/2))<min(w,a.bbox[2])*.65 and abs(y+h/2-(a.bbox[1]+a.bbox[3]/2))<min(h,a.bbox[3])*.65 for a in accepted): continue
            accepted.append(action)
        return accepted

    def _numeric_crop(self,image,rect,*,allow_fraction=False,contrast=False):
        x,y,w,h = rect
        left,top = max(0,round(x)),max(0,round(y))
        right,bottom = min(image.shape[1],left+round(w)),min(image.shape[0],top+round(h))
        if right <= left or bottom <= top or not hasattr(self.ocr,"recognize_crop"):
            return None
        crop=image[top:bottom,left:right]
        values=[self.ocr.recognize_crop(crop)]
        if contrast:
            cv=_cv()
            gray=cv.cvtColor(crop,cv.COLOR_BGR2GRAY)
            binary=cv.cvtColor((gray>150).astype(np.uint8)*255,cv.COLOR_GRAY2BGR)
            values.append(self.ocr.recognize_crop(cv.copyMakeBorder(binary,3,3,5,5,cv.BORDER_CONSTANT)))
        for value,confidence in values:
            value=value.strip().translate(str.maketrans({"Ⅰ":"1","Ｏ":"0","O":"0","o":"0"}))
            if confidence>=0.87 and re.fullmatch(r"\d{1,3}(?:[/／]\d{1,3})?" if allow_fraction else r"\d{1,3}",value):
                return tuple(int(p) for p in re.split(r"[/／]",value))
        return None

    def _hud_resources(self,image,spans):
        """Locate HUD labels/icons, then read their neighboring *current* values.

        Relative label/icon offsets follow actual HUD elements, including scale.
        No frame edge, fixed 1280 ROI, old state, or forecast participates.
        """
        resources={}
        height,width=image.shape[:2]
        for span in spans:
            x,y,w,h=span.bbox
            label=_clean(span.text)
            if span.confidence<0.9:
                continue
            if label in {"目标生命值","目标生命","生命值"} and y<height*.2:
                rect=(x-2,y+h*.95,w*.65,h*1.7)
                value=self._numeric_crop(image,rect,allow_fraction=True)
                if value:
                    resources["hp"]=value[0]
                    if len(value)>1 and value[1]>=value[0]: resources["max_hp"]=value[1]
            elif label=="行动力" and x>width*.6 and y<height*.35:
                value=self._numeric_crop(image,(x-w*.4,y+h*1.5,w*1.3,h*3.5))
                if value: resources["action_points"]=value[0]
            elif label=="零件箱" and y>height*.7:
                value=self._numeric_crop(image,(x-w*.20,y+h,w*1.55,h*1.90),allow_fraction=True)
                if value: resources["parts"]=value[0]
            elif label in {"收藏品","藏品"} and y>height*.7:
                value=self._numeric_crop(image,(x+w*.22,y-h*1.35,w*.59,h*1.35),contrast=True)
                if value: resources["relics"]=value[0]
        for field in ("hope","gold"):
            hits=self.templates.match(image,field+"_icon.png",threshold=.88)
            if not hits: continue
            x,y,w,h=hits[0].bbox
            if y>height*.18: continue
            rect=(x+w*2.24,y+h*.25,w*.75,h*.85) if field=="hope" else (x+w*2.19,y+h*.4,w*.90,h*.64)
            value=self._numeric_crop(image,rect)
            if value: resources[field]=value[0]
        return resources

    def _map(self,image: np.ndarray,spans: tuple[OCRSpan,...]):
        height,width = image.shape[:2]
        candidates: list[tuple[str,tuple[float,float,float,float],float]] = []
        scales = []
        for spec in self.templates.node_specs:
            role = spec.get("role", "ordinary")
            if role not in {"empty","special","current_marker"}:
                continue
            if role == "current_marker":
                continue
            hits = self.templates.match(image,spec["file"],threshold=max(0.78,float(spec.get("threshold",0.8))),maximum=60)
            for hit in hits:
                x,y,w,h = hit.bbox
                if height*0.08 < y+h/2 < height*0.87:
                    candidates.append((spec.get("node_type","empty"),hit.bbox,hit.confidence))
                    scales.append(hit.scale)
        map_scale = float(np.median(scales)) if scales else min(width/1280,height/720)
        # Ordinary map nodes use the visible node title, as in MAA's manifest.
        for span in spans:
            node_type = self.templates.labels.get(_clean(span.text))
            x,y,w,h = span.bbox
            if node_type and span.confidence >= 0.84 and height*0.09 < y < height*0.82 and x < width*0.88:
                size = 42*map_scale
                center = (x+w/2,y+h/2-29*map_scale)
                candidates.append((node_type,(center[0]-size/2,center[1]-size/2,size,size),span.confidence))
        merged = []
        for candidate in sorted(candidates,key=lambda c:c[2],reverse=True):
            x,y,w,h = candidate[1]
            if any(math.dist((x+w/2,y+h/2),(other[1][0]+other[1][2]/2,other[1][1]+other[1][3]/2)) < 30*map_scale for other in merged):
                continue
            merged.append(candidate)
        if len(merged) < 2:
            return (),(),None,["map_nodes_insufficient"]
        xs = sorted(x+w/2 for _,(x,y,w,h),_ in merged)
        ys = sorted(y+h/2 for _,(x,y,w,h),_ in merged)
        def cluster(values):
            groups = []
            for value in values:
                if groups and value-np.mean(groups[-1]) < 35*map_scale:
                    groups[-1].append(value)
                else:
                    groups.append([value])
            return [float(np.mean(g)) for g in groups]
        columns,rows = cluster(xs),cluster(ys)
        nodes = []
        for kind,bbox,score in merged:
            x,y,w,h = bbox
            col = int(np.argmin([abs(x+w/2-value) for value in columns]))
            row = int(np.argmin([abs(y+h/2-value) for value in rows]))
            revealed=kind not in {"hide_invisible","hide_battle","unclassified"}
            mapped={"shop":"SCRAP_SHOP","informant":"STORY","sacrifice":"SACRIFICE","battle_mid_boss_shsgzd":"BATTLE_BOSS","battle_mid_boss_shwksc":"BATTLE_BOSS","battle_boss_cadejo":"BATTLE_BOSS"}.get(kind,kind.upper()) if revealed else kind
            nodes.append(ObservedNode(f"r{row}c{col}",mapped,row,col,bbox,score,revealed,False))
        # Duplicate/overlapping grid assignments make a route unsafe to execute.
        if len({n.node_id for n in nodes}) != len(nodes):
            return tuple(nodes),(),None,["ambiguous_map_grid"]
        marker_spec = next((s for s in self.templates.node_specs if s.get("role")=="current_marker"),None)
        current = None
        if marker_spec:
            hits = self.templates.match(image,marker_spec["file"],threshold=0.8,maximum=2)
            if len(hits) == 1:
                x,y,w,h = hits[0].bbox
                closest = sorted(nodes,key=lambda n:math.dist((x+w/2,y+h/2),(n.bbox[0]+n.bbox[2]/2,n.bbox[1]+n.bbox[3]/2)))
                n = closest[0]
                if math.dist((x+w/2,y+h/2),(n.bbox[0]+n.bbox[2]/2,n.bbox[1]+n.bbox[3]/2)) < 58*map_scale:
                    current=n.node_id
        candidate_edges,pairs = [],[]
        for index,a in enumerate(nodes):
            for b in nodes[index+1:]:
                if abs(a.row-b.row)+abs(a.col-b.col) != 1:
                    continue
                pa,pb = (a.bbox[0]+a.bbox[2]/2,a.bbox[1]+a.bbox[3]/2),(b.bbox[0]+b.bbox[2]/2,b.bbox[1]+b.bbox[3]/2)
                if math.dist(pa,pb) > 145*map_scale:
                    continue
                candidate_edges.append((a.node_id,b.node_id)); pairs.append((pa,pb))
        probabilities=self.corridor.score(image,pairs)
        threshold=self.corridor.config["decision"]["probability_threshold"]
        edges=tuple(pair for pair,p in zip(candidate_edges,probabilities) if p>=threshold)
        notes=[] if current else ["current_node_unrecognized"]
        if not edges:
            notes.append("no_confident_corridors")
        return tuple(nodes),edges,current,notes


ScreenshotObserver = VisionPipeline
