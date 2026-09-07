import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
with (ROOT / "source_data" / "roguelike_topic_table_full.json").open(encoding="utf-8") as f:
    data = json.load(f)

d = data["details"]["rogue_6"]
s = data["modules"]["rogue_6"]["scrap"]


def props(obj):
    return list(obj) if isinstance(obj, dict) else []


summary = {
    "counts": {
        "choices": len(d["choices"]),
        "choiceScenes": len(d["choiceScenes"]),
        "nodeTypeData": len(d["nodeTypeData"]),
        "subTypeData": len(d["subTypeData"]),
        "stages": len(d["stages"]),
        "relics": len(d["relics"]),
        "items": len(d["items"]),
    },
    "node_types": d["nodeTypeData"],
    "subtypes": d["subTypeData"],
    "scrap_keys": props(s),
    "scrap_types": s["scrapTypeData"],
    "goods_scrap_sample": dict(list(s["goodsScrapData"].items())[:5]),
    "move_scrap_sample": dict(list(s["moveScrapData"].items())[:5]),
    "relic_sample": dict(list(d["relics"].items())[:5]),
}

(ROOT / "source_data" / "rogue6_inspection.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
