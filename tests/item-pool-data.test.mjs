import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = (await readFile(
  new URL("../app/item-pool-data.ts", import.meta.url),
  "utf8",
)).replace(/\r\n/g, "\n");

function exportedArray(name) {
  const declaration = `export const ${name}`;
  const start = source.indexOf(declaration);
  assert.notEqual(start, -1, `${name} export should exist`);
  const valueStart = source.indexOf("= ", start) + 2;
  const valueEnd = source.indexOf(";\n", valueStart);
  return JSON.parse(source.slice(valueStart, valueEnd));
}

const items = exportedArray("BLACKFLOW_POOL_ITEMS");
const pools = exportedArray("BLACKFLOW_POOLS");
const itemById = new Map(items.map((item) => [item.id, item]));

test("2026-08-31 pool snapshot is complete and internally consistent", () => {
  assert.equal(items.length, 315);
  assert.equal(pools.length, 44);
  assert.equal(itemById.size, items.length, "item ids must be unique");
  assert.equal(new Set(pools.map((pool) => pool.id)).size, pools.length);

  for (const pool of pools) {
    assert.equal(pool.itemCount, pool.items.length, `${pool.name} item count`);
    for (const occurrence of pool.items) {
      assert.ok(itemById.has(occurrence.id), `${occurrence.id} must be searchable`);
    }
    for (const sourceEntry of pool.sources) {
      assert.equal(sourceEntry.path.at(-1), pool.name);
    }
  }
});

test("迷藏 is represented as an upstream source of 板藤", () => {
  const boardVine = items.find((item) => item.name === "板藤");
  assert.ok(boardVine);

  const pool = pools.find((entry) => entry.id === "pool_scrap_7");
  assert.ok(pool);
  assert.ok(pool.items.some((entry) => entry.id === boardVine.id));
  assert.ok(
    pool.sources.some(
      (entry) => entry.path.join(" → ") === "迷藏 → 藏果地 → 零件池7",
    ),
  );
});

test("today's newly published random processed-part pool is included", () => {
  const pool = pools.find(
    (entry) => entry.id === "rogue_6:不期而遇：被歌颂的影子（随机加工品）",
  );
  assert.ok(pool);
  assert.equal(pool.poolType, "零件池");
  assert.equal(pool.items.length, 8);
  assert.ok(pool.items.every((entry) => itemById.get(entry.id)?.type === "零件"));
});
