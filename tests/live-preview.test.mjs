import assert from "node:assert/strict";
import test from "node:test";
import { previewIdentity, readConsistentPreview } from "../lib/live-preview.ts";

const frame = (id = "frame-a", time = 123) => ({ has_preview: true, observation: { frame_id: id, captured_at: time } });
const blob = new Blob(["saved-jpeg"]);

test("a preview is published only between matching observed frame identities", async () => {
  const calls = [];
  const before = frame();
  const after = { ...frame(), state: "paused" };
  const result = await readConsistentPreview(before, async () => { calls.push("image"); return blob; },
    async () => { calls.push("status"); return after; }, () => true);
  assert.deepEqual(calls, ["image", "status"]);
  assert.equal(result.blob, blob);
  assert.equal(result.status, after);
});

test("a new observation during JPEG download discards the image", async () => {
  for (const next of [frame("frame-b"), frame("frame-a", 124), { has_preview: false }]) {
    assert.equal(await readConsistentPreview(frame(), async () => blob, async () => next, () => true), null);
  }
});

test("missing or invalid preview identity never fetches an untagged image", async () => {
  for (const status of [{ has_preview: false }, { has_preview: true },
    { has_preview: true, observation: { captured_at: 123 } }, frame("frame-a", NaN)]) {
    assert.equal(previewIdentity(status), null);
    assert.equal(await readConsistentPreview(status, () => assert.fail("no image read"),
      () => assert.fail("no status read"), () => true), null);
  }
});

test("token replacement or command cancellation during download cannot repopulate preview", async () => {
  let current = true;
  assert.equal(await readConsistentPreview(frame(), async () => { current = false; return blob; },
    () => assert.fail("no old-session status request"), () => current), null);
  current = true;
  assert.equal(await readConsistentPreview(frame(), async () => blob,
    async () => { current = false; return frame(); }, () => current), null);
});

test("transport failures do not produce a successful preview", async () => {
  await assert.rejects(readConsistentPreview(frame(), async () => { throw new Error("frame failed"); },
    () => assert.fail("no status read"), () => true), /frame failed/);
  await assert.rejects(readConsistentPreview(frame(), async () => blob,
    async () => { throw new Error("status failed"); }, () => true), /status failed/);
});
