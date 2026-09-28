export type PreviewStatus = {
  has_preview: boolean;
  observation?: { frame_id?: string; captured_at: number };
};

export function previewIdentity(status: PreviewStatus): string | null {
  const observation = status.observation;
  return status.has_preview && observation?.frame_id && Number.isFinite(observation.captured_at)
    ? `${observation.frame_id}:${observation.captured_at}` : null;
}

// The v2 bridge returns an untagged JPEG. Its observation and image are committed
// together under one lock, so bracket the image read with matching unique frame
// identities. Discard a racing frame instead of assigning it another timestamp.
export async function readConsistentPreview<T extends PreviewStatus>(
  before: T,
  readImage: () => Promise<Blob>,
  readStatus: () => Promise<T>,
  current: () => boolean,
): Promise<{ blob: Blob; status: T; identity: string } | null> {
  const identity = previewIdentity(before);
  if (!identity || !current()) return null;
  const blob = await readImage();
  if (!current()) return null;
  const after = await readStatus();
  if (!current() || identity !== previewIdentity(after)) return null;
  return { blob, status: after, identity };
}
