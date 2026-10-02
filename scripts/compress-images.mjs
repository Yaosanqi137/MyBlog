import { readdirSync, statSync, writeFileSync } from "node:fs";
import { join, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const argv = process.argv.slice(2).filter((a) => a !== "--dry-run");
const DRY_RUN = process.argv.includes("--dry-run");
const ROOT = join(fileURLToPath(new URL("..", import.meta.url)), "src");
const SKIP_DIRS = new Set([".cache", ".temp", "dist", "node_modules", ".vite"]);
const EXT_MAP = {
  ".png": (img) => img.png({ quality: 85, palette: true, effort: 7 }),
  ".jpg": (img) => img.jpeg({ quality: 80 }),
  ".jpeg": (img) => img.jpeg({ quality: 80 }),
  ".webp": (img) => img.webp({ quality: 80 }),
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function walk(dir, files = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (!SKIP_DIRS.has(entry.name)) walk(p, files);
    } else if (EXT_MAP[extname(entry.name).toLowerCase()]) {
      files.push(p);
    }
  }
  return files;
}

const files = argv.length
  ? argv.map((a) => resolve(a))
  : walk(ROOT);

async function compress(file, preset) {
  let lastErr;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      return await preset(sharp(file, { rotate: true })).toBuffer();
    } catch (e) {
      lastErr = e;
      await sleep(500);
    }
  }
  throw lastErr;
}

let done = 0;
let totalBefore = 0;
let totalAfter = 0;

for (const file of files) {
  const ext = extname(file).toLowerCase();
  const preset = EXT_MAP[ext];
  if (!preset) continue;
  const before = statSync(file).size;
  totalBefore += before;
  let after = before;
  try {
    const out = await compress(file, preset);
    if (out.length < before) {
      if (!DRY_RUN) writeFileSync(file, out);
      after = out.length;
      done += 1;
      const kb = (n) => `${(n / 1024).toFixed(1)}KB`;
      console.log(`✓ ${file.replace(ROOT, "src")}  ${kb(before)} -> ${kb(out.length)}`);
    }
  } catch (e) {
    console.warn(`! skipped ${file}: ${e.message}`);
  }
  totalAfter += after;
}

const mb = (n) => `${(n / 1024 / 1024).toFixed(2)}MB`;
console.log(
  `\n${DRY_RUN ? "[dry-run] " : ""}scanned ${files.length} images, compressed ${done}, ` +
    `${mb(totalBefore)} -> ${mb(totalAfter)}`,
);
