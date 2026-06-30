/*
 * In-browser, on-device object detection for the phone demo (BONUS — not a project
 * requirement). The model.onnx exported by src/edge/export_onnx.py is a full torchvision
 * Faster R-CNN: it already does normalize + resize + NMS internally, so all we do here is:
 *   capture a frame -> RGB float[0,1] CHW tensor -> session.run -> draw boxes.
 *
 * UX notes:
 *  - The 72 MB model is PRELOADED on page open with a real download progress bar (streamed
 *    fetch), then the session is created from the bytes — so the wait overlaps the user reading
 *    the "What is this?" blurb instead of stalling after a button press.
 *  - We warm up the session with one dummy inference so the first real camera frame isn't slow.
 *  - Execution provider: WASM (CPU), single-threaded (GitHub Pages sends no COOP/COEP headers
 *    that multi-threaded WASM needs). FRCNN uses RoiAlign / NMS / TopK, reliably covered by
 *    ort-web's WASM kernels. Expect a few FPS — that's the on-device flex.
 */

const MODEL_URL = "model.onnx";
let procMax = 512; // longest side fed to the model — live-tunable via the input-size slider
const CLASS = { 1: { name: "person", color: "#dc2828" },
                2: { name: "vehicle", color: "#2878dc" } };

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
ort.env.wasm.numThreads = 1;

const video = document.getElementById("view");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("start");
const flipBtn = document.getElementById("flip");
const snapBtn = document.getElementById("snap");
const thr = document.getElementById("thr");
const thrVal = document.getElementById("thrVal");
const res = document.getElementById("res");
const resVal = document.getElementById("resVal");
const fpsEl = document.getElementById("fps");
const barfill = document.getElementById("barfill");
const loadtxt = document.getElementById("loadtxt");
const loadBox = document.getElementById("load");

const proc = document.createElement("canvas");
const pctx = proc.getContext("2d", { willReadFrequently: true });

let session = null;
let running = false;
let lastT = performance.now();
let facing = "environment"; // rear camera by default
let stream = null;

thr.addEventListener("input", () => (thrVal.textContent = (+thr.value).toFixed(2)));
res.addEventListener("input", () => { procMax = +res.value; resVal.textContent = res.value; });

function setStatus(msg) { statusEl.textContent = msg; }

/* ---- Preload + progress: stream the model bytes, then build the session + warm up ---- */
async function preload() {
  try {
    setStatus("Downloading model…");
    const resp = await fetch(MODEL_URL);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const total = +resp.headers.get("content-length") || 0;
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      const pct = total ? Math.round((received / total) * 100) : null;
      barfill.style.width = (pct ?? 60) + "%";
      const mb = (received / 1048576).toFixed(1);
      loadtxt.textContent = pct !== null
        ? `Downloading model… ${pct}% (${mb} MB)`
        : `Downloading model… ${mb} MB`;
    }
    const bytes = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { bytes.set(c, off); off += c.length; }

    loadtxt.textContent = "Initialising on-device runtime…";
    barfill.style.width = "100%";
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"], graphOptimizationLevel: "all",
    });

    loadtxt.textContent = "Warming up…";
    await warmup();

    loadBox.hidden = true;
    startBtn.disabled = false;
    setStatus("Ready — tap “Start camera” and point at a person or a vehicle.");
  } catch (err) {
    loadtxt.textContent = "";
    setStatus("Couldn't load the model: " + err.message);
  }
}

async function warmup() {
  // one dummy inference so ort-web allocates/JITs its WASM kernels before the first real frame
  const w = 128, h = 96;
  const dummy = new ort.Tensor("float32", new Float32Array(3 * w * h), [3, h, w]);
  try { await session.run({ input: dummy }); } catch { /* non-fatal */ }
}

async function startCamera() {
  startBtn.disabled = true;
  try {
    await openStream();
    await loadModelReady();
    running = true;
    flipBtn.hidden = false; snapBtn.hidden = false; fpsEl.hidden = false;
    setStatus("Running — point at a person or a vehicle.");
    requestAnimationFrame(loop);
  } catch (err) {
    setStatus("Error: " + err.message + " (camera needs HTTPS — use the GitHub Pages URL).");
    startBtn.disabled = false;
  }
}

async function openStream() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: facing } }, audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

async function loadModelReady() {
  // session was preloaded; if preload failed, this gives a clear message
  if (!session) throw new Error("model not loaded yet");
}

/* Build a [3,H,W] float32 tensor (RGB, /255) from the offscreen canvas. */
function frameToTensor() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.min(1, procMax / Math.max(vw, vh));
  const w = Math.round(vw * scale), h = Math.round(vh * scale);
  proc.width = w; proc.height = h;
  pctx.drawImage(video, 0, 0, w, h);
  const { data } = pctx.getImageData(0, 0, w, h); // RGBA
  const out = new Float32Array(3 * w * h);
  const plane = w * h;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    out[p] = data[i] / 255;
    out[p + plane] = data[i + 1] / 255;
    out[p + 2 * plane] = data[i + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", out, [3, h, w]), w, h };
}

function drawBoxes(r, procW, procH) {
  const dispW = video.clientWidth, dispH = video.clientHeight;
  overlay.width = dispW; overlay.height = dispH;
  const sx = dispW / procW, sy = dispH / procH;

  const boxes = r.boxes.data, scores = r.scores.data, labels = r.labels.data;
  const minScore = +thr.value;
  octx.clearRect(0, 0, dispW, dispH);
  octx.lineWidth = 3;
  octx.font = "16px system-ui, sans-serif";
  octx.textBaseline = "bottom";

  let n = 0;
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] < minScore) continue;
    const cls = CLASS[labels[i]] || { name: String(labels[i]), color: "#22c55e" };
    const x1 = boxes[i * 4] * sx, y1 = boxes[i * 4 + 1] * sy;
    const x2 = boxes[i * 4 + 2] * sx, y2 = boxes[i * 4 + 3] * sy;
    octx.strokeStyle = cls.color;
    octx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const tag = `${cls.name} ${scores[i].toFixed(2)}`;
    octx.fillStyle = cls.color;
    const tw = octx.measureText(tag).width + 8;
    octx.fillRect(x1, Math.max(0, y1 - 20), tw, 20);
    octx.fillStyle = "#fff";
    octx.fillText(tag, x1 + 4, Math.max(18, y1 - 2));
    n++;
  }
  return n;
}

async function loop() {
  if (!running || !video.videoWidth) { requestAnimationFrame(loop); return; }
  const { tensor, w, h } = frameToTensor();
  try {
    const r = await session.run({ input: tensor });
    const n = drawBoxes(r, w, h);
    const now = performance.now();
    const fps = 1000 / (now - lastT); lastT = now;
    fpsEl.textContent = `${fps.toFixed(1)} FPS`;
    setStatus(`${n} detections · ${w}×${h} input · on-device (WASM)`);
  } catch (err) {
    running = false;
    setStatus("Inference error: " + err.message +
      " — an op may be unsupported in ort-web. See docs/08 (try the RetinaNet export).");
    return;
  }
  requestAnimationFrame(loop);
}

flipBtn.addEventListener("click", async () => {
  facing = facing === "environment" ? "user" : "environment";
  try { await openStream(); } catch (e) { setStatus("Flip failed: " + e.message); }
});

snapBtn.addEventListener("click", () => {
  const c = document.createElement("canvas");
  c.width = video.clientWidth; c.height = video.clientHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(video, 0, 0, c.width, c.height);
  ctx.drawImage(overlay, 0, 0);
  const a = document.createElement("a");
  a.href = c.toDataURL("image/png");
  a.download = "edge-detection.png";
  a.click();
});

startBtn.addEventListener("click", startCamera);

// kick off the download immediately on page load
preload();
