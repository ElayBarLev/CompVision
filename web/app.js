/*
 * In-browser, on-device object detection for the Pixel 6a demo (BONUS — not a project
 * requirement). The model.onnx exported by src/edge/export_onnx.py is a full torchvision
 * Faster R-CNN: it already does normalize + resize + NMS internally, so all we do here is:
 *   capture a frame -> RGB float[0,1] CHW tensor -> session.run -> draw boxes.
 *
 * Execution provider: WASM (CPU). FRCNN uses RoiAlign / NonMaxSuppression / TopK, which are
 * reliably covered by ort-web's WASM kernels but only partially on WebGPU — so we stick to
 * WASM for a demo that "just works". Expect a few FPS; that's fine for a flex.
 */

const MODEL_URL = "model.onnx";
const PROC_MAX = 512; // longest side fed to the model (matches export --min-size ballpark)
const CLASS = { 1: { name: "person", color: "#dc2828" },
                2: { name: "vehicle", color: "#2878dc" } };

// Point ort-web at the CDN for its .wasm binaries; single-threaded (GitHub Pages doesn't
// send the COOP/COEP headers that multi-threaded WASM needs). SIMD still kicks in.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
ort.env.wasm.numThreads = 1;

const video = document.getElementById("view");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("start");
const thr = document.getElementById("thr");
const thrVal = document.getElementById("thrVal");

// offscreen canvas we draw the video into to read pixels
const proc = document.createElement("canvas");
const pctx = proc.getContext("2d", { willReadFrequently: true });

let session = null;
let running = false;
let lastT = performance.now();

thr.addEventListener("input", () => (thrVal.textContent = (+thr.value).toFixed(2)));

function setStatus(msg) { statusEl.textContent = msg; }

async function loadModel() {
  if (session) return session;
  setStatus("Loading model… (first load downloads model.onnx)");
  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return session;
}

async function startCamera() {
  startBtn.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } }, audio: false,
    });
    video.srcObject = stream;
    await video.play();
    await loadModel();
    running = true;
    setStatus("Running — point at a person or a vehicle.");
    requestAnimationFrame(loop);
  } catch (err) {
    setStatus("Error: " + err.message + " (camera needs HTTPS — use the GitHub Pages URL).");
    startBtn.disabled = false;
  }
}

// Build a [3,H,W] float32 tensor (RGB, /255) from the offscreen canvas.
function frameToTensor() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.min(1, PROC_MAX / Math.max(vw, vh));
  const w = Math.round(vw * scale), h = Math.round(vh * scale);
  proc.width = w; proc.height = h;
  pctx.drawImage(video, 0, 0, w, h);
  const { data } = pctx.getImageData(0, 0, w, h); // RGBA
  const out = new Float32Array(3 * w * h);
  const plane = w * h;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    out[p] = data[i] / 255;             // R
    out[p + plane] = data[i + 1] / 255; // G
    out[p + 2 * plane] = data[i + 2] / 255; // B
  }
  return { tensor: new ort.Tensor("float32", out, [3, h, w]), w, h };
}

function drawBoxes(res, procW, procH) {
  // size overlay to the displayed video and scale boxes from proc-space to display-space
  const dispW = video.clientWidth, dispH = video.clientHeight;
  overlay.width = dispW; overlay.height = dispH;
  const sx = dispW / procW, sy = dispH / procH;

  const boxes = res.boxes.data, scores = res.scores.data, labels = res.labels.data;
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
    const res = await session.run({ input: tensor });
    const n = drawBoxes(res, w, h);
    const now = performance.now();
    const fps = 1000 / (now - lastT); lastT = now;
    setStatus(`${n} detections · ${fps.toFixed(1)} FPS · input ${w}×${h} (on-device, WASM)`);
  } catch (err) {
    running = false;
    setStatus("Inference error: " + err.message +
      " — an op may be unsupported in ort-web. See docs/08 (try the RetinaNet export).");
    return;
  }
  requestAnimationFrame(loop);
}

startBtn.addEventListener("click", startCamera);
