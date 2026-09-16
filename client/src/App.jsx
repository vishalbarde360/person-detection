import { useEffect, useRef, useState } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";
import * as faceapi from "@vladmandic/face-api";

import {
  Activity,
  Camera,
  CameraOff,
  Clock3,
  Download,
  Eye,
  Images,
  ScanFace,
  ScanLine,
  ShieldCheck,
  Trash2,
  UserPlus,
  Volume2,
  VolumeX,
} from "lucide-react";

import { describeActivity } from "./activityRules";

const API_BASE = (
  import.meta.env.VITE_API_URL || ""
).replace(/\/+$/, "");

const ACTIVITY_API = `${API_BASE}/api/activities`;
const PEOPLE_API = `${API_BASE}/api/people`;

/*
 * Phone च्या CPU/GPU वर desktop इतकं जड model + zero-delay
 * loop चालवला की app freeze/lag होतो. म्हणून mobile device
 * वर आपोआप हलकी settings वापरतो, desktop वर जास्त अचूक/वेगवान.
 */
const IS_MOBILE =
  typeof navigator !== "undefined" &&
  (/Android|iPhone|iPad|iPod/i.test(
    navigator.userAgent,
  ) ||
    (typeof window !== "undefined" &&
      window.matchMedia?.("(pointer: coarse)").matches));

const FACE_OPTIONS = new faceapi.TinyFaceDetectorOptions({
  inputSize: IS_MOBILE ? 160 : 416,
  scoreThreshold: 0.5,
});

/*
 * Face recognition (landmark + 128-value descriptor काढणं) हा
 * संपूर्ण pipeline मधला सगळ्यात जड भाग आहे. Phone वर तो प्रत्येक
 * cycle ला न चालवता दर काही cycles नंतर चालवतो — यामुळे object
 * detection आणि एकूण responsiveness खूप सुधारते, आणि चेहरा
 * ओळखणं तरीही सेकंदाभरातच update होत राहतं.
 */
const FACE_RECOGNITION_INTERVAL = IS_MOBILE ? 3 : 1;

const OBJECT_MODEL_BASE = IS_MOBILE
  ? "lite_mobilenet_v2"
  : "mobilenet_v2";

/*
 * मागचं detection संपल्यावर पुढचं सुरू करण्याआधीचा gap.
 * Desktop वर 0 (शक्य तितकं वेगवान), phone वर थोडा gap
 * ठेवून CPU/GPU ला विश्रांती — नाहीतर overheating/hang होतं.
 */
const ANALYZE_LOOP_GAP = IS_MOBILE ? 200 : 0;

/*
 * Camera resolution — phone वर कमी ठेवली की decode +
 * detection चा भार कमी होतो, त्यामुळे smooth चालतं.
 */
const CAMERA_WIDTH_IDEAL = IS_MOBILE ? 640 : 1280;
const CAMERA_HEIGHT_IDEAL = IS_MOBILE ? 480 : 720;

/*
 * Auto-capture साठी cooldown — नाहीतर प्रत्येक cycle ला
 * screenshot घेत राहील आणि memory भरून जाईल.
 */
const UNKNOWN_CAPTURE_COOLDOWN_MS = 10000;
const MOTION_CAPTURE_COOLDOWN_MS = 10000;
const MAX_CAPTURES = 24;
const MOTION_CAPTURE_THRESHOLD = 22;

/*
 * Motion sampling साठी downscaled frame size आणि
 * त्यावर टाकलेली grid (movement zones शोधण्यासाठी).
 */
const SAMPLE_WIDTH = 160;
const SAMPLE_HEIGHT = 90;
const GRID_COLS = 8;
const GRID_ROWS = 5;
const CELL_WIDTH = SAMPLE_WIDTH / GRID_COLS;
const CELL_HEIGHT = SAMPLE_HEIGHT / GRID_ROWS;
const ZONE_MOTION_THRESHOLD = 16;

function faceDistance(first, second) {
  return Math.sqrt(
    first.reduce(
      (total, value, index) =>
        total + (value - second[index]) ** 2,
      0,
    ),
  );
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const objectModelRef = useRef(null);

  const streamRef = useRef(null);
  const previousFrameRef = useRef(null);
  const timerRef = useRef(null);
  const motionRafRef = useRef(null);
  const lastPredictionsRef = useRef([]);
  const motionScoreRef = useRef(0);
  const brightnessRef = useRef(100);
  const cellStabilityRef = useRef(
    new Uint8Array(GRID_COLS * GRID_ROWS),
  );
  const lastMotionUiUpdateRef = useRef(0);
  const analyzeCycleRef = useRef(0);
  const lastFaceResultRef = useRef({
    match: null,
    hasFace: false,
  });
  const lastUnknownCaptureRef = useRef(0);
  const lastMotionCaptureRef = useRef(0);

  const busyRef = useRef(false);
  const lastEventRef = useRef("");
  const recognizedRef = useRef("");
  const knownPeopleRef = useRef([]);
  const voiceRef = useRef(true);

  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);

  const [message, setMessage] = useState("Camera is off");
  const [kind, setKind] = useState("idle");

  const [objects, setObjects] = useState([]);
  const [motion, setMotion] = useState(0);
  const [activeZones, setActiveZones] = useState(0);
  const [captures, setCaptures] = useState([]);

  const [voice, setVoice] = useState(true);
  const [events, setEvents] = useState([]);

  const [people, setPeople] = useState([]);
  const [recognized, setRecognized] = useState(null);

  const [tab, setTab] = useState("live");
  const [name, setName] = useState("");
  const [details, setDetails] = useState("");
  const [consent, setConsent] = useState(false);

  const [enrolling, setEnrolling] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    loadPeople();

    fetch(`${ACTIVITY_API}/recent?limit=8`)
      .then((response) => (response.ok ? response.json() : []))
      .then(setEvents)
      .catch(() => {});

    return () => stopCamera();
  }, []);

  useEffect(() => {
    voiceRef.current = voice;
  }, [voice]);

  /*
   * Live Recognition आणि Register Person tab बदलल्यावर
   * नवीन video elementला existing camera stream जोडतो.
   */
  useEffect(() => {
    const video = videoRef.current;

    if (running && streamRef.current && video) {
      video.srcObject = streamRef.current;
      video.play().catch(() => {});
    }
  }, [tab, running]);

  async function loadPeople() {
    try {
      const response = await fetch(PEOPLE_API);
      const data = response.ok ? await response.json() : [];

      setPeople(data);
      knownPeopleRef.current = data;
    } catch {
      setPeople([]);
      knownPeopleRef.current = [];
    }
  }

  async function startCamera() {
    setError("");
    setLoading(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: CAMERA_WIDTH_IDEAL },
          height: { ideal: CAMERA_HEIGHT_IDEAL },
        },
        audio: false,
      });

      streamRef.current = stream;

      const video = videoRef.current;

      if (!video) {
        throw new Error("Camera preview element is not available.");
      }

      video.srcObject = stream;
      await video.play();

      setRunning(true);

      /*
       * Motion loop लगेच सुरू — AI models load होण्याची वाट न
       * बघता, जेणेकरून movement detection ला अजिबात delay जाणवू नये.
       */
      motionRafRef.current = requestAnimationFrame(motionTick);

      setMessage("Loading AI models…");

      await Promise.all([
        cocoSsd
          .load({
            base: OBJECT_MODEL_BASE,
          })
          .then((model) => {
            objectModelRef.current = model;
          }),

        faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
        faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
        faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
      ]);

      setMessage("Looking for activity and known faces…");

      analyzeLoop();
    } catch (exception) {
      const message =
        exception.name === "NotAllowedError"
          ? "Camera permission was denied. Allow it in browser settings and try again."
          : `Could not start camera or AI models: ${exception.message}`;

      setError(message);
      stopCamera();
    } finally {
      setLoading(false);
    }
  }

  function stopCamera() {
    if (timerRef.current) {
      window.clearTimeout(timerRef.current);
    }

    timerRef.current = null;

    if (motionRafRef.current) {
      cancelAnimationFrame(motionRafRef.current);
    }

    motionRafRef.current = null;

    streamRef.current
      ?.getTracks()
      .forEach((track) => track.stop());

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    streamRef.current = null;
    previousFrameRef.current = null;
    busyRef.current = false;
    recognizedRef.current = "";
    lastPredictionsRef.current = [];
    motionScoreRef.current = 0;
    brightnessRef.current = 100;
    cellStabilityRef.current.fill(0);
    analyzeCycleRef.current = 0;
    lastFaceResultRef.current = {
      match: null,
      hasFace: false,
    };
    lastUnknownCaptureRef.current = 0;
    lastMotionCaptureRef.current = 0;

    const overlay = overlayCanvasRef.current;

    if (overlay) {
      overlay
        .getContext("2d")
        ?.clearRect(0, 0, overlay.width, overlay.height);
    }

    setRunning(false);
    setObjects([]);
    setMotion(0);
    setActiveZones(0);
    setRecognized(null);
    setMessage("Camera is off");
    setKind("idle");
  }

  function frameMetrics() {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video || !video.videoWidth) {
      return {
        motionScore: 0,
        brightness: 100,
        cells: [],
      };
    }

    const context = canvas.getContext("2d", {
      willReadFrequently: true,
    });

    canvas.width = SAMPLE_WIDTH;
    canvas.height = SAMPLE_HEIGHT;

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height,
    );

    const pixels = context.getImageData(
      0,
      0,
      canvas.width,
      canvas.height,
    ).data;

    let brightness = 0;
    let difference = 0;
    let count = 0;

    const currentFrame = new Uint8Array(
      pixels.length / 4,
    );

    /*
     * प्रत्येक grid cell साठी वेगळा motion accumulator,
     * जेणेकरून frame मध्ये नक्की कुठे हालचाल झाली ते कळेल.
     */
    const cellTotals = new Float32Array(
      GRID_COLS * GRID_ROWS,
    );
    const cellCounts = new Float32Array(
      GRID_COLS * GRID_ROWS,
    );

    for (
      let pixelIndex = 0, frameIndex = 0;
      pixelIndex < pixels.length;
      pixelIndex += 4, frameIndex += 1
    ) {
      const value =
        (pixels[pixelIndex] +
          pixels[pixelIndex + 1] +
          pixels[pixelIndex + 2]) /
        3;

      currentFrame[frameIndex] = value;
      brightness += value;

      const x = frameIndex % canvas.width;
      const y = Math.floor(frameIndex / canvas.width);
      const cellX = Math.min(
        GRID_COLS - 1,
        Math.floor(x / CELL_WIDTH),
      );
      const cellY = Math.min(
        GRID_ROWS - 1,
        Math.floor(y / CELL_HEIGHT),
      );
      const cellIndex = cellY * GRID_COLS + cellX;

      if (previousFrameRef.current) {
        const pixelDifference = Math.abs(
          value - previousFrameRef.current[frameIndex],
        );

        difference += pixelDifference;
        cellTotals[cellIndex] += pixelDifference;
      }

      cellCounts[cellIndex] += 1;
      count += 1;
    }

    previousFrameRef.current = currentFrame;

    /*
     * Hysteresis: एका frame मध्ये threshold ओलांडला की लगेच
     * "active" न ठरवता, सलग 2 frames मध्ये ओलांडला तरच active
     * ठरवतो — यामुळे camera noise/light flicker मुळे होणारे
     * false movement zones कमी होतात (जास्त अचूक movement detection).
     */
    const stability = cellStabilityRef.current;
    const cells = [];

    for (let row = 0; row < GRID_ROWS; row += 1) {
      for (let col = 0; col < GRID_COLS; col += 1) {
        const cellIndex = row * GRID_COLS + col;
        const cellScore = cellCounts[cellIndex]
          ? cellTotals[cellIndex] / cellCounts[cellIndex]
          : 0;

        if (cellScore >= ZONE_MOTION_THRESHOLD) {
          stability[cellIndex] = Math.min(
            3,
            stability[cellIndex] + 1,
          );
        } else {
          stability[cellIndex] = 0;
        }

        if (stability[cellIndex] >= 2) {
          cells.push({ col, row, score: cellScore });
        }
      }
    }

    return {
      motionScore: count ? difference / count : 0,
      brightness: count ? brightness / count : 100,
      cells,
    };
  }

  /*
   * detected objects (bounding boxes) आणि motion zones
   * हे थेट video वर overlay canvas वर काढतो, जेणेकरून
   * object आणि movement detection प्रत्यक्ष "दिसेल".
   */
  function drawOverlay(predictions, cells) {
    const overlay = overlayCanvasRef.current;
    const video = videoRef.current;

    if (!overlay || !video || !video.videoWidth) {
      return;
    }

    const rect = video.getBoundingClientRect();
    const displayWidth = Math.round(rect.width) || video.videoWidth;
    const displayHeight = Math.round(rect.height) || video.videoHeight;

    if (
      overlay.width !== displayWidth ||
      overlay.height !== displayHeight
    ) {
      overlay.width = displayWidth;
      overlay.height = displayHeight;
    }

    const context = overlay.getContext("2d");
    context.clearRect(0, 0, overlay.width, overlay.height);

    /*
     * video वर object-fit: cover लागू आहे, त्यामुळे native
     * video resolution ते displayed box असे mapping
     * scale + centered crop offset वापरून काढतो.
     */
    const scale = Math.max(
      overlay.width / video.videoWidth,
      overlay.height / video.videoHeight,
    );
    const drawnWidth = video.videoWidth * scale;
    const drawnHeight = video.videoHeight * scale;
    const offsetX = (overlay.width - drawnWidth) / 2;
    const offsetY = (overlay.height - drawnHeight) / 2;

    // 1) Movement zones: सक्रिय grid cells हलक्या rectangles ने highlight.
    cells.forEach(({ col, row, score }) => {
      const cellVideoX = (col * CELL_WIDTH * video.videoWidth) / SAMPLE_WIDTH;
      const cellVideoY = (row * CELL_HEIGHT * video.videoHeight) / SAMPLE_HEIGHT;
      const cellVideoW = (CELL_WIDTH * video.videoWidth) / SAMPLE_WIDTH;
      const cellVideoH = (CELL_HEIGHT * video.videoHeight) / SAMPLE_HEIGHT;

      const rawX = offsetX + cellVideoX * scale;
      const boxY = offsetY + cellVideoY * scale;
      const boxW = cellVideoW * scale;
      const boxH = cellVideoH * scale;

      /*
       * video CSS मध्ये mirror (scaleX(-1)) आहे, त्यामुळे
       * overlay वरच्या प्रत्येक box चा x हा horizontally
       * स्वतः mirror करावा लागतो — canvas स्वतः mirror
       * केलं तर त्यावरचा text उलटा दिसतो, म्हणून हे टाळलं.
       */
      const boxX = overlay.width - rawX - boxW;

      const intensity = Math.min(1, score / 60);

      context.fillStyle = `rgba(255, 176, 32, ${0.12 + intensity * 0.28})`;
      context.fillRect(boxX, boxY, boxW, boxH);
      context.strokeStyle = "rgba(255, 176, 32, 0.55)";
      context.lineWidth = 1;
      context.strokeRect(boxX, boxY, boxW, boxH);
    });

    // 2) Object detection boxes.
    predictions.forEach((prediction) => {
      if (prediction.score <= 0.5) {
        return;
      }

      const [x, y, width, height] = prediction.bbox;

      const rawX = offsetX + x * scale;
      const boxY = offsetY + y * scale;
      const boxW = width * scale;
      const boxH = height * scale;
      const boxX = overlay.width - rawX - boxW;

      const color =
        prediction.class === "person" ? "#3ddc84" : "#4fa8ff";

      context.strokeStyle = color;
      context.lineWidth = 2;
      context.strokeRect(boxX, boxY, boxW, boxH);

      const label = `${prediction.class} ${Math.round(
        prediction.score * 100,
      )}%`;

      context.font = "600 13px Arial";
      const textWidth = context.measureText(label).width;

      context.fillStyle = color;
      context.fillRect(boxX, Math.max(0, boxY - 18), textWidth + 10, 18);

      context.fillStyle = "#0d1116";
      context.fillText(label, boxX + 5, Math.max(12, boxY - 5));
    });
  }

  /*
   * सध्याचा video frame JPEG screenshot म्हणून capture करतो.
   * हे फक्त browser च्या memory मध्ये राहतं — कुठल्याही
   * server ला पाठवलं जात नाही (privacy-friendly).
   */
  function captureSnapshot(label) {
    const video = videoRef.current;

    if (!video || !video.videoWidth) {
      return;
    }

    const snap = document.createElement("canvas");
    snap.width = video.videoWidth;
    snap.height = video.videoHeight;

    const context = snap.getContext("2d");

    /*
     * screenshot user ला स्क्रीनवर जसं दिसतं तसंच
     * (mirrored) दिसावं म्हणून येथेही horizontal flip.
     */
    context.translate(snap.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, snap.width, snap.height);

    const dataUrl = snap.toDataURL("image/jpeg", 0.7);
    const timestamp = new Date();

    setCaptures((previous) =>
      [
        {
          id: `${timestamp.getTime()}-${Math.random()
            .toString(36)
            .slice(2, 8)}`,
          dataUrl,
          label,
          timestamp: timestamp.toISOString(),
        },
        ...previous,
      ].slice(0, MAX_CAPTURES),
    );
  }

  async function recognizeFace() {
    const video = videoRef.current;

    /*
     * fromPixels(null) error टाळण्यासाठी आवश्यक check.
     */
    if (
      !video ||
      !video.srcObject ||
      video.readyState < 2
    ) {
      return null;
    }

    const detection = await faceapi
      .detectSingleFace(video, FACE_OPTIONS)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      return { match: null, hasFace: false };
    }

    if (!knownPeopleRef.current.length) {
      return { match: null, hasFace: true };
    }

    let bestMatch = null;

    for (const person of knownPeopleRef.current) {
      if (
        !Array.isArray(person.faceDescriptor) ||
        person.faceDescriptor.length !== 128
      ) {
        continue;
      }

      const distance = faceDistance(
        Array.from(detection.descriptor),
        person.faceDescriptor,
      );

      if (
        !bestMatch ||
        distance < bestMatch.distance
      ) {
        bestMatch = {
          ...person,
          distance,
        };
      }
    }

    /*
     * कमी distance म्हणजे चांगला match.
     * 0.50 conservative threshold आहे.
     */
    const match =
      bestMatch && bestMatch.distance <= 0.5
        ? bestMatch
        : null;

    return { match, hasFace: true };
  }

  /*
   * मागचा analyze() (object detection + face recognition)
   * पूर्ण होताच लगेच पुढचा सुरू — मध्ये कुठलाही artificial
   * delay न ठेवता, hardware जितक्या वेगाने चालेल तितक्या वेगाने.
   */
  async function analyzeLoop() {
    await analyze();

    if (streamRef.current) {
      timerRef.current = window.setTimeout(
        analyzeLoop,
        ANALYZE_LOOP_GAP,
      );
    }
  }

  /*
   * Motion + movement-zone overlay साठी वेगळा, हलका loop —
   * हा प्रत्येक display frame ला (rAF, ~60fps) चालतो आणि
   * जड object/face models च्या वेगावर अजिबात अवलंबून नाही,
   * त्यामुळे movement detection मध्ये जाणवण्याइतका delay राहत नाही.
   */
  function motionTick() {
    if (!streamRef.current) {
      return;
    }

    const metrics = frameMetrics();

    motionScoreRef.current = metrics.motionScore;
    brightnessRef.current = metrics.brightness;

    drawOverlay(lastPredictionsRef.current, metrics.cells);

    const now = performance.now();

    if (now - lastMotionUiUpdateRef.current > 120) {
      lastMotionUiUpdateRef.current = now;
      setMotion(Math.round(metrics.motionScore));
      setActiveZones(metrics.cells.length);
    }

    /*
     * मोठी हालचाल झाली की screenshot capture (cooldown सह,
     * नाहीतर सतत हलत असताना खूप screenshots जमा होतील).
     */
    if (metrics.motionScore > MOTION_CAPTURE_THRESHOLD) {
      const nowMs = Date.now();

      if (
        nowMs - lastMotionCaptureRef.current >
        MOTION_CAPTURE_COOLDOWN_MS
      ) {
        lastMotionCaptureRef.current = nowMs;
        captureSnapshot("Active movement detected");
      }
    }

    motionRafRef.current = requestAnimationFrame(motionTick);
  }

  async function analyze() {
    const video = videoRef.current;

    /*
     * Register tabवर video element बदलताना null असू शकतो.
     */
    if (
      busyRef.current ||
      !objectModelRef.current ||
      !video ||
      !video.srcObject ||
      video.readyState < 2
    ) {
      return;
    }

    busyRef.current = true;

    try {
      const predictions =
        await objectModelRef.current.detect(
          video,
          12,
          0.42,
        );

      lastPredictionsRef.current = predictions;

      analyzeCycleRef.current += 1;

      const shouldRunFace =
        analyzeCycleRef.current %
          FACE_RECOGNITION_INTERVAL ===
        0;

      if (shouldRunFace) {
        lastFaceResultRef.current =
          await recognizeFace();
      }

      const { match, hasFace } =
        lastFaceResultRef.current;

      const activity = describeActivity(
        predictions,
        motionScoreRef.current,
        brightnessRef.current,
        hasFace,
      );

      setRecognized(match || null);

      /*
       * ओळख नसलेली व्यक्ती दिसली की screenshot capture —
       * फक्त तेव्हाच जेव्हा किमान एक व्यक्ती आधीच register
       * केलेली आहे (नाहीतर प्रत्येकच "unknown" ठरेल).
       */
      const personPresent =
        hasFace ||
        predictions.some(
          (prediction) =>
            prediction.class === "person" &&
            prediction.score > 0.5,
        );

      if (
        personPresent &&
        !match &&
        knownPeopleRef.current.length > 0
      ) {
        const nowMs = Date.now();

        if (
          nowMs - lastUnknownCaptureRef.current >
          UNKNOWN_CAPTURE_COOLDOWN_MS
        ) {
          lastUnknownCaptureRef.current = nowMs;
          captureSnapshot("Unknown person detected");
        }
      }

      const label = match
        ? `${match.name} detected · ${activity.label}`
        : activity.label;

      setObjects(
        predictions
          .filter((prediction) => prediction.score > 0.5)
          .map((prediction) => ({
            name: prediction.class,
            score: prediction.score,
          })),
      );

      setMessage(label);
      setKind(activity.kind);

      /*
       * एकच नाव सतत repeat होऊ नये म्हणून
       * नाव बदलल्यावरच voice announcement.
       */
      if (match?.name !== recognizedRef.current) {
        recognizedRef.current = match?.name || "";

        if (
          match &&
          voiceRef.current &&
          "speechSynthesis" in window
        ) {
          window.speechSynthesis.cancel();

          window.speechSynthesis.speak(
            new SpeechSynthesisUtterance(
              `${match.name} detected`,
            ),
          );
        }
      }

      if (label !== lastEventRef.current) {
        lastEventRef.current = label;

        const event = {
          label,
          kind: activity.kind,
          confidence: activity.confidence,
          motionScore: Math.round(
            motionScoreRef.current,
          ),
          objects: predictions
            .filter(
              (prediction) =>
                prediction.score > 0.5,
            )
            .map(
              (prediction) =>
                prediction.class,
            ),
          recognizedPersonName: match?.name || "",
          recognizedPersonId: match?._id || "",
        };

        setEvents((oldEvents) => [
          {
            ...event,
            createdAt: new Date().toISOString(),
          },
          ...oldEvents,
        ].slice(0, 8));

        fetch(ACTIVITY_API, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(event),
        }).catch(() => {});
      }
    } catch (exception) {
      console.error("Analysis failed:", exception);
    } finally {
      busyRef.current = false;
    }
  }

  async function enrollPerson() {
    if (!running) {
      setError(
        "Start the camera before registering a person.",
      );
      return;
    }

    if (!name.trim()) {
      setError("Enter the person’s name.");
      return;
    }

    if (!consent) {
      setError(
        "Confirm that the person consented to biometric enrollment.",
      );
      return;
    }

    const video = videoRef.current;

    if (
      !video ||
      !video.srcObject ||
      video.readyState < 2
    ) {
      setError(
        "Camera is not ready. Wait a moment and try again.",
      );
      return;
    }

    setError("");
    setEnrolling(true);

    try {
      const detections = await faceapi
        .detectAllFaces(video, FACE_OPTIONS)
        .withFaceLandmarks()
        .withFaceDescriptors();

      if (detections.length !== 1) {
        throw new Error(
          "Exactly one face must be clearly visible.",
        );
      }

      const response = await fetch(PEOPLE_API, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: name.trim(),
          details: details.trim(),
          consentGiven: true,
          faceDescriptor: Array.from(
            detections[0].descriptor,
          ),
        }),
      });

      if (!response.ok) {
        const result = await response.json();

        throw new Error(
          result.error || "Registration failed",
        );
      }

      await loadPeople();

      setName("");
      setDetails("");
      setConsent(false);
      setTab("live");

      setMessage(
        "Person registered. Live recognition is ready.",
      );
    } catch (exception) {
      setError(exception.message);
    } finally {
      setEnrolling(false);
    }
  }

  async function deletePerson(personId) {
    const confirmed = window.confirm(
      "Delete this biometric profile?",
    );

    if (!confirmed) {
      return;
    }

    await fetch(`${PEOPLE_API}/${personId}`, {
      method: "DELETE",
    });

    await loadPeople();
  }

  return (
    <main className="shell">
      <header>
        <div className="brand">
          <div className="logo">
            <Eye size={22} />
          </div>

          <div>
            <strong>ActivityLens</strong>
            <span>
              Private activity + face recognition
            </span>
          </div>
        </div>

        <div className="privacy">
          <ShieldCheck size={16} />
          Frames stay on this device
        </div>
      </header>

      <section className="hero">
        <div>
          <span className="eyebrow">
            LIVE VISION · CONSENT FIRST
          </span>

          <h1>
            Recognize activity.
            <br />
            <em>Greet known people.</em>
          </h1>

          <p>
            Objects and motion are analyzed locally.
            Consented face profiles are matched against
            private database records.
          </p>
        </div>

        <div
          className={`status ${running ? "live" : ""}`}
        >
          <span />
          {running ? "Camera active" : "Camera off"}
        </div>
      </section>

      {error && (
        <div className="error">{error}</div>
      )}

      <nav
        className="tabs"
        aria-label="Workspace"
      >
        <button
          className={tab === "live" ? "active" : ""}
          onClick={() => setTab("live")}
        >
          <ScanFace />
          Live recognition
        </button>

        <button
          className={
            tab === "register" ? "active" : ""
          }
          onClick={() => setTab("register")}
        >
          <UserPlus />
          Register person
        </button>

        <button
          className={
            tab === "captures" ? "active" : ""
          }
          onClick={() => setTab("captures")}
        >
          <Images />
          Captures
          {captures.length > 0 && (
            <span className="captureCount">
              {captures.length}
            </span>
          )}
        </button>
      </nav>

      {tab === "captures" ? (
        <section className="capturesPanel">
          <div className="capturesHeader">
            <div>
              <h2>Auto-captured screenshots</h2>
              <p>
                Unknown person or big movement triggers
                an automatic screenshot. These stay only
                in this browser tab — nothing is
                uploaded to any server.
              </p>
            </div>

            {captures.length > 0 && (
              <button
                className="ghostDanger"
                onClick={() => setCaptures([])}
              >
                <Trash2 size={16} />
                Clear all
              </button>
            )}
          </div>

          {captures.length === 0 ? (
            <div className="capturesEmpty">
              <Images size={28} />
              <p>
                No captures yet. Start the camera on
                Live recognition — screenshots will
                appear here automatically.
              </p>
            </div>
          ) : (
            <div className="capturesGrid">
              {captures.map((capture) => (
                <figure
                  key={capture.id}
                  className="captureCard"
                >
                  <img
                    src={capture.dataUrl}
                    alt={capture.label}
                  />

                  <figcaption>
                    <span className="captureLabel">
                      {capture.label}
                    </span>
                    <span className="captureTime">
                      {new Date(
                        capture.timestamp,
                      ).toLocaleString()}
                    </span>
                  </figcaption>

                  <a
                    className="captureDownload"
                    href={capture.dataUrl}
                    download={`capture-${capture.timestamp}.jpg`}
                    title="Download"
                  >
                    <Download size={16} />
                  </a>
                </figure>
              ))}
            </div>
          )}
        </section>
      ) : tab === "live" ? (
        <section className="grid">
          <div className="cameraCard">
            <div className="viewport">
              <video
                ref={videoRef}
                playsInline
                muted
                className={running ? "show" : ""}
              />

              <canvas ref={canvasRef} className="sampleCanvas" />
              <canvas ref={overlayCanvasRef} className="overlayCanvas" />

              {!running && (
                <div className="empty">
                  <div className="emptyIcon">
                    <CameraOff size={30} />
                  </div>

                  <h2>Your camera is private</h2>

                  <p>
                    It starts only after you choose to
                    allow access.
                  </p>
                </div>
              )}

              {running && (
                <>
                  <div className="scan">
                    <ScanLine size={18} />
                    Local analysis
                  </div>

                  <div
                    className={`activityBubble ${kind}`}
                  >
                    <Activity size={18} />
                    <span>{message}</span>
                  </div>
                </>
              )}
            </div>

            <div className="controls">
              <button
                className={
                  running ? "danger" : "primary"
                }
                onClick={
                  running
                    ? stopCamera
                    : startCamera
                }
                disabled={loading}
              >
                {running ? (
                  <>
                    <CameraOff />
                    Stop camera
                  </>
                ) : (
                  <>
                    <Camera />
                    {loading
                      ? "Loading…"
                      : "Start camera"}
                  </>
                )}
              </button>

              <button
                className="secondary"
                onClick={() =>
                  setVoice((value) => !value)
                }
              >
                {voice ? <Volume2 /> : <VolumeX />}
                {voice ? "Voice on" : "Voice off"}
              </button>
            </div>
          </div>

          <aside>
            <div className="identityCard">
              <div
                className={`avatar ${
                  recognized ? "known" : ""
                }`}
              >
                <ScanFace />
              </div>

              <small>IDENTITY MATCH</small>

              <h2>
                {recognized?.name ||
                  (running
                    ? "Unknown person"
                    : "Waiting")}
              </h2>

              <p>
                {recognized?.details ||
                  (recognized
                    ? "Known profile"
                    : "No matching consented profile")}
              </p>

              {recognized && (
                <span className="confidence">
                  Match distance{" "}
                  {recognized.distance.toFixed(3)}
                </span>
              )}
            </div>

            <div className="metricCard">
              <div className="cardTitle">
                <span>Live signals</span>
                <Activity size={18} />
              </div>

              <div className="bigActivity">
                {message}
              </div>

              <div className="metrics">
                <div>
                  <small>MOTION</small>
                  <strong>
                    {running ? motion : "—"}
                  </strong>
                </div>

                <div>
                  <small>MOVE ZONES</small>
                  <strong>
                    {running ? activeZones : "—"}
                  </strong>
                </div>

                <div>
                  <small>OBJECTS</small>
                  <strong>
                    {running
                      ? objects.length
                      : "—"}
                  </strong>
                </div>

                <div>
                  <small>KNOWN</small>
                  <strong>{people.length}</strong>
                </div>
              </div>

              <div className="chips">
                {objects.length ? (
                  objects
                    .slice(0, 6)
                    .map((object, index) => (
                      <span
                        key={`${object.name}-${index}`}
                      >
                        {object.name} ·{" "}
                        {Math.round(
                          object.score * 100,
                        )}
                        %
                      </span>
                    ))
                ) : (
                  <span className="quiet">
                    No objects yet
                  </span>
                )}
              </div>
            </div>

            <div className="history">
              <div className="cardTitle">
                <span>Recent activity</span>
                <Clock3 size={18} />
              </div>

              {events.length ? (
                events
                  .slice(0, 5)
                  .map((event, index) => (
                    <div
                      className="event"
                      key={`${event.createdAt}-${index}`}
                    >
                      <span
                        className={`dot ${event.kind}`}
                      />

                      <div>
                        <strong>{event.label}</strong>

                        <small>
                          {new Date(
                            event.createdAt,
                          ).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </small>
                      </div>
                    </div>
                  ))
              ) : (
                <div className="noEvents">
                  Start the camera to create an
                  activity timeline.
                </div>
              )}
            </div>
          </aside>
        </section>
      ) : (
        <section className="registerGrid">
          <div className="enrollCard">
            <span className="eyebrow">
              BIOMETRIC ENROLLMENT
            </span>

            <h2>
              Register one consenting person
            </h2>

            <p>
              Face photos are converted into a
              128-value mathematical descriptor.
              Raw photos are not stored.
            </p>

            <div className="enrollPreview">
              <video
                ref={videoRef}
                playsInline
                muted
                className={running ? "show" : ""}
              />

              {!running && (
                <span>
                  Start the camera in Live
                  recognition first
                </span>
              )}
            </div>

            <label>
              Name

              <input
                value={name}
                onChange={(event) =>
                  setName(event.target.value)
                }
                maxLength={80}
                placeholder="e.g. Vishal"
              />
            </label>

            <label>
              Details

              <textarea
                value={details}
                onChange={(event) =>
                  setDetails(event.target.value)
                }
                maxLength={500}
                placeholder="Role, notes, or contact details"
              />
            </label>

            <label className="consent">
              <input
                type="checkbox"
                checked={consent}
                onChange={(event) =>
                  setConsent(
                    event.target.checked,
                  )
                }
              />

              <span>
                This person has explicitly consented
                to face enrollment and recognition.
              </span>
            </label>

            <button
              className="primary enroll"
              onClick={enrollPerson}
              disabled={enrolling}
            >
              {enrolling
                ? "Capturing face…"
                : "Capture and register face"}
            </button>
          </div>

          <div className="peopleCard">
            <div className="cardTitle">
              <span>Registered people</span>
              <strong>{people.length}</strong>
            </div>

            {people.length ? (
              people.map((person) => (
                <div
                  className="personRow"
                  key={person._id}
                >
                  <div className="miniAvatar">
                    {person.name
                      .slice(0, 1)
                      .toUpperCase()}
                  </div>

                  <div>
                    <strong>{person.name}</strong>

                    <small>
                      {person.details ||
                        "No details"}
                    </small>
                  </div>

                  <button
                    onClick={() =>
                      deletePerson(person._id)
                    }
                  >
                    Delete
                  </button>
                </div>
              ))
            ) : (
              <div className="noEvents">
                No biometric profiles registered.
              </div>
            )}
          </div>
        </section>
      )}

      <footer>
        Face matching can be wrong. Never use it as
        the sole basis for access, attendance,
        policing, employment, or safety decisions.
      </footer>
    </main>
  );
}
