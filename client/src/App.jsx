import { useEffect, useRef, useState } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";
import * as faceapi from "@vladmandic/face-api";

import {
  Activity,
  Camera,
  CameraOff,
  Clock3,
  Eye,
  ScanFace,
  ScanLine,
  ShieldCheck,
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
const FACE_OPTIONS = new faceapi.TinyFaceDetectorOptions({
  inputSize: 224,
  scoreThreshold: 0.5,
});

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
  const objectModelRef = useRef(null);

  const streamRef = useRef(null);
  const previousFrameRef = useRef(null);
  const timerRef = useRef(null);

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
          width: { ideal: 1280 },
          height: { ideal: 720 },
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
      setMessage("Loading AI models…");

      await Promise.all([
        cocoSsd
          .load({
            base: "lite_mobilenet_v2",
          })
          .then((model) => {
            objectModelRef.current = model;
          }),

        faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
        faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
        faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
      ]);

      setMessage("Looking for activity and known faces…");

      timerRef.current = window.setInterval(analyze, 1800);
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
      window.clearInterval(timerRef.current);
    }

    timerRef.current = null;

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

    setRunning(false);
    setObjects([]);
    setMotion(0);
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
      };
    }

    const context = canvas.getContext("2d", {
      willReadFrequently: true,
    });

    canvas.width = 160;
    canvas.height = 90;

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
      pixels.length / 16,
    );

    for (
      let pixelIndex = 0, frameIndex = 0;
      pixelIndex < pixels.length;
      pixelIndex += 16, frameIndex += 1
    ) {
      const value =
        (pixels[pixelIndex] +
          pixels[pixelIndex + 1] +
          pixels[pixelIndex + 2]) /
        3;

      currentFrame[frameIndex] = value;
      brightness += value;

      if (previousFrameRef.current) {
        difference += Math.abs(
          value -
            previousFrameRef.current[frameIndex],
        );
      }

      count += 1;
    }

    previousFrameRef.current = currentFrame;

    return {
      motionScore: count ? difference / count : 0,
      brightness: count ? brightness / count : 100,
    };
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

    if (
      !detection ||
      !knownPeopleRef.current.length
    ) {
      return null;
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
    return bestMatch && bestMatch.distance <= 0.5
      ? bestMatch
      : null;
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

      const metrics = frameMetrics();

      const activity = describeActivity(
        predictions,
        metrics.motionScore,
        metrics.brightness,
      );

      const match = await recognizeFace();

      setRecognized(match || null);

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

      setMotion(Math.round(metrics.motionScore));
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
            metrics.motionScore,
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
      </nav>

      {tab === "live" ? (
        <section className="grid">
          <div className="cameraCard">
            <div className="viewport">
              <video
                ref={videoRef}
                playsInline
                muted
                className={running ? "show" : ""}
              />

              <canvas ref={canvasRef} />

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