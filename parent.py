#!/usr/bin/env python3
"""Ideal-parent-figure voice guide: push-to-talk, local Phonon-2 + Kokoro, Claude Sonnet 5.

Live:    uv run python parent.py [--prompt imf-prompt-2.txt]   (default: ipf-prompt-2.txt)
Replay:  uv run python parent.py --replay      (feeds your speech from therapy-sessions/session01, no mic/cam/speakers)

Each session exports <prompt>_<date>_<hhmm>.mp4 (360p video + mixed audio), .transcript.vtt and .prompt.txt
(the prompt it used) into export/, e.g. ipf-1_2026-10-05_1530. To import them, open Claude Code in
../therapy-sessions and run the /import-ideal-parent-session skill, which numbers them its own way.
"""
from __future__ import annotations

import argparse
import json
import re
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from queue import Queue

warnings.filterwarnings("ignore")

import numpy as np
import sounddevice as sd
import soundfile as sf
from anthropic import Anthropic
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
THERAPY = HERE.parent / "therapy-sessions"
EXPORT = HERE / "export"
load_dotenv(HERE.parent / ".env")

MODEL = "claude-sonnet-5"
VOICE, SPEED = "af_nicole", 1.25
USER, AI = "Corbin", "Claude"
PHONON_MODEL = "phonon-2"
MIN_SPEECH_RMS = 0.004  # below this a push-to-talk clip counts as silence
TTS_RATE, STT_RATE = 24000, 16000
SENTENCE_GAP = 0.2  # seconds of silence appended to each spoken sentence
SILENT_TURN = "[stayed silent, ready to continue]"
MIC_PREFERENCE = ["hollyland", "macbook pro microphone"]  # first match wins; compared against macOS device name and manufacturer
IDLE_SECONDS = 60  # no push-to-talk for this long -> Claude checks in
IDLE = object()  # record_ptt() result when the user said nothing for IDLE_SECONDS
SENT_END = re.compile(r"[.!?…]+[\"')\]]*\s+")
# a prompt whose first line is "# induction-seconds: 240" is built for a timed, app-driven breathing induction
INDUCTION_HEADER = re.compile(r"\A#\s*induction-seconds:\s*(\d+)[^\S\n]*\n")
PAUSE_TAG = re.compile(r"\[PAUSE:\s*(\d+)\s*\]")  # Claude ends each induction line with this; the app waits that long, unspoken
DEFAULT_PAUSE, MIN_PAUSE, MAX_PAUSE = 8, 3, 20


class Clock:
    """Session clock. Live: wall time since the recorder's first frame. Replay: virtual, advanced by audio durations."""

    def __init__(self, virtual: bool):
        self.virtual, self._t, self._t0 = virtual, 0.0, time.monotonic()

    def reset(self):
        self._t0 = time.monotonic()

    def now(self) -> float:
        return self._t if self.virtual else time.monotonic() - self._t0

    def advance(self, dt: float):
        if self.virtual:
            self._t += dt


# ---------- video + mic recording (live only) ----------

class Recorder:
    """ffmpeg records the webcam (scaled to 640x360), video only. The audio is built afterwards from the clips themselves
    (mix_audio): ffmpeg's avfoundation mic capture drops chunks under load, which time-compressed Corbin's side of the mix."""

    def __init__(self, path: Path, clock: Clock, cam: str):
        self.path, self.clock, self.cam = path, clock, cam
        self.started = threading.Event()

    def start(self):
        cmd = ["ffmpeg", "-y", "-hide_banner", "-stats_period", "0.1", "-f", "avfoundation", "-framerate", "30", "-video_size", "1280x720",
               "-i", f"{self.cam}:none", "-vf", "scale=640:360", "-r", "25",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-an", str(self.path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
        self.stderr_tail = b""
        threading.Thread(target=self._watch, daemon=True).start()
        if not self.started.wait(30):
            raise RuntimeError(f"camera did not start (allow camera access?):\n{self.stderr_tail.decode(errors='ignore')[-600:]}")

    def _watch(self):
        while chunk := self.proc.stderr.read(256):
            self.stderr_tail = (self.stderr_tail + chunk)[-2000:]
            if not self.started.is_set() and b"frame=" in self.stderr_tail:
                self.clock.reset()
                self.started.set()

    def stop(self):
        try:
            self.proc.stdin.write(b"q")
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.terminate()


# ---------- speech to text ----------

class Transcriber:
    """Phonon-2 (fermion-research, Parakeet on MLX), loaded from the local cache. Takes ~15 s to load, then runs far faster than real time."""

    def __init__(self):
        from fermion._speech import backends, fetch
        from fermion.transcribe import _resolve  # private to fermion-research; version is pinned in pyproject
        repo, key, pin, local_dir = _resolve(PHONON_MODEL)
        model_dir = local_dir or fetch.ensure(repo, key, pin)
        self.speech = backends.load(backends.resolve("parent.py"), model_dir, profile=key, backend=pin["backend"], quiet=True)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:  # first decode compiles MLX kernels (~4 s); do it now, not mid-session
            sf.write(f.name, np.random.default_rng(0).normal(0, 0.01, STT_RATE).astype(np.float32), STT_RATE)
            self.speech.transcribe_detailed(f.name)

    def __call__(self, audio: np.ndarray) -> list[tuple[float, float, str]]:
        """-> [(start_s, end_s, text)] cue-sized chunks, relative to the start of the clip"""
        if len(audio) < STT_RATE * 0.3 or np.sqrt(np.mean(audio ** 2)) < MIN_SPEECH_RMS:
            return []
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio, STT_RATE)
            result = self.speech.transcribe_detailed(f.name)
        words = result.words if result.timed else None
        if not words:
            return [(s["start"], s["end"], t) for s in result.segments if (t := clean(s["text"]))]
        # group word timings into cue-sized chunks (sentence ends, or ~7 s), so the viewer's highlighting stays accurate
        cues, chunk = [], []
        for w in words:
            chunk.append(w)
            if re.search(r"[.!?]$", w["text"]) or w["end"] - chunk[0]["start"] > 7:
                cues.append(chunk)
                chunk = []
        if chunk:
            cues.append(chunk)
        return [(c[0]["start"], c[-1]["end"], clean(" ".join(w["text"] for w in c))) for c in cues]


# ---------- text to speech + playback ----------

class Voice:
    def __init__(self, clock: Clock, play: bool):
        from kokoro import KPipeline
        self.pipe = KPipeline(lang_code=VOICE[0], repo_id="hexgrad/Kokoro-82M")
        self.clock, self.play = clock, play

    def synth(self, text: str) -> np.ndarray:
        audio = np.concatenate([np.asarray(a, dtype=np.float32) for _, _, a in self.pipe(text, voice=VOICE, speed=SPEED)])
        return np.concatenate([audio, np.zeros(int(SENTENCE_GAP * TTS_RATE), np.float32)])

    def speak(self, sentences) -> list[tuple[str, float, float, np.ndarray]]:
        """Synthesize sentences as they stream in and play them. -> [(text, start, end, audio)]"""
        q: Queue = Queue(maxsize=3)
        err: list[BaseException] = []

        def worker():
            try:
                for s in sentences:
                    q.put((s, self.synth(s)))
            except BaseException as e:
                err.append(e)
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        spoken = []
        while (item := q.get()) is not None:
            text, audio = item
            start = self.clock.now()
            if self.play:
                sd.play(audio, TTS_RATE)
                sd.wait()
            else:
                self.clock.advance(len(audio) / TTS_RATE)
            spoken.append((text, start, self.clock.now(), audio))
        if err:
            raise err[0]
        return spoken


# ---------- claude ----------

class Guide:
    def __init__(self, prompt_path: Path):
        self.client = Anthropic()
        text = prompt_path.read_text()
        header = INDUCTION_HEADER.match(text)
        self.induction_seconds = int(header.group(1)) if header else 0
        self.last_pause = DEFAULT_PAUSE  # seconds of silence Claude asked for at the end of its last line
        self.system = [{"type": "text", "text": text[header.end():] if header else text, "cache_control": {"type": "ephemeral"}}]
        self.history = [{"role": "user", "content": "[The session is starting. Corbin has just sat down.]"}]

    def say(self, user_text: str | None = None):
        """Add the user's turn (if any), stream Claude's reply, yield it sentence by sentence."""
        if user_text is not None:
            self.history.append({"role": "user", "content": user_text})
        parts: list[str] = []
        buf = full = ""
        with self.client.messages.stream(model=MODEL, max_tokens=300, system=self.system, thinking={"type": "disabled"},
                                         messages=self.history) as stream:
            for delta in stream.text_stream:
                buf += delta
                full += delta
                while (end := sentence_end(buf)) is not None:
                    sentence, buf = clean(buf[:end]), buf[end:]
                    if sentence:
                        parts.append(sentence)
                        yield sentence
            final = stream.get_final_message()
        if tail := clean(buf):
            parts.append(tail)
            yield tail
        if final.stop_reason == "refusal" or not parts:
            fallback = "I'm right here. Take your time."
            parts.append(fallback)
            yield fallback
        pause = PAUSE_TAG.search(full)
        self.last_pause = min(max(int(pause.group(1)), MIN_PAUSE), MAX_PAUSE) if pause else DEFAULT_PAUSE
        self.history.append({"role": "assistant", "content": full.strip() or " ".join(parts)})  # raw, so Claude keeps seeing its own tags


def sentence_end(buf: str) -> int | None:
    """Index just past the first full sentence in buf; ellipses are soft pauses, not sentence ends."""
    for m in SENT_END.finditer(buf):
        if not m.group().rstrip().endswith(("...", "…")):
            return m.end()
    return None


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]*(?:\]|$)|[*_#`]|\([^)]*\)", "", text)).strip()  # also drops [PAUSE:N] and bracket fragments


# ---------- user input ----------

def pick_mic(preferences: list[str]) -> tuple[str, int, int]:
    """-> (device name, sounddevice index, channels). The Hollyland receiver is named just "Wireless Microphone" and only
    its manufacturer says Hollyland, so match against both, using what macOS reports."""
    out = subprocess.run(["system_profiler", "SPAudioDataType", "-json"], capture_output=True, text=True).stdout
    try:
        listed = [(i["_name"], i.get("coreaudio_device_manufacturer", "")) for g in json.loads(out)["SPAudioDataType"]
                  for i in g.get("_items", []) if i.get("coreaudio_device_input")]
    except (ValueError, KeyError):
        listed = []
    usable = {d["name"]: (i, d["max_input_channels"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"]}
    for pref in preferences:
        for name, maker in listed:
            if name in usable and (pref in name.lower() or pref in maker.lower()):
                index, channels = usable[name]
                return name, index, min(channels, 2)
    raise SystemExit(f"no microphone matching {preferences} found. Inputs available: {', '.join(usable) or 'none'} (pick one with --mic)")


def record_ptt(mic_device, mic_channels: int, idle_seconds: float, clock: Clock, started: bool = False):
    """Enter to start, Enter to stop. Returns (16 kHz mono audio, session-clock time of its first sample), None if the user
    typed q, or IDLE after idle_seconds of nothing. The time is when recording began, not when the prompt appeared.
    started=True: the caller already consumed the starting Enter (used during the induction)."""
    if not started:
        print("\n  [enter] to talk (q to quit) > ", end="", flush=True)
        if not select.select([sys.stdin], [], [], idle_seconds)[0]:
            print()
            return IDLE
        if sys.stdin.readline().strip().lower() == "q":
            return None
    frames: list[np.ndarray] = []
    first: list[float] = []

    def on_audio(data, *_):
        if not first:
            first.append(clock.now() - len(data) / STT_RATE)  # the callback fires once the first block has been captured
        frames.append(data.copy())

    with sd.InputStream(samplerate=STT_RATE, channels=mic_channels, dtype="float32", device=mic_device, callback=on_audio):
        input("  recording... [enter] to send > ")
    if not frames:
        return np.zeros(0, np.float32), clock.now()
    audio = np.concatenate(frames)
    return audio[:, np.argmax(np.sqrt((audio ** 2).mean(axis=0)))], max(first[0], 0.0)  # stereo receivers: use the channel with the voice


def load_replay_turns(n: int) -> list[tuple[float, np.ndarray, str]]:
    """Corbin's own speech from therapy-sessions/session01 as stand-in PTT clips: [(duration, audio16k, vtt_text)]"""
    sys.path.insert(0, str(THERAPY))
    from build_library import parse_vtt
    cues = parse_vtt(THERAPY / "session01.transcript.vtt")
    turns, cur = [], None
    for c in cues:
        if c["speaker"] != USER:
            cur = None
            continue
        if cur and c["start"] - cur["end"] < 2.0 and c["end"] - cur["start"] < 25:
            cur["end"], cur["text"] = c["end"], cur["text"] + " " + c["text"]
        else:
            cur = {"start": c["start"], "end": c["end"], "text": c["text"]}
            turns.append(cur)
    turns = [t for t in turns if t["end"] - t["start"] >= 6][:n]
    out = []
    for t in turns:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t["start"]), "-t", str(t["end"] - t["start"]), "-i",
                              str(THERAPY / "session01.m4a"), "-ac", "1", "-ar", str(STT_RATE), "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        out.append((t["end"] - t["start"], np.frombuffer(raw, np.float32), t["text"]))
    return out


# ---------- export ----------

def vtt_ts(t: float) -> str:
    h, rem = divmod(max(t, 0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def write_vtt(cues: list[tuple[float, float, str, str]], path: Path):
    """cues: (start, end, speaker, text), written in the Zoom style build_library.py parses"""
    cues = sorted(cues)
    lines = ["WEBVTT", ""]
    for i, (start, end, who, text) in enumerate(cues, 1):
        lines += [str(i), f"{vtt_ts(start)} --> {vtt_ts(max(end, start + 0.2))}", f"{who}: {text}", ""]
    path.write_text("\n".join(lines))


def probe_duration(path: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                                capture_output=True, text=True, check=True).stdout)


def place(clips: list[tuple[float, np.ndarray]], rate: int, total: float) -> np.ndarray:
    """Lay clips onto a silent track at their session-clock times."""
    track = np.zeros(max([int(total * rate)] + [int(t * rate) + len(a) for t, a in clips]), np.float32)
    for t, a in clips:
        i = int(t * rate)
        track[i:i + len(a)] += a
    return np.clip(track, -1, 1)


def mix_audio(raw_mp4: Path, user_clips: list[tuple[float, np.ndarray]], claude_clips: list[tuple[float, np.ndarray]], out_mp4: Path):
    """Put the session's audio under the silent video: Corbin's push-to-talk clips and Claude's speech, each at the session-clock
    time it happened, so they can't drift against each other, the captions, or the video."""
    total = probe_duration(raw_mp4)
    with tempfile.NamedTemporaryFile(suffix=".wav") as mic, tempfile.NamedTemporaryFile(suffix=".wav") as ai:
        sf.write(mic.name, place(user_clips, STT_RATE, total), STT_RATE)
        sf.write(ai.name, place(claude_clips, TTS_RATE, total), TTS_RATE)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw_mp4), "-i", mic.name, "-i", ai.name, "-filter_complex",
                        "[1:a]aresample=48000[a1];[2:a]aresample=48000[a2];[a1][a2]amix=inputs=2:duration=longest:normalize=0[a]",
                        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
                        str(out_mp4)], check=True)


def synthetic_raw_video(path: Path, duration: float):
    """Replay mode stand-in for the webcam recording: black 640x360 video, no audio."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=25",
                    "-t", f"{duration:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                    "-an", str(path)], check=True)


def session_name(prompt: Path, started: datetime) -> str:
    """This app's own naming, independent of therapy-sessions' numbering: <prompt>_<date>_<hhmm>, e.g. ipf-1_2026-10-05_1530"""
    kind = re.sub(r"[^a-z0-9]+", "-", prompt.stem.lower().replace("-prompt", "")).strip("-") or "session"
    base = f"{kind}_{started:%Y-%m-%d_%H%M}"
    name, n = base, 2
    while (EXPORT / f"{name}.mp4").exists():
        name, n = f"{base}-{n}", n + 1
    return name


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="test mode: use your speech from therapy-sessions/session01")
    ap.add_argument("--turns", type=int, default=4, help="replay: number of your turns to feed in")
    ap.add_argument("--idle-seconds", type=float, default=IDLE_SECONDS, help="silence before Claude checks in")
    ap.add_argument("--idle-at", default="", help="replay: comma-separated turn counts after which to simulate a silent minute, e.g. 0,2")
    ap.add_argument("--prompt", type=Path, default=HERE / "ipf-prompt-2.txt", help="system prompt file")
    ap.add_argument("--induction-seconds", type=int, default=None,
                    help="length of the timed breathing induction; default comes from the prompt's '# induction-seconds:' line, 0 turns it off")
    ap.add_argument("--cam", default="FaceTime HD Camera")
    ap.add_argument("--mic", default=None, help="mic name to use instead of the default order (Hollyland, then MacBook Pro Microphone)")
    args = ap.parse_args()

    out_dir = EXPORT / "replay-test" if args.replay else EXPORT
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = "replay-test" if args.replay else session_name(args.prompt, datetime.now())
    raw_mp4 = out_dir / f"{sid}.raw.mp4"
    shutil.copyfile(args.prompt, out_dir / f"{sid}.prompt.txt")  # which prompt this session used, for the therapy-sessions viewer

    clock = Clock(virtual=args.replay)
    print("loading phonon-2 + kokoro (about 15 s)...")
    stt, voice, guide = Transcriber(), Voice(clock, play=not args.replay), Guide(args.prompt)

    recorder = None
    mic_device, mic_channels = None, 1
    if not args.replay:
        mic_name, mic_device, mic_channels = pick_mic([args.mic.lower()] if args.mic else MIC_PREFERENCE)
        print(f"  mic: {mic_name}")
        recorder = Recorder(raw_mp4, clock, args.cam)
        recorder.start()
        print(f"  recording to {raw_mp4.name}; wear headphones so the mic doesn't hear Claude")

    cues: list[tuple[float, float, str, str]] = []
    clips: list[tuple[float, np.ndarray]] = []       # Claude audio, placed on the session timeline
    user_clips: list[tuple[float, np.ndarray]] = []  # Corbin's push-to-talk audio, placed on the session timeline
    replay = iter(load_replay_turns(args.turns)) if args.replay else None
    idle_at = {int(x) for x in args.idle_at.split(",") if x.strip()}
    user_turns = 0

    def claude_turn(user_text: str | None):
        t_ask = time.monotonic()
        first = True
        sentences = guide.say(user_text)

        def timed():
            nonlocal first
            for s in sentences:
                if first:
                    print(f"  (first sentence from claude after {time.monotonic() - t_ask:.1f}s)")
                    first = False
                print(f"  {AI}: {s}")
                yield s
        for text, start, end, audio in voice.speak(timed()):
            cues.append((start, end - SENTENCE_GAP, AI, text))
            clips.append((start, audio))

    def respond(audio, start: float, suffix: str = ""):
        """Turn one push-to-talk clip (or an idle timeout) into a message for Claude and speak the reply."""
        if audio is IDLE:
            text = f"[no response for about {round(args.idle_seconds)} seconds; automatic check-in]"
            print(f"  (nothing from {USER} for {round(args.idle_seconds)}s, checking in)")
        else:
            user_clips.append((start, audio))
            segs = stt(audio)
            if segs:
                text = " ".join(s[2] for s in segs)
                for s0, s1, t in segs:
                    cues.append((start + s0, start + s1, USER, t))
            else:
                text = SILENT_TURN
            print(f"  {USER}: {text}")
        claude_turn(text + suffix)

    def stamp(seconds: float) -> str:
        return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"

    def run_induction(t0: float, total: float) -> bool:
        """Timed breathing induction. Claude can't keep time, so the app does: it sends [continue: m:ss of m:ss] after each
        line's pause, and only releases Claude to the next stage once `total` seconds have passed. False if the user quit."""
        if not args.replay:
            print("  (breathing induction: [enter] to speak, s+enter to skip ahead, q+enter to quit)")
        while clock.now() - t0 < total:
            elapsed, pause = clock.now() - t0, guide.last_pause
            print(f"  (breathing {stamp(elapsed)} of {stamp(total)}; pausing {pause}s)")
            if args.replay:
                clock.advance(pause)
                action = None
            elif not select.select([sys.stdin], [], [], pause)[0]:
                action = None
            else:
                line = sys.stdin.readline().strip().lower()
                action = "skip" if line == "s" else "quit" if line == "q" else "talk"
            if action == "quit":
                return False
            if action == "skip":
                break
            elapsed = clock.now() - t0
            progress = f"{stamp(elapsed)} of {stamp(total)}"
            if action == "talk":
                audio, start = record_ptt(mic_device, mic_channels, args.idle_seconds, clock, started=True)
                respond(audio, start, suffix=f"\n[breathing time so far: {progress}]")
            elif elapsed < total:
                claude_turn(f"[continue: {progress} of breathing]")
        claude_turn("[breathing time is complete; move on to the next stage now]")
        return True

    induction_total = guide.induction_seconds if args.induction_seconds is None else args.induction_seconds
    if induction_total and not guide.induction_seconds:
        print(f"  warning: {args.prompt.name} has no '# induction-seconds:' line, so it may not know how to run the induction")

    try:
        t_start = clock.now()
        claude_turn(None)
        ended = bool(induction_total) and not run_induction(t_start, induction_total)
        while not ended:
            if replay:
                if user_turns in idle_at:
                    idle_at.discard(user_turns)
                    clock.advance(args.idle_seconds)
                    audio = IDLE
                else:
                    item = next(replay, None)
                    if item is None:
                        break
                    user_turns += 1
                    clock.advance(1.0)
                    dur, audio, _ = item
                    start = clock.now()
                    clock.advance(dur)
            else:
                got = record_ptt(mic_device, mic_channels, args.idle_seconds, clock)
                if got is None:
                    break
                audio, start = (IDLE, clock.now()) if got is IDLE else got
            respond(audio, start)
    except KeyboardInterrupt:
        print("\n  ending session...")
    finally:
        end_t = clock.now() + 1.0
        if recorder:
            recorder.stop()
        else:
            synthetic_raw_video(raw_mp4, end_t)
        vtt = out_dir / f"{sid}.transcript.vtt"
        write_vtt(cues, vtt)
        final = out_dir / f"{sid}.mp4"
        mix_audio(raw_mp4, user_clips, clips, final)
        raw_mp4.unlink(missing_ok=True)

    print(f"\n  exported {final} ({probe_duration(final) / 60:.1f} min) and {vtt.name} (prompt: {args.prompt.name})")
    if not args.replay:
        print(f"  to import: open Claude Code in {THERAPY} and run /import-ideal-parent-session (it picks the next session number)")


if __name__ == "__main__":
    main()
