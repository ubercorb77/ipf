# ideal-parent

A voice-guided **Ideal Parent Figure (IPF)** imagery practice that runs locally on a Mac. You talk, Claude answers out loud, and the whole session is recorded to a video with captions.

IPF is the attachment-imagery protocol from Brown & Elliott's *Attachment Disturbances in Adults*. You imagine being a young child with ideal, entirely fictional parents who protect, attune, soothe, and delight you. Claude plays the guide, helping you build and stay inside the imagery and notice how it feels in your body. This is an imagery practice, not therapy.

## How it works

`parent.py` is the whole app:

- **Push-to-talk.** Press Enter to start talking and Enter again to send. If you say nothing for 60 seconds, Claude checks in.
- **Speech to text.** Phonon-2 (via `fermion-research`, Parakeet on MLX) transcribes your clips locally.
- **The guide.** Claude (`claude-sonnet-5`) is driven by a system prompt file and streams its reply sentence by sentence.
- **Text to speech.** Kokoro (voice `af_nicole`) speaks each sentence as it arrives, played to your headphones.
- **Timed breathing induction.** Prompts that start with `# induction-seconds: N` get an app-driven breathing stretch. Claude can't keep time, so the app does. Claude ends each line with `[PAUSE:N]` and the app waits that long in silence. Enter lets you speak, `s` skips ahead, and `q` quits.
- **Recording.** ffmpeg records the webcam (360p). The audio is rebuilt afterward from your clips and Claude's clips, each placed at its session-clock time, and mixed under the video.
- **Export.** Each session writes three files to `export/`:
  - `<prompt>_<date>_<hhmm>.mp4`, the video with mixed audio
  - `.transcript.vtt`, the captions
  - `.prompt.txt`, a copy of the prompt used

  To import them, open Claude Code in `../therapy-sessions` and run `/import-ideal-parent-session`.

## Prompts

| File | What it is |
|---|---|
| `ipf-prompt-1.txt`, `ipf-prompt-2.txt` | Standard IPF with fictional ideal parents (`ipf-prompt-2.txt` is the default) |
| `ipf-scenes-prompt-1.txt` | IPF variant that brings the ideal parents into real childhood scenes, once a base of felt security is in place. This deliberately departs from the book |
| `imf-prompt-1.txt` to `imf-prompt-3.txt` | Ideal *Manager* Figure: the same idea applied to your relationship with work |

`book.txt` and `ch8.txt` are source text from the book.

## Setup

Requires Python 3.11, [uv](https://github.com/astral-sh/uv), ffmpeg, a mic, and a webcam. Put `ANTHROPIC_API_KEY` in `../.env`.

```bash
uv sync
```

## Usage

Wear headphones so the mic doesn't hear Claude.

```bash
uv run python parent.py                                  # live session with the default prompt
uv run python parent.py --prompt ipf-scenes-prompt-1.txt
uv run python parent.py --mic "MacBook Pro Microphone" --cam "FaceTime HD Camera"
```

Replay mode feeds in your speech from `../therapy-sessions/session01` with no mic, camera, or speakers. It's for testing and writes to `export/replay-test/`.

```bash
uv run python parent.py --replay --turns 4 --idle-at 0,2
```

Mic order is Hollyland first, then the MacBook Pro microphone. Other flags are `--idle-seconds` and `--induction-seconds`.

`make_samples.py` generates Kokoro voice samples into `samples/` to compare voices.

## Known bugs

- **Exported audio is out of sync.** In the exported video, the audio isn't placed where it should be. Syncing your clips and Claude's speech to the right spot on the timeline (and against the video and captions) is messed up.
- Not a bug, but I want it to be more concise and use more language from the book.
