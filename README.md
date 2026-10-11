# dissect

**Reverse-engineer any video into a frame-accurate blueprint you can rebuild from, then score your rebuild against it.**

Paste a link from YouTube, Shorts, TikTok, Instagram Reels, X, Facebook, Vimeo, Reddit and more. You get a Markdown file you, an editor or an AI can follow to make a *new* video in the same format.

```text
/dissect https://www.tiktok.com/@zachking/video/6768504823336815877
```

## What's new in 2.0

dissect now measures a video frame by frame. In a side-by-side test (same brief, same model, same tools), a rebuild made from a v2 blueprint scored 82/100 against the original, against 62 from a v1 blueprint; the builders said the frames mattered most.

- **Frame-level timeline** (`data/timeline.json`): every shot with in and out frames; the transition into each shot (cut, dissolve, fade, flash, whip, zoom, wipe) and its length in frames; hidden cuts and dips inside shots; camera and element moves as numbers (pan, tilt, zoom, roll, segments, easing); blur, focus and animation cadence; palette and highlights per shot; the brightness curve; on-screen text with times, boxes and cap heights; the beat grid with an honest confidence, key, sections, a cue sheet of hits and whooshes keyed to shots; loudness.
- **Four kinds of frame sheets**: the hook (first 3 s at 0.25 s), one frame per shot, transition strips (14 consecutive frames around every cut and change) and the whole film at fixed steps. Every tile is labelled with shot, time and frame.
- **Measured blueprint sections** (`measured.md`): shot table in frames, camera and transitions, type with a phone minimum, palette, brightness, per-phrase speech timing and the sound grid, pasted verbatim into the blueprint by `dissect.py assemble`, which also checks the prose against the measurements.
- **A new blueprint template**: numbers for every move, a type spec and layout boxes, a material card, an original music bed, two production routes, construction notes for the signature move, aspect and length adaptation, and a recipe prompt with brand placeholders. The blueprint ships with `<slug>.timeline.json`.
- **`check`**: `dissect.py check your-render.mp4 --ref <dissection>` measures your render and scores it out of 100 (cuts, brightness curve, motion, pace, loudness, format), with fixes in frames: "Cut 2 lands 4 f late at f131; move it to f127".

## What it does

1. **Downloads** the video with yt-dlp. It falls back to browser impersonation and the newest yt-dlp release when a site pushes back, and uses your browser login only if you say yes.
2. **Measures** what machines measure well:
   - every cut, transition and hidden change, in frames, and the pace across the video
   - camera and element moves, type sizes and timings, palette and brightness, the beat grid and sound cues
   - the opening, frame by frame, at 4–6 frames a second
   - a transcript with word timings: words per minute, pauses (and what comes after them), emphasised words, pitch and sentence endings
   - the music bed and its level under the voice, sound effects on cuts, loudness
   - platform stats, engagement ratios, top comments, and YouTube's most-replayed moments
   - labelled frame sheets: hook, one frame per shot, transition strips, the whole timeline
3. **Watches and interprets.** Claude looks at every frame sheet, lines up what's on screen with what's being said, and works out why the video works.
4. **Writes the blueprint** (plus a timeline JSON a coding agent can build to):
   - format card, hook breakdown, beat sheet, shot table in frames, camera and transitions with numbers
   - type spec with a phone minimum, layout boxes, material card, voice direction, sound grid and an original music bed, packaging
   - a fill-in-the-blanks script, a production plan
   - prompts for script, voice, visuals, captions and music
   - construction notes for the signature move, two production routes, aspect and length adaptation, a QA checklist and a runnable recipe prompt
5. **Checks your rebuild** against the reference and tells you what to fix, in frames.

Give it several links and it also writes `formula.md`: what stays constant across them, and what the best performer did differently.

## Install

### Claude Code (plugin)

```text
/plugin marketplace add 0xpratzyy/dissect
/plugin install dissect@dissect
```

From a local clone, point the marketplace at the folder instead: `/plugin marketplace add ./dissect`.

### Any agent that reads `SKILL.md` files (manual)

Copy the skill folder into your skills directory:

```bash
cp -r plugins/dissect/skills/dissect ~/.claude/skills/            # personal, every project
cp -r plugins/dissect/skills/dissect .claude/skills/              # this project only
```

```powershell
Copy-Item -Recurse plugins\dissect\skills\dissect "$HOME\.claude\skills\"   # Windows
```

### Requirements

| Tool | Why | Install |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Runs the script and installs its Python packages automatically | `brew install uv` · `winget install astral-sh.uv` · `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| ffmpeg | Frames, audio, loudness | `brew install ffmpeg` · `winget install Gyan.FFmpeg` · `sudo apt install ffmpeg` |
| yt-dlp | Downloads (optional: uv can fetch it on demand) | `brew install yt-dlp` · `winget install yt-dlp.yt-dlp` · `uv tool install yt-dlp` |
| Deno (recommended) | Lets yt-dlp unlock every YouTube format | `brew install deno` · `winget install DenoLand.Deno` |
| tesseract (Linux/Windows, optional) | Reads on-screen text off macOS | `sudo apt install tesseract-ocr` · `winget install UB-Mannheim.TesseractOCR` |

Check everything, with install hints for your OS:

```bash
python3 plugins/dissect/skills/dissect/scripts/doctor.py --warmup
```

On first use, uv installs the Python packages (about 600 MB, a few minutes) and Whisper downloads a speech model: about 140 MB for quick, 480 MB for standard (the default), 1.5 GB for deep.

## Use

Just talk to Claude:

- `dissect this reel and tell me how to make one like it: <link>`
- `why did this go viral? <link>`
- `break down the first 3 minutes of this MrBeast video, I want his pacing: <link>`
- `here are 4 TikToks from the same creator, what's their formula? <links>`
- `recreate this launch film for my product: <link>` then `how close is my render? out/video.mp4`

Or call it directly: `/dissect <link or file> [more links] [quick|deep]`.

Each dissection lands in `./dissections/<date>-<platform>-<title>/`:

```text
<slug>.blueprint.md   the reusable blueprint
<slug>.timeline.json  the measured timeline plus the writer's plan, for coding agents
facts.md              every measurement, the shot list with what's said in each shot, the transcript
measured.md           the measured blueprint sections, in frames
sheets/               hook.jpg, shots*.jpg (one frame per shot), transitions-*.jpg, timeline-*.jpg,
                      most-replayed.jpg, cover.jpg
data/                 timeline.json, analysis.json, transcript.json / .txt, info.json, comments.txt
```

The downloaded video is deleted afterwards (`--keep-video` keeps it), so a dissection is a couple of MB.

### Script options

`uv run plugins/dissect/skills/dissect/scripts/dissect.py <link-or-file> [options]`

| Option | Default | |
|---|---|---|
| `--depth quick\|standard\|deep` | standard | frames on the sheets (36 / 72 / 160) and Whisper model (base / small / medium) |
| `--out DIR` | `./dissections` (or `$DISSECT_OUT`) | where dissections go |
| `--max-minutes N` | whole video | analyse only the opening N minutes |
| `--lang CODE` | auto | spoken language, e.g. `en`, `hi`, `es` |
| `--model NAME` | by depth | any faster-whisper model, e.g. `large-v3`, `turbo` |
| `--cookies-from-browser NAME` / `--cookies FILE` | off | use a logged-in session (Claude asks you first) |
| `--keep-video` | off | keep the source video |
| `--no-comments` | off | skip YouTube comments |
| `--cut-threshold X` | 0.30 | scene-change score that counts as a cut |
| `--timeline-minutes N` | 10 | measure the frame-level timeline over at most the first N minutes |
| `--no-timeline` | off | skip the frame-level timeline (v1 output only, faster) |

Two more commands:

```bash
uv run plugins/dissect/skills/dissect/scripts/dissect.py assemble <dissection folder>          # draft + measured sections -> final blueprint
uv run plugins/dissect/skills/dissect/scripts/dissect.py check <render.mp4> --ref <dissection folder> [--target x|youtube|tiktok|web]
```

A 40 s film takes about 3 minutes to measure on a laptop. On-screen text is read with Apple Vision on macOS, or tesseract elsewhere (`apt install tesseract-ocr`); without either, text sizes and times are skipped.

## Refs

Want the blueprints without measuring? [Refs](https://refs.video) is a hosted library of 3,000+ launch films already dissected this way, with frame sheets and timelines, for Claude Code, Codex and Cursor.

## Privacy and fair use

- Everything runs on your computer. Nothing is uploaded anywhere.
- Browser cookies are read only when you approve it, for that one download.
- Blueprints quote the video sparingly and teach the *format*. They don't reproduce the script, the footage or anyone's likeness.
- Respect platform terms and creators' copyright: study, don't re-upload.

Troubleshooting downloads: [`references/downloading.md`](plugins/dissect/skills/dissect/references/downloading.md).

## License

MIT
