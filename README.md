# dissect

**Reverse-engineer any video into a blueprint you can reuse.**

Paste a link from YouTube, Shorts, TikTok, Instagram Reels, X, Facebook, Vimeo, Reddit and more. You get a Markdown file you, an editor or an AI can follow to make a *new* video in the same format.

```text
/dissect https://www.tiktok.com/@zachking/video/6768504823336815877
```

## What it does

1. **Downloads** the video with yt-dlp. It falls back to browser impersonation and the newest yt-dlp release when a site pushes back, and uses your browser login only if you say yes.
2. **Measures** what machines measure well:
   - every cut and on-screen change (captions, graphics, dissolves) and the pace across the video
   - the opening, frame by frame, at 4–6 frames a second
   - a transcript with word timings: words per minute, pauses (and what comes after them), emphasised words, pitch and sentence endings
   - the music bed and its level under the voice, sound effects on cuts, loudness
   - platform stats, engagement ratios, top comments, and YouTube's most-replayed moments
   - labelled contact sheets of the frames
3. **Watches and interprets.** Claude looks at every frame sheet, lines up what's on screen with what's being said, and works out why the video works.
4. **Writes the blueprint**:
   - format card, hook breakdown, beat sheet, shot grammar
   - visual style guide (captions, fonts, colours), voice direction, sound design, packaging
   - a fill-in-the-blanks script, a production plan
   - prompts for script, voice, visuals, captions and music
   - an edit recipe and a QA checklist

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

Check everything, with install hints for your OS:

```bash
python3 plugins/dissect/skills/dissect/scripts/doctor.py --warmup
```

On first use, Whisper downloads a speech model: about 140 MB for quick, 480 MB for standard (the default), 1.5 GB for deep.

## Use

Just talk to Claude:

- `dissect this reel and tell me how to make one like it: <link>`
- `why did this go viral? <link>`
- `break down the first 3 minutes of this MrBeast video, I want his pacing: <link>`
- `here are 4 TikToks from the same creator, what's their formula? <links>`

Or call it directly: `/dissect <link or file> [more links] [quick|deep]`.

Each dissection lands in `./dissections/<date>-<platform>-<title>/`:

```text
<slug>.blueprint.md   the reusable blueprint
facts.md              every measurement, the shot list with what's said in each shot, the transcript
sheets/               hook.jpg (dense opening), shots-*.jpg, most-replayed.jpg, cover.jpg
data/                 analysis.json, transcript.json / .txt, info.json, comments.txt
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

## Privacy and fair use

- Everything runs on your computer. Nothing is uploaded anywhere.
- Browser cookies are read only when you approve it, for that one download.
- Blueprints quote the video sparingly and teach the *format*. They don't reproduce the script, the footage or anyone's likeness.
- Respect platform terms and creators' copyright: study, don't re-upload.

Troubleshooting downloads: [`references/downloading.md`](plugins/dissect/skills/dissect/references/downloading.md).

## License

MIT
