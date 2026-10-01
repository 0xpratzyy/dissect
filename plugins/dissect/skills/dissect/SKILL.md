---
name: dissect
description: Reverse-engineer any online video into a reusable Markdown blueprint for making new videos in the same format. Takes links from YouTube (videos and Shorts), TikTok, Instagram Reels, X/Twitter, Facebook, Vimeo, Reddit and the other sites yt-dlp supports, or a local file. Downloads it, finds every cut and on-screen change, captures the hook frame by frame, transcribes with word timings, profiles the voice (pace, pauses, pitch, emphasis), music, sound effects and loudness, reads platform stats and YouTube's most-replayed moments, then writes a blueprint with beat sheet, shot list, style guide, fill-in script and ready-to-paste AI prompts. Use it whenever someone shares a video link and wants it broken down, wants to know why it works or went viral, wants to copy its style, editing or hook, says "make a video like this", or wants a creator's formula, even if they never say "dissect".
license: MIT
compatibility: Needs internet access, ffmpeg, and uv (or Python 3.9+ with numpy, pillow, faster-whisper and praat-parselmouth). yt-dlp is fetched on demand if missing. scripts/doctor.py checks everything and prints install commands.
metadata:
  version: "1.1.0"
---

# Dissect

Turn a reference video into a blueprint that a person or an AI can follow to make a *new* video in the same format. The work has two halves:

1. **Measure.** `scripts/dissect.py` downloads the video and measures what machines measure well: every cut and on-screen change, pace, the transcript with word timings, voice pitch and emphasis, the music bed, loudness and platform stats. It also renders labelled contact sheets of the frames.
2. **Watch and interpret.** You look at the sheets, read the numbers and the transcript, work out *why* the video works, and write the blueprint.

Numbers alone can't explain a video, and looking alone misses timing. Doing both, and tying each claim to a timestamp or a number, is what makes the blueprint trustworthy and reusable.

## Input

The user gives one or more links or file paths, optionally with `quick` or `deep` (when invoked as `/dissect`, they arrive as `$ARGUMENTS`). If there's no link, ask for one.
- One video → one blueprint.
- Several videos → one blueprint each, plus a formula file across them (see "Several videos").

## Step 1: Measure

Run this for one video at a time:

```bash
uv run <this skill's folder>/scripts/dissect.py "<link or file>" [--depth quick|standard|deep] [--out <folder>]
```

- Output goes to `./dissections/<date>-<platform>-<title>/` unless `--out` or `$DISSECT_OUT` sets a folder. The script prints a JSON summary with the folder, the sheets and a suggested blueprint path. Progress lines go to stderr.
- On first use, uv installs the Python packages (about a minute) and Whisper downloads a speech model: about 140 MB for quick, 480 MB for standard, 1.5 GB for deep. Tell the user it's a one-time wait.
- Depth: `quick` is for batches of 4+ videos (~36 frames, faster model). `standard` is the default (~72 frames). Use `deep` when someone wants maximum detail on a short video (~160 frames, best transcript, slow on long videos).
- **Long videos** (over ~20 minutes): ask whether to analyse all of it or only the opening (`--max-minutes N`). The opening is usually what people want to learn from. Transcription on a CPU takes roughly a tenth to a third of the video's running time.
- **Speech not in English**: Whisper detects the language. Pass `--lang hi` (or `es`, `pt`, etc.) if facts.md shows the wrong one.
- **No `uv`**: run `python3 <skill>/scripts/doctor.py`. It checks every requirement and prints install commands for the user's OS. Ask before installing anything.

If the download fails, the JSON gives the reason (`status`) and what to do (`fix`). `references/downloading.md` has platform notes. The common cases:
- **Exit 3, login needed** (common on Instagram, sometimes X, age-restricted YouTube). Ask before touching the user's browser cookies, for example: "This reel only downloads when you're logged in. May I let yt-dlp read your Chrome cookies for this one download?" Only after a clear yes, rerun with `--cookies-from-browser chrome` (or their browser). A keychain or password prompt may appear; that's normal.
- **Exit 2, anything else.** The script has already retried with browser impersonation and the newest yt-dlp. Suggest the user download the file with any downloader they trust and give you the path.

## Step 2: Watch

Read `facts.md` in full. It has every measurement, the shot list with what's said during each shot, the hook word by word, the transcript, and audience signals.

Then look at every sheet with the Read tool, in this order: `hook.jpg`, `shots*.jpg`, `most-replayed.jpg` (if present), `cover.jpg`.
- Every tile is labelled with its time.
- On the shot sheets, `S7` is the middle of shot 7, and `S7+` is the moment just after something changed inside shot 7: a caption or graphic popping on, a new card, a dissolve.
- `S7?` follows an *uncertain* change, which is either a real cut or just camera shake, smoke, a flash or fast motion.

For each tile, write down what you actually see:
- **Shot type and framing**: talking head, b-roll, screen recording, motion graphic, stock, meme or UGC; close-up, medium or wide; the camera angle.
- **On-screen text**: the words if legible, how the font looks, its colour, size and position, and how it animates (word-by-word captions, pop-ins, highlighted words).
- **Overlays**: emoji, arrows, circles, stickers, progress bars, logos, split screens.
- **Motion**: compare neighbouring tiles to spot zooms, pans, punch-ins and camera moves.
- **Look**: colour grade, lighting, set, wardrobe, props.
- **Transitions**: hard cut, punch-in, dissolve, whip, match cut. The facts show where the cuts and step changes fall.

Then line up the pictures with the words; facts.md lists what's said during each shot. What the editor puts on screen *while* each line is spoken is the heart of the format, and the part people can't copy by just watching.

`references/field-guide.md` has vocabulary for hooks, beats, shots, captions, retention devices and sound, plus rough benchmarks for the numbers. Skim it before writing.

**Don't re-measure.** The facts and sheets are built so you can write from them directly, and the value you add is interpretation. Cut detection can't always tell a fast camera move or a smoke puff from a cut, so facts.md gives the pace as a range with the uncertain moments listed. Report that range, or a best estimate after a glance at the `?` tiles; checking every change by eye costs a lot of time and changes the conclusions little. Pull extra frames with ffmpeg only to answer a specific question, such as what text is on screen at 0:42 or what happens in the first second. Roughly, a short video should take 10–15 minutes from link to blueprint, and a long one 20–30.

## Step 3: Interpret

Answer these from evidence, citing times or numbers:

1. What does the video promise, and when does it make that promise?
2. How does the hook hold attention in the first 1–3 seconds (picture, words, text, sound)?
3. What's the structure? Name the beats, where attention gets won back (re-hooks), the payoff, the call to action, and whether the ending loops.
4. What's the visual grammar? Look at the shot mix, how cuts and on-screen changes track the speech, and the caption style.
5. How is the voice delivered? Cover pace and how it shifts, pauses before reveals, emphasised words, pitch movement and sentence endings.
6. How does the sound help? Note the music bed's level and mood, effects on cuts, and any use of silence.
7. What do the audience signals add (most-replayed moments, top comments)?
8. What belongs only to this creator (face, name, catchphrases, running jokes, signature music), and what is the format anyone could reuse?

Mark anything you infer rather than see or measure as "(inferred)". The heuristics (music-like bed, sound effects on cuts, emphasis) are rough. When they disagree with what you see, trust the frames and say so.

## Step 4: Write the blueprint

Write `<slug>.blueprint.md` in the dissection folder (the script suggests the path), following `assets/blueprint-template.md` section by section. Aim for a blueprint that is:
- **Self-contained**: useful to someone who never saw the video or the sheets.
- **Concrete**: times, counts, colours, sizes, the closest free fonts, words per minute. "Captions: white Montserrat ExtraBold, key word in yellow, 2–3 words at a time, lower third" is useful; "bold captions" isn't.
- **Reusable**: templates with `{{SLOTS}}`, rules of thumb, and prompts that work for a new topic.

Keep quotes from the video short (a hook line or key phrase, under about 15 words each) and paraphrase the rest. The full transcript stays in `data/transcript.txt` for the user's own reference. That keeps the blueprint a format guide people can share, not a copy of someone's script.

Describe voices and looks as archetypes, such as "warm, quick, conversational female narrator, mid-20s". Never tell the reader to clone a real person's voice, face or name.

## Step 5: Wrap up

- The script has already deleted the downloaded video and scratch files, so a dissection takes up only a couple of MB. Pass `--keep-video` if the user wants the source kept.
- Reply with three things: a link to the blueprint, the three strongest reasons the video works (one line each, with evidence), and a 6–8 line format card. Then offer the natural next step: "Want me to write a script for your topic using this blueprint?"

## Several videos

Measure the videos one after another, not in parallel; transcription is CPU-heavy, and parallel runs can freeze a laptop. Use `quick` for 4 or more. Write a blueprint for each, then write `formula.md` next to the dissection folders. It covers:
- what stays the same across the videos (that's the real format)
- what varies, and what the best performer did differently (compare views per follower and likes per 1,000 views from the facts)
- one merged template built from the constants

## Ground rules

- The goal is to learn a format, not to copy a video. Don't help re-upload someone's video or pass off their script as new.
- Use browser cookies only after the user says yes, and only for that download.
- Keep heavy work sequential.
