---
name: dissect
description: Reverse-engineer any online video into a reusable, frame-accurate blueprint for making new videos in the same format, and score a rebuild against it. Works with YouTube (videos and Shorts), TikTok, Instagram Reels, X/Twitter, Vimeo, Reddit and any other site yt-dlp supports, or a local file. Finds every cut and transition in frames, measures camera and element moves, type sizes, palette, beat grid, sound cues and loudness, transcribes with word timings, profiles the voice, reads platform stats, draws frame sheets, then writes a blueprint with numbers for every move, a timeline JSON, sound and type plates, and ready-to-paste build prompts. Use it whenever someone shares a video link and wants it broken down, wants to know why it works or went viral, wants to copy its style, editing, hook or motion, says "make a video like this" or "recreate this", wants a creator's formula, or wants to check how close their render is to a reference, even if they never say "dissect".
license: MIT
compatibility: Needs internet access, ffmpeg, and uv (or Python 3.10+ with numpy, pillow, faster-whisper, praat-parselmouth, opencv-python-headless, librosa, soundfile and scipy). yt-dlp is fetched on demand if missing. On-screen text is read with Apple Vision on macOS, or tesseract elsewhere if installed. scripts/doctor.py checks everything and prints install commands.
metadata:
  version: "2.0.1"
---

# Dissect

Turn a reference video into a blueprint that a person or an AI can follow to make a *new* video in the same format, then check the new video against the reference. Three parts:

1. **Measure.** `scripts/dissect.py` downloads the video and measures it frame by frame: every cut and transition (type and length in frames), hidden cuts inside shots, camera and element motion with numbers, on-screen text with times, boxes and cap heights, palette and brightness over time, the beat grid, sound cues and loudness, the transcript with word timings and the voice. It draws four kinds of labelled frame sheets and writes `measured.md`, the measured blueprint sections ready to paste.
2. **Look and write.** You look at the sheets, read the numbers, work out *why* it works and *how each move is built*, and write the blueprint around the measured sections.
3. **Check.** `dissect.py check <render> --ref <folder>` scores a rebuild against the reference and prints concrete fixes in frames.

Rebuilds from v2 blueprints score far closer to the original than from v1 (62 → 82 out of 100 on the same brief), and the builders said the frames mattered most: one frame per shot and the transition strips. So look at the sheets, and ship them with the blueprint.

## Input

One or more links or file paths, optionally with `quick` or `deep` (as `/dissect`, they arrive as `$ARGUMENTS`). No link: ask for one.
- One video → one blueprint.
- Several videos → one blueprint each, plus a formula file across them (see "Several videos").
- "How close is my render to this?" → Step 6 (check).

## Step 1: Measure

```bash
uv run <this skill's folder>/scripts/dissect.py "<link or file>" [--depth quick|standard|deep] [--out <folder>]
```

- Output goes to `./dissections/<date>-<platform>-<title>/` unless `--out` or `$DISSECT_OUT` sets a folder. The script prints a JSON summary with the folder, `sheets_in_reading_order`, `measured`, `timeline` and the `draft` path to write. Progress goes to stderr. Run it in the background for anything over a minute and wait for it.
- First use: uv installs the Python packages (a few minutes, about 600 MB) and Whisper downloads a speech model (about 140 MB for quick, 480 MB for standard, 1.5 GB for deep). Tell the user it's a one-time wait.
- Depth: `quick` for batches of 4+ (~36 shot frames, 48 timeline tiles, base model, OCR every other sample). `standard` is the default. `deep` for maximum detail on a short video (more tiles, more transition strips, medium model).
- Time: a 40 s film takes about 3 minutes on a laptop (the sound grid and text reading are the slow parts). The frame-level timeline covers the first 10 minutes (`--timeline-minutes N`); `--no-timeline` skips it and gives the v1 output only.
- **Long videos** (over ~20 minutes): ask whether to analyse all of it or only the opening (`--max-minutes N`).
- **Speech not in English**: pass `--lang hi` (or `es`, `pt`, ...) if facts.md shows the wrong language.
- **No `uv`**: run `python3 <skill>/scripts/doctor.py`. It checks every requirement and prints install commands. Ask before installing anything.
- A harmless macOS warning about `AVFFrameReceiver` being implemented twice (two bundled copies of ffmpeg's device library) can be ignored.

If the download fails, the JSON gives the reason (`status`) and what to do (`fix`); `references/downloading.md` has platform notes.
- **Exit 3, login needed** (common on Instagram, sometimes X, age-restricted YouTube). Ask before touching browser cookies: "This reel only downloads when you're logged in. May I let yt-dlp read your Chrome cookies for this one download?" Only after a clear yes, rerun with `--cookies-from-browser chrome` (or their browser).
- **Exit 2, anything else.** The script already retried with browser impersonation and the newest yt-dlp. Suggest the user download the file another way and give you the path.

## Step 2: Read the numbers

1. `facts.md` in full: platform stats, pace, speech and voice, sound, audience signals, the shot list with what's said in each shot, the transcript, and a "Timeline" summary.
2. `measured.md` in full. Its first block, **Timing facts**, lists every cut, continuous boundary, change inside a shot, fade and brightness step in frames; every claim you write about timing must agree with it. **Aspect conversions** says what a centre crop to other aspects keeps. The other blocks (shots, sheets, camera, transitions, type, palette, brightness, speech, sound) go into the blueprint verbatim.
3. `data/timeline.json` when you need a detail the tables leave out (per-shot speed curves, element tracks, all text reads, every beat). Its `notes` say what is uncertain.

## Step 3: Look at every sheet, in this order

Read them with the Read tool. Every tile is labelled with its shot, time and frame (`S3 0:04.2 f126`).

1. `sheets/hook.jpg`: the first 3 s at 0.25 s steps (deep: 5 s at 6 per second). What the viewer sees before deciding to stay.
2. `sheets/shots*.jpg`: **one frame per shot**, from its middle; `S7+` = just after a change inside shot 7; `S7?` = after an uncertain change (a cut, or just shake, smoke, a flash). This is the film's picture vocabulary: framing, type, materials, layout.
3. `sheets/transitions-*.jpg`: 14 consecutive frames around each transition and change inside a shot, the change frame outlined, titled with the measured type and length. This is how the moves are actually built: the blur on a whip, the frames of a dissolve, a mask, a dip to black, a morph. Trust what you see here over a low-confidence label.
4. `sheets/timeline-*.jpg`: the whole film at fixed steps. Use it for text animation (word-by-word, karaoke highlight, typing, counters), element moves between cuts, and anything after the hook the shot sheet misses.
5. `sheets/most-replayed.jpg` (YouTube, if present) and `sheets/cover.jpg`.

For each shot write down what you see: shot type and framing; on-screen text, its look, position and how it animates; overlays; motion between tiles; look (grade, light, set, materials); and how it hands off to the next shot. Line the pictures up with the words (facts.md and the Speech timing block): what goes on screen *while* each line is spoken is the heart of the format.

**Don't re-measure.** The numbers are done; your value is interpretation and construction notes. Pull extra frames with ffmpeg only for a specific question the sheets can't answer. A short film should take 15–25 minutes from link to blueprint.

`references/field-guide.md` has vocabulary for hooks, beats, shots, captions, retention devices and sound, plus benchmarks. Skim it before writing.

## Step 4: Interpret

Answer from evidence (times, frames, numbers):

1. What does the video promise, and when?
2. How does the hook hold attention in the first 1–3 s (picture, words, text, sound)?
3. Structure: the beats, re-hooks, payoff, call to action, whether it loops.
4. What drives it: **narration-led** (words cover most of it: cuts on phrases, captions follow the words), **music-led** (only when plate 8 says the pulse is steady *and* the cuts follow the beat) or **picture-led** (moves and cues set the timing). This drives plates 8, 10 and 15 and the recipe.
5. The signature move: what it does in the film (not the biggest number), and how it is built: layers, masks, camera trick, easing, frames.
6. Voice delivery, sound design, audience signals.
7. What belongs only to this creator or brand (face, name, palette, logo, music) vs. the format anyone can reuse.

Mark readings "(inferred)". When a heuristic disagrees with the frames, trust the frames, keep the number, and say both.

## Step 5: Write the blueprint

1. Copy `assets/blueprint-template.md` to the `draft` path the script printed (`<slug>.blueprint.draft.md` in the dissection folder) and fill it plate by plate. Its writer notes are the rules; the key ones:
   - Keep every `{{measured:KEY}}` line once, where the template puts it. Don't retype measured tables; cite them ("the ×1.39 push in shot 4, plate 5").
   - Numbers for every move: start and end frames, scale and travel, keyframes for big moves, easing as cubic-bezier, cadence, blur.
   - Plate 6: a **type spec** at or above the phone minimum, **layout boxes** with a source and confidence, a **material card** with CSS values (or a grade and shoot card for live action).
   - Plate 8: an **original music bed** (tempo, key, instruments, sections keyed to shots, cue frames, a one-line music prompt; never the reference's track).
   - Plate 11: **two routes** (Minimum: Claude alone in code; Full: the real toolkit with file handoffs) and the asset-honesty rule.
   - Plate 13: **How it's built**, signature move first, every change inside a shot with its method and length in frames.
   - Plate 15: slots, what the film avoids, a brand-distance warning, aspect and length adaptation.
   - Brand: the reference's colours, fonts, logo and copy are evidence, not the format. Use slots (`{{PRODUCT}} {{BRAND_FONT}} {{BRAND_BG}} {{BRAND_ACCENTS}} {{LOGO}} {{CTA_URL}}` for a product film; `{{TOPIC}} {{HOST}} {{CHANNEL_LOOK}} {{CTA}}` for a creator video). Match the reference's light, not its hues.
2. Write `plan.json` next to it:

```json
{
  "why_it_works": "2-3 sentences citing measured times, frames or values",
  "scales": [{"shot": 1, "scales": "fixed | scales | optional", "why": "max 8 words"}],
  "avoid": ["4-8 things the film deliberately leaves out"],
  "routes": {"minimum": {"stack": ["Remotion"], "builds": "...", "gives_up": "..."},
             "full": {"stack": ["Blender", "Remotion"], "builds": "...", "handoffs": ["Blender → Remotion: PNG sequence with alpha, 1920×1080, 30 fps"]}},
  "recipe_prompt": "150-230 words, rules below"
}
```

   `scales`, one per measured shot: `fixed` keeps its length at any film length (hook, beat-locked sting, end-card hold), `scales` stretches with it, `optional` goes in the 15 s cut. The signature move is never optional; the fixed shots must add up to well under 15 s.

   **Recipe prompt**: our own runnable prompt for Claude Code or any coding agent to make a NEW film in this format. It opens "Make a <length> s <aspect> film for {{PRODUCT}} in the format of <format name>. Brand: font {{BRAND_FONT}}, background {{BRAND_BG}}, accents {{BRAND_ACCENTS}}, logo {{LOGO}}, call to action {{CTA_URL}}." It tells the agent to load `<slug>.timeline.json` first and build to its shot frames, camera numbers, type boxes and cue frames (beat grid only if steady); output 1080p in the film's aspect (1920×1080, 1080×1920, 1080×1080) at the reference's fps, H.264 MP4; the render stack from the toolkit with the Minimum route as fallback; the beats with times and shot numbers, the signature move in a few numbers, type at or above the phone minimum, the sound and the loudness target; a narration-led film's word count, pace and caption rule; a text call to action (no QR codes or store badges unless asked); asset honesty; never the reference's hex colours, fonts, logo or music; and to finish by running `dissect.py check`.
3. Assemble and fix:

```bash
uv run <skill>/scripts/dissect.py assemble <dissection folder>
```

   It writes `<slug>.blueprint.md` (measured blocks pasted in, scales filled, length-cut and aspect tables added, recipe appended) and `<slug>.timeline.json` (the timeline plus your plan), and prints warnings: missing parts, recipe rule breaks, and **contradictions** between your prose and the measurements. Fix every `CONTRADICTION` and `FIX` warning in the draft and assemble again.

Keep quotes from the video under about 15 words each and paraphrase the rest; the full transcript stays in `data/` for the user's own reference. Describe voices and looks as archetypes; never tell the reader to clone a real person's voice, face or name.

## Step 6: Check a render

When the user (or you) has built a video from the blueprint:

```bash
uv run <skill>/scripts/dissect.py check <render.mp4> --ref <dissection folder> [--target x|youtube|tiktok|web] [--json]
```

It measures the render (cuts, brightness curve, motion, loudness, format) and scores it out of 100 against the reference's timeline: cuts 35, brightness curve 20, motion 15, pace 10, loudness 10, format 10. The fixes are in the render's own frames ("Cut 2 lands 4 f late at f131; move it to f127"). Work through the top fixes, re-render, check again. As a guide, the original film scores about 93–95 against itself, a faithful rebuild 80+, a loose one around 60. Different on purpose (another aspect, a planned 15 s cut) is fine: say so rather than chase the score.

## Step 7: Wrap up

- The script deletes the downloaded video (`--keep-video` keeps it), so a dissection is a few MB.
- Reply with: a link to the blueprint and the timeline JSON, the three strongest reasons the video works (one line each, with evidence), the signature move in one line with its numbers, and a 6–8 line format card. Offer the next step: "Want me to build one for your product with the recipe prompt, and check it against the original?"

## Several videos

Measure one after another, never in parallel (transcription and the timeline pass are CPU-heavy). Use `quick` for 4 or more. Write a blueprint for each, then `formula.md` next to the folders: what stays the same across them (the real format), what varies and what the best performer did differently (views per follower, likes per 1,000 views), and one merged template.

## Ground rules

- The goal is to learn a format, not to copy a video. Don't help re-upload someone's video or pass off their script, brand or likeness as new.
- Use browser cookies only after the user says yes, and only for that download.
- Keep heavy work sequential.
