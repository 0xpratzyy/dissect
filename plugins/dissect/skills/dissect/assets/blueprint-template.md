# {{Format name}}: video blueprint

> Reverse-engineered from **"{{title}}"** by {{creator}} ({{platform}}, {{length}} s, {{views}} views, posted {{date}}). Measured frame by frame {{today}}.
> Use it to make a **new** video in the same format with your own product or topic, brand, words and music. **(measured)** = read off the video itself; **(inferred)** = our reading of it.

<!-- WRITER NOTES. `dissect.py assemble` strips every HTML comment, so leave them in the draft.
- Save your draft as <slug>.blueprint.draft.md in the dissection folder, and plan.json next to it (shape in SKILL.md).
  `dissect.py assemble <folder>` writes the final <slug>.blueprint.md and <slug>.timeline.json and lists problems.
- Plates are the "## N. Title" sections, numbered 1-15. Inside a plate use "###" or bold, never "##".
- {{measured:KEY}} lines are placeholders. Keep each exactly once, on its own line, where this template puts it. The
  assembler replaces it with the block of the same name from measured.md, verbatim. Never retype, round or "correct" a
  measured number anywhere else; cite it ("the ×1.39 push in shot 4, plate 5").
- {{scales:N}} cells in the shot table are filled from plan.json. Don't touch them.
- Your additions are labelled "(inferred)" in their heading or first words, and must agree with the measured blocks
  (measured.md's "Timing facts" block lists every cut, fade and brightness step: check claims against it). Where the
  frames disagree with a measurement, keep the number and add one line: "(inferred) The frames suggest ...; the
  measurement says ...".
- Say each beat once. Plate 4 owns the beats; plates 10-13 refer to them by number ("beat 2", "shots 3-5").
- Slots: the reference's colours, fonts, logo, mascot, face and copy are evidence, not the format. A product or brand
  film uses {{PRODUCT}} {{BRAND_FONT}} {{BRAND_BG}} {{BRAND_ACCENTS}} {{LOGO}} {{CTA_URL}}; a creator video uses
  {{TOPIC}} {{HOST}} {{CHANNEL_LOOK}} {{CTA}}; add film-specific slots ({{FEATURE}}, {{UI_ASSETS}}) when needed and
  declare every slot you use in plate 15. Name reference values only as "the reference uses ...".
- Colour rule: match the reference's LIGHT (its brightness curve, contrast and where highlights sit), never its HUES.
  Write "follow its light", not "follow its colour".
- Concrete beats vague: frames, px at 1080p, %, ms, cubic-bezier values. Quotes from the video stay under 15 words.
  Describe people as archetypes; never tell anyone to imitate a real person's face, voice or name.
- Known traps (from a real rebuild): count every distinct on-screen word or card from the Type table AND the timeline
  sheet (one reference had three benefit words, not two); describe shapes from the transition strips, not from a label
  (a "whip" with confidence under 0.6 may be a blur swap); write down text animation states you see (word-by-word
  highlight, karaoke scroll, typing); give every big move (×1.2 scale or 10% travel and up) as keyframes, not a total
  plus an ease name; build to the measured frame count, not duration × fps; a "fade from black" on shot 1 can be a dark
  opening where elements appear: check the hook sheet before calling it a fade.
-->

## How to use this blueprint

Plates are the numbered sections below: "plate 5" means section 5, Shot grammar.

1. Fill the slots once: `{{PRODUCT}}` · `{{BRAND_FONT}}` · `{{BRAND_BG}}` · `{{BRAND_ACCENTS}}` · `{{LOGO}}` · `{{CTA_URL}}` <!-- or the creator slots, plus this film's own -->.
2. Load `{{slug}}.timeline.json` next to this file. It has every shot's frames, camera numbers, text boxes, palette, the beat grid and the cue sheet; the (measured) tables below are rendered from it. Look at the frame sheets in `sheets/` in this order: hook, shots, transitions, timeline.
3. Pick a route in plate 11, build in plate 13's order, check against plate 14, then score your render: `uv run <dissect>/scripts/dissect.py check your-render.mp4 --ref <this folder>`. Plate 15 adapts the aspect and the length.

**Your video:** `{{PRODUCT}}` or topic ____ · audience ____ · platform ____ · {{aspect}} at {{deliverable px}}, {{fps}} fps · about {{length}} s (15, 30 and 60 s cuts in plate 15)

<!-- Plate 2's Script row says what drives the film: narration-led (cuts on phrases, captions follow the words),
music-led (cuts on the beat, only when plate 8 says the pulse is steady and the cuts follow it) or picture-led.
Plates 8, 10 and 15 and the recipe follow it. -->

## 1. The format in one breath

{{2-3 sentences: what the video is, what it promises, and the engine that makes it work.}}

**Why it works**
- {{reason with evidence: a time, a frame number or a measured value}}
- {{reason with evidence}}
- {{reason with evidence}}

## 2. Format card

| | |
|---|---|
| Platform · aspect · length | {{platform · aspect · seconds and frames · source px · fps}} |
| Pace | {{shots, transitions, median shot length, cuts/min (measured)}} |
| Script | {{narration-led, music-led or picture-led · words, wpm; or "wordless"}} |
| Voice | {{archetype · register · energy, or none}} |
| On screen | {{shot mix, e.g. 60% UI animation · 25% 3D product · 15% end card}} |
| Type | {{levels and sizes from plate 6 (cap px at 1080p), case, one line on animation}} |
| Sound | {{bed and mood · tempo if steady, or "no steady pulse" · cues · measured loudness}} |
| Structure | {{number of beats · when the payoff and the call to action land}} |
| Tools | {{what a builder needs by role (code, 3D, footage, screen capture, voice, music) · Claude alone: yes / partly / no}} |

## 3. The hook, frame by frame (first 3 seconds)

{{measured:hook_sheet}}

| Time · frame | Picture | On-screen text | Words and sound | What it does |
|---|---|---|---|---|
| 0.00 · f0 | | | | |

**Hook formula:** {{type}}: `{{one-line template with {{SLOTS}}}}`
**Why it holds:** {{1-3 sentences}}

## 4. Structure (beat sheet)

<!-- The one place the beats are described. "Shots" ties each beat to the measured shot table below. -->

| # | Time (s) · frames | Shots | Beat | What happens | Technique |
|---|---|---|---|---|---|
| 1 | | | Hook | | |

**Retention map:** re-hooks at {{times}} · payoff at {{time}} · call to action at {{time}} · ending {{loops / hard stop / holds}}.

{{measured:shots}}

{{measured:sheets}}

## 5. Shot grammar

**Mix:** {{share of each shot type}}
**Rhythm:** {{what the measured tables show, in one or two sentences: where cuts cluster, how long the holds are}}

**Rules this editor follows** (inferred):
- {{rule tied to evidence}}

{{measured:camera}}

{{measured:transitions}}

<details><summary>What each shot shows (inferred from the sheets)</summary>

| # | Type | Framing | What's on screen | Text / overlay and how it animates |
|---|---|---|---|---|

</details>

## 6. Visual style guide

- **Look:** {{grade, light, materials, depth; the reference's palette by role (background, surface, accent) as evidence only}}
- **Graphics and overlays:** {{style, when they appear, how they move}}
- **Captions and text animation:** {{word-by-word, karaoke highlight, typing, scroll: what the timeline sheet shows, with frame counts}}
- **Transitions:** {{in words; the frame counts are in plate 5}}
- **Branding and end card:** {{what the end card holds and for how long; {{LOGO}} and {{CTA_URL}} as plain text}}

{{measured:type}}

### Type spec (inferred)

<!-- One row per level of the measured type scale. Cap height vs font size explicit (state the ratio you assume);
tracking in em; case; colour pair as roles ({{BRAND_BG}} / text colour); how it animates (frames and easing). Words
the viewer must read stay at or above the phone minimum above, even where the reference went smaller. Closest free
font only as a fallback for {{BRAND_FONT}}. -->

| Level | Cap px at 1080p | Font size | Weight · case · tracking | Colour pair | Motion |
|---|---|---|---|---|---|

{{measured:palette}}

{{measured:brightness}}

### Layout boxes (inferred)

<!-- Key elements as % of the frame, x · y · w · h from the top-left. Measured text boxes and element regions first
(high: copy plates 5 and 6); then a named sheet tile (medium: a tile is the whole frame, so an edge at three quarters
of the tile is 75%); estimates last (low). Drop any box you can't support. Always include the end card. -->

| Element | Shots | Box x · y · w · h (%) | Source · confidence | Notes |
|---|---|---|---|---|

**End card:** {{background, {{LOGO}} box, CTA text box and size, hold in frames (it ends where the measured picture ends or fades)}}

### Material card (inferred)

<!-- The signature surfaces as copy-paste values. CSS for a code build (colours as var(--brand-bg),
var(--brand-accent) ...); Blender values where the look is dimensional (Principled BSDF, world, EEVEE/Cycles, lens mm).
Every 3D value gets its 2D equivalent in px at 1080p: depth of field = blur px outside a focus box (x · y · w · h %,
use plate 5's measured focus), bloom = glow radius and opacity, falloff = vignette. A flat 2D film: one line saying
no 3D is needed. Live action, footage-led or a talking head: call it "Grade and shoot card (inferred)": lens mm,
camera height and angle, depth of field, key and fill light and colour temperature, the grade, the shooting
resolution (4K when the edit punches in), audio to record, and how many clips of what. -->

```css
/* {{surface name}} */
```

## 7. Voice and delivery

**Measured:** {{wpm and how it moves, pauses, pitch movement, emphasis, sentence endings (facts.md); or "no speech"}}
**Voice direction:** {{one paragraph for a voice actor or a TTS voice designer, or "no voice: the picture carries it"}}

{{measured:speech}}

## 8. Sound design

- **Music:** {{mood, instruments, where it enters, drops and lands, keyed to shots}}
- **Sound effects:** {{tie them to the cue sheet's shot numbers and frames}}
- **Silence and space:** {{drops, held air before reveals}}

{{measured:sound}}

### Original music bed (inferred)

<!-- An ORIGINAL bed that fits this picture: tempo (the measured BPM when the pulse is steady; otherwise choose one
and say why), key, instruments, a section plan keyed to shot numbers, which cues land on which frames, and a one-line
prompt for a music model. Never name the reference's track or artist. -->

## 9. Packaging

- **Title:** {{formula and template}}
- **Cover / thumbnail:** {{composition, text, colours}}
- **Post copy:** {{pattern}}
- **Audience signals:** {{views per follower, likes per 1k views, most-replayed moments, top comments; or what was not available}}

## 10. Script template

<!-- Words only: spoken lines and on-screen text per beat, with word budgets (seconds × the reference's words per
second from plate 7, or the measured on-screen time). Refer to beats by number. -->

```text
[beat 1 · 0.0–2.3 s · f0–68 · shot 1 · on screen ≤ 4 words] {{...}}
```

## 11. Production plan

### Two routes (inferred)

**Minimum: Claude alone.** {{the simplest single stack that reaches ~80% of the look (for UI, device and dark-macro
films: Remotion or HTML/CSS with CSS 3D), what it builds, what it simplifies, what it gives up, and the inputs the user
must supply (voice, footage, screen recordings)}}

**Full: with the toolkit.** {{each tool by role, in build order, with the handoffs: which file passes between tools,
at what size and fps (camera keyframes as JSON, PNG sequences with alpha, a WAV with a click track)}}

**Asset honesty:** real product captures or your own UI rebuilt in code; where no image exists, a type tile says it
plainly. Never generated product UI, invented screenshots, fake metrics, fake reviews or fake results. {{one film-specific line}}

## 12. Prompts

<!-- Only prompts this film needs: script; visual prompts only for shots an image or video model makes; a caption
spec when there is on-screen text; a music prompt. Use the slots. -->

**Script prompt**
```text
{{...}}
```

## 13. Edit recipe

### How it's built (inferred)

<!-- The construction notes a builder would otherwise guess, signature move first: how it is actually made (camera
trick, layer order, mask, morph), with start and end frames, scale or travel from plate 5, a keyframe list for every
big move (frame → scale / x / y, at least start, the speed peak and end), easing as cubic-bezier or spring values, the
measured cadence (ones, twos), motion blur method and amount, and the gotchas. Every change inside a shot, hand-off and
"no visible cut" boundary in the Timing facts gets its method and its length in frames. -->

### Build order

1. {{step; refer to beats and shots by number}}

## 14. QA checklist

- [ ] The render has the measured frame count (plate 4) at the reference's fps, or scales every frame number with it
- [ ] Cuts land on the measured frames (plate 4), within ±1 frame
- [ ] Every word the viewer must read is at or above the phone minimum in plate 6
- [ ] {{signature-move check with a number}}
- [ ] {{sound check: cues on the cue-sheet frames; the beat grid only if the pulse is steady}}
- [ ] Loudness at the target for the platform you post to (plate 8), true peak ≤ −1 dBTP
- [ ] Nothing of the reference left: colours, fonts, logo, mascot, face, copy and music are yours
- [ ] `dissect.py check` scores 80 or more, and its top fixes are done or deliberately skipped

## 15. Make it yours

**Slots**

| Slot | Replaces in the reference | Notes |
|---|---|---|
| `{{PRODUCT}}` | | |
| `{{BRAND_FONT}}` | | |
| `{{BRAND_BG}}` | | |
| `{{BRAND_ACCENTS}}` | | |
| `{{LOGO}}` | | |
| `{{CTA_URL}}` | | |

**What this film avoids** (keep these out too):
- {{what is deliberately absent and why it matters, e.g. no voiceover, no stock footage, no feature list}}

**Brand-distance warning:** {{the combination that would make a copy read as the original creator or brand, and the one change that fixes it; or "low risk" with one line why}}

**Adapt the aspect** (one line for each of 16:9, 9:16 and 1:1 that isn't the source's aspect; the assembler appends what a centre crop keeps)
- **16:9:** {{only when the source isn't 16:9}}
- **9:16:** {{what moves where, type sizes against the 9:16 phone minimum, the platform UI safe zones}}
- **1:1:** {{crop or re-layout}}

**Adapt the length** (uses the `scales` flags in plate 4; the assembler appends the computed frame counts, so no durations here)
- **15 s:** {{which beats stay, which optional shots go}}
- **30 s:** {{...}}
- **60 s:** {{where the extra time goes: new beats, not longer holds}}

**The creator's signature (don't copy):** {{face, name, mascot, catchphrases, signature music, logo and palette}}
**The transferable format (do copy):** {{structure, pacing, devices, the signature move's construction}}

---
<sub>Source: {{link}} · measured with dissect 2.0 (timeline v2.1) on {{date}} · (measured) = read off the video · (inferred) = our reading</sub>
