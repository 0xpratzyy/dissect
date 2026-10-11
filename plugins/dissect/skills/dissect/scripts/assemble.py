"""assemble.py - turn the writer's blueprint draft into the final blueprint, and check it.

    uv run dissect.py assemble <dissection folder>

Reads, in the folder: <slug>.blueprint.draft.md (Claude's draft, from assets/blueprint-template.md), plan.json (the
writer's scales flags, recipe prompt, routes and avoid list), measured.md and data/timeline.json (from dissect.py).
Writes <slug>.blueprint.md (every {{measured:KEY}} replaced verbatim by its measured block, the shot table's
{{scales:N}} filled, computed length-cut and aspect tables in plate 15, the recipe prompt at the end, no HTML comments)
and <slug>.timeline.json (the measured timeline plus the plan: what a builder loads alongside the markdown).
Prints warnings: missing plates or parts, recipe rule breaks, and prose that contradicts the measurements.
Ported from the Refs blueprint-v2 assembler. Pure standard library.
"""
import json, os, re, sys

from plates import parse_blocks, Film

PLATE_OF = {'hook_sheet': 3, 'shots': 4, 'sheets': 4, 'camera': 5, 'transitions': 5, 'type': 6, 'palette': 6,
            'brightness': 6, 'speech': 7, 'sound': 8}
SLOTS = ['{{PRODUCT}}', '{{BRAND_FONT}}', '{{BRAND_BG}}', '{{BRAND_ACCENTS}}', '{{LOGO}}', '{{CTA_URL}}']
FLAGS = ('fixed', 'scales', 'optional')


def load(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def strip_comments(md):
    return re.sub(r'<!--.*?-->\n?', '', md, flags=re.S)


def plate_spans(md):
    """[(n, start, end)] for every '## N.' plate in md."""
    heads = [(m.start(), m.group(1)) for m in re.finditer(r'(?m)^## (?:(\d+)\.|\S)', md)]
    spans = []
    for i, (pos, n) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(md)
        if n:
            spans.append((int(n), pos, end))
    return spans


def insert_at_plate_end(md, n, text):
    for k, a, b in plate_spans(md):
        if k == n:
            body = md[a:b].rstrip()
            # keep a trailing footer rule (---) after plate 15 below the insert
            foot = ''
            m = re.search(r'\n---\n<sub>.*\Z', body, re.S)
            if m:
                foot, body = body[m.start():], body[:m.start()].rstrip()
            return md[:a] + body + '\n\n' + text.strip() + '\n' + (foot + '\n' if foot else '') + '\n' + md[b:].lstrip('\n')
    return md


def length_table(film, flags):
    """Shot lengths for 15, 30 and 60 s cuts, computed from the scales flags (frames at the reference fps)."""
    S = film.shots
    if not S or len(S) > 25 or not film.dur:
        return None
    fps = film.fps
    dur = {s['n']: (s.get('dur') or 0) for s in S}
    total = sum(dur.values())
    fl = {s['n']: flags.get(s['n'], 'scales') for s in S}
    F = sum(d for n, d in dur.items() if fl[n] == 'fixed')
    O = sum(d for n, d in dur.items() if fl[n] == 'optional')
    Sc = sum(d for n, d in dur.items() if fl[n] == 'scales')
    cols, notes = {}, []
    MAXK = 1.6   # beyond this a stretched shot drags: the rest of a longer cut has to be new beats
    extra = {}
    for T in (15, 30, 60):
        if abs(T - total) < 1.0 / fps:
            cols[T] = {n: d for n, d in dur.items()}
            notes.append(f'{T} s is the reference length: keep the measured lengths')
            continue
        dropped = set()
        if T < total:
            # optional shots go first, longest first, as many as the cut needs; then the scaling shots share the rest
            for n_ in sorted((n for n in dur if fl[n] == 'optional'), key=lambda n: -dur[n]):
                if F + Sc + sum(dur[n] for n in dur if fl[n] == 'optional' and n not in dropped) <= T:
                    break
                dropped.add(n_)
            Ok = sum(dur[n] for n in dur if fl[n] == 'optional' and n not in dropped)
            k = min(1.0, (T - F - Ok) / Sc) if Sc else None
            if k is None or k <= 0:
                cols[T] = None
                notes.append(f'{T} s: the fixed shots alone run {F:.1f} s: cut a beat first')
                continue
            if k < 0.35:
                notes.append(f'{T} s: scaling shots shrink to ×{k:.2f}, too fast: drop a beat instead')
        else:
            k = min(MAXK, (T - F - O) / Sc) if Sc else 1.0
            got = F + O + Sc * k
            if T - got > 0.5:
                extra[T] = T - got
                notes.append(f'{T} s: stretching tops out at ×{MAXK} ({got:.1f} s); add {T - got:.1f} s of new beats '
                             f'(more of the scaling beats\' pattern with new content), never a longer hold on the fixed shots')
        cols[T] = {n: (d if fl[n] == 'fixed' else (None if n in dropped else d) if fl[n] == 'optional' else d * k) for n, d in dur.items()}
    if all(v is None for v in cols.values()):
        return None
    head = ['#', 'flag', f'reference (f @{film.fpsl})'] + [f'{T} s cut' for T in (15, 30, 60)]
    rows = []
    for s in S:
        n = s['n']
        r = [n, fl[n], f"{int(round(dur[n] * fps))}"]
        for T in (15, 30, 60):
            c = cols[T]
            r.append('–' if c is None else 'drop' if c[n] is None else str(int(round(c[n] * fps))))
        rows.append(r)
    tot = ['total', '', f'{int(round(total * fps))} f · {total:.1f} s']
    for T in (15, 30, 60):
        c = cols[T]
        if c is None:
            tot.append('–')
        else:
            t = sum(v for v in c.values() if v is not None)
            tot.append(f'{int(round(t * fps))} f · {t:.1f} s' + (f' + {extra[T]:.1f} s new beats' if T in extra else ''))
    rows.append(tot)
    L = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---' for _ in head) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    out = (f'**Length cuts (computed from the scales flags):** frames per shot at {film.fpsl} fps; fixed shots keep their length, '
           f'optional shots go first when the cut is shorter, the scaling shots share what is left. Snap each cut to the '
           f'nearest beat when plate 8 has a steady pulse.\n\n' + '\n'.join(L))
    if notes:
        out += '\n\n' + ' · '.join(notes) + '.'
    return out


def check_recipe(rp, film, tl, warn):
    if not rp:
        warn.append('recipe: missing')
        return
    words = len(rp.split())
    if words < 120 or words > 360:
        warn.append(f'recipe: {words} words (want 150-300)')
    for s in ([] if '{{TOPIC}}' in rp else SLOTS):
        if s not in rp:
            warn.append(f'recipe: no {s}')
    if 'timeline.json' not in rp:
        warn.append('recipe: does not say to load the timeline JSON (<slug>.timeline.json)')
    res = f'{film.W}×{film.H}'
    if res not in rp and f'{film.W}x{film.H}' not in rp:
        warn.append(f'recipe: does not name the {res} deliverable')
    for bad in re.findall(r'\b(?:1280|1440|854|640)\s*[×x]\s*(?:720|480|360)\b', rp):
        warn.append(f'recipe: names a sub-1080p size "{bad}"')
    low = rp.lower()
    for term in ('qr', 'app store', 'google play', 'badge'):
        for m in re.finditer(r'\b' + re.escape(term), low):
            ctx = low[max(0, m.start() - 60):m.start()]
            if not re.search(r'\b(no|not|never|without|unless|avoid|don\'t|nor|omit|omits|leave out|skip|exclude|drop)\b', ctx):
                warn.append(f'recipe: mentions "{term}" without ruling it out')
                break
    hexes = set(h.upper() for h in re.findall(r'#[0-9A-Fa-f]{6}\b', rp))
    if hexes:
        warn.append('recipe: hard-codes colours ' + ', '.join(sorted(hexes)))
    tk = None
    return words


def check_numbers(text, tl, film, warn):
    """Soft checks that the inferred text doesn't contradict the measured numbers."""
    a = (tl.get('audio') or {})
    t = a.get('tempo') or {}
    bpm, conf = t.get('bpm'), t.get('confidence')
    if bpm and conf is not None and conf >= 0.45:
        for m in re.finditer(r'(\d{2,3}(?:\.\d+)?)\s*BPM', text):
            v = float(m.group(1))
            if not any(abs(v - x) <= max(1.5, 0.02 * x) for x in (bpm, bpm / 2, bpm * 2)):
                ctx = text[max(0, m.start() - 80):m.start()].lower()
                if 'original' in ctx or 'your' in ctx or 'new' in ctx or 'choose' in ctx or 'pick' in ctx:
                    continue
                warn.append(f'numbers: inferred text says {v:g} BPM; measured {bpm:g} BPM')
    elif bpm:
        # only statements that give the REFERENCE a tempo ("the reference's pulse is 84 BPM"), not tempos chosen for a new bed
        about_ref = (r"(?i)\b(?:reference|source|original film)(?:'s)?\s+(?:music\s+|track\s+|bed\s+)?(?:tempo|pulse|beat|bpm)\b[^.\n]{0,40}?"
                     r"(\d{2,3}(?:\.\d+)?)\s*BPM|(\d{2,3}(?:\.\d+)?)\s*BPM\s+(?:in|of)\s+the\s+(?:reference|source)")
        for m in re.finditer(about_ref, text):
            ctx = text[max(0, m.start() - 60):m.end() + 40].lower()
            if not re.search(r'no (reliable |clear )?steady|not steady|no (reliable |clear )?pulse|best guess|confidence|weak', ctx):
                warn.append(f'numbers: a {m.group(1) or m.group(2)} BPM claim about the reference, but the measured pulse is not steady')
    lufs = (a.get('loudness') or {}).get('integrated_lufs')
    ok = {-14, -16, -23, -24, -10, -11, -12, -13, -9, -15, -18}
    if lufs is not None:
        ok.add(round(lufs))
    for m in re.finditer(r'[−-]\s?(\d{1,2}(?:\.\d)?)\s*LUFS', text):
        v = -float(m.group(1))
        if lufs is not None and abs(v - lufs) < 0.6:
            continue
        if round(v) not in ok:
            warn.append(f'numbers: inferred text says {v:g} LUFS; the reference measured {lufs} and the targets are −14/−16/−23')
    seen = set()
    for m in re.finditer(r'(?<![\d.])(\d{2,3}(?:\.\d+)?)\s*fps\b', text):
        v = float(m.group(1))
        ctx = text[max(0, m.start() - 50):m.end() + 30].lower()
        if re.search(r'unique|new pictures|cadence|on (?:ones|twos|threes)|avoid|not |never|don\'t|hold', ctx):
            continue   # a cadence ("about 20 new pictures a second") or a warning, not a delivery frame rate
        if not any(abs(v - x) < 0.05 for x in (23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, film.fps)) and v not in seen:
            seen.add(v)
            warn.append(f'numbers: {v:g} fps mentioned; the reference runs at {film.fpsl} fps')


def contradictions(text, film):
    """Inferred statements about holds, fades and cuts that the measurements contradict (the builder of REF-3279 was
    told to hold the end card through frame 900 while the picture fades from about frame 818)."""
    out = []
    fps = film.fps
    nfr = film.nfr
    fo = film.fade_out()
    ff = int(round(fo[0] * fps)) if fo else None
    tol = max(2, int(round(0.25 * fps)))
    bounds = {s['start_frame']: (s.get('transition_in') or {}).get('type') for s in film.shots[1:]}
    low = text.replace('’', "'")
    def negated(m):   # "do not hold to the end", "never cut at", "no fade": a rule, not a claim
        return bool(re.search(r"(?i)\b(?:not|no|never|don't|avoid|without|instead of|rather than)\b[^.\n]{0,40}$", low[max(0, m.start() - 60):m.start() + 1]))
    def frame_of(num_, unit):
        return int(round(float(num_) * fps)) if unit and unit.strip().startswith('s') else int(num_)
    num_re = r'(?:frame\s*|f)(\d{1,5})\b|(\d{1,3}(?:\.\d+)?)\s*(s)\b'
    # 1. holds that run past the measured fade
    if fo:
        for m in re.finditer(r'(?i)\bhold(?:s|ing)?\b[^.\n|]{0,90}?\b(?:through|until|till|to)\s+(?:the\s+)?(?:' + num_re + r'|(end|last frame|final frame))', low):
            if negated(m):
                continue
            if m.group(4):
                out.append(f'says something holds to the {m.group(4)}, but the picture fades to black from about frame {ff} (plate 6)')
                continue
            n = int(m.group(1)) if m.group(1) else frame_of(m.group(2), m.group(3))
            if n > ff + tol:
                out.append(f'says something holds through frame {n}, but the picture fades to black from about frame {ff} (plate 6)')
    # 2. fades to black at the wrong time, or that aren't there
    for m in re.finditer(r'(?i)\bfade(?:s|d)?\s+(?:to\s+black|out)\b[^.\n|]{0,50}?\b(?:from|at|starting at|begins at|after)\s+(?:about\s+)?(?:' + num_re + r')', low):
        if negated(m):
            continue
        n = int(m.group(1)) if m.group(1) else frame_of(m.group(2), m.group(3))
        if fo and abs(n - ff) > int(round(0.5 * fps)):
            out.append(f'puts the fade to black at frame {n}; it is measured from about frame {ff} (plate 6)')
        elif not fo and n >= (film.shots[-1]['start_frame'] if film.shots else 0):
            out.append(f'describes a fade to black at frame {n}, but the measurements show no fade at the end')
    # 3. cuts where there is none, or none where there are
    maybe = set(film.possible_cuts())
    for s_ in film.shots:   # v2.1: cuts v1 missed (and other hidden changes) are listed inside their shots
        for c_ in s_.get('internal_changes') or []:
            if isinstance(c_, dict) and c_.get('frame') is not None and c_.get('type') in ('cut', 'reframe', 'dip_to_dark', 'flash'):
                maybe.add(int(c_['frame']))
    for m in re.finditer(r'(?i)\b(?:hard\s+)?cut(?:s)?\b(?:\s+[a-z-]+){0,3}?\s+(?:at|on)\s+(?:' + num_re + r')', low):
        n = int(m.group(1)) if m.group(1) else frame_of(m.group(2), m.group(3))
        if n <= 0 or n >= nfr:
            continue
        if negated(m) or re.search(r'(?i)possible|uncertain|maybe|might|could', low[max(0, m.start() - 40):m.start() + 12]) or \
                any(abs(n - x) <= 2 for x in maybe):
            continue   # the measurement itself flags these as possible cuts
        near = [b for b in bounds if abs(b - n) <= 2]
        if not near:
            out.append(f'puts a cut at frame {n}; the measured boundaries are at frames '
                       + (', '.join(str(b) for b in sorted(bounds)[:12]) or 'none (one continuous shot)'))
        elif all(bounds[b] == 'unknown' for b in near):
            out.append(f'calls frame {n} a cut; the measurement shows no visible cut there (the picture continues)')
    hard = [b for b, t_ in bounds.items() if t_ == 'cut'] + film.hidden_cuts()
    if hard and re.search(r'(?i)\b(?:one|a single)\s+continuous\s+(?:shot|take)\b', low):
        out.append(f'calls the film one continuous shot, but it has {len(hard)} hard cuts (plate 4)')
    seen, res = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            res.append(c)
    return res


def lint(md, rd, film, flags, tl, warn, d='.'):
    """Things the reviewers caught by hand: unfilled <…> placeholders, slots used but never declared, a 15 s cut that
    can't exist, a beat grid promised when there is no steady pulse."""
    rp = rd.get('recipe_prompt') or ''
    html = {'details', 'summary', 'sub', 'sup', 'br', 'b', 'i', 'em', 'strong', 'code', 'kbd'}
    left = sorted({m for m in re.findall(r'<([^<>\n]{2,30})>', rp) if m.strip('/').lower() not in html})
    left += sorted(set(re.findall(r'<(length|aspect|REF id|format name|product|brand)>', md)))
    if left:
        warn.append('unfilled placeholders ' + ', '.join(f'<{x}>' for x in left[:6]))
    # script-template fill-ins (plates 9, 10 and 12) are blanks for the writer's own words, not brand slots
    md_slots = re.sub(r'(?ms)^## (?:9|10|12)\.[^\n]*\n.*?(?=^## )', '', md)
    used = set(re.findall(r'\{\{([A-Z][A-Z0-9_]*)\}\}', md_slots + ' ' + rp))
    p15 = re.search(r'(?ms)^## 15\.[^\n]*\n(.*?)(?=^## |\Z)', md)
    declared = set(re.findall(r'\{\{([A-Z][A-Z0-9_]*)\}\}', p15.group(1))) if p15 else set()
    howto = re.search(r'(?ms)^## How to use this blueprint\n(.*?)(?=^## )', md)
    declared |= set(re.findall(r'\{\{([A-Z][A-Z0-9_]*)\}\}', howto.group(1))) if howto else set()
    undeclared = sorted(used - declared - {'PRODUCT', 'BRAND_FONT', 'BRAND_BG', 'BRAND_ACCENTS', 'LOGO', 'CTA_URL'})
    if undeclared:
        warn.append('slots used but not declared in plate 15: ' + ', '.join(undeclared[:8]))
    words_ = len(rp.split())
    if words_ > 230:
        warn.append(f'FIX recipe is {words_} words: keep it at 230 or fewer, and leave frame lists, cue lists and boxes to the JSON')
    mat = re.search(r'(?ims)^### [^\n]*(?:material|grade|shoot)[^\n]*card[^\n]*\n(.*?)(?=^##)', md)
    three_d = re.search(r'(?i)\b\d{2,3}\s?mm\b|\bf/\d|\barea lights?\b|\bprincipled\b|\bdepth of field\b', md)
    shoot_card = bool(re.search(r'(?im)^### [^\n]*(?:grade|shoot)[^\n]*card', md))
    if three_d and not shoot_card and not (mat and re.search(r'\d\s?px\b', mat.group(1)) and re.search(r'(?i)\b2D\b|css', mat.group(1))
                        and re.search(r'(?i)blur|glow|vignette', mat.group(1))):
        warn.append('FIX 3D terms (lens mm, f-stop, area lights, depth of field) need 2D equivalents in the material card: '
                    'blur px at 1080p outside the focus box, glow radius and opacity, vignette')
    lb = re.search(r'(?ms)^### Layout boxes[^\n]*\n(.*?)(?=^###|^## )', md)
    if lb and not re.search(r'(?i)confidence|source', lb.group(1).split('\n|---')[0] if '|---' in lb.group(1) else lb.group(1)[:600]):
        warn.append('FIX layout boxes need a source/confidence column (measured box, sheet tile, estimate); drop boxes you cannot support')
    mn = ((rd.get('routes') or {}).get('minimum') or {})
    mstack = ' '.join(str(x) for x in (mn.get('stack') or [])).lower()
    if 'blender' in mstack and 'remotion' in mstack:
        p11 = re.search(r'(?ms)^## 11\.[^\n]*\n(.*?)(?=^## )', md)
        if not (p11 and re.search(r'(?i)\b(refract|caustic|physical light|real 3D geometry)', p11.group(1))):
            warn.append('FIX the Minimum route uses Blender without saying why 2D/CSS 3D cannot reach the look: Minimum is the '
                        'simplest single stack that gets ~80% of it (often Remotion with CSS 3D)')
    p15 = re.search(r'(?ms)^## 15\.[^\n]*\n(.*?)(?=^## |\Z)', md)
    if film.w and film.h and abs(film.w / film.h - 16 / 9) / (16 / 9) > 0.035 and p15 and '16:9' not in p15.group(1).split('Aspect conversions')[0]:
        warn.append('FIX the source is not 16:9: Adapt the aspect needs a 16:9 line')
    fixed = sum((s.get('dur') or 0) for s in film.shots if flags.get(s['n']) == 'fixed')
    if fixed > 13:
        warn.append(f'the fixed shots add up to {fixed:.1f} s: a 15 s cut cannot keep them all')
    t = (tl.get('audio') or {}).get('tempo') or {}
    if 'beat grid' in rp.lower() and not (t.get('bpm') and (t.get('confidence') or 0) >= 0.5):
        warn.append('recipe mentions a beat grid, but plate 8 has no steady pulse')


def assemble(folder):
    d = os.path.abspath(os.path.expanduser(folder))
    warn, fatal = [], []
    drafts = sorted(p for p in os.listdir(d) if p.endswith('.blueprint.draft.md'))
    if not drafts:
        return dict(ok=False, fatal=[f'no <slug>.blueprint.draft.md in {d}: write the draft from assets/blueprint-template.md first'])
    slug = drafts[0][:-len('.blueprint.draft.md')]
    md = open(os.path.join(d, drafts[0]), encoding='utf-8').read()
    tl = load(os.path.join(d, 'data', 'timeline.json'))
    meas = os.path.join(d, 'measured.md')
    if not tl or not os.path.exists(meas):
        return dict(ok=False, fatal=['no data/timeline.json or measured.md: this folder was measured without the timeline '
                                     '(dissect 1.x or --no-timeline); fill the template by hand instead'])
    blocks = parse_blocks(open(meas, encoding='utf-8').read())
    plan = load(os.path.join(d, 'plan.json'), {}) or {}
    words = []
    tp = os.path.join(d, 'data', 'transcript.json')
    if os.path.exists(tp):
        words = [w for s in (load(tp, {}) or {}).get('segments', []) for w in s.get('words', [])]
    film = Film(tl, words)

    md = strip_comments(md)
    inferred_only = re.sub(r'\{\{measured:\w+\}\}', '', md)
    for key, plate in PLATE_OF.items():
        ph = '{{measured:%s}}' % key
        n = md.count(ph)
        if key not in blocks:
            md = md.replace(ph, '')
            continue
        if n == 0:
            warn.append(f'placeholder {ph} missing: block appended to plate {plate}')
            md = insert_at_plate_end(md, plate, blocks[key][1])
        else:
            if n > 1:
                warn.append(f'placeholder {ph} appears {n} times; used once')
                first = md.index(ph) + len(ph)
                md = md[:first] + md[first:].replace(ph, '')
            md = md.replace(ph, blocks[key][1])

    flags = {}
    for x in plan.get('scales') or []:
        try:
            n = int(x.get('shot'))
        except Exception:
            continue
        f = str(x.get('scales') or '').strip().lower()
        if f in FLAGS:
            flags[n] = f
    missing = [s['n'] for s in film.shots if s['n'] not in flags]
    if missing:
        warn.append(f'plan.json scales: no flag for shots {missing[:12]}' + (' …' if len(missing) > 12 else ''))
    md = re.sub(r'\{\{scales:(\d+)\}\}', lambda m: flags.get(int(m.group(1)), '–'), md)
    lt = length_table(film, flags) if flags else None
    if lt:
        m = re.search(r'(?m)^(?:\*\*|###? )Adapt the length.*\n(?:\n?[ \t]*[-*] .*\n?|[ \t]{2,}.*\n?)*', md)
        md = (md[:m.end()].rstrip('\n') + '\n\n' + lt + '\n\n' + md[m.end():].lstrip('\n')) if m else insert_at_plate_end(md, 15, lt)
    at = film.aspect_table()
    if at:
        m = re.search(r'(?m)^(?:\*\*|###? )Adapt the aspect.*\n(?:\n?[ \t]*[-*] .*\n?|[ \t]{2,}.*\n?)*', md)
        md = (md[:m.end()].rstrip('\n') + '\n\n' + at + '\n\n' + md[m.end():].lstrip('\n')) if m else insert_at_plate_end(md, 15, at)
    rp = (plan.get('recipe_prompt') or '').strip()
    if rp and 'Recipe prompt' not in md:
        foot = re.search(r'\n---\n<sub>.*\Z', md, re.S)
        tail = md[foot.start():] if foot else ''
        md = (md[:foot.start()] if foot else md).rstrip() + '\n\n## Recipe prompt\n\nPaste this into Claude Code (or any coding ' \
            'agent) with this file and `' + slug + '.timeline.json` next to it:\n\n```text\n' + rp + '\n```\n' + tail
    md = re.sub(r'\n{3,}', '\n\n', md).strip() + '\n'

    nums = [int(x) for x in re.findall(r'(?m)^## (\d+)\.', md)]
    if nums != list(range(1, 16)):
        miss = sorted(set(range(1, 16)) - set(nums))
        (fatal if miss else warn).append(f'plates found {nums}' + (f', missing {miss}' if miss else ' (order or duplicates)'))
    for leftover in sorted(set(re.findall(r'\{\{(?:measured|scales):[^}]*\}\}', md))):
        fatal.append(f'unresolved {leftover}')
    if len(re.findall(r'\(inferred\)', inferred_only, re.I)) < 6:
        warn.append('few "(inferred)" labels in the written parts')
    mdn = md.replace('’', "'").lower()
    if not re.search(r'(?m)^### .*(?:material|grade|shoot).*card', mdn):
        warn.append('missing part: Material card (or Grade and shoot card)')
    for need in ('Two routes', "How it's built", 'Layout boxes', 'Type spec', 'Original music bed', 'What this film avoids',
                 'Brand-distance', 'Adapt the aspect', 'Adapt the length', 'Asset honesty'):
        if need.lower() not in mdn:
            warn.append(f'missing part: {need}')
    if rp:
        check_recipe(rp, film, tl, warn)
    else:
        warn.append('plan.json has no recipe_prompt')
    check_numbers(inferred_only, tl, film, warn)
    lint(md, plan, film, flags, tl, warn, d)
    for c in (contradictions(inferred_only + '\n' + rp, film) if film.shots else []):
        warn.append('CONTRADICTION ' + c)
    if fatal:
        return dict(ok=False, fatal=fatal, warnings=warn)

    out_md = os.path.join(d, f'{slug}.blueprint.md')
    open(out_md, 'w', encoding='utf-8').write(md)
    tlo = dict(tl)
    tlo['shots'] = [dict(s, scales=flags.get(s['n'])) for s in tl.get('shots') or []]
    tlo['plan'] = {k: plan[k] for k in ('scales', 'recipe_prompt', 'routes', 'avoid', 'why_it_works') if k in plan}
    out_tl = os.path.join(d, f'{slug}.timeline.json')
    open(out_tl, 'w', encoding='utf-8').write(json.dumps(tlo, indent=1, ensure_ascii=False))
    return dict(ok=True, blueprint=out_md, timeline=out_tl, lines=md.count('\n'), warnings=warn)


def main(argv):
    if not argv:
        print(__doc__)
        sys.exit(1)
    res = assemble(argv[0])
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if not res.get('ok'):
        sys.exit(2)
