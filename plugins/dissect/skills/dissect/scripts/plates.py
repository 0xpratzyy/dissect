#!/usr/bin/env python3
"""plates.py - render the MEASURED sections of a dissect blueprint from timeline.json.

Imported by dissect.py, which writes measured.md: a set of blocks, each wrapped in
<!-- measured:KEY plate=N --> ... <!-- /measured:KEY -->. The blueprint template has a {{measured:KEY}} line where each
block goes; `dissect.py assemble <folder>` pastes them in verbatim. Every block heading says "(measured)". Nothing here
is interpreted: it formats numbers, and says so plainly when a number is missing or too uncertain to use.
The shot table's "scales" column holds {{scales:N}} placeholders that the writer fills (fixed | scales | optional).
Ported from the Refs blueprint-v2 renderer. Pure standard library.
"""
import datetime, difflib, json, math, re, statistics

VERSION = 'plates 2.0'
SPARK = '▁▂▃▄▅▆▇█'

# Confidence below which a number is reported as unusable instead of printed as fact.
TEMPO_MIN, TEMPO_SOLID = 0.5, 0.65   # B: only tell builders to cut on the grid from about 0.5
KEY_MIN = 0.25                      # B flags key confidence (margin over the runner-up) under 0.25 as low
LOW = 0.5          # camera / transition confidence under this gets a "?"

# Phone legibility minimum, as a share of frame WIDTH (see type_block for the reasoning).
PHONE_MIN_W = 0.02


# ------------------------------------------------------------------ small helpers

def num(x, nd=1, sign=False):
    if x is None:
        return '–'
    if abs(x) < 0.5 * 10 ** -nd:
        x = 0.0   # no "−0.0"
    s = f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'
    return s.replace('-', '−')


def secs(x, nd=2):
    return '–' if x is None else f'{x:.{nd}f}'


def spark(vals, lo=0.0, hi=1.0):
    out = []
    for v in vals:
        if v is None:
            out.append(' ')
            continue
        u = 0.0 if hi <= lo else (v - lo) / (hi - lo)
        out.append(SPARK[max(0, min(7, int(u * 7.999)))])
    return ''.join(out)


def resample(vals, n):
    vals = [v for v in vals if v is not None]
    if not vals:
        return []
    if len(vals) <= n:
        return vals
    out = []
    for k in range(n):
        a, b = int(k * len(vals) / n), int((k + 1) * len(vals) / n)
        seg = vals[a:max(b, a + 1)]
        out.append(sum(seg) / len(seg))
    return out


def binned(vals, step, width):
    """Average a series sampled every `step` s into bins of `width` s."""
    k = max(1, int(round(width / step)))
    out = []
    for i in range(0, len(vals), k):
        seg = [v for v in vals[i:i + k] if v is not None]
        out.append(sum(seg) / len(seg) if seg else None)
    return out


def table(head, rows):
    L = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---' for _ in head) + '|']
    L += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(L)


def short_text(s, words=8, chars=48):
    """On-screen text, kept well under the 15-word quote limit."""
    s = re.sub(r'\s+', ' ', (s or '').replace('|', '/')).strip()
    w = s.split(' ')
    t = ' '.join(w[:words])
    if len(t) > chars:
        t = t[:chars - 1].rstrip()
    return t + ('…' if t != s else '')


STD_ASPECTS = ((16, 9, 1920, 1080), (9, 16, 1080, 1920), (1, 1, 1080, 1080), (4, 5, 1080, 1350),
               (2, 1, 2160, 1080), (4, 3, 1440, 1080), (3, 4, 1080, 1440), (21, 9, 2520, 1080))


def deliverable(w, h):
    """The 1080p deliverable in the film's aspect: the short side is 1080."""
    if not w or not h:
        return 1920, 1080
    r = w / h
    for a, b, W, H in STD_ASPECTS:
        if abs(r - a / b) / (a / b) < 0.035:
            return W, H
    if r >= 1:
        return int(round(1080 * r / 2) * 2), 1080
    return 1080, int(round(1080 / r / 2) * 2)


def aspect_name(w, h, given=None):
    """The aspect a builder would deliver: a standard one when the source is within 3.5% of it (1400×720 is 2:1)."""
    if w and h:
        r = w / h
        for a, b, _, _ in STD_ASPECTS:
            if abs(r - a / b) / (a / b) < 0.035:
                return f'{a}:{b}'
    if given:
        return given
    if not w or not h:
        return '?'
    g = math.gcd(int(w), int(h))
    return f'{int(w) // g}:{int(h) // g}'


def ease_name(e):
    return {'ease_in': 'ease-in', 'ease_out': 'ease-out', 'ease_in_out': 'ease-in-out'}.get(e, e or '–')


def move_name(m):
    return (m or '–').replace('_', ' ')


def frame(t, fps):
    return None if t is None else int(round(t * fps))


def shot_at(shots, t):
    for s in shots:
        if s['start'] <= t < s['end']:
            return s['n']
    return shots[-1]['n'] if shots else None


def hexrgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


_WORDS = None


def words():
    """The system word list (macOS /usr/share/dict/words), lower-cased; empty when there is none."""
    global _WORDS
    if _WORDS is None:
        try:
            _WORDS = {w.strip().lower() for w in open('/usr/share/dict/words') if w.strip()}
        except Exception:
            _WORDS = set()
    return _WORDS


def wordlike(text, extra=frozenset()):
    """Does an OCR read look like real text? Most of its 3+ letter words are dictionary words or names from the post,
    or it is a clean number ('213.00', '$1,250', '30%'). Non-Latin scripts can't be checked and pass."""
    W = words()
    if not W:
        return True
    t = (text or '').strip()
    if any(ord(c) > 0x24F for c in t if c.isalpha()):
        return True
    toks = re.findall(r'[A-Za-z\u00C0-\u024F]+', t)
    long_ = [x for x in toks if len(x) >= 3]
    if long_:
        ok = [x for x in long_ if x.lower() in W or x.lower() in extra or (x.lower().endswith('s') and x.lower()[:-1] in W)]
        return len(ok) >= max(1, len(long_) / 2)
    return bool(re.fullmatch(r'[$€£+\-−]?\d{1,3}(?:[,.:]\d+)*(?:\.\d+)?\s?%?', t)) or bool(re.fullmatch(r'(19|20)\d\d', t))


def fmt_highlights(h):
    """v2.1 per-shot highlights: the colour of the brightest pixels (hue and temperature), which says more than the
    near-black average on dark films. Tolerant of a dict or a list of dicts."""
    if not h:
        return '–'
    items = h if isinstance(h, list) else [h]
    out = []
    for x in items[:3]:
        if not isinstance(x, dict):
            continue
        bits = []
        if x.get('hex'):
            bits.append(f"`{str(x['hex']).upper()}`")
        if x.get('hue_deg', x.get('hue')) is not None:
            bits.append(f"hue {x.get('hue_deg', x.get('hue')):.0f}°")
        k = x.get('kelvin', x.get('temp_k', x.get('temperature_k')))
        if isinstance(k, (int, float)):
            bits.append(f'≈{k:.0f} K')
        elif x.get('temperature') or x.get('temp'):
            bits.append(str(x.get('temperature') or x.get('temp')))
        if x.get('share') is not None:
            bits.append(f"{round(x['share'] * 100)}%")
        if bits:
            out.append(' '.join(bits))
    return ' · '.join(out) or '–'


def fmt_total(tot, W, H):
    """B's v2.1 `total` of a move ({dx_pct, dy_pct, scale, roll_deg[, seconds]}), with px at the deliverable."""
    dx, dy = tot.get('dx_pct') or 0, tot.get('dy_pct') or 0
    z, rr = tot.get('scale') or 1.0, tot.get('roll_deg') or 0
    out = []
    if abs(dx) >= 0.5:
        out.append(f'x {num(dx, 1, True)}% ({num(dx / 100 * W, 0, True)} px)')
    if abs(dy) >= 0.5:
        out.append(f'y {num(dy, 1, True)}% ({num(dy / 100 * H, 0, True)} px)')
    if abs(z - 1) >= 0.01:
        out.append(f'×{z:.2f} scale')
    if abs(rr) >= 0.5:
        out.append(f'roll {num(rr, 1, True)}°')
    return ' · '.join(out) or 'none'


def seg_sum(segs, W, H):
    """Total of consecutive moves: translations add, scales multiply."""
    dx = sum((g.get('pan_x_pct_s') or 0) * (g['end'] - g['start']) for g in segs)
    dy = sum((g.get('pan_y_pct_s') or 0) * (g['end'] - g['start']) for g in segs)
    z = 1.0
    for g in segs:
        if g.get('zoom_per_s'):
            z *= g['zoom_per_s'] ** (g['end'] - g['start'])
    rr = sum((g.get('roll_deg_s') or 0) * (g['end'] - g['start']) for g in segs)
    out = []
    if abs(dx) >= 0.5:
        out.append(f'x {num(dx, 1, True)}% ({num(dx / 100 * W, 0, True)} px)')
    if abs(dy) >= 0.5:
        out.append(f'y {num(dy, 1, True)}% ({num(dy / 100 * H, 0, True)} px)')
    if abs(z - 1) >= 0.01:
        out.append(f'×{z:.2f} scale')
    if abs(rr) >= 0.5:
        out.append(f'roll {num(rr, 1, True)}°')
    return ' · '.join(out) or 'none'


def seg_total(g, W, H):
    """Total displacement of a move, with px at the deliverable size: B's `total` when it has one (v2.1), else rate × length."""
    if isinstance(g.get('total'), dict):
        return fmt_total(g['total'], W, H)
    d = max(0.0, (g.get('end') or 0) - (g.get('start') or 0))
    out = []
    px, py, z, rr = g.get('pan_x_pct_s'), g.get('pan_y_pct_s'), g.get('zoom_per_s'), g.get('roll_deg_s')
    if px is not None and abs(px * d) >= 0.5:
        out.append(f'x {num(px * d, 1, True)}% ({num(px * d / 100 * W, 0, True)} px)')
    if py is not None and abs(py * d) >= 0.5:
        out.append(f'y {num(py * d, 1, True)}% ({num(py * d / 100 * H, 0, True)} px)')
    if z and z > 0 and abs(z ** d - 1) >= 0.01:
        out.append(f'×{z ** d:.2f} scale')
    if rr and abs(rr * d) >= 0.5:
        out.append(f'roll {num(rr * d, 1, True)}°')
    return ' · '.join(out) or 'none'


TRANS_NAME = {'unknown': 'no visible cut (continuous)'}


def tname(t):
    return TRANS_NAME.get(t, move_name(t))


def block(key, plate, body):
    return f'<!-- measured:{key} plate={plate} -->\n{body.strip()}\n<!-- /measured:{key} -->'


# ------------------------------------------------------------------ blocks

class Film:
    def __init__(self, tl, words=None, extra_words=()):
        self.tl = tl
        v = tl.get('video') or {}
        self.w, self.h = v.get('width'), v.get('height')
        self.fps = float(v.get('fps') or 30)
        self.dur = float(v.get('duration_s') or 0)
        self.aspect = aspect_name(self.w, self.h, v.get('aspect'))
        self.W, self.H = deliverable(self.w, self.h)
        self.shots = sorted(tl.get('shots') or [], key=lambda s: s.get('n') or 0)
        for s in self.shots:
            s.setdefault('start_frame', frame(s.get('start'), self.fps))
            s.setdefault('end_frame', frame(s.get('end'), self.fps))
            if s.get('dur') is None and s.get('start') is not None and s.get('end') is not None:
                s['dur'] = s['end'] - s['start']
        self.ref = 'the reference'
        self.extra_words = {w.lower() for w in extra_words}
        self.fpsl = f'{self.fps:g}'
        self.nfr = int(v.get('frames') or round(self.dur * self.fps))   # decoded frames (the count to build to)
        self.words = sorted((float(w[0]), float(w[1])) for w in (words or []) if len(w) >= 4 and (w[3] or 0) >= 0.4)
        self.raw_words = [w for w in (words or []) if len(w) >= 3]

    # -- measured facts shared by the cue sheet, the timing facts for the writer, and the prose check
    def fade_out(self):
        """(start_s, how) of the closing fade to black, or None. B's transition_out when it has one, else the brightness
        curve: the end falls below 40% of the level 1.5-4 s before it (relative, so a dark end card fading counts)."""
        S = self.shots
        out = (S[-1].get('transition_out') or {}) if S else {}
        if str(out.get('type') or '').startswith('fade') and out.get('frames'):
            return self.dur - out['frames'] / self.fps, 'measured transition'
        b = self.tl.get('brightness') or {}
        v, step = b.get('mean') or [], b.get('step_s') or 0.25
        n = len(v)
        if n < 12 or any(x is None for x in v[-16:]):
            return None
        win = sorted(v[max(0, n - 16):n - 6])
        L = win[len(win) // 2]
        if L <= 0.003 or v[-1] > 0.4 * L:
            return None
        j = n - 1
        while j > 0 and v[j] < 0.9 * L:
            j -= 1
        start = (j + 0.5) * step
        return (start, 'brightness curve') if self.dur - start >= 0.4 else None

    def fade_in(self):
        b = self.tl.get('brightness') or {}
        v, step = b.get('mean') or [], b.get('step_s') or 0.25
        if len(v) < 12 or any(x is None for x in v[:12]):
            return None
        win = sorted(v[4:12])
        L = win[len(win) // 2]
        if L <= 0.003 or v[0] > 0.4 * L:
            return None
        j = 0
        while j < len(v) - 1 and v[j] < 0.9 * L:
            j += 1
        return (j - 0.5) * step if j * step >= 0.4 else None

    def internal_changes(self, s):
        """B's shots[].internal_changes (v2.1), normalised to [(t, label)]; tolerant of the exact field names."""
        out = []
        for c in s.get('internal_changes') or []:
            if not isinstance(c, dict):
                continue
            t = c.get('t', c.get('time', c.get('start')))
            if t is None and c.get('frame') is not None:
                t = c['frame'] / self.fps
            if t is None:
                continue
            kind = c.get('kind') or c.get('type') or 'change'
            what = c.get('what') or c.get('desc') or c.get('label') or ''
            fr = c.get('frames')
            out.append((float(t), move_name(kind) + (f' over {fr} f' if fr else '') + (f': {what}' if what else '')))
        return out

    def hidden_cuts(self):
        """Frames of cuts B found inside v1 shots (v2.1 internal_changes of type cut)."""
        return sorted(int(c['frame']) for s in self.shots for c in (s.get('internal_changes') or [])
                      if isinstance(c, dict) and c.get('type') == 'cut' and c.get('frame') is not None)

    def possible_cuts(self):
        """Frames B flags in notes as possible cuts it did not split on (uncertain boundaries inside shots)."""
        out = []
        for n in self.tl.get('notes') or []:
            if 'possible cut' in str(n).lower():
                out += [int(x) for x in re.findall(r'\(f(\d+)\)', str(n))]
        return sorted(set(out))

    def events(self):
        """Visual events the picture shows: [(t, tolerance_s, label)]. Frame-accurate ones get ±3 frames; ones read off
        sampled data (text every OCR step, brightness every 0.25 s) get half a sample more."""
        f3 = 3.0 / self.fps
        ev = []
        for s in self.shots[1:]:
            ti = s.get('transition_in') or {}
            ev.append((s['start'], f3, f"{tname(ti.get('type') or 'cut')} into shot {s['n']}"))
        for s in self.shots:
            for t, lab in self.internal_changes(s):
                ev.append((t, f3, f'shot {s["n"]}: {lab}'))
            c = s.get('camera') or {}
            for j in c.get('reframes') or []:
                if j.get('t') is not None:
                    ev.append((j['t'], f3, f'jump ×{j.get("scale", 1):.2f} in shot {s["n"]}'))
            for g in (c.get('segments') or [])[1:]:
                if g.get('start') is not None and g.get('move') != 'static':
                    ev.append((g['start'], f3 + 0.05, f'camera {move_name(g.get("move"))} starts (shot {s["n"]})'))
            e = s.get('element_motion') or {}
            segs = e.get('segments') or []
            for g in segs:
                if g.get('start') is not None and g.get('move') != 'static':
                    ev.append((g['start'], f3 + 0.05, f'element {move_name(g.get("move"))} starts (shot {s["n"]})'))
            if segs and segs[-1].get('end') is not None and segs[-1]['end'] < s['end'] - 0.2 and segs[-1].get('move') != 'static':
                ev.append((segs[-1]['end'], f3 + 0.05, f'element settles (shot {s["n"]})'))
        step_t = 0.5 if self.dur > 30 else 0.25
        for s in self.shots:
            for t in s.get('text') or []:
                txt = (t.get('text') or '').strip()
                held = (t.get('last') or 0) - (t.get('first') or 0)
                letters = len(re.findall(r'[A-Za-z\u00C0-\u024F]', txt))
                number = bool(re.fullmatch(r'[$€£]?\d[\d,.:]*%?', txt)) and len(re.findall(r'\d', txt)) >= 2
                if t.get('first') is None or not wordlike(txt) or not (letters >= 3 or number):
                    continue
                if held < 0.9 * step_t and len(txt.split()) < 3:   # seen in a single OCR sample: too often a misread
                    continue
                if (t.get('first') or 0) > s['start'] + 0.05:
                    ev.append((t['first'], f3 + step_t / 2, f'"{short_text(txt, 4, 26)}" appears'))
        b = self.tl.get('brightness') or {}
        mean, bstep = b.get('mean') or [], b.get('step_s') or 0.25
        rng_ = (max(v for v in mean if v is not None) - min(v for v in mean if v is not None)) if any(v is not None for v in mean) else 0
        for i in range(1, len(mean)):
            a0, a1 = mean[i - 1], mean[i]
            if a0 is None or a1 is None:
                continue
            d = a1 - a0
            if abs(d) >= max(0.06, 0.25 * rng_) or (max(a0, a1) > 0.02 and abs(d) / max(a0, a1) >= 0.4 and abs(d) >= 0.01):
                ev.append(((i - 0.5) * bstep, f3 + bstep / 2, 'picture gets ' + ('brighter' if d > 0 else 'darker')))
        fo = self.fade_out()
        if fo:
            ev.append((fo[0], f3 + 0.125, 'fade to black begins'))
        return sorted(ev)

    def on_screen(self, t, ev=None):
        ev = self.events() if ev is None else ev
        best = None
        for te, tol, lab in ev:
            d = abs(te - t)
            if d <= tol and (best is None or d < best[0]):
                best = (d, lab)
        return best[1] if best else 'no visible event'

    def timing_facts(self):
        """A plain list for the writer (plates.md block 'facts', not pasted into the blueprint): every inferred claim
        about cuts, holds, fades and brightness has to agree with it."""
        S, fps = self.shots, self.fps
        f = lambda t: int(round(t * fps))
        L = [f'Timing facts at {self.fpsl} fps ({self.nfr} frames, 0–{self.nfr - 1}). '
             f'Every inferred statement about cuts, holds, fades or brightness must agree with these.']
        cuts = [s for s in S[1:] if (s.get('transition_in') or {}).get('type') == 'cut']
        cont = [s for s in S[1:] if (s.get('transition_in') or {}).get('type') == 'unknown']
        other = [s for s in S[1:] if (s.get('transition_in') or {}).get('type') not in ('cut', 'unknown')]
        hc = self.hidden_cuts()
        if len(S) <= 1 and not hc:
            L.append('- One continuous shot: no cuts anywhere.')
        elif len(S) <= 1:
            L.append(f'- One detected shot, but with {len(hc)} hidden hard cut{"s" if len(hc) > 1 else ""} inside it (listed below): it is '
                     f'not one continuous take.')
        if cuts:
            L.append('- Hard cuts at frames ' + ', '.join(str(s['start_frame']) for s in cuts[:60]) + '.')
        if cont:
            L.append('- Boundaries with no visible cut (the picture continues): frames ' + ', '.join(str(s['start_frame']) for s in cont) + '.')
        for s in other[:30]:
            ti = s['transition_in']
            L.append(f'- {tname(ti["type"]).capitalize()}' + (f' over {ti.get("frames")} f' if ti.get('frames') else '') + f' into shot {s["n"]} at frame {s["start_frame"]}.')
        for s in S:
            for t, lab in self.internal_changes(s)[:8]:
                L.append(f'- Inside shot {s["n"]}, frame {f(t)}: {lab}.')
        pc = self.possible_cuts()
        if pc:
            L.append('- Possible cuts the measurement did not split on (uncertain, left inside their shots): frames '
                     + ', '.join(str(x) for x in pc[:20]) + '. Call them possible, not cuts.')
        if len(S) > 1:
            L.append(f'- The last shot starts at frame {S[-1]["start_frame"]} ({S[-1]["start"]:.2f} s).')
        fi = self.fade_in()
        if fi is not None:
            L.append(f'- The film opens from black: it is up by about frame {f(fi)} ({fi:.2f} s).')
        fo = self.fade_out()
        if fo:
            L.append(f'- The picture fades to black from about frame {f(fo[0])} ({fo[0]:.2f} s, {fo[1]}) to the end: '
                     f'nothing holds to the last frame.')
        else:
            L.append('- No fade to black at the end.')
        steps = [(te, lab) for te, tol, lab in self.events() if lab.startswith('picture gets')]
        if steps:
            L.append('- Brightness steps: ' + ', '.join(f'{lab.split()[-1]} at frame {f(te)}' for te, lab in steps[:12]) + '.')
        return '\n'.join(L)

    def aspect_table(self):
        """What a centre crop to each other aspect keeps, measured against the film's text boxes and moving elements."""
        if not self.w or not self.h:
            return None
        src = self.w / self.h
        items, _ = self.clean_text()
        boxes = [(f'"{short_text(it["t"].get("text"), 4, 24)}"', it['t'].get('box')) for it in items if it['t'].get('box') and len(it['t']['box']) == 4]
        for s in self.shots:
            e = s.get('element_motion') or {}
            if e.get('region') and len(e['region']) == 4:
                boxes.append((f'moving element, shot {s["n"]}', e['region']))
        rows = []
        for name, a, b in (('16:9', 16, 9), ('9:16', 9, 16), ('1:1', 1, 1)):
            tgt = a / b
            if abs(tgt - src) / tgt < 0.035:
                continue
            W, H = (1920, 1080) if tgt > 1 else (1080, 1920) if tgt < 1 else (1080, 1080)
            if tgt < src:   # narrower: keep the full height, crop the sides
                k = tgt / src
                x0, x1, y0, y1 = (1 - k) / 2, (1 + k) / 2, 0.0, 1.0
                keep = f'middle {k * 100:.1f}% of the width (cut {(1 - k) / 2 * 100:.1f}% each side)'
            else:           # wider: keep the full width, crop top and bottom
                k = src / tgt
                x0, x1, y0, y1 = 0.0, 1.0, (1 - k) / 2, (1 + k) / 2
                keep = f'middle {k * 100:.1f}% of the height (cut {(1 - k) / 2 * 100:.1f}% top and bottom)'
            lost = [n for n, (bx, by, bw, bh) in boxes if bx < x0 - 0.005 or bx + bw > x1 + 0.005 or by < y0 - 0.005 or by + bh > y1 + 0.005]
            kept = len(boxes) - len(lost)
            verdict = ('a straight centre crop works' if not lost else
                       'centre crop, then move the lost boxes inside' if len(lost) <= max(1, len(boxes) // 4) else 're-lay out: a crop loses too much')
            rows.append([name, f'{W}×{H}', keep, f'{kept} of {len(boxes)}' if boxes else '–',
                         (', '.join(lost[:4]) + (' …' if len(lost) > 4 else '')) if lost else 'none', f'{0.02 * W:.0f} px cap', verdict])
        if not rows:
            return None
        head = ['to', 'deliver at', 'a centre crop keeps', 'measured boxes kept', 'boxes a crop loses', 'phone minimum', 'so']
        L = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---' for _ in head) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
        note = ''
        if src > 16 / 9 * 1.035:
            note = (f' This film is wider than 16:9 ({self.aspect}): at 1920×1080, centre-crop the sides (scale the 1080 px height as is) '
                    f'rather than letterboxing, which would leave a {1920 / src:.0f} px tall picture with bars in a feed.')
        return (f'**Aspect conversions (computed from the measured boxes):** a centre crop of the reference frame, checked against its '
                f'text boxes and moving elements.{note}\n\n' + '\n'.join(L))


    def short_text(t, n=24):
        t = re.sub(r'\s+', ' ', (t or '').replace('|', '/')).strip()
        return t if len(t) <= n else t[:n - 1].rstrip() + '…'

    # -- plate 3
    def hook_sheet(self):
        # No sheet images or /api/frames URLs in plates: the film page shows every sheet in its own "Frame by frame"
        # block (an embed would show the hook twice) and the CDN caches those URLs for a week. Point at get_frames.
        sh = self.tl.get('sheets') or {}
        if not sh.get('hook'):
            return block('hook_sheet', 3, '**Hook sheet (measured):** not drawn for this film.')
        return block('hook_sheet', 3, f"**Hook sheet (measured):** `sheets/{sh['hook']}`: the opening at {sh.get('hook_step', '0.25 s')} steps, "
                                      f"each tile labelled with its time and frame. Read it next to this table.")

    # -- plate 4
    def shots_block(self):
        S = self.shots
        if not S:
            return block('shots', 4, '### Shots (measured)\n\nNo shot data in the timeline.')
        cuts = len(S) - 1
        rows = []
        cap = 200   # every shot: this table is the film's structure
        for s in S[:cap]:
            ti = s.get('transition_in') or {}
            t = ti.get('type') or 'unknown'
            if s['n'] == S[0]['n'] or t == 'start':
                tr = 'start'
            elif t == 'unknown':
                tr = tname(t)
            else:
                fr = ti.get('frames') or 0
                q = '?' if (ti.get('confidence') is not None and ti['confidence'] < LOW) else ''
                tr = (move_name(t) + ''.join(f' {ti[k]}' for k in ('direction', 'axis') if ti.get(k)) + q + (f' · {fr} f' if fr else ''))
            sf, ef = s.get('start_frame'), s.get('end_frame')
            frames = f'{sf}–{ef - 1} · {ef - sf} f' if sf is not None and ef is not None else '–'
            row = [s['n'], f"{secs(s.get('start'))}–{secs(s.get('end'))}", frames, secs(s.get('dur')), tr]
            if any(x.get('internal_changes') for x in S):
                ic = self.internal_changes(s)
                row.append(' · '.join(f'f{int(round(t * self.fps))} {lab.split(":")[0]}' for t, lab in ic[:5]) + (' …' if len(ic) > 5 else '') if ic else '–')
            rows.append(row + ['{{scales:%d}}' % s['n']])
        head = ['#', 'in–out (s)', f'frames @{self.fpsl} (first–last · count)', 'dur (s)', 'transition in'] + \
            (['changes inside'] if any(x.get('internal_changes') for x in S) else []) + ['scales (inferred)']
        lens = [s['dur'] for s in S if s.get('dur')]
        total = f'{self.dur:.2f} s = {self.nfr} frames at {self.fpsl} fps (frames 0–{self.nfr - 1})'
        if len(S) == 1 and self.hidden_cuts():
            lead = (f'One detected shot with {len(self.hidden_cuts())} hidden hard cut{"s" if len(self.hidden_cuts()) > 1 else ""} inside it '
                    f'(see "changes inside"), {total}.')
        elif len(S) == 1:
            lead = f'One continuous shot, {total}.'
        elif lens:
            lead = (f'{len(S)} shots · {cuts} transitions · {total} · shot length median {statistics.median(lens):.2f} s '
                    f'(shortest {min(lens):.2f}, longest {max(lens):.2f}).')
        else:
            lead = f'{len(S)} shots · {total}.'
        lead += (f' Frames are at {self.fpsl} fps, first and last inclusive: a shot "33–70 · 38 f" is a Remotion '
                 f'`<Sequence from={{33}} durationInFrames={{38}}>`; at another frame rate, use the seconds.')
        L = ['### Shots (measured)', '', lead, '', table(head, rows)]
        if len(S) > cap:
            L += ['', f'… {len(S) - cap} more shots in `timeline.json`.']
        L += ['', '`scales`: **fixed** = keep this shot\'s length when the film gets longer or shorter · **scales** = stretch or '
                  'shrink it with the length · **optional** = drop it for the 15 s cut.']
        notes = [n for n in (self.tl.get('notes') or []) if n and not n.startswith('cap_h_px_1080 is estimated')]
        if notes:
            L += ['', '**Measurement notes:** ' + ' · '.join(short_text(n, 40, 220) for n in notes[:6])]
        return block('shots', 4, '\n'.join(L))

    def sheets_block(self):
        sh = self.tl.get('sheets') or {}
        parts = []
        if sh.get('shots'):
            parts.append(f"`{', '.join('sheets/' + p for p in sh['shots'][:3])}`{' …' if len(sh['shots']) > 3 else ''}: one labelled frame per "
                         "shot, from the middle of the shot (labels `S<n> m:ss.s f<frame>`; `+` = just after a change inside the shot)")
        tls = sh.get('timeline') or []
        if tls:
            parts.append(f"`sheets/timeline-01.jpg`{f' … `timeline-{len(tls):02d}.jpg`' if len(tls) > 1 else ''}: the whole film at "
                         f"{sh.get('timeline_step_s', 0.5)} s steps (labels `S<n> m:ss.s f<frame>`)")
        trs = sh.get('transitions') or []
        if trs:
            idx = [x for x in (sh.get('transitions_index') or []) if isinstance(x, dict) and x.get('frame') is not None]
            what = (': ' + ', '.join(f'f{x["frame"]} {move_name(x.get("type") or "change")}' for x in idx[:12])) if idx else ''
            parts.append(f'`sheets/transitions-NN.jpg` ({len(trs)} page{"s" if len(trs) > 1 else ""}): 14 consecutive frames around '
                         f'each cut and change{what}, the change frame outlined (labels `S<n> m:ss.sss f<frame>`)')
        if not parts:
            return block('sheets', 4, '**Frame sheets (measured):** none were drawn.')
        return block('sheets', 4, '**Frame sheets (measured):** ' + ' · '.join(parts) + '. Ship them with this blueprint: '
                     'a builder rebuilding the film should look at the shot and transition sheets before writing any code.')

    # -- plate 5
    def camera_block(self):
        S = self.shots
        rows, moved = [], []
        cap = 30   # long films: the rest is in the timeline JSON
        layered = []
        for s in S[:cap]:
            c = s.get('camera')
            cad = (s.get('cadence') or {}).get('label') or '–'
            if not c:
                rows.append([s['n'], 'not measured', '–', '–', '–', '–', '–', cad, secs(s.get('blur')), secs(s.get('motion_energy')), '–'])
                continue
            q = '?' if (c.get('confidence') is not None and c['confidence'] < LOW) else ''
            d = s.get('dur') or 0
            px, py, z, roll = c.get('pan_x_pct_s'), c.get('pan_y_pct_s'), c.get('zoom_per_s'), c.get('roll_deg_s')
            tot = []
            if px is not None and abs(px * d) >= 0.5:
                tot.append(f'x {num(px * d, 1, True)}% ({num(px * d / 100 * self.W, 0, True)} px)')
            if py is not None and abs(py * d) >= 0.5:
                tot.append(f'y {num(py * d, 1, True)}% ({num(py * d / 100 * self.H, 0, True)} px)')
            if z is not None and z > 0 and abs(z ** d - 1) >= 0.01:
                tot.append(f'×{z ** d:.2f} scale')
            if roll and abs(roll * d) >= 0.5:
                tot.append(f'roll {num(roll * d, 1, True)}°')
            if isinstance(c.get('total'), dict):   # v2.1: B's own total (segment totals add up to it)
                ft = fmt_total(c['total'], self.W, self.H)
                tot = [] if ft == 'none' else [ft]
            pan = f'{num(px, 1, True)} · {num(py, 1, True)}' if (px is not None or py is not None) else '–'
            zz = f'×{z:.3f}' if z is not None else '–'
            sp = spark(resample(c.get('speed') or [], max(4, min(16, int(round(d / 0.5))))))  # about one character per 0.5 s
            pk = c.get('peak_speed_pct_s')
            if sp and pk:
                sp += f' · {pk:.0f}%/s'
            mv = move_name(c.get('move')) + q
            if c.get('move') == 'static' and s.get('element_motion'):
                mv += ' · element moves ↓'
            if c.get('shake_pct_s'):
                mv += f' (shake {c["shake_pct_s"]:.1f}%/s)'
            if c.get('layers'):
                mv += ' ²'
                layered.append(s['n'])
            for j in (c.get('reframes') or [])[:4]:   # instant punch-ins / jumps inside the shot
                tot.append(f'jump at f{j.get("frame")}: ×{j.get("scale", 1):.2f}' +
                           (f', x {num(j.get("dx_pct"), 1, True)}%' if abs(j.get('dx_pct') or 0) >= 1 else '') +
                           (f', y {num(j.get("dy_pct"), 1, True)}%' if abs(j.get('dy_pct') or 0) >= 1 else ''))
            rows.append([s['n'], mv, pan, zz, ' · '.join(tot) or 'none', ease_name(c.get('ease')), sp or '–', cad,
                         secs(s.get('blur')), secs(s.get('motion_energy')), secs(c.get('confidence'))])
            for g in (c.get('segments') or [])[:6]:   # the move changes direction inside the shot
                rows.append(['', f'↳ {secs(g.get("start"))}–{secs(g.get("end"))} s: {move_name(g.get("move"))}',
                             f'{num(g.get("pan_x_pct_s"), 1, True)} · {num(g.get("pan_y_pct_s"), 1, True)}',
                             f'×{g["zoom_per_s"]:.3f}' if g.get('zoom_per_s') is not None else '–',
                             seg_total(g, self.W, self.H), ease_name(g.get('ease')), '', '', '', '', ''])
            if c.get('move') not in (None, 'static') and (c.get('confidence') or 0) >= LOW:
                mag = abs((px or 0) * d) + abs((py or 0) * d) + abs(((z or 1) ** d - 1) * 100) + abs((roll or 0) * d)
                if mag >= 2:
                    moved.append((mag, s, c))
        # consecutive locked-off shots share one row ("4–9 · static"), which keeps long films short
        merged = []
        for r in rows:
            if r[0] == '':
                merged.append(['row', r])
                continue
            if r[1] == 'static' and r[4] == 'none' and r[0] != '' and merged and merged[-1][0] == 'static-run':
                merged[-1][1].append(r)
            elif r[1] == 'static' and r[4] == 'none' and r[0] != '':
                merged.append(['static-run', [r]])
            else:
                merged.append(['row', r])
        rows = []
        for kind, r in merged:
            if kind == 'row':
                rows.append(r)
            elif len(r) == 1:
                rows.append(r[0])
            else:
                def rng(k):
                    v = [x[k] for x in r if x[k] not in ('–', None)]
                    return '–' if not v else (min(v) if min(v) == max(v) else f'{min(v)}–{max(v)}')
                cads = sorted({x[7] for x in r if x[7] != '–'})
                rows.append([f'{r[0][0]}–{r[-1][0]}', 'static', '–', '–', 'none', 'hold', '–', ', '.join(cads) or '–',
                             rng(8), rng(9), f'≥ {min(x[10] for x in r)}'])
        head = ['#', 'camera move', 'pan x · y (%/s)', 'zoom (/s)', 'over the shot', 'ease', 'speed · peak', 'cadence', 'blur', 'motion', 'conf']
        L = ['### Camera and motion (measured)', '',
             f'Per shot, measured on the frames. "Camera move" names the camera\'s move; the numbers are the image\'s motion: pan in % '
             f'of frame width (x) or height (y) per second, +x = content moves right (so a pan left), +y = content moves down; '
             f'zoom per second > 1 = content grows (push in). "Over the shot" is the total, with px at {self.W}×{self.H}. '
             f'Speed is the move through the shot, about one character per 0.5 s (▁ still → █ its peak), then the peak in % of '
             f'width per second. Cadence: "ones" = a new picture every frame, "twos" = each picture held two frames. Blur is '
             f'relative to this film (0 = its sharpest shots, 1 = its blurriest); motion = share of the frame changing, 0–1. '
             f'"?" = confidence under {LOW}: a hint, check the sheets. The labels describe how the image moves, not where a 3D '
             f'camera went: an object flying toward a still camera can read as a push in.', '']
        nel = sum(1 for s in S if s.get('element_motion'))
        if nel:
            L += [f'In {nel} of {len(S)} shots the camera holds still and something inside the frame moves: see **Element motion** '
                  f'right under this table.', '']
        L += [table(head, rows)]
        if layered:
            L += ['', f'² Shots {", ".join(str(n) for n in layered)}: a second layer moves differently (elements over a background); '
                      f'the numbers follow the dominant layer.']
        el = [(s, s['element_motion']) for s in S[:cap] if s.get('element_motion')][:16]
        if el:
            erows = []
            for s, e in el:
                d = s.get('dur') or 0
                q = '?' if (e.get('confidence') is not None and e['confidence'] < LOW) else ''
                reg = ' · '.join(f'{round(v * 100) or 0:.0f}' for v in (e.get('region') or [])) or '–'
                tr_ = [b_ for b_ in (e.get('track') or []) if isinstance(b_, dict) and b_.get('box') or isinstance(b_, list)]
                if tr_:   # v2.1: the moving box every 0.5 s; show where it starts and ends
                    bx = lambda b_: b_.get('box') if isinstance(b_, dict) else b_
                    if len(bx(tr_[0]) or []) == 4 and len(bx(tr_[-1]) or []) == 4:
                        reg = (' · '.join(f'{round(v * 100) or 0:.0f}' for v in bx(tr_[0])) + ' → ' +
                               ' · '.join(f'{round(v * 100) or 0:.0f}' for v in bx(tr_[-1])))
                segs = [g for g in (e.get('segments') or []) if g.get('start') is not None and g.get('end') is not None]
                if isinstance(e.get('total'), dict):   # v2.1: B's total, which its segments add up to
                    tot = fmt_total(e['total'], self.W, self.H)
                    if e['total'].get('seconds') and tot != 'none':
                        tot += f' in {e["total"]["seconds"]:.1f} s'
                elif segs:   # the segments are the measurement; their sum is the move (the shot-wide average can disagree)
                    tot = seg_sum(segs, self.W, self.H)
                    tot = (tot + ' (sum of the ↳ moves)') if tot != 'none' else tot
                else:
                    dm = d * (e.get('moving_share') if e.get('moving_share') is not None else 1.0)   # the time it actually moves
                    tot = seg_total({'start': 0, 'end': dm, 'pan_x_pct_s': e.get('dx_pct_s'), 'pan_y_pct_s': e.get('dy_pct_s'),
                                     'zoom_per_s': e.get('zoom_per_s'), 'roll_deg_s': e.get('roll_deg_s')}, self.W, self.H)
                    tot = (tot + f' in ≈{dm:.1f} s') if tot != 'none' else tot
                sp = spark(resample(e.get('speed') or [], max(4, min(16, int(round(d / 0.5))))))
                if sp and e.get('peak_speed_pct_s'):
                    sp += f' · {e["peak_speed_pct_s"]:.0f}%/s'
                erows.append([s['n'], reg, move_name(e.get('move')) + q,
                              f'{num(e.get("dx_pct_s"), 1, True)} · {num(e.get("dy_pct_s"), 1, True)}',
                              f'×{e["zoom_per_s"]:.3f}' if e.get('zoom_per_s') is not None else '–', tot, ease_name(e.get('ease')),
                              sp or '–', secs(e.get('moving_share')), secs(e.get('confidence'))])
                for g in (e.get('segments') or [])[:6]:
                    erows.append(['', '', f'↳ {secs(g.get("start"))}–{secs(g.get("end"))} s: {move_name(g.get("move"))}',
                                  f'{num(g.get("pan_x_pct_s"), 1, True)} · {num(g.get("pan_y_pct_s"), 1, True)}',
                                  f'×{g["zoom_per_s"]:.3f}' if g.get('zoom_per_s') is not None else '–', seg_total(g, self.W, self.H),
                                  ease_name(g.get('ease')), '', '', ''])
            L += ['', '**Element motion (measured):** in these shots the frame holds still and something inside it moves: a '
                      'card, a UI panel, a product, a word. Region = where it moves (x · y · w · h % of the frame); numbers as above, '
                      'for the moving region; "moving" = share of the shot in motion; ↳ = the move changes direction.', '',
                  table(['#', 'region x·y·w·h %', 'element move', 'x · y (%/s)', 'zoom (/s)', 'total while moving', 'ease', 'speed · peak',
                         'moving', 'conf'], erows)]
        foc = []
        for s in S[:cap]:
            fo_ = s.get('focus') or {}
            parts = [p_ for p_ in (fo_.get('by_part') or []) if isinstance(p_, dict) and p_.get('box')] if isinstance(fo_, dict) else []
            items_ = [(f'{p_["start"]:.2f}–{p_["end"]:.2f} s', p_) for p_ in parts] if len(parts) > 1 else [('', fo_)]
            for when, f_ in items_:
                if not isinstance(f_, dict) or not (f_.get('box') or f_.get('blur_px') is not None):
                    continue
                bx = ' · '.join(f'{round(v * 100) or 0:.0f}' for v in (f_.get('box') or [])) or '–'
                bp = f_.get('blur_px', f_.get('blur_px_1080'))
                foc.append(f'shot {s["n"]}' + (f' {when}' if when else '') + f': sharp inside {bx}'
                           + (f', about {bp:.0f} px blur outside at 1080p' if isinstance(bp, (int, float)) else ', no defocus (deep focus)'))
        if foc:
            L += ['', '**Focus (measured):** ' + ' · '.join(foc[:12]) + '. Box = x · y · w · h % of the frame. In 2D, blur the layers '
                      'outside the box by that many px; in 3D, set the focus distance on the box and open the aperture until the '
                      'background blur matches.']
        if len(S) > cap:
            L += ['', f'… {len(S) - cap} more shots in the timeline JSON.']
        moved.sort(key=lambda m: -m[0])
        if moved:
            bits = []
            for mag, s, c in moved[:3]:
                d = s.get('dur') or 0
                z = c.get('zoom_per_s')
                desc = move_name(c.get('move'))
                if c.get('move') in ('push_in', 'pull_out') and z:
                    desc += f' ×{z ** d:.2f} over {d:.2f} s'
                elif c.get('move') in ('pan_left', 'pan_right') and c.get('pan_x_pct_s') is not None:
                    desc += f', content {num(c["pan_x_pct_s"] * d, 1, True)}% of width over {d:.2f} s'
                elif c.get('move') in ('tilt_up', 'tilt_down') and c.get('pan_y_pct_s') is not None:
                    desc += f', content {num(c["pan_y_pct_s"] * d, 1, True)}% of height over {d:.2f} s'
                else:
                    desc += f' over {d:.2f} s'
                bits.append(f'shot {s["n"]}: {desc}, {ease_name(c.get("ease"))}')
            L += ['', '**Biggest moves:** ' + ' · '.join(bits) + '.']
        static = sum(1 for s in S if (s.get('camera') or {}).get('move') == 'static' and ((s.get('camera') or {}).get('confidence') or 0) >= LOW)
        untracked = sum(1 for s in S if not s.get('camera') or ((s['camera'].get('confidence') or 0) < LOW and s['camera'].get('move') == 'static'))
        if S:
            L += [f'**Locked-off shots:** {static} of {len(S)}' + (f'; {untracked} more could not be tracked (shown as "static?" or '
                                                                    f'"not measured": check the sheets)' if untracked else '') + '.']
        cad = (self.tl.get('video') or {}).get('cadence') or {}
        if cad.get('label'):
            lab = cad['label']
            desc = {'ones': 'a new picture every frame (on ones)', 'twos': 'each picture held for two frames (on twos)',
                    'threes': 'each picture held for three frames (on threes)'}.get(lab, f'about {cad.get("unique_fps")} new pictures a second')
            mvf = cad.get('moving_frames')
            low = bool(cad.get('optional')) or (cad.get('confidence') is not None and cad['confidence'] < 0.4) or (mvf is not None and mvf < 30)
            if low:
                L += [f'**Animation cadence (low confidence):** {desc}, but it rests on only {mvf} moving frames'
                      + (f' (confidence {cad["confidence"]:.2f})' if cad.get('confidence') is not None else '') +
                      f': optional, check the sheets before matching it.']
            else:
                L += [f'**Animation cadence:** {desc}: {cad.get("unique_fps")} unique frames per second of {self.fpsl} while things move '
                      f'(held-frame share {cad.get("hold_share")}, {mvf} moving frames). Match it, or the motion will feel smoother or '
                      f'choppier than the reference.']
        return block('camera', 5, '\n'.join(L))

    def transitions_block(self):
        S = self.shots
        T = []
        for s in S[1:]:
            ti = s.get('transition_in') or {}
            T.append((s, ti))
        out = (S[-1].get('transition_out') or {}) if S else {}
        out_line = ''
        if out.get('type'):
            fr = out.get('frames') or 0
            out_line = (f'**Out:** the last shot ends with a {move_name(out["type"])}' + (f' over {fr} f ({fr / self.fps:.2f} s)' if fr else '')
                        + (f', from frame {S[-1]["end_frame"] - fr}' if fr and S[-1].get('end_frame') is not None else '') + '.')
        if not T:
            ic_rows = [[s['n'], secs(t), int(round(t * self.fps)), lab] for s in S for t, lab in self.internal_changes(s)]
            hc = self.hidden_cuts()
            lead = ('One detected shot, with hidden cuts and changes inside it' if hc else
                    'One continuous shot: no cuts. Changes happen inside the shot (camera, light, layout)')
            body = ('### Transitions (measured)\n\n' + lead
                    + ('; the measured ones:\n\n' + table(['shot', 'at (s)', f'frame @{self.fpsl}', 'change'], ic_rows[:40]) if ic_rows else '.'))
            return block('transitions', 5, body + ('\n\n' + out_line if out_line else ''))
        kinds = {}
        for s, ti in T:
            kinds.setdefault(ti.get('type') or 'unknown', []).append(ti.get('frames') or 0)
        plural = {'flash': 'flashes', 'fade_from_black': 'fades from black', 'fade_from_white': 'fades from white',
                  'unknown': 'boundaries with no visible cut (continuous)'}
        name = lambda k, n: (plural.get(k, move_name(k) + 's') if n > 1 else tname(k))
        summ = ' · '.join(f'{len(v)} {name(k, len(v))}' + (f' ({min(v)}–{max(v)} f)' if max(v) and min(v) != max(v) else f' ({max(v)} f)' if max(v) else '')
                          for k, v in sorted(kinds.items(), key=lambda kv: -len(kv[1])))
        L = ['### Transitions (measured)', '', f'{len(T)} transitions: {summ}. Frames are at {self.fpsl} fps; '
             f'a transition of n frames lasts n ÷ {self.fpsl} s. Hard cuts are reliable; the blur types (whip versus zoom) are the '
             f'weakest labels, so read their confidence and check the sheets.', '']
        cutsT = [(s, ti) for s, ti in T if ti.get('type') == 'cut' and (ti.get('confidence') is None or ti['confidence'] >= LOW)]
        rest = [(s, ti) for s, ti in T if (s, ti) not in cutsT] if len(T) > 8 else T
        if len(T) > 8 and cutsT:
            fr = ', '.join(str(s['start_frame']) for s, _ in cutsT[:60]) + (' …' if len(cutsT) > 60 else '')
            L.append(f'**Hard cuts** (0 frames), at frames {fr} (@{self.fpsl}).')
            if rest:
                L.append('')
        if rest:
            rows = []
            for s, ti in rest[:40]:
                fr = ti.get('frames') or 0
                ty = tname(ti.get('type')) + ''.join(f' {ti[k]}' for k in ('direction', 'axis') if ti.get(k))
                rows.append([s['n'], secs(s.get('start')), s.get('start_frame', '–'), ty,
                             f'{fr} f · {fr / self.fps:.2f} s' if fr else '0 f', secs(ti.get('confidence'))])
            L.append(table(['into #', 'at (s)', f'frame @{self.fpsl}', 'type', 'length', 'conf'], rows))
            if len(rest) > 40:
                L += ['', f'… {len(rest) - 40} more in the timeline JSON.']
        ic_rows = []
        for s in S:
            for t, lab in self.internal_changes(s):
                ic_rows.append([s['n'], secs(t), int(round(t * self.fps)), lab])
        if ic_rows:
            L += ['', '**Changes inside shots** (the picture changes without a cut: a reframe, a hand-off behind black, a mask):', '',
                  table(['shot', 'at (s)', f'frame @{self.fpsl}', 'change'], ic_rows[:40])]
        if out_line:
            L += ['', out_line]
        return block('transitions', 5, '\n'.join(L))

    # -- plate 6
    def clean_text(self):
        """The on-screen text the type table shows, before ranking: one item per line per shot, OCR fragments and
        misreads removed, counters merged. Returns (items, noise); each item is {t, shots, first, last}."""
        raw = []
        for s in self.shots:
            for t in s.get('text') or []:
                if (t.get('text') or '').strip():
                    raw.append((s['n'], t))
        # one row per distinct text per shot (OCR sees the same words on many frames)
        seen = {}
        for n, t in sorted(raw, key=lambda x: (x[1].get('first') or 0)):
            k = (n, re.sub(r'[^a-z0-9]+', '', (t.get('text') or '').lower()))
            if k in seen:
                it = seen[k]
                it['last'] = max(it['last'] or 0, t.get('last') or 0)
                if (t.get('cap_h_px_1080') or 0) > (it['t'].get('cap_h_px_1080') or 0):
                    it['t'] = t
            else:
                seen[k] = {'t': t, 'shots': [n], 'first': t.get('first'), 'last': t.get('last')}
        items = list(seen.values())
        px0 = lambda it: it['t'].get('cap_h_px_1080') or 0
        keyof = lambda it: re.sub(r'[^a-z0-9]+', '', (it['t'].get('text') or '').lower())
        # OCR of animating text leaves fragments: drop reads under 3 letters, and partial reads of a longer line
        # in the same shot at about the same size (a word typing on, a line wiping in)
        def covered(k, ko):   # share of k's letters found, in order, in the longer ko (OCR misreads included)
            return sum(b.size for b in difflib.SequenceMatcher(None, k, ko, autojunk=False).get_matching_blocks()) / max(1, len(k))
        keep = []
        for it in items:
            k = keyof(it)
            if len(k) < 3:
                continue
            if any(o is not it and o['shots'][0] == it['shots'][0] and len(keyof(o)) > len(k)
                   and (k in keyof(o) or covered(k, keyof(o)) >= 0.8)
                   and px0(o) and abs(px0(it) - px0(o)) <= 0.35 * px0(o) for o in items):
                continue
            keep.append(it)
        # the same line held across a cut is one row ("1–3")
        keep.sort(key=lambda it: (it['first'] or 0))
        joined = []
        for it in keep:
            b = it['t'].get('box') or [0, 0, 0, 0]
            prev = next((o for o in reversed(joined) if keyof(o) == keyof(it) and o['shots'][-1] in (it['shots'][0] - 1, it['shots'][0])), None)
            if prev:
                pb = prev['t'].get('box') or [0, 0, 0, 0]
                if abs(pb[0] - b[0]) < 0.03 and abs(pb[1] - b[1]) < 0.03:
                    if it['shots'][0] not in prev['shots']:
                        prev['shots'].append(it['shots'][0])
                    prev['last'] = max(prev['last'] or 0, it['last'] or 0)
                    continue
            joined.append(it)
        items = joined
        # a number counting up or down in one place becomes one row: "135.03 → 213.00"
        isnum = lambda it: bool(re.fullmatch(r'[\d.,:%$€£+\-\s]+', (it['t'].get('text') or '').strip())) and any(ch.isdigit() for ch in it['t'].get('text') or '')
        merged_items = []
        for it in sorted(items, key=lambda it: (it['shots'][0], it['first'] or 0)):
            prev = next((o for o in reversed(merged_items) if o.get('_num') and o['shots'][0] == it['shots'][0]
                         and (it['first'] or 0) - (o['last'] or 0) <= 1.0), None)
            if prev and isnum(it):
                b0, b1 = prev['t'].get('box') or [0, 0, 0, 0], it['t'].get('box') or [0, 0, 0, 0]
                c0, c1 = (b0[0] + b0[2] / 2, b0[1] + b0[3] / 2), (b1[0] + b1[2] / 2, b1[1] + b1[3] / 2)
                if abs(c0[0] - c1[0]) < 0.12 and abs(c0[1] - c1[1]) < 0.3 and px0(prev) and abs(px0(it) - px0(prev)) <= 0.3 * px0(prev):
                    prev['_last_text'] = it['t'].get('text')
                    prev['last'] = max(prev['last'] or 0, it['last'] or 0)
                    if px0(it) > px0(prev):
                        prev['t'] = dict(prev['t'], cap_h_px_1080=px0(it))
                    continue
            if isnum(it):
                it = dict(it, _num=True)
            merged_items.append(it)
        for it in merged_items:
            if it.get('_last_text'):
                it['t'] = dict(it['t'], text=f"{it['t'].get('text')} → {it['_last_text']} (counting)")
        items = merged_items
        # reads that aren't words (blurred or moving text the OCR misread) and one-frame reads of short text stay out of
        # the table and the type scale; they are counted, and all of them remain in the timeline JSON
        extra = set()
        extra = self.extra_words
        step = 0.5 if self.dur > 30 else 0.25
        noise = []
        clean = []
        for it in items:
            one = (it['last'] or 0) - (it['first'] or 0) < step * 0.9 and len((it['t'].get('text') or '').split()) < 3
            if not wordlike(it['t'].get('text'), extra) or one:
                noise.append(it)
            else:
                clean.append(it)
        return clean, noise

    def type_block(self):
        items, noise = self.clean_text()
        minpx = PHONE_MIN_W * self.W
        px = lambda it: (it['t'].get('cap_h_px_1080') or 0) * self.H / 1080
        held = lambda it: max(0.1, (it['last'] or 0) - (it['first'] or 0))
        hidden = []
        if len(items) > 18:
            # keep the designed type: lines of 2+ words held 0.5 s at or above the phone minimum, the first line, the end
            # card and the largest read; then the most prominent of the rest. Labels inside interfaces fall to "hidden".
            last_shot = self.shots[-1]['n'] if self.shots else None
            must = []
            for it in sorted(items, key=lambda it: it['first'] or 0):
                nw = len((it['t'].get('text') or '').split())
                if (nw >= 2 and held(it) >= 0.5 and px(it) >= minpx) or it['shots'][0] == last_shot:
                    must.append(it)
            first_it = min(items, key=lambda it: it['first'] or 0)
            big_it = max(items, key=lambda it: px(it))
            for it in (first_it, big_it):
                if it not in must:
                    must.append(it)
            must = sorted(must, key=lambda it: -(px(it) * held(it) ** 0.5))[:18]
            rest = sorted([it for it in items if it not in must], key=lambda it: -(px(it) * held(it) ** 0.5))
            room = max(0, 18 - len(must))
            items, hidden = must + rest[:room], rest[room:]
        items.sort(key=lambda it: (it['first'] or 0))
        L = ['### Type (measured)', '']
        if not items:
            L.append('No clean on-screen text was measured in this film.' if noise else 'No on-screen text was measured in this film.')
        else:
            L.append(f'Sizes are **cap height** (the height of a capital letter) in px at the 1080p deliverable, {self.W}×{self.H}. '
                     f'Font size ≈ cap ÷ 0.71 for a typical sans (Inter 0.73, Geist 0.71, SF Pro 0.70, Instrument Sans 0.72): '
                     f'check your own font\'s cap-height ratio. Cap heights are estimated from text-recognition boxes (median error about '
                     f'11% on real text; text under about 13 px at 1080p and non-Latin scripts are not read). Box = x · y · w · h as % of '
                     f'the frame, from the top-left.')
            L.append('')
            rows = []
            for it in items:
                t = it['t']
                cap = t.get('cap_h_px_1080')
                cpx = cap * self.H / 1080 if cap is not None else None
                box = t.get('box') or None
                pos = '–'
                if box and len(box) == 4:
                    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
                    hx = 'left' if cx < 0.36 else 'right' if cx > 0.64 else 'centre'
                    vy = 'top' if cy < 0.22 else 'upper' if cy < 0.42 else 'middle' if cy < 0.58 else 'lower third' if cy < 0.8 else 'bottom'
                    edge = min(box[0], box[1], 1 - box[0] - box[2], 1 - box[1] - box[3])
                    pos = f'{hx}, {vy}' + (' (inside 5% of the edge)' if edge < 0.05 else '')
                    box = ' · '.join(f'{round(v * 100) or 0:.0f}' for v in box)
                if t.get('angle_deg') is not None and abs(t['angle_deg']) >= 5:   # v2.1: size measured on the rotated line
                    pos = (pos if pos != '–' else '') + f' · tilted {t["angle_deg"]:+.0f}°'
                sh = str(it['shots'][0]) if len(it['shots']) == 1 else f"{it['shots'][0]}–{it['shots'][-1]}"
                rows.append([f'"{short_text(t.get("text"))}"', sh, f"{secs(it['first'], 1)}–{secs(it['last'], 1)}",
                             '–' if cpx is None else f'{cpx:.0f}', '–' if cpx is None else f'{cpx / 0.71:.0f}',
                             '–' if cap is None else f'{cap / 1080 * 100:.1f}', box or '–', pos])
            L.append(table(['text', 'shot', 'on screen (s)', 'cap px', '≈ font px', 'cap % of height', 'box x·y·w·h %', 'position'], rows))
            if hidden:
                hp = sorted(px(it) for it in hidden if px(it))
                rng_ = f', cap {hp[0]:.0f}–{hp[-1]:.0f} px' if hp else ''
                L += ['', f'Plus {len(hidden)} smaller or briefer text items{rng_}, mostly labels inside interfaces: all in the timeline JSON.']
            caps = sorted({round(px(it)) for it in items if px(it)}, reverse=True)
            if caps:
                # sizes within 12% of each other are one level: the measurement itself is good to about 11%
                steps = []
                for c in caps:
                    if not steps or c < steps[-1][-1] * 0.88:
                        steps.append([c])
                    else:
                        steps[-1].append(c)
                scale = ' · '.join(f'{s[0]}' if len(s) == 1 else f'{s[-1]}–{s[0]}' for s in steps)
                L += ['', f'**Type scale (cap px):** {scale}' + (f' ({len(steps)} levels)' if len(steps) > 1 else '')
                      + (' among the text above' if hidden else '') + '.']
        if noise:
            ex = ', '.join(f'"{short_text(it["t"].get("text"), 3, 18)}"' for it in noise[:3])
            L += ['', f'Left out: {len(noise)} reads that are not clean words or were seen in a single frame ({ex}…): blurred, '
                      f'moving or partly hidden text the recognizer misread. They stay in the timeline JSON, not in the type scale.']
        allit = items + hidden
        below = [it for it in items if it['t'].get('cap_h_px_1080') is not None and px(it) < minpx]
        # copy before labels: multi-word lines first, then the longest held
        below.sort(key=lambda it: (-(len((it['t'].get('text') or '').split()) >= 3), -held(it)))
        below_hidden = sum(1 for it in hidden if it['t'].get('cap_h_px_1080') is not None and px(it) < minpx)
        why = (f'**Phone minimum: cap height ≥ 2% of frame width = {minpx:.0f} px at {self.W}×{self.H}.** Why: in a phone feed the film '
               f'fills the screen width, about 390 pt, and 2% of that is 7.8 pt, the cap height of 11 pt text, Apple\'s smallest '
               f'recommended text size. Words the viewer must read go at or above it; smaller text can only be decoration.')
        if not allit:
            L += ['', why]
        elif below or below_hidden:
            names = ', '.join(f'"{short_text(it["t"].get("text"), 4, 28)}" ({px(it):.0f} px, shot {it["shots"][0]})' for it in below[:6])
            more = len(below) - 6 if len(below) > 6 else 0
            tail = []
            if more:
                tail.append(f'{more} more above')
            if below_hidden:
                tail.append(f'{below_hidden} of the smaller items')
            lst = (names + (' and ' + ' and '.join(tail) if tail else '')) if names else ' and '.join(tail)
            L += ['', why + f' **Below it in the reference:** {lst}. Size these up in your version, or treat them as decoration.']
        else:
            L += ['', why + ' Every measured text item in the reference clears it.']
        return block('type', 6, '\n'.join(L))

    def palette_block(self):
        S = self.shots
        rows = []
        agg = []
        def near(a, b):
            (r1, g1, b1), (r2, g2, b2) = hexrgb(a), hexrgb(b)
            return abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2) < 60

        def alike(p, q):  # the two main colours of each shot appear in the other's top three
            tp, tq = [x['hex'] for x in p[:3] if x.get('hex')], [x['hex'] for x in q[:3] if x.get('hex')]
            return bool(tp and tq) and all(any(near(a, b) for b in tq) for a in tp[:2]) and all(any(near(a, b) for b in tp) for a in tq[:2])

        groups = []   # consecutive shots with alike palettes share one row (the longest shot's colours)
        for s in S:
            pal = s.get('palette') or []
            lu = s.get('luma') or {}
            if not pal and not lu:
                continue
            for p in pal:
                if p.get('hex') and p.get('share') is not None:
                    agg.append((p['hex'].upper(), p['share'] * (s.get('dur') or 0)))
            if groups and pal and alike(groups[-1][-1].get('palette') or [], pal):
                groups[-1].append(s)
            else:
                groups.append([s])
        for g in groups[:20]:
            best = max(g, key=lambda s: s.get('dur') or 0)
            pal, lus = best.get('palette') or [], [s.get('luma') or {} for s in g]
            means = [l['mean'] for l in lus if l.get('mean') is not None]
            p95s = [l['p95'] for l in lus if l.get('p95') is not None]
            lum = (f"{sum(means) / len(means):.2f} · {max(p95s):.2f}" if means and p95s else '–')
            label = str(g[0]['n']) if len(g) == 1 else f"{g[0]['n']}–{g[-1]['n']}"
            row = [label, lum, ' · '.join(f"`{p['hex'].upper()}` {round(p['share'] * 100)}%" for p in pal[:5]) or '–']
            if any(x.get('highlights') for x in S):
                row.append(fmt_highlights(best.get('highlights')))
            rows.append(row)
        if not rows:
            return block('palette', 6, '### Palette by shot (measured)\n\nNo colour data in the timeline.')
        # time-weighted film palette, merging near-identical colours
        merged = []
        for hx, wgt in sorted(agg, key=lambda a: -a[1]):
            r, g, b = hexrgb(hx)
            for m in merged:
                R, G, B = hexrgb(m[0])
                if abs(R - r) + abs(G - g) + abs(B - b) < 48:
                    m[1] += wgt
                    break
            else:
                merged.append([hx, wgt])
        tot = sum(m[1] for m in merged) or 1
        film = ' · '.join(f'`{m[0]}` {m[1] / tot * 100:.0f}%' for m in sorted(merged, key=lambda m: -m[1])[:6])
        L = ['### Palette by shot (measured)', '',
             'Colours measured on the frames, with their share of the frame; luma 0 = black, 1 = white. These are the '
             'reference\'s brand colours: map them onto {{BRAND_BG}} and {{BRAND_ACCENTS}} by role (background, '
             'surface, accent), don\'t reuse them as the format. Neighbouring shots with the same palette share a row.', '',
             table(['#', 'luma mean · p95', 'colours (share of frame)'] + (['highlights (bright pixels)'] if any(x.get('highlights') for x in S) else []), rows)]
        if len(groups) > 20:
            L += ['', f'… {len(groups) - 20} more rows (shots {groups[20][0]["n"]}–{S[-1]["n"]}) in the timeline JSON.']
        L += ['', f'**Whole film (time-weighted):** {film}.']
        return block('palette', 6, '\n'.join(L))

    def brightness_block(self):
        b = self.tl.get('brightness') or {}
        mean, p95, step = b.get('mean') or [], b.get('p95') or [], b.get('step_s') or 0.25
        if not mean:
            return block('brightness', 6, '### Brightness curve (measured)\n\nNot measured.')
        width = 1 if self.dur <= 120 else 2 if self.dur <= 240 else 5 if self.dur <= 600 else 10
        m1, p1 = binned(mean, step, width), binned(p95, step, width) if p95 else []
        top = max([v for v in (p1 or m1) if v is not None] + [0.2])
        top = min(1.0, math.ceil(top * 10) / 10)
        def rows_of(vals):
            groups = [vals[i:i + 10] for i in range(0, len(vals), 10)]
            return ' '.join(spark(g, 0.0, top) for g in groups)
        ruler = ''
        for i in range(0, len(m1), 10):
            lab = f'{i * width}s'
            ruler += lab.ljust(11)
        L = ['### Brightness curve (measured)', '',
             f'Mean luma per {width} s, one character each, grouped in tens; p95 = the bright highlights. Scale: ▁ = 0 (black) → '
             f'█ = {top:.1f}' + (' (white)' if top >= 1 else ' (nothing in this film is brighter)') + '; luma runs 0–1.', '',
             '```text',
             '      ' + ruler.rstrip(),
             'mean  ' + rows_of(m1)]
        if p1:
            L.append('p95   ' + rows_of(p1))
        L.append('```')
        vals = [(i * step, v) for i, v in enumerate(mean) if v is not None]
        lo = min(vals, key=lambda x: x[1])
        hi = max(vals, key=lambda x: x[1])
        avg = sum(v for _, v in vals) / len(vals)
        facts = [f'average {avg:.2f}', f'darkest {lo[1]:.2f} at {lo[0]:.1f} s', f'brightest {hi[1]:.2f} at {hi[0]:.1f} s']
        jumps = []
        for i in range(1, len(mean)):
            if mean[i] is not None and mean[i - 1] is not None and abs(mean[i] - mean[i - 1]) >= 0.18:
                t = i * step
                if not jumps or t - jumps[-1][0] > 0.6:
                    jumps.append((t, mean[i] - mean[i - 1]))
        if jumps:
            facts.append('sudden changes at ' + ', '.join(f'{t:.2f} s ({"brighter" if d > 0 else "darker"} by {abs(d):.2f})' for t, d in jumps[:8]))
        # fade to / from black
        fo = self.fade_out()
        if fo:
            facts.append(f'fades to black from about {fo[0]:.2f} s (frame {int(round(fo[0] * self.fps))}) to the end')
        fi = self.fade_in()
        if fi is not None:
            facts.append(f'opens from black, up by about {fi:.2f} s')
        elif mean[0] is not None and mean[0] < 0.03:
            facts.append('opens on black')
        L += ['', '**Curve facts:** ' + ' · '.join(facts) + '.']
        return block('brightness', 6, '\n'.join(L))

    # -- plate 7
    def speech_block(self):
        """Per-phrase speech times in frames: narration-led rebuilds cut and caption on these."""
        W_ = self.raw_words
        if not W_:
            return block('speech', 7, '### Speech timing (measured)\n\nNo speech in this film.')
        phrases, cur = [], [W_[0]]
        for a, b in zip(W_, W_[1:]):
            if b[0] - a[1] >= 0.25 or re.search(r'[.!?,;:…]$', str(a[2])) or len(cur) >= 12:
                phrases.append(cur)
                cur = []
            cur.append(b)
        phrases.append(cur)
        fps = self.fps
        rows = []
        for i, ph in enumerate(phrases[:80], 1):
            a, b = float(ph[0][0]), float(ph[-1][1])
            txt = ' '.join(str(w[2]).strip() for w in ph)
            wps = len(ph) / max(0.1, b - a)
            rows.append([i, f'{a:.2f}–{b:.2f}', f'{int(round(a * fps))}–{int(round(b * fps))}', shot_at(self.shots, a) if self.shots else '–',
                         len(ph), f'{wps:.1f}', f'"{short_text(txt, 7, 44)}"'])
        L = ['### Speech timing (measured)', '',
             f'Each spoken phrase (split at pauses of 0.25 s or more and at punctuation) with its times and frames at {self.fpsl} fps, '
             f'from the transcript\'s word timings. Captions and cuts in a narration-led film land on these; the words are '
             f'truncated here (the full transcript is in data/transcript.txt, for reference only).', '',
             table(['#', 'time (s)', f'frames @{self.fpsl}', 'shot', 'words', 'words/s', 'starts with'], rows)]
        if len(phrases) > 80:
            L += ['', f'… {len(phrases) - 80} more phrases in data/transcript.json.']
        return block('speech', 7, '\n'.join(L))

    # -- plate 8
    def sound_block(self):
        a = self.tl.get('audio') or {}
        if not a or a.get('has_audio') is False:
            return block('sound', 8, '### Sound grid (measured)\n\nNo audio track: the reference plays silent. If your version '
                                     'needs sound, plate 8 says what fits; the platform loudness targets below still apply.\n\n'
                         + self.loudness_table(None))
        L = ['### Sound grid (measured)', '']
        # speech first: in a narration-led film the words set the cuts, and the grid comes second
        if self.words and self.dur:
            spoken = sum(max(0.0, e - b_) for b_, e in self.words)
            share = min(1.0, spoken / self.dur)
            kind = ('narration-led: the words carry it, cut on phrases and keep effects off the stressed words' if share >= 0.5
                    else 'some speech over picture and music' if share >= 0.15 else 'little speech')
            L += [f'**Speech:** words cover {share * 100:.0f}% of the film ({len(self.words)} words in the transcript): {kind}.', '']
        t = a.get('tempo') or {}
        bpm, conf = t.get('bpm'), t.get('confidence')
        first = t.get('first_downbeat_s')
        steady = bpm and conf is not None and conf >= TEMPO_MIN
        if not bpm:
            L.append('**Tempo:** no steady pulse was found. Time the edit to the cue sheet below, not to a BPM.')
        elif not steady:
            L.append(f'**Tempo:** no steady pulse. The best guess, {bpm:g} BPM, has confidence {conf:.2f}, too low to build on: '
                     f'time the edit to the cue sheet below, not to a BPM.')
        else:
            P = t.get('beat_period_s') or 60.0 / bpm
            weak = conf < TEMPO_SOLID
            felt = bpm / 2 if bpm > 140 else bpm * 2 if bpm < 70 else None
            octave = (f' (felt as {felt:.1f} BPM: count {"every other beat" if bpm > 140 else "twice per beat"})' if felt else '')
            dbc = t.get('downbeat_confidence')
            L.append(f'**Tempo:** {bpm:g} BPM{octave}, confidence {conf:.2f}' + (' (weak: half or double time is possible)' if weak else '')
                     + (f' · first downbeat {first:.3f} s' + (f' (bar start confidence {dbc:.2f}' + (', uncertain)' if dbc < 0.5 else ')')
                                                              if dbc is not None else '') if first is not None else '')
                     + f' · beat {P:.4f} s · bar of 4 = {4 * P:.3f} s.')
            constant = str(t.get('grid') or '').startswith('constant')
            beats = t.get('beats') or []
            downs = t.get('downbeats') or []
            if constant and first is not None:
                downs = [first + 4 * P * k for k in range(int((self.dur - first) / (4 * P)) + 1)]
                beats = [first + P * k for k in range(-int(first / P), int((self.dur - first) / P) + 1)]
            fpss = []
            for f in (self.fps, 30.0, 60.0):
                if all(abs(f - g) > 0.01 for g in fpss):
                    fpss.append(f)
            rows = []
            for f in fpss:
                fb = P * f
                dl = ' · '.join(str(int(round(d * f))) for d in downs[:8]) + (' …' if len(downs) > 8 else '')
                rows.append([f'{f:g} fps' + (' (this film)' if f == self.fps else ''), f'{fb:.2f} f', f'{4 * fb:.2f} f',
                             '–' if first is None else int(round(first * f)), dl or '–'])
            how = ('A constant grid fitted to the music: beat k is at round((first downbeat + k × beat) × fps); round each beat on '
                   'its own, since adding a rounded beat length drifts.' if constant else
                   'Tracked beats, not a constant grid (the tempo drifts): use the listed times, not a formula.')
            L += ['', f'**Beat grid in frames.** {how}', '',
                  table(['', 'per beat', 'per bar', 'first downbeat (frame)', f'downbeats (frame), first {min(8, len(downs))}'], rows)]
            if beats:
                L += ['', 'Beats in seconds: ' + ' · '.join(f'{b_:.3f}' for b_ in beats[:12])
                      + (f' … ({len(beats)} in all, `audio.tempo.beats` in the timeline JSON).' if len(beats) > 12 else '.')]
            if self.shots and len(self.shots) > 1 and beats:
                tol = 1.5 / self.fps
                n = len(self.shots) - 1
                on = sum(1 for s_ in self.shots[1:] if min(abs(b_ - s_['start']) for b_ in beats) <= tol)
                chance = n * min(1.0, 2 * tol / P)
                verdict = ('so the edit is cut to the beat' if on >= max(2, 2.5 * chance) and on >= 0.4 * n else
                           'so the edit does not follow the beat')
                L.append(f'**Cuts on the beat:** {on} of {n} land within ±1.5 frames of a beat; chance alone would give about '
                         f'{chance:.1f}, {verdict}.')
        k = a.get('key') or {}
        if k.get('name') and (k.get('confidence') or 0) >= KEY_MIN:
            L += ['', f'**Key:** {k["name"]} (confidence {k["confidence"]:.2f}' + (f'; runner-up {k["runner_up"]}' if k.get('runner_up') else '') + ').']
        elif k.get('name'):
            L += ['', f'**Key:** unclear: {k["name"]}' + (f' or {k["runner_up"]}' if k.get('runner_up') else '')
                  + f' (confidence {k.get("confidence") or 0:.2f}); pick your own.']
        else:
            L += ['', '**Key:** none reported (little tonal music).']
        secs_ = a.get('sections') or []
        if secs_:
            rows = []
            for s_ in secs_[:16]:
                sh = [x['n'] for x in self.shots if x['start'] < s_['end'] and x['end'] > s_['start']]
                shs = (f'{sh[0]}' if len(sh) == 1 else f'{sh[0]}–{sh[-1]}') if sh else '–'
                rows.append([s_.get('label') or '–', f"{secs(s_.get('start'), 1)}–{secs(s_.get('end'), 1)}", shs,
                             f"{num(s_.get('rms_db'), 0)} dB" if s_.get('rms_db') is not None else '–'])
            L += ['', '**Sections** (by energy of the whole mix' + (', voice included' if self.words else '') + '):', '',
                  table(['section', 'time (s)', 'shots', 'level (RMS dBFS, not LUFS)'], rows)]
        en = a.get('energy') or {}
        rms, est = en.get('rms_db') or [], en.get('step_s') or 0.5
        if rms:
            width = 1 if self.dur <= 120 else 2 if self.dur <= 240 else 5 if self.dur <= 600 else 10
            r1 = binned(rms, est, width)
            vals = [v for v in r1 if v is not None]
            lo, hi = min(vals), max(vals)
            groups = [r1[i:i + 10] for i in range(0, len(r1), 10)]
            L += ['', f'**Energy map** (RMS per {width} s, ▁ = {num(lo, 0)} dB → █ = {num(hi, 0)} dB):', '',
                  '```text', ' '.join(spark(g, lo, hi) for g in groups), '```']
        cues = a.get('cues') or []
        if cues:
            def voiced(tc):
                return any(b_ - 0.08 <= tc <= e for b_, e in self.words)
            top = sorted(cues, key=lambda c: -(c.get('strength') or 0))[:24]
            top.sort(key=lambda c: c.get('t') or 0)
            rows = []
            nv = 0
            ev = self.events()
            for c in top:
                sh = c.get('shot') or (shot_at(self.shots, c['t']) if self.shots else '–')
                v = self.words and c.get('t') is not None and voiced(c['t'])
                nv += 1 if v else 0
                rows.append([secs(c.get('t')), frame(c.get('t'), self.fps), sh, (c.get('kind') or '–') + (' · voice?' if v else ''),
                             secs(c.get('strength')), self.on_screen(c['t'], ev) if c.get('t') is not None else '–',
                             '✓' if c.get('on_beat') else ''])
            nc = sum(1 for c in cues if c.get('near_cut'))
            nb = sum(1 for c in cues if c.get('on_beat'))
            L += ['', f'**Cue sheet** ({len(cues)} cues: {nc} at a cut, {nb} on a beat'
                      + (f'; the {len(top)} strongest shown' if len(cues) > len(top) else '') + '). "On screen" names the '
                      'picture event the cue hits within ±3 frames (a cut, a jump, a move starting, a line of text appearing, '
                      'a brightness step), or says there is none:', '',
                  table(['t (s)', f'frame @{self.fpsl}', 'shot', 'kind', 'strength', 'on screen (±3 f)', 'on beat'], rows)]
            if nv:
                L += ['', f'"voice?" = the cue sits on a spoken word ({nv} of {len(top)}): that is the narration, not an effect to copy.']
        else:
            L += ['', '**Cue sheet:** no distinct hits, whooshes or clicks were found.']
        L += ['', self.loudness_table(a.get('loudness'))]
        return block('sound', 8, '\n'.join(L))

    def loudness_table(self, loud):
        L = []
        if loud and loud.get('integrated_lufs') is not None:
            L.append(f"**Loudness of the reference (measured):** {num(loud['integrated_lufs'], 1)} LUFS integrated, true peak "
                     f"{num(loud.get('true_peak_dbfs'), 1)} dBTP, range {num(loud.get('range_lu'), 1)} LU. That is how this film was "
                     f"mastered, not a target" + (" (its peak is above −1 dBTP: don't copy that)" if loud.get('true_peak_dbfs') is not None and loud['true_peak_dbfs'] > -1.0 else '') + '.')
        else:
            L.append('**Loudness of the reference:** not measured.')
        plat = (self.tl.get('source') or '').lower()
        here = 'YouTube' if 'youtu' in plat else 'X, LinkedIn' if ('x.com' in plat or 'twitter' in plat) else \
            'TikTok, Reels, Shorts' if ('tiktok' in plat or 'instagram' in plat) else None
        rows = [['YouTube', '−14 LUFS', '−1 dBTP', 'louder masters are turned down; quieter ones are not turned up'],
                ['X, LinkedIn', '−14 LUFS', '−1 dBTP', 'no published normalisation: the file plays as mastered'],
                ['TikTok, Reels, Shorts', '−14 to −10 LUFS', '−1 dBTP', 'played on phone speakers; short-form is often mastered hotter'],
                ['Website or app (autoplay)', '−16 LUFS', '−1 dBTP', 'often starts muted; keep the unmuted level comfortable'],
                ['Broadcast (EBU R128)', '−23 LUFS', '−1 dBTP', 'only when the brief asks for it']]
        for r in rows:
            if here and r[0] == here:
                r[0] = f'**{r[0]}** (this reference)'
        L += ['', '**Loudness targets for your version** (common practice per platform; the brief overrides them):', '',
              table(['where it plays', 'integrated', 'true peak max', 'note'], rows)]
        return '\n'.join(L)


def render(tl, words=None, extra_words=()):
    f = Film(tl, words, extra_words)
    v = tl.get('video') or {}
    head = (f'<!-- measured.md · {VERSION} · rendered {datetime.datetime.now().isoformat(timespec="seconds")} from timeline.json measured '
            f'{tl.get("measured_at")} · source {v.get("width")}×{v.get("height")} {f.aspect} at {f.fpsl} fps, {f.dur:.2f} s · deliverable '
            f'{f.W}×{f.H} -->\n<!-- Each block between "measured:KEY plate=N" markers is pasted verbatim into plate N of the blueprint '
            f'by `dissect.py assemble`. Blocks with plate=0 are for the writer only. -->')
    at = f.aspect_table()
    parts = [head, block('facts', 0, f.timing_facts())] + ([block('aspect', 0, at)] if at else []) + [f.hook_sheet(), f.shots_block(), f.sheets_block(), f.camera_block(),
             f.transitions_block(), f.type_block(), f.palette_block(), f.brightness_block(), f.speech_block(), f.sound_block()]
    return '\n\n'.join(parts) + '\n'


def parse_blocks(md):
    """{key: (plate, body)} from a measured.md."""
    out = {}
    for m in re.finditer(r'<!-- measured:(\w+) plate=(\d+) -->\n(.*?)\n<!-- /measured:\1 -->', md, re.S):
        out[m.group(1)] = (int(m.group(2)), m.group(3))
    return out
