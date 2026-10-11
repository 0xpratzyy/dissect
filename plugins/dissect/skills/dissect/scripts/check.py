"""check.py - score a render against a dissected reference.

    uv run dissect.py check <your-render.mp4> --ref <dissection folder> [--target x|youtube|tiktok|web] [--json]

Measures the render (cuts, brightness, motion, loudness: the same measures as the Refs starter kit's measure.mjs) and
scores it against the reference's data/timeline.json with the same logic as Refs' check_render: cuts 35, brightness
curve 20, motion 15, pace 10, loudness 10, format 10. Prints a score out of 100 and concrete fixes in the RENDER's
frames ("Cut 2 lands 4 f late at f131; move it to f127"). Needs numpy and ffmpeg/ffprobe only.
"""
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

STEP_S = 0.25      # luma and motion bins
MERGE_S = 0.25     # detections closer than this are one cut
SCENE_MIN = 0.3    # ffmpeg scene score that counts as a cut
LONG = 256         # analysis frame, long side (px), grey
SCENE_LONG = 320


def _r(x, d=3):
    return None if x is None or not math.isfinite(x) else round(float(x), d)


def clip01(x):
    return max(0.0, min(1.0, float(x)))


def _rate(s):
    try:
        a, b = str(s).split('/')
        return float(a) / float(b) if float(b) else float(a)
    except Exception:
        try:
            return float(s)
        except Exception:
            return float('nan')


def _sane(x):
    return math.isfinite(x) and 1 < x <= 240


def probe(file):
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(file)],
                       capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f'check: ffprobe cannot read {file}: {p.stderr.strip()[-200:]}')
    info = json.loads(p.stdout or '{}')
    st = info.get('streams') or []
    v = next((s for s in st if s.get('codec_type') == 'video' and not (s.get('disposition') or {}).get('attached_pic')), None)
    if not v:
        raise SystemExit(f'check: {file} has no video stream')
    r, avg = _rate(v.get('r_frame_rate')), _rate(v.get('avg_frame_rate'))
    use_r = _sane(r) and (not _sane(avg) or abs(r - avg) / avg < 0.02)
    fps = r if use_r else avg if _sane(avg) else 30.0
    fps_arg = v.get('r_frame_rate') if use_r else v.get('avg_frame_rate') if _sane(avg) else '30'
    w, h = int(v.get('width') or 0), int(v.get('height') or 0)
    rot = 0
    for sd in v.get('side_data_list') or []:
        if 'rotation' in sd:
            rot = abs(int(float(sd['rotation']))) % 180
    sar = 1.0
    try:
        a, b = (v.get('sample_aspect_ratio') or '1:1').split(':')
        sar = float(a) / float(b) if float(b) and float(a) else 1.0
    except ValueError:
        pass
    if math.isfinite(sar) and sar > 0 and abs(sar - 1) > 0.01:
        w = round(w * sar)
    if rot == 90:
        w, h = h, w
    return dict(fps=fps, fps_arg=fps_arg, width=w, height=h, has_audio=any(s.get('codec_type') == 'audio' for s in st))


def loudness(file):
    p = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-nostats', '-i', str(file), '-map', '0:a:0', '-vn',
                        '-af', 'ebur128=peak=true', '-f', 'null', '-'], capture_output=True, text=True, errors='replace')
    s = p.stderr[p.stderr.rfind('Summary:'):] if 'Summary:' in p.stderr else ''

    def g(rx):
        m = re.search(rx, s)
        try:
            v = float(m.group(1)) if m else None
        except ValueError:
            v = None
        return None if v is None or not math.isfinite(v) else round(v, 1)
    return dict(integrated_lufs=g(r'I:\s+(-?[\d.]+|-?inf)\s+LUFS'), true_peak_dbtp=g(r'Peak:\s+(-?[\d.]+|-?inf)\s+dBFS'),
                range_lu=g(r'LRA:\s+(-?[\d.]+)\s+LU'))


def _mult4(x):
    return max(4, 4 * round(x / 4))


def _persistence(a, b):
    """Of the thumbnail blocks with detail in a, the share whose pattern survives into b (1 px shift allowed)."""
    qh, qw = a.shape
    gw, gh = (8, 5) if qw >= qh else (5, 8)
    active = kept = 0
    pad = np.pad(b, 1, mode='edge')
    for by in range(gh):
        for bx in range(gw):
            y0, y1 = by * qh // gh, (by + 1) * qh // gh
            x0, x1 = bx * qw // gw, (bx + 1) * qw // gw
            A = a[y0:y1, x0:x1]
            if A.size == 0 or A.std() < 2:
                continue
            active += 1
            za = (A - A.mean()) / A.std()
            best = -1.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    B = pad[y0 + 1 + dy:y1 + 1 + dy, x0 + 1 + dx:x1 + 1 + dx]
                    if B.std() < 1:
                        continue
                    best = max(best, float((za * (B - B.mean())).mean() / B.std()))
            if best >= 0.5:
                kept += 1
    return kept / max(1, active), active


def _blend_fit(a, m, b):
    d = b - a
    ab = float((d * d).sum())
    if ab / a.size < 4:
        return None
    alpha = float(((m - a) * d).sum()) / ab
    e = m - a - alpha * d
    return alpha, math.sqrt(float((e * e).sum()) / ab)


def _corr_dist(a, b):
    sa, sb = a.std(), b.std()
    if sa < 2 or sb < 2:
        return None
    return 1 - float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def measure_render(file):
    """The render's cuts, brightness, motion and loudness (the same measures as the Refs starter kit's measure.mjs)."""
    info = probe(file)
    W, H = info['width'], info['height']
    land = W >= H
    aw = LONG if land else _mult4(LONG * W / H)
    ah = _mult4(LONG * H / W) if land else LONG
    sw = SCENE_LONG if land else 2 * round(SCENE_LONG * W / H / 2)
    sh = 2 * round(SCENE_LONG * H / W / 2) if land else SCENE_LONG
    graph = (f"[0:v:0]fps={info['fps_arg']},split=2[sc][an];"
             f"[sc]scale={sw}:{sh}:flags=area,select='gte(scene,0)',metadata=print:key=lavfi.scene_score,nullsink;"
             f"[an]scale={aw}:{ah}:flags=area,format=gray[g]")
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(['ffmpeg', '-nostdin', '-hide_banner', '-nostats', '-v', 'info', '-i', str(file),
                                 '-filter_complex', graph, '-map', '[g]', '-an', '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1'],
                                stdout=subprocess.PIPE, stderr=errf)
        n = aw * ah
        Y, P95, D, S, thumbs = [], [], [], [], []
        prev = prev_half = None
        pm = psd = 0.0
        hw, hh = aw // 2, ah // 2
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            f = np.frombuffer(buf, np.uint8).reshape(ah, aw).astype(np.float32)
            half = f[:hh * 2, :hw * 2].reshape(hh, 2, hw, 2).mean(axis=(1, 3))
            m, sd = float(half.mean()), float(half.std())
            struct = 0.0
            if prev_half is not None:
                fl = 0.004 * 255
                if sd < fl and psd < fl:
                    struct = 0.0
                elif sd < fl or psd < fl:
                    struct = 1.0 if abs(m - pm) > 0.003 * 255 else 0.0
                else:
                    struct = max(0.0, min(2.0, 1 - (float((half * prev_half).mean()) - m * pm) / (sd * psd)))
            q = half[:(hh // 2) * 2, :(hw // 2) * 2].reshape(hh // 2, 2, hw // 2, 2).mean(axis=(1, 3))
            Y.append(float(f.mean()) / 255)
            P95.append(float(np.percentile(f, 95)) / 255)
            D.append(float(np.abs(f - prev).mean()) / 255 if prev is not None else 0.0)
            S.append(struct)
            thumbs.append(np.round(q))
            prev, prev_half, pm, psd = f, half, m, sd
        proc.wait()
        errf.seek(0)
        err = errf.read().decode('utf-8', 'replace')
    nF = len(Y)
    if nF < 2:
        raise SystemExit(f'check: ffmpeg could not decode the picture of {file}')
    fps = info['fps']
    scored = [(float(a), float(b)) for a, b in re.findall(r'pts_time:([\d.]+)[^\n]*\n[^\n]*lavfi\.scene_score=([\d.]+)', err)]
    scene = [0.0] * nF
    for t, v in scored:
        k = round((t - (scored[0][0] if scored else 0)) * fps)
        if 0 <= k < nF:
            scene[k] = v
    cuts = _find_cuts(np.array(Y), np.array(D), np.array(S), thumbs, scene, fps)

    def bins(series, digits):
        nb = max(1, round(nF / fps / STEP_S))
        s_, c_ = np.zeros(nb), np.zeros(nb)
        for k in range(nF):
            i = min(nb - 1, int((k / fps + STEP_S / 2) // STEP_S))
            s_[i] += series[k]
            c_[i] += 1
        out, last = [], 0.0
        for i in range(nb):
            last = round(s_[i] / c_[i], digits) if c_[i] else last
            out.append(last)
        return out
    aud = loudness(file) if info['has_audio'] else dict(integrated_lufs=None, true_peak_dbtp=None, range_lu=None)
    return dict(kind='refs-render-measure', version=1, file=Path(file).name, duration_s=_r(nF / fps), fps=_r(fps),
                width=W, height=H, cuts=cuts, luma=dict(step_s=STEP_S, mean=bins(Y, 3), p95=bins(P95, 3)),
                motion=dict(step_s=STEP_S, diff=bins(D, 4)), audio=dict(has_audio=info['has_audio'], **aud))


def _pct(arr, p):
    if not len(arr):
        return 0.0
    a = np.sort(np.asarray(arr, float))
    return float(a[min(len(a) - 1, max(0, round(p * (len(a) - 1))))])


def _find_cuts(Y, Dd, S, thumbs, scene, fps):
    n = len(Y)
    Wn = max(6, round(0.4 * fps))
    ev, hard = [], set()

    def local(arr, k, p):
        nb = [arr[j] for j in range(max(1, k - Wn), min(n - 1, k + Wn) + 1) if abs(j - k) >= 2]
        return _pct(nb, p)

    def add_hard(k, score, kind):
        if k >= n - 2 or max(Y[k], Y[k - 1]) < 0.005:
            return
        ev.append(dict(k=k, a=k, b=k, score=clip01(score), kind=kind, hard=True))
        hard.add(k)

    def fades_out(k, level):
        rest = float(Y[k:].max()) if k < n else 0.0
        return Y[n - 1] < 0.3 * level and rest < 0.5 * level

    for k in range(1, n):
        if scene[k] > SCENE_MIN:
            add_hard(k, max(scene[k], S[k]), 'scene')
    rel = np.zeros(n)
    for k in range(1, n):
        rel[k] = Dd[k] / max(0.02, (Y[k] + Y[k - 1]) / 2)
    for k in range(1, n):
        if k in hard or S[k] < 0.25 or k >= n - 2:
            continue
        rs = rel[k] / (local(rel, k, 0.8) + 0.01) if rel[k] >= 0.08 else 0
        ss = S[k] / (local(S, k, 0.9) + 0.02) if S[k] >= 0.3 else 0
        if rs < 5 and ss < 3:
            continue
        spike = max(rs, ss)
        share, active = _persistence(thumbs[k - 1], thumbs[k])
        ok = (share < 0.4 or (S[k] >= 0.7 and share < 0.6) or (spike >= 20 and share < 0.65)) if active >= 3 else \
            (active >= 1 and share == 0 and S[k] >= 0.9 and spike >= 10)
        if ok:
            add_hard(k, S[k], 'step' if rs >= 5 else 'struct')
    V = max(2, round(0.1 * fps))
    side = max(4, round(0.5 * fps))
    for k in range(1, n - 1):
        seg = Y[max(0, k - V):min(n, k + V + 1)]
        is_min, is_max = Y[k] <= seg.min(), Y[k] >= seg.max()
        if not is_min and not is_max:
            continue
        before = Y[max(0, k - side - V):max(0, k - V)]
        after = Y[min(n, k + V + 1):min(n, k + V + 1 + side)]
        if len(before) < 2 or len(after) < 2:
            continue
        if is_min:
            pre, post = _pct(before, 0.8), _pct(after, 0.8)
            low = min(pre, post)
            if not (Y[k] <= 0.5 * low and low - Y[k] >= 0.01):
                continue
            mid = (Y[k] + low) / 2
            a = b = k
            while a - 1 >= 0 and Y[a - 1] < mid:
                a -= 1
            while b + 1 < n and Y[b + 1] < mid:
                b += 1
            if b - a + 1 > round(1.0 * fps) or fades_out(k, pre):
                continue
            while a - 1 >= max(0, k - side) and Y[a - 1] < pre - 0.1 * (pre - Y[k]):
                a -= 1
            while b + 1 < min(n, k + side) and Y[b + 1] < post - 0.1 * (post - Y[k]):
                b += 1
            ev.append(dict(k=k, a=max(1, a), b=b, score=clip01(1 - Y[k] / max(low, 1e-4)), kind='dip', hard=False))
        else:
            pre, post = _pct(before, 0.2), _pct(after, 0.2)
            high = max(pre, post)
            rise = Y[k] - high
            if not (rise >= 0.05 and (rise / max(0.05, 1 - high) >= 0.4 or Y[k] >= 2.5 * high)):
                continue
            mid = high + rise / 2
            a = b = k
            while a - 1 >= 0 and Y[a - 1] > mid:
                a -= 1
            while b + 1 < n and Y[b + 1] > mid:
                b += 1
            if b - a + 1 > max(3, round(0.3 * fps)):
                continue
            while a - 1 >= max(0, k - side) and Y[a - 1] > high + 0.1 * rise:
                a -= 1
            while b + 1 < min(n, k + side) and Y[b + 1] > high + 0.1 * rise:
                b += 1
            ev.append(dict(k=k, a=max(1, a), b=b, score=clip01(rise / max(0.05, 1 - high)), kind='flash', hard=False))
    halves = sorted({max(2, min(h, round(0.6 * fps))) for h in (2, 4, 8, 16)})
    hits = []
    for k in range(2, n - 2):
        for h in halves:
            if k - h < 1 or k + h >= n:
                continue
            if any(j in hard for j in range(k - h + 1, k + h + 1)):
                continue
            A, B = thumbs[k - h], thumbs[k + h]
            dist = _corr_dist(A, B)
            if dist is None or dist < 0.5:
                continue
            share, active = _persistence(A, B)
            if active < 3 or share >= 0.4:
                continue
            fits = [_blend_fit(A, thumbs[j], B) for j in (k - (h >> 1), k, k + (h >> 1))]
            if any(f is None or f[1] > 0.35 for f in fits):
                continue
            f0, f1, f2 = fits
            if not (f0[0] < f1[0] < f2[0] and 0.25 < f1[0] < 0.75):
                continue
            if fades_out(k, Y[k - h]):
                continue
            hits.append(dict(k=k, h=h, score=clip01(1 - (f0[1] + f1[1] + f2[1]) / 3)))
            break
    others = list(ev)
    i = 0
    while i < len(hits):
        j = i
        while j + 1 < len(hits) and hits[j + 1]['k'] - hits[j]['k'] <= 2:
            j += 1
        run = hits[i:j + 1]
        a, b = run[0]['k'] - run[0]['h'], run[-1]['k'] + run[-1]['h']
        i = j + 1
        if any(e['a'] - b < 0.5 * fps and a - e['b'] < 0.5 * fps for e in others):
            continue
        ev.append(dict(k=run[len(run) // 2]['k'], a=a, b=b, score=max(h['score'] for h in run), kind='blend', hard=False))
    near = MERGE_S * fps
    groups = []
    for e in sorted([x for x in ev if x['hard']], key=lambda x: x['k']):
        g = groups[-1] if groups else None
        if g and e['k'] - g['first'] < near:
            g['list'].append(e)
            g['b'] = max(g['b'], e['b'])
        else:
            groups.append(dict(first=e['k'], a=e['a'], b=e['b'], hard=True, list=[e]))
    hard_groups = list(groups)
    soft_groups = []
    for e in sorted([x for x in ev if not x['hard']], key=lambda x: (x['a'], x['k'])):
        best, bd = None, float('inf')
        for g in hard_groups:
            d = e['a'] - g['first'] if g['first'] < e['a'] else g['first'] - e['b'] if g['first'] > e['b'] else 0
            if d < near and d < bd:
                best, bd = g, d
        if best:
            best['list'].append(e)
            continue
        g = soft_groups[-1] if soft_groups else None
        if g and (e['a'] - g['a'] < near or e['a'] <= g['b']):
            g['list'].append(e)
            g['b'] = max(g['b'], e['b'])
        else:
            soft_groups.append(dict(first=e['k'], a=e['a'], b=e['b'], hard=False, list=[e]))
    out = []
    for g in hard_groups + soft_groups:
        if g['hard']:
            at = max((e for e in g['list'] if e['hard']), key=lambda e: S[e['k']])['k']
        else:
            m = max(range(g['a'], g['b'] + 1), key=lambda j: S[j])
            at = m if S[m] >= 0.3 else g['list'][0]['k']
        out.append(dict(t=_r(at / fps), frame=int(at), score=_r(max(e['score'] for e in g['list']), 2)))
    out.sort(key=lambda c: c['frame'])
    res = []
    for c in out:
        if not res or c['frame'] - res[-1]['frame'] >= near:
            res.append(c)
    return res


# ---------------------------------------------------------------- scoring (port of Refs check-render.ts)

WEIGHTS = dict(cuts=35, brightness=20, motion=15, pace=10, loudness=10, format=10)
NAMES = dict(cuts='Cuts', brightness='Brightness curve', motion='Motion', pace='Pace', loudness='Loudness', format='Format')
VISIBLE_TRANSITIONS = {'cut', 'dissolve', 'fade_from_black', 'fade_from_white', 'flash', 'whip', 'zoom', 'wipe', 'morph'}
VISIBLE_INSIDE = {'cut', 'dip_to_dark', 'flash'}
SEEN_BY_MOTION = {'whip', 'zoom', 'wipe', 'morph', 'dissolve', 'flash', 'fade_from_black', 'fade_from_white', 'dip_to_dark'}
SEEN_BY_LIGHT = {'flash', 'fade_from_black', 'fade_from_white', 'dip_to_dark'}
SEEN_CREDIT = 0.6
TYPE_TEXT = dict(cut='hard cut', dissolve='dissolve', fade_from_black='fade from black', fade_from_white='fade from white',
                 flash='flash', whip='whip', zoom='zoom transition', wipe='wipe', morph='morph', dip_to_dark='dip to dark',
                 reframe='reframe', focus_pull='focus pull', unknown='unclassified change', text='on-screen text change')
TARGETS = dict(x=(-14, -14, 'X: −14 LUFS, true peak ≤ −1 dBTP'), youtube=(-14, -14, 'YouTube: −14 LUFS, true peak ≤ −1 dBTP'),
               tiktok=(-14, -10, 'TikTok: −14 to −10 LUFS, true peak ≤ −1 dBTP'), web=(-16, -16, 'web: −16 LUFS, true peak ≤ −1 dBTP'))
PLATFORM_NAME = dict(x='X', youtube='YouTube', tiktok='TikTok', web='the web')
ASPECTS = [('16:9', 16 / 9), ('9:16', 9 / 16), ('1:1', 1), ('4:5', 4 / 5), ('2:1', 2), ('1:2', 0.5), ('21:9', 21 / 9),
           ('2.39:1', 2.39), ('4:3', 4 / 3), ('3:4', 3 / 4), ('3:2', 1.5), ('1.85:1', 1.85)]


def _fin(a):
    return [x for x in a if x is not None and math.isfinite(x)]


def _median(a):
    a = sorted(_fin(a))
    if not a:
        return float('nan')
    m = len(a) // 2
    return a[m] if len(a) % 2 else (a[m - 1] + a[m]) / 2


def _mean(a):
    a = _fin(a)
    return sum(a) / len(a) if a else float('nan')


def _pearson(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return None
    ma, mb = _mean(a[:n]), _mean(b[:n])
    sab = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    saa = sum((a[i] - ma) ** 2 for i in range(n))
    sbb = sum((b[i] - mb) ** 2 for i in range(n))
    return sab / math.sqrt(saa * sbb) if saa > 1e-12 and sbb > 1e-12 else None


def _ranks(a):
    idx = sorted(range(len(a)), key=lambda i: a[i])
    out = [0.0] * len(a)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and a[idx[j + 1]] == a[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            out[idx[k]] = (i + j) / 2
        i = j + 1
    return out


def _spearman(a, b):
    return _pearson(_ranks(a), _ranks(b))


def _at(series, step, t):
    if not series:
        return float('nan')
    x = max(0.0, min(len(series) - 1, t / step))
    i = int(math.floor(x))
    f = x - i
    return series[i] * (1 - f) + series[i + 1] * f if i + 1 < len(series) else series[i]


def _filled(v):
    out, last = [], float('nan')
    for x in v or []:
        if isinstance(x, (int, float)) and math.isfinite(x):
            last = x
        out.append(last)
    first = next((x for x in out if math.isfinite(x)), 0.0)
    return [x if math.isfinite(x) else first for x in out]


def aspect_name(w, h):
    r = w / h
    best = min(ASPECTS, key=lambda a: abs(math.log(r / a[1])))
    return best[0] if abs(math.log(r / best[1])) < 0.02 else f'{r:.2f}:1'


def delivery(w, h):
    r = w / h
    return f'{2 * round(1080 * r / 2)}×1080' if r >= 1 else f'1080×{2 * round(1080 / r / 2)}'


def s2(t):
    return f'{t:.2f} s'


def rng(a, b):
    return f'{a:.1f}–{b:.1f} s'


def signed(x, d=1):
    return f"{'+' if x > 0 else '−' if x < 0 else '±'}{abs(x):.{d}f}"


def minus(x, d=1):
    return f"{'−' if x < 0 else ''}{abs(x):.{d}f}"


def plural(n, one, many=None):
    return f'{n} {one if n == 1 else (many or one + "s")}'


def describe(e):
    return (('hidden ' if e['inside'] else '') + TYPE_TEXT.get(e['type'], e['type'])
            + (f" over {e['frames']} f" if e['frames'] > 0 else '') + (f" inside shot {e['shot']}" if e['inside'] else ''))


def merge_events(events):
    out = []
    for e in sorted(events, key=lambda e: (e['t'], e['soft'])):
        if out and e['t'] - out[-1]['t'] < MERGE_S:
            if out[-1]['soft'] and not e['soft']:
                out[-1] = e
            continue
        out.append(e)
    n = 0
    for e in out:
        if not e['soft']:
            n += 1
            e['n'] = n
    return out


def ref_model(tl):
    video = tl.get('video') or {}
    fps = float(video.get('fps') or 30)
    duration = float(video.get('duration_s') or 0)
    shots = [s for s in tl.get('shots') or [] if isinstance(s, dict)]
    if not duration or not shots:
        raise SystemExit('check: the reference timeline has no shots')
    m = re.search(r'every ([\d.]+) s', str((tl.get('method') or {}).get('text') or ''))
    ocr = float(m.group(1)) if m else (0.5 if duration > 30 else 0.25)
    events, rshots = [], []
    for i, s in enumerate(shots):
        n = s.get('n') or i + 1
        start = s.get('start') or 0.0
        end = s.get('end') or (shots[i + 1].get('start') if i + 1 < len(shots) else duration)
        if i > 0:
            tin = s.get('transition_in') or {}
            typ = tin.get('type') or 'cut'
            soft = typ not in VISIBLE_TRANSITIONS
            events.append(dict(n=0, t=start, frame=s.get('start_frame', round(start * fps)), type=typ,
                               frames=max(0, tin.get('frames') or 0), shot=n, inside=False, soft=soft, w=0 if soft else 1))
        for ic in s.get('internal_changes') or []:
            typ = ic.get('type') or 'unknown'
            t = ic.get('t') if ic.get('t') is not None else (ic.get('frame') or 0) / fps
            soft = typ not in VISIBLE_INSIDE
            events.append(dict(n=0, t=t, frame=ic.get('frame', round(t * fps)), type=typ, frames=max(0, ic.get('frames') or 0),
                               shot=n, inside=True, soft=soft, w=0 if soft else 0.5))
        for tx in s.get('text') or []:
            first, last = tx.get('first'), tx.get('last')
            for t in ([first - ocr] if first is not None and first - ocr > start + 0.1 else []) + \
                     ([last] if last is not None and last < end - 0.1 else []):
                events.append(dict(n=0, t=t, frame=round(t * fps), type='text', frames=round(ocr * fps), shot=n, inside=True,
                                   soft=True, w=0, tolF=3))
        cam, el = s.get('camera') or {}, s.get('element_motion') or {}
        noisy = (not cam.get('move') or cam.get('move') == 'static') and el.get('move') == 'jitter'
        rshots.append(dict(n=n, start=start, end=end, motion=s.get('motion_energy'), scales=s.get('scales'), noisy=noisy))
    b = tl.get('brightness') or {}
    audio = tl.get('audio') or {}
    cad = video.get('cadence') or {}
    flags = {}
    for s in rshots:
        if s['scales']:
            flags.setdefault(s['scales'], []).append(s['n'])
    sl = lambda ns: f"shot{'s' if len(ns) > 1 else ''} {', '.join(map(str, ns[:6]))}" + (f' and {len(ns) - 6} more' if len(ns) > 6 else '')
    scales = ' and '.join(f"{sl(flags[f])} as {'fixed (keep their length)' if f == 'fixed' else 'optional (can be dropped)'}"
                          for f in ('fixed', 'optional') if flags.get(f)) or None
    return dict(measured=f"timeline v{tl.get('version') or 2}", duration=duration, fps=fps, width=video.get('width'),
                height=video.get('height'), events=merge_events(events), shots=rshots, hidden=True,
                brightness=dict(step=b.get('step_s') or 0.25, mean=_filled(b.get('mean')), p95=_filled(b.get('p95'))) if b.get('mean') else None,
                hasAudio=audio.get('has_audio') if isinstance(audio.get('has_audio'), bool) else None,
                lufs=(audio.get('loudness') or {}).get('integrated_lufs'),
                cadence=dict(label=cad['label'], optional=cad.get('optional')) if cad.get('label') else None, scales=scales)


def _fr(ctx, t):
    return round(t * ctx['ren']['fps'])


def _status(score, p, w):
    return 'pass' if score >= p else 'warn' if score >= w else 'fail'


def _seen_in(ren, typ, a, b):
    def pick(xs, step, lo, hi):
        return [x for i, x in enumerate(xs) if lo - 1e-9 <= i * step <= hi + 1e-9]
    md, ms = ren['motion']['diff'], ren['motion']['step_s']
    near = pick(md, ms, a, b)
    peak = max(near) if near else 0
    if peak >= 0.015 and peak >= 1.6 * _median(pick(md, ms, a - 2, b + 2)):
        return True
    if typ not in SEEN_BY_LIGHT:
        return False
    ls = ren['luma']['step_s']
    lum = pick(ren['luma']['mean'], ls, a - ls, b + ls)
    return len(lum) >= 2 and max(lum) - min(lum) >= 0.05


def match_cuts(ref, ren, factor, span):
    tol = max(3 / ren['fps'], 0.015 * ren['duration_s'])
    within = lambda t: t < span - 0.5 / ren['fps']
    mapped = [dict(e=e, a=e['t'] * factor, b=(e['t'] + e['frames'] / ref['fps']) * factor,
                   tol=e['tolF'] / ren['fps'] if e.get('tolF') is not None else tol) for e in ref['events']]
    mapped = [m for m in mapped if within(m['a'])]
    cuts = [c for c in ren['cuts'] if c['t'] > 0.5 / ren['fps'] and within(c['t'])]
    pairs = []
    for i, m in enumerate(mapped):
        for j, c in enumerate(cuts):
            off = c['t'] - m['a'] if c['t'] < m['a'] else c['t'] - m['b'] if c['t'] > m['b'] else 0.0
            d = abs(off)
            if d <= m['tol']:
                pairs.append(dict(i=i, j=j, d=d + abs(c['t'] - m['a']) * 1e-3, off=off,
                                  target=m['a'] if c['t'] < m['a'] else m['b'] if c['t'] > m['b'] else c['t']))
    pairs.sort(key=lambda p: (mapped[p['i']]['e']['soft'], p['d']))
    usedE, usedC, matched, absorbed = set(), set(), [], []
    for p in pairs:
        if p['i'] in usedE or p['j'] in usedC:
            continue
        usedE.add(p['i'])
        usedC.add(p['j'])
        e = mapped[p['i']]['e']
        if e['soft']:
            absorbed.append(e)
        else:
            matched.append(dict(e=e, c=cuts[p['j']], target=p['target'], off=p['off']))
    expected = [m['e'] for m in mapped if not m['e']['soft']]
    seen, missing = [], []
    for i, m in enumerate(mapped):
        if m['e']['soft'] or i in usedE:
            continue
        if m['e']['type'] in SEEN_BY_MOTION and _seen_in(ren, m['e']['type'], m['a'] - m['tol'], m['b'] + m['tol']):
            seen.append(m['e'])
        else:
            missing.append(m['e'])
    extra = [c for j, c in enumerate(cuts) if j not in usedC]
    credit = sum(2 * m['e']['w'] * (1 - 0.5 * min(1, abs(m['off']) / tol)) for m in matched) + sum(SEEN_CREDIT * e['w'] for e in seen)
    denom = sum(e['w'] for e in expected) + sum(m['e']['w'] for m in matched) + len(extra)
    score = credit / denom if denom else 1.0
    matched.sort(key=lambda m: m['e']['t'])
    return dict(score=clip01(score), matched=matched, seen=seen, missing=missing, extra=extra, absorbed=absorbed,
                expected=expected, cuts=cuts)


def cuts_check(ctx, m):
    ref, ren = ctx['ref'], ctx['ren']
    f = lambda t: f'f{_fr(ctx, t)}'
    offF = lambda x: _fr(ctx, x['c']['t']) - _fr(ctx, x['target'])
    late = sorted([x for x in m['matched'] if abs(offF(x)) >= 2], key=lambda x: -abs(offF(x)))
    fixes = ctx['fixes']
    if len(late) > 5:
        rest = late[5:]
        offs = [abs(offF(x)) for x in rest]
        fixes.append(('cuts', 2, f"{plural(len(rest), 'more cut')} {'is' if len(rest) == 1 else 'are'} {min(offs)}–{max(offs)} f off: "
                      + ', '.join(f"cut {x['e']['n']} {'+' if offF(x) > 0 else '−'}{abs(offF(x))} f → {f(x['target'])}" for x in rest[:8])
                      + (', …' if len(rest) > 8 else '') + '.'))
    for x in late[:5]:
        off = offF(x)
        span = f", {f(x['e']['t'] * ctx['factor'])}–{f((x['e']['t'] + x['e']['frames'] / ref['fps']) * ctx['factor'])}" if x['e']['frames'] > 0 else ''
        fixes.append(('cuts', 2 + min(3.5, abs(off) / 2),
                      f"Cut {x['e']['n']} lands {abs(off)} f {'late' if off > 0 else 'early'} at {f(x['c']['t'])}; move it to "
                      f"{f(x['target'])} ({s2(x['target'])}) (reference: {describe(x['e'])}{span})."))
    if len(m['missing']) > 4:
        fixes.append(('cuts', 6.5, f"Add the reference's {len(m['missing'])} missing cuts: " + ', '.join(
            f"cut {e['n']} at {f(e['t'] * ctx['factor'])}" + ('' if e['type'] == 'cut' else f" ({TYPE_TEXT.get(e['type'], e['type'])})")
            for e in m['missing'][:10]) + (', …' if len(m['missing']) > 10 else '') + '.'))
    else:
        for e in m['missing']:
            fixes.append(('cuts', 6, f"Add cut {e['n']} at {f(e['t'] * ctx['factor'])} ({s2(e['t'] * ctx['factor'])}): the reference's {describe(e)}."))
    if m['extra']:
        times = [f"{f(c['t'])} ({s2(c['t'])})" for c in m['extra']]
        fixes.append(('cuts', min(7, 2 + len(m['extra'])),
                      f"Remove the cut{'s' if len(m['extra']) > 1 else ''} at {', '.join(times)}: the reference has no change there."
                      if len(m['extra']) <= 4 else
                      f"You cut {len(m['extra'])} times where the reference doesn't ({', '.join(times[:6])}{', …' if len(times) > 6 else ''}): hold those shots."))
    on = sum(1 for x in m['matched'] if abs(offF(x)) <= 1)
    if not m['expected'] and not m['cuts']:
        summary = 'one continuous shot, like the reference'
    elif not m['expected']:
        summary = f"the reference has no visible cut; you cut {plural(len(m['cuts']), 'time')}"
    else:
        summary = (f"{len(m['matched'])} of {plural(len(m['expected']), 'reference cut')} matched ({on} within 1 f)"
                   + (f", {len(m['seen'])} seen as motion or light" if m['seen'] else '')
                   + (f", {len(m['missing'])} missing" if m['missing'] else '') + (f", {len(m['extra'])} extra" if m['extra'] else ''))
    return dict(key='cuts', status=_status(m['score'], 0.85, 0.6), score=round(m['score'], 2), summary=summary)


def pace_check(ctx, m):
    def lens(cuts, end):
        b = [0.0] + [t for t in cuts if 0 < t < end] + [end]
        return [b[i + 1] - b[i] for i in range(len(b) - 1) if b[i + 1] - b[i] > 1e-6]
    ref_l = lens(sorted(e['t'] * ctx['factor'] for e in m['expected'] + m['absorbed']), ctx['span'])
    ren_l = lens([c['t'] for c in m['cuts']], ctx['span'])
    nR, nY = len(ref_l), len(ren_l)
    mR, mY = _median(ref_l), _median(ren_l)
    score = clip01(0.5 * min(nR, nY) / max(nR, nY) + 0.5 * min(mR, mY) / max(mR, mY))
    if score < 0.8 and len(m['missing']) + len(m['extra']) > 4:
        ctx['fixes'].append(('pace', 3 * (1 - score) + 1, f"{'Cut more' if nY < nR else 'Cut less'}: you have {plural(nY, 'shot')} "
                             f"(median {s2(mY)}); the reference{', scaled to your length,' if ctx['factor'] != 1 else ''} has {nR} (median {s2(mR)})."))
    return dict(key='pace', status=_status(score, 0.8, 0.6), score=round(score, 2),
                summary=f"{plural(nY, 'shot')}, median {s2(mY)} vs {nR}, median {s2(mR)}")


def brightness_check(ctx):
    ref, ren = ctx['ref'], ctx['ren']
    b = ref['brightness']
    N = max(20, min(400, round(ctx['span'] / 0.125)))
    ts = [(i + 0.5) / N * ctx['span'] for i in range(N)]
    yours = [_at(ren['luma']['mean'], ren['luma']['step_s'], t) for t in ts]
    theirs = [_at(b['mean'], b['step'], t / ctx['factor']) for t in ts]
    yp = [_at(ren['luma']['p95'], ren['luma']['step_s'], t) for t in ts]
    tp = [_at(b['p95'], b['step'], t / ctx['factor']) for t in ts]
    my, mt = _mean(yours), _mean(theirs)
    ratio = (my + 0.01) / (mt + 0.01)
    r = _pearson(yours, theirs)
    sd = lambda a: math.sqrt(_mean([(v - _mean(a)) ** 2 for v in a]))
    flat = sd(theirs) < 0.01 or sd(yours) < 0.01
    shape = 1 - min(1, _mean([abs(v - theirs[i]) for i, v in enumerate(yours)]) / max(0.05, mt)) if flat or r is None else max(0, r)
    level = 1 / (1 + max(0, abs(math.log2(ratio)) - 0.2))
    score = clip01(0.5 * shape + 0.5 * level)
    dt = ctx['span'] / N
    runs = []
    for i in range(N):
        d = yours[i] - theirs[i]
        q = (max(yours[i], theirs[i]) + 0.01) / (min(yours[i], theirs[i]) + 0.01)
        sg = (1 if d > 0 else -1) if abs(d) >= 0.04 and q >= 1.4 else 0
        if not sg:
            continue
        if runs and runs[-1]['sign'] == sg and i - runs[-1]['b'] <= 2:
            runs[-1]['b'] = i
            runs[-1]['area'] += abs(d) * dt
        else:
            runs.append(dict(a=i, b=i, sign=sg, area=abs(d) * dt))
    wins = sorted([w for w in runs if max(0.4, 0.03 * ctx['span']) <= (w['b'] - w['a'] + 1) * dt <= 0.8 * ctx['span']],
                  key=lambda w: -w['area'])[:3]
    x = lambda q: f"{q:.{0 if q >= 3 else 1}f}×"
    if ratio >= 1.4 or ratio <= 1 / 1.4:
        ctx['fixes'].append(('brightness', 2 + min(4, 2 * abs(math.log2(ratio))),
                             f"Your film is on average {x(ratio)} as bright as the reference (mean luma {my:.3f} vs {mt:.3f}): darken the grade, background and fills."
                             if ratio >= 1 else
                             f"Your film is on average {x(1 / ratio)} darker than the reference (mean luma {my:.3f} vs {mt:.3f}): lift the exposure, background and highlights."))
    for w in wins:
        a, e = w['a'] * dt, (w['b'] + 1) * dt
        yv, tv = _mean(yours[w['a']:w['b'] + 1]), _mean(theirs[w['a']:w['b'] + 1])
        q = (yv + 0.01) / (tv + 0.01)
        ctx['fixes'].append(('brightness', 1.5 + min(3, abs(math.log2(q))),
                             f"{rng(a, e)} (f{_fr(ctx, a)}–f{_fr(ctx, e)}): yours {yv:.2f} vs the reference's {tv:.2f} mean luma "
                             f"({x(q) + ' brighter' if q > 1 else x(1 / q) + ' darker'}); {'darken' if q > 1 else 'brighten'} this stretch."))
    if not flat and r is not None and r < 0.5:
        ctx['fixes'].append(('brightness', 2.5, f"Follow the reference's light: its brightness curve and yours move differently (r = {r:.2f}). "
                                                f"Compare them on the timeline sheet."))
    pr = (_mean(yp) + 0.01) / (_mean(tp) + 0.01)
    if 0.8 < ratio < 1.25 and (pr >= 1.6 or pr <= 1 / 1.6):
        ctx['fixes'].append(('brightness', 1.5, f"Highlights: your brightest areas reach {_mean(yp):.2f} (95th percentile) vs the "
                                                f"reference's {_mean(tp):.2f}; {'pull them down' if pr > 1 else 'let them glow brighter'}."))
    return dict(key='brightness', status=_status(score, 0.75, 0.5), score=round(score, 2),
                summary=(f'r {r:.2f}; ' if r is not None and not flat else '') + f"yours {ratio:.2f}× the reference's mean luma ({my:.3f} vs {mt:.3f})")


def motion_check(ctx):
    ref, ren = ctx['ref'], ctx['ren']
    step = ren['motion']['step_s']
    rate = [d * (ren['fps'] / 30) for d in ren['motion']['diff']]

    def win_mean(a, b):
        v = [rate[i] for i in range(len(rate)) if a - 1e-9 <= i * step < b]
        return _mean(v) if v else _at(rate, step, (a + b) / 2)
    allS = [dict(s, a=s['start'] * ctx['factor'], b=min(s['end'] * ctx['factor'], ctx['span'])) for s in ref['shots']]
    allS = [s for s in allS if s['b'] - s['a'] > 0.05 and s['motion'] is not None]
    clean = [s for s in allS if not s['noisy']]
    shots = clean if len(clean) >= 3 else allS
    if len(shots) >= 3:
        theirs = [s['motion'] for s in shots]
        yours = [win_mean(s['a'], s['b']) for s in shots]
        rho = _spearman(theirs, yours)
        rt = [v / (len(shots) - 1) for v in _ranks(theirs)]
        ry = [v / (len(shots) - 1) for v in _ranks(yours)]
        flags = []
        for i, s in enumerate(shots):
            if (rt[i] >= 0.67 and ry[i] <= 0.33) or (theirs[i] >= 0.2 and yours[i] < 0.002):
                flags.append((s, 'too static'))
            elif rt[i] <= 0.33 and ry[i] >= 0.67 and theirs[i] < 0.2:
                flags.append((s, 'too busy'))
        score = clip01((((rho or 0) + 0.2) / 1.2)) * (1 - min(0.3, 0.08 * len(flags)))
        for s, v in flags[:4]:
            ctx['fixes'].append(('motion', 1.5 + (s['b'] - s['a']) / max(1, ctx['span']) * 4,
                                 f"Shot {s['n']} ({rng(s['a'], s['b'])}, f{_fr(ctx, s['a'])}–f{_fr(ctx, s['b'])}) is {v}: "
                                 + ("the reference moves a lot here; add the camera or element move from the timeline's camera/element_motion for this shot"
                                    if v == 'too static' else "the reference is calm here; hold the picture and let one element move") + '.'))
        if rho is not None and rho < 0.3 and not flags:
            ctx['fixes'].append(('motion', 2, f"Your motion doesn't rise and fall with the reference's (rank correlation {rho:.2f} over {len(shots)} shots): busy shots should be busy, still ones still."))
        summ = f"rank correlation {'n/a' if rho is None else f'{rho:.2f}'} over {len(shots)} shots"
        for v in ('too static', 'too busy'):
            ns = [s['n'] for s, w in flags if w == v]
            if ns:
                summ += f"; {v}: shot{'s' if len(ns) > 1 else ''} {', '.join(map(str, ns[:4]))}"
        return dict(key='motion', status=_status(score, 0.75, 0.55), score=round(score, 2), summary=summ)
    if not shots:
        return None
    per = []
    for s in shots:
        est = 12 * win_mean(s['a'], s['b'])
        q = (est + 0.02) / (s['motion'] + 0.02)
        per.append((s, est, q, 1 / (1 + max(0, abs(math.log2(q)) - 1))))
    score = _mean([p[3] for p in per])
    for s, est, q, sc in per:
        if sc < 0.75:
            ctx['fixes'].append(('motion', 1.5 + 2 * (1 - sc), f"Shot {s['n']} ({rng(s['a'], s['b'])}) is {'too busy' if q > 1 else 'too static'}: "
                                 f"about {est:.2f} motion vs the reference's {s['motion']:.2f} (share of the frame changing)."))
    return dict(key='motion', status=_status(score, 0.75, 0.55), score=round(score, 2),
                summary='; '.join(f"shot {s['n']}: about {est:.2f} vs {s['motion']:.2f}" for s, est, _, _ in per) + ' (levels compared, roughly)')


def loudness_check(ctx, platform):
    ref, ren = ctx['ref'], ctx['ren']
    lo, hi, label = TARGETS[platform]
    I, TP = ren['audio']['integrated_lufs'], ren['audio']['true_peak_dbtp']
    ref_note = (f". The reference itself sits at {minus(ref['lufs'])} LUFS; don't copy that, the target follows the platform"
                if ref['lufs'] is not None and (ref['lufs'] < lo - 3 or ref['lufs'] > hi + 3) else '')
    if not ren['audio']['has_audio'] or I is None or I <= -69:
        silent = ren['audio']['has_audio'] and I is not None
        if ref['hasAudio'] is False:
            return dict(key='loudness', status='skip', score=None, summary='no audio, like the reference')
        ctx['fixes'].append(('loudness', 5, f"Your render has {'a silent audio track' if silent else 'no audio'}"
                             f"{'; the reference has sound' if ref['hasAudio'] else ''}. Add a soundtrack and master it to {label}."))
        return dict(key='loudness', status='fail', score=0.0, summary='silent audio track' if silent else 'no audio track')
    gain = lo - I if I < lo else hi - I if I > hi else 0
    peak_after = TP + gain if TP is not None else None
    near = ref['lufs'] is not None and lo - 3 <= ref['lufs'] < lo
    low = ref['lufs'] - 1 if near else lo
    as_ref = near and gain > 0 and I >= low
    err = 0 if as_ref else abs(low - I if I < low else gain)
    score = 1 if err <= 1 else 0.75 if err <= 2 else 0.45 if err <= 4 else max(0, 0.45 - (err - 4) * 0.1)
    if TP is not None and TP > -1:
        score = max(0, score - 0.3)
    target = f'{minus(lo, 0)} LUFS' if lo == hi else f'{minus(lo, 0)} to {minus(hi, 0)} LUFS'
    cmd = f"ffmpeg -i in.mp4 -c:v copy -af loudnorm=I={lo if gain > 0 else hi}:TP=-1:LRA=11 -c:a aac -b:a 192k out.mp4"
    limit = f', then limit: peaks would reach {signed(peak_after)} dBTP, keep them at −1' if peak_after is not None and peak_after > -1 else ''
    if as_ref and abs(gain) > 1:
        ctx['fixes'].append(('loudness', 0.8, f"Master is {minus(I)} LUFS, close to the reference's own {minus(ref['lufs'])}, so it passes. "
                             f"{PLATFORM_NAME[platform]} posts usually sit at {target}: raise it by {signed(gain)} dB if you want it as loud as the rest of the feed{limit} (e.g. {cmd})."))
    elif err > 1:
        ctx['fixes'].append(('loudness', 1 + min(4, err / 1.5), f"Master is {minus(I)} LUFS; {'raise' if gain > 0 else 'lower'} it by {signed(gain)} dB to "
                             f"{minus(I + gain, 0)} LUFS for {PLATFORM_NAME[platform]}{'' if lo == hi else f' (its window is {target})'}{limit} (e.g. {cmd}){ref_note}."))
    elif TP is not None and TP > -1:
        ctx['fixes'].append(('loudness', 2, f"True peak is {signed(TP)} dBTP: limit it to −1 dBTP so platform encoders don't clip (e.g. {cmd})."))
    status = 'pass' if err <= 1 and (TP is None or TP <= -1) else 'warn' if err <= 3 else 'fail'
    return dict(key='loudness', status=status, score=round(score, 2),
                summary=f"{minus(I)} LUFS, true peak {'?' if TP is None else minus(TP) + ' dBTP'}; "
                        + (f"matches the reference's {minus(ref['lufs'])} ({PLATFORM_NAME[platform]} level {target})" if as_ref
                           else f"target {target}" + (f' ({signed(gain)} dB)' if gain else '')))


def format_check(ctx):
    ref, ren = ctx['ref'], ctx['ren']
    score = wsum = 0.0
    notes = []
    ya = aspect_name(ren['width'], ren['height'])
    ra = aspect_name(ref['width'], ref['height']) if ref['width'] and ref['height'] else None
    if ra:
        same = abs(math.log(ren['width'] / ren['height'] / (ref['width'] / ref['height']))) < 0.02
        score += 0.5 if same else 0
        wsum += 0.5
        if not same:
            ctx['fixes'].append(('format', 4, f"Your film is {ya} ({ren['width']}×{ren['height']}); the reference is {ra}. Render {ra} at "
                                 f"{delivery(ref['width'], ref['height'])}, or re-lay out every shot for {ya} on purpose (plate 15)."))
            notes.append(f"{ya} vs the reference's {ra}")
        else:
            notes.append(f'{ya} like the reference')
    short = min(ren['width'], ren['height'])
    score += 0.25 if short >= 1080 else 0.1 if short >= 720 else 0
    wsum += 0.25
    if short < 1080:
        ctx['fixes'].append(('format', 2.5, f"Render at 1080 px or more on the short side (yours {ren['width']}×{ren['height']}); e.g. {delivery(ren['width'], ren['height'])}."))
        notes.append(f'{short}p')
    same_fps = abs(ren['fps'] - ref['fps']) / ref['fps'] < 0.01
    score += 0.25 if same_fps else 0.15 if abs(ren['fps'] - ref['fps']) / ref['fps'] < 0.3 else 0.1
    wsum += 0.25
    if not same_fps:
        ctx['fixes'].append(('format', 1.5, f"You render at {ren['fps']:.2f} fps; the reference runs at {ref['fps']:.2f} fps. Match it so frame timings line up (or keep yours and use the seconds)."))
        notes.append(f"{ren['fps']:.2f} vs {ref['fps']:.2f} fps")
    else:
        notes.append(f"{ren['fps']:.2f} fps")
    cad = ref['cadence']
    if cad and cad.get('optional') is False and cad['label'] != 'ones':
        held = 2 if cad['label'] == 'twos' else 3 if cad['label'] == 'threes' else None
        ctx['fixes'].append(('format', 1, f"The reference animates on {cad['label']}: each picture is held {held} frames at {ref['fps']:.2f} fps. "
                                          f"Step your animation the same way (e.g. Remotion: frame = Math.floor(frame / {held}) * {held})."
                             if held else f"The reference animates at about {cad['label'].replace('~', '')} inside {ref['fps']:.2f} fps (pictures are held); step your animation to match."))
    s = clip01(score / wsum)
    return dict(key='format', status=_status(s, 0.99, 0.6), score=round(s, 2), summary=', '.join(notes))


def platform_of(name):
    n = (name or '').lower()
    return 'youtube' if 'youtube' in n else 'x' if ('twitter' in n or n == 'x') else 'tiktok' if ('tiktok' in n or 'instagram' in n) else 'web'


def check_render(tl, ren, platform='web', title='the reference'):
    ref = ref_model(tl)
    notes = []
    ratio = ren['duration_s'] / ref['duration']
    factor, mode = 1.0, 'as is'
    lengths = f"your film is {s2(ren['duration_s'])}, the reference {s2(ref['duration'])}"
    if abs(ratio - 1) <= 0.02:
        note = f'Lengths match within 2% ({lengths}), so reference times are used as they are.'
    else:
        scaled = match_cuts(ref, ren, ratio, ren['duration_s'])
        as_is = match_cuts(ref, ren, 1, min(ren['duration_s'], ref['duration']))
        if len(as_is['matched']) > len(scaled['matched']) and as_is['score'] >= 0.5 and as_is['score'] >= scaled['score'] + 0.15:
            note = f"{lengths[0].upper()}{lengths[1:]}. Your cuts follow the reference's own times better than scaled ones, so they are compared as they are."
        else:
            factor, mode = ratio, 'scaled'
            note = f"{lengths[0].upper()}{lengths[1:]}: reference times are scaled ×{ratio:.3f} to your length."
        if mode == 'scaled' and ref['scales']:
            note += f" Even scaling is approximate: the blueprint marks {ref['scales']}."
    span = ren['duration_s'] if mode == 'scaled' else min(ren['duration_s'], ref['duration'])
    if span < ren['duration_s'] - 0.5:
        notes.append(f"The last {s2(ren['duration_s'] - span)} of your film run past the reference's length and aren't compared.")
    ctx = dict(ref=ref, ren=ren, factor=factor, span=span, fixes=[])
    m = match_cuts(ref, ren, factor, span)
    checks = [cuts_check(ctx, m)]
    skip = lambda k, why: dict(key=k, status='skip', score=None, summary=why)
    checks.append(brightness_check(ctx) if ref['brightness'] and ref['brightness']['mean'] else skip('brightness', 'no brightness curve in the timeline'))
    mo = motion_check(ctx) if any(s['motion'] is not None for s in ref['shots']) else None
    checks.append(mo or skip('motion', 'too little measured motion to compare'))
    checks += [pace_check(ctx, m), loudness_check(ctx, platform), format_check(ctx)]
    swaps = sum(1 for e in m['absorbed'] if e['type'] == 'text')
    unsure = len(m['absorbed']) - swaps
    if swaps:
        notes.append(f"{plural(swaps, 'cut')} of yours {'falls' if swaps == 1 else 'fall'} where the reference's on-screen text changes: a big text swap can read as a cut, so {'it isn' if swaps == 1 else 'they aren'}'t scored either way.")
    if unsure:
        notes.append(f"{plural(unsure, 'cut')} of yours land on reference changes of uncertain kind (unclassified, reframe or focus pull), not scored either way.")
    if m['seen']:
        notes.append('No hard cut was found for ' + ', '.join(f"cut {e['n']} ({TYPE_TEXT.get(e['type'], e['type'])} at f{_fr(ctx, e['t'] * factor)})" for e in m['seen'][:4])
                     + ", but your film's motion or brightness changes there, as a blur or fade transition does: partial credit. Check by eye.")
    for c in checks:
        c['name'], c['weight'] = NAMES[c['key']], WEIGHTS[c['key']]
    scored = [c for c in checks if c['score'] is not None]
    total = sum(c['weight'] for c in scored)
    score = round(100 * sum(c['weight'] * c['score'] for c in scored) / max(1, total))
    fixes = sorted(ctx['fixes'], key=lambda f: -f[1])
    top = list(dict.fromkeys(f[0] for f in fixes))[:2]
    names = ' and '.join(NAMES[k].lower() for k in top)
    many = len(top) > 1 or (top and top[0] == 'cuts')
    verdict = (f"Close to {title}{'; polish the ' + names if top else ''}." if score >= 85 else
               f"Recognisably {title}'s format; fix the {names or 'details below'} first." if score >= 70 else
               f"Partly follows {title}: the {names} {'are' if many else 'is'} off." if score >= 50 else
               f"Doesn't follow {title} yet: the {names} {'differ' if many else 'differs'}.")
    return dict(kind='dissect-render-check', version=1, score=score, verdict=verdict,
                reference=dict(duration_s=ref['duration'], fps=ref['fps'], width=ref['width'], height=ref['height']),
                render=dict(file=ren.get('file'), duration_s=ren['duration_s'], fps=ren['fps'], width=ren['width'], height=ren['height']),
                timing=dict(mode=mode, factor=round(factor, 3), note=note), checks=checks, fixes=[f[2] for f in fixes], notes=notes)


def format_report(r):
    mark = {'pass': 'pass', 'warn': 'warn', 'fail': 'FAIL', 'skip': 'n/a'}
    ren, ref = r['render'], r['reference']
    L = [f"check: {ren['file']} vs the reference",
         f"Score {r['score']}/100. {r['verdict']}",
         f"Reference: {s2(ref['duration_s'])}, {ref['fps']:.2f} fps, {ref['width']}×{ref['height']}. Yours: {s2(ren['duration_s'])}, "
         f"{ren['fps']:.2f} fps, {ren['width']}×{ren['height']}.", r['timing']['note'], '',
         '| Check | Weight | Status | Score | Found |', '|---|---|---|---|---|']
    for c in r['checks']:
        sc = '–' if c['score'] is None else f"{c['score']:.2f}"
        L.append(f"| {c['name']} | {c['weight']} | {mark[c['status']]} | {sc} | {c['summary']} |")
    L += ['', 'Fixes, most important first (frames at your frame rate):' if r['fixes'] else 'No fixes needed.']
    L += [f'{i + 1}. {f}' for i, f in enumerate(r['fixes'][:12])]
    if len(r['fixes']) > 12:
        L.append(f"…and {len(r['fixes']) - 12} smaller fixes (--json shows all).")
    L += [f'Note: {n}' for n in r['notes']]
    L += ['', 'After fixing, re-render and run check again.']
    return '\n'.join(L)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog='dissect.py check', description='Score a render against a dissected reference.')
    ap.add_argument('render', help='your rendered video file')
    ap.add_argument('--ref', required=True, help='the dissection folder (or its data/timeline.json)')
    ap.add_argument('--target', choices=['x', 'youtube', 'tiktok', 'web'], help='loudness target (default: the reference\'s platform)')
    ap.add_argument('--json', action='store_true', help='print the full report as JSON')
    a = ap.parse_args(argv)
    ref = Path(a.ref).expanduser()
    tlp = ref if ref.suffix == '.json' else ref / 'data' / 'timeline.json'
    if not tlp.exists():
        raise SystemExit(f'check: no timeline at {tlp}. Dissect the reference with dissect 2.0 first (it writes data/timeline.json).')
    tl = json.loads(tlp.read_text(encoding='utf-8'))
    platform = a.target
    title = 'the reference'
    an = tlp.parent / 'analysis.json'
    if an.exists():
        A = json.loads(an.read_text(encoding='utf-8'))
        platform = platform or platform_of(A.get('platform_name'))
        title = f"\"{(A.get('title') or 'the reference')[:60]}\""
    ren = measure_render(a.render)
    rep = check_render(tl, ren, platform or 'web', title)
    print(json.dumps(dict(rep, render_measure=ren), indent=1, ensure_ascii=False) if a.json else format_report(rep))
    return rep
