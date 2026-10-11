"""timeline.py - the frame-level measurement pass of dissect (timeline v2.1).

Imported by dissect.py; not run on its own. Given the downloaded video and dissect's own shot list, it measures what
the v1 pass does not: frame-exact shot boundaries, the transition into every shot (cut, dissolve, fade, flash, whip,
zoom, wipe) with its length in frames, hidden cuts and changes inside shots, camera and element motion per shot
(pan, tilt, zoom, roll as numbers, with segments and easing), blur, focus, animation cadence, palette, highlights and
brightness over time, on-screen text with times, boxes and cap heights (Apple Vision on macOS, tesseract elsewhere if
installed), and the sound grid: tempo with an honest confidence, beats and downbeats, key, energy sections, and a cue
sheet of hits, whooshes and risers keyed to shots. It also draws the full-length timeline sheet and the transition
close-up strips. Ported from the Refs blueprint-v2 measurer (measure2.py 2.1).
"""
import os

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_k, '2')

import collections
import datetime as dt
import difflib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

VERSION = '2.1.0'
TIMELINE_VERSION = 2.1

AN_LONG = 480            # camera / blur / cadence frames: long side in px
TR_LONG = 192            # transition-window frames: long side in px
B_STEP = 0.25            # colour pass: one frame every 0.25 s (brightness, palette, text, sheets)
OCR_MIN_H = 0.012        # Vision minimumTextHeight, as a fraction of frame height
CAP_A, CAP_B = 1.091, 2.0  # Vision line box h = CAP_A * extent(text) * cap height + CAP_B px (at 720 px tall); fitted on
#                            700 reads of 10 typefaces at 9-80 px caps: median error 7 %, p90 19 %.

LIM = dict(cut_spike=4.0, sig_total=2.5, sig_rate=1.0, jitter=1.2)


class Fail(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def run(cmd, timeout=None):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(cmd, 124, '', f'timed out: {e}')
    except FileNotFoundError as e:
        return subprocess.CompletedProcess(cmd, 127, '', str(e))


def r(x, n=3):
    return None if x is None else round(float(x), n)


def clip01(x):
    return float(min(1.0, max(0.0, x)))


def probe_rate(video):
    p = run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=avg_frame_rate,r_frame_rate', '-of', 'json', str(video)])
    try:
        st = json.loads(p.stdout)['streams'][0]
    except Exception:
        return None
    for key in ('avg_frame_rate', 'r_frame_rate'):
        try:
            fr = Fraction(st.get(key, '0/1'))
            if 1 <= float(fr) <= 240:
                return fr.limit_denominator(1001)
        except (ValueError, ZeroDivisionError):
            continue
    return None


def even(x):
    return max(2, int(round(x / 2)) * 2)


def fit_long(w, h, long_side):
    if w >= h:
        return even(long_side), even(long_side * h / w)
    return even(long_side * w / h), even(long_side)


# ---------------------------------------------------------------- image measures

CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)


def block_change(a, b, block=10, thr=0.035):
    """Share of block x block tiles whose mean absolute change exceeds thr (v1's 'near' motion measure)."""
    h, w = a.shape
    hh, ww = h // block * block, w // block * block
    d = np.abs(a[:hh, :ww] - b[:hh, :ww]).reshape(hh // block, block, ww // block, block).mean(axis=(1, 3))
    return float((d > thr).mean())


def sharpness(g8):
    """Edge sharpness independent of contrast: mean(Laplacian^2) / mean(|gradient|^2) (1/px^2). Falls as blur grows.
    Also the x and y gradient energies, for directional (whip) blur."""
    g = g8.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ex, ey = float((gx * gx).mean()), float((gy * gy).mean())
    if ex + ey < 4.0:   # nearly flat frame: no edges to judge
        return None, ex, ey
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float((lap * lap).mean() / (ex + ey)), ex, ey


def similarity_step(a8, b8, w, h, max_corners=300, fb_max=1.0, ransac=1.0):
    """Global similarity motion from frame a to b: sparse corners + pyramidal LK (forward-backward checked) +
    RANSAC estimateAffinePartial2D. Returns motion of the frame centre in % of width/height, log scale, rotation
    (deg), and how much of the tracked points agree (inlier ratio) or form a second motion layer."""
    p0 = cv2.goodFeaturesToTrack(a8, maxCorners=max_corners, qualityLevel=0.01, minDistance=7, blockSize=7)
    if p0 is None or len(p0) < 8:
        return dict(ok=False, n=0 if p0 is None else len(p0), why='features')
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a8, b8, p0, None, winSize=(21, 21), maxLevel=3, criteria=CRIT)
    if p1 is None:
        return dict(ok=False, n=0, why='track')
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(b8, a8, p1, None, winSize=(21, 21), maxLevel=3, criteria=CRIT)
    a, b = p0.reshape(-1, 2), p1.reshape(-1, 2)
    fb = np.linalg.norm(a - p0r.reshape(-1, 2), axis=1)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_max)
    a, b = a[good], b[good]
    n = len(a)
    if n < 8:
        return dict(ok=False, n=n, why='track')
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=ransac, maxIters=1000,
                                         confidence=0.99, refineIters=10)
    if M is None or inl is None:
        return dict(ok=False, n=n, why='ransac')
    inl = inl.ravel().astype(bool)
    s = math.hypot(M[0, 0], M[1, 0])
    if not (0.2 < s < 5) or not np.isfinite(M).all():
        return dict(ok=False, n=n, why='degenerate')
    th = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    cx, cy = w / 2, h / 2
    dx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx
    dy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy
    def bbox(pts):
        x0, x1 = np.percentile(pts[:, 0], [5, 95])
        y0, y1 = np.percentile(pts[:, 1], [5, 95])
        return [float(x0 / w), float(y0 / h), float((x1 - x0) / w), float((y1 - y0) / h)], float((x1 - x0) * (y1 - y0) / (w * h))

    def params(M_):
        s_ = math.hypot(M_[0, 0], M_[1, 0])
        dx_ = M_[0, 0] * cx + M_[0, 1] * cy + M_[0, 2] - cx
        dy_ = M_[1, 0] * cx + M_[1, 1] * cy + M_[1, 2] - cy
        return dict(dx=dx_ / w * 100, dy=dy_ / h * 100, ls=math.log(max(1e-6, s_)),
                    rot=math.degrees(math.atan2(M_[1, 0], M_[0, 0])))
    second, layer2 = 0.0, None
    out_a, out_b = a[~inl], b[~inl]
    if len(out_a) >= 8 and len(out_a) >= 0.1 * n:
        M2, inl2 = cv2.estimateAffinePartial2D(out_a, out_b, method=cv2.RANSAC, ransacReprojThreshold=ransac,
                                               maxIters=500, confidence=0.99)
        if M2 is not None and inl2 is not None and inl2.sum() >= 6 and np.isfinite(M2).all() and \
                0.2 < math.hypot(M2[0, 0], M2[1, 0]) < 5:
            second = float(inl2.sum()) / n
            bx2, cv2_ = bbox(out_b[inl2.ravel().astype(bool)])
            layer2 = dict(params(M2), box=bx2, cover=cv2_, share=second)
    disp = np.linalg.norm(b - a, axis=1)
    box, cover = bbox(b[inl])
    return dict(ok=True, n=n, inl=int(inl.sum()), ratio=float(inl.mean()), second=second, layer2=layer2,
                dx=dx / w * 100, dy=dy / h * 100, ls=math.log(max(1e-6, s)), rot=th,
                still=float((disp < 0.25).mean()), box=box, cover=cover)


_SIFT = None


def jump_step(a8, b8, w, h):
    """Large instant reframes inside a shot (punch-in, jump) that LK cannot follow: SIFT + ratio test + RANSAC
    similarity. Only accepted when the matches spread over a real part of the frame (not one small element)."""
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=1000)
    k1, d1_ = _SIFT.detectAndCompute(a8, None)
    k2, d2_ = _SIFT.detectAndCompute(b8, None)
    if d1_ is None or d2_ is None or len(k1) < 12 or len(k2) < 12:
        return None
    good = []
    for m in cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1_, d2_, k=2):
        if len(m) == 2 and m[0].distance < 0.8 * m[1].distance:
            good.append(m[0])
    if len(good) < 8:
        return None
    pa = np.float32([k1[m.queryIdx].pt for m in good])
    pb = np.float32([k2[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000,
                                         confidence=0.99)
    if M is None or inl is None or inl.sum() < 8 or inl.mean() < 0.35:
        return None
    # the transform must explain the whole picture (a reframe), not one small element: warp a onto b and compare
    wa = cv2.warpAffine(a8, M, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.ones_like(a8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
    if valid.mean() < 0.2:
        return None
    fa, fb_, fw = a8.astype(np.float32), b8.astype(np.float32), wa.astype(np.float32)
    res_m = float(np.abs(fw - fb_)[valid].mean())
    res_i = float(np.abs(fa - fb_)[valid].mean())
    if res_i < 4.0 or res_m > 0.8 * res_i:
        return None
    s = math.hypot(M[0, 0], M[1, 0])
    if not (0.05 < s < 20) or not np.isfinite(M).all():
        return None
    cx, cy = w / 2, h / 2
    dx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx
    dy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy
    if abs(math.log(s)) < 0.1 and math.hypot(dx / w, dy / h) < 0.08:
        return None   # not a jump: an ordinary step LK failed on
    return dict(scale=s, dx=dx / w * 100, dy=dy / h * 100, rot=math.degrees(math.atan2(M[1, 0], M[0, 0])),
                inl=int(inl.sum()), ratio=float(inl.mean()))


# ---------------------------------------------------------------- pass A: every frame, small grey

def struct_change(a, b, ma, sa, mb, sb):
    """Luma-normalised frame difference: 1 - correlation of two frames after removing each one's own brightness and
    contrast. Sees cuts in dark or low-contrast films that absolute differences miss. 0 = same picture, ~1 = new one."""
    floor = 0.004
    if sa < floor and sb < floor:
        return 0.0
    if sa < floor or sb < floor:
        return 1.0 if abs(ma - mb) > 0.003 else 0.0
    return float(1.0 - ((a - ma) * (b - mb)).mean() / (sa * sb))


def focus_map(g8, grid):
    """Defocus per block from the edge re-blur ratio (Zhuo & Sim 2011): sigma = s0 / sqrt(R^2 - 1), R = gradient at an
    edge over the gradient after a known Gaussian blur s0. Returns (sigma per block in px of this frame or NaN where a
    block has too few edges, edge count per block)."""
    g = g8.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m0 = cv2.magnitude(gx, gy)
    hi = max(24.0, float(np.percentile(m0, 99)) * 0.3)
    edges = cv2.Canny(g8, hi * 0.4, hi, L2gradient=True) > 0
    s0 = 1.0
    g1 = cv2.GaussianBlur(g, (0, 0), s0)
    m1 = cv2.magnitude(cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3))
    valid = edges & (m1 > 1.0)
    R = np.zeros_like(m0)
    R[valid] = m0[valid] / m1[valid]
    sig = np.full(m0.shape, np.nan, np.float32)
    ok = valid & (R > 1.02)
    sig[ok] = s0 / np.sqrt(R[ok] ** 2 - 1.0)
    sig[valid & ~ok] = 7.0
    sig = np.clip(sig, 0.0, 7.0)
    gh, gw = grid
    h, w = g8.shape
    out = np.full((gh, gw), np.nan, np.float32)
    cnt = np.zeros((gh, gw), np.int32)
    for by in range(gh):
        for bx in range(gw):
            blk = sig[by * h // gh:(by + 1) * h // gh, bx * w // gw:(bx + 1) * w // gw]
            v = blk[~np.isnan(blk)]
            cnt[by, bx] = len(v)
            if len(v) >= 12:
                out[by, bx] = float(np.median(v))
    return out, cnt


def pass_a(video, rate, dur, aw, ah, flow_of, win_set, tw, th, threads, bound_frames, log):
    fps = float(rate)
    cmd = ['ffmpeg', '-v', 'error', '-threads', str(threads), '-i', str(video), '-map', '0:v:0', '-an',
           '-t', f'{dur:.3f}', '-vf', f'fps={rate.numerator}/{rate.denominator},scale={aw}:{ah}:flags=area,format=gray',
           '-f', 'rawvideo', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = aw * ah
    sw, sh = even(aw / 3), even(ah / 3)
    lagk = max(1, int(round(fps * 0.1)))
    ring = collections.deque(maxlen=lagk + 1)
    d1, d1p, near, ylum, ds, ystat = [], [], [], [], [], []
    steps, sharp, win, big = [], {}, {}, {}
    prev = None
    # hidden changes: a change that stands far above its neighbourhood, checked DLY frames late so the frames after
    # it are known; WC frames either side are kept for classifying it
    DLY = 10
    WC = int(min(24, max(8, round(0.4 * fps))))
    ringw = collections.deque(maxlen=WC + DLY + 3)
    ringb = collections.deque(maxlen=DLY + 4)
    near_b = set()
    for b in bound_frames:
        near_b.update(range(b - 3, b + 4))
    cands, keep_until, keepb_until = [], -1, -1
    fgrid = (7, 12) if aw >= ah else (12, 7)
    focus = []
    smalls = {}

    def check(k):
        """Is frame k a hidden change? Uses ds/relative-brightness around it."""
        if k < 2 or k in near_b:
            return None
        lo, hi = max(1, k - 10), min(len(ds) - 1, k + 10)
        nb = [ds[j] for j in range(lo, hi + 1) if abs(j - k) >= 2]
        nbm = max(nb) if nb else 0.0
        ya, yb = ylum[k - 1], ylum[k]
        rb = abs(yb - ya) / max(0.006, min(ya, yb))
        nbr = [abs(ylum[j] - ylum[j - 1]) / max(0.006, min(ylum[j], ylum[j - 1])) for j in range(lo, hi + 1) if abs(j - k) >= 2]
        nbrm = max(nbr) if nbr else 0.0
        struct = ds[k] >= 0.3 and ds[k] >= 3.0 * (nbm + 0.02)
        bright = rb >= 0.35 and abs(yb - ya) >= 0.004 and rb >= 3.0 * (nbrm + 0.02)
        if struct or bright:
            return dict(k=k, ds=round(ds[k], 3), rb=round(rb, 3), spike=round(ds[k] / (nbm + 0.02), 1))
        return None

    i = 0
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        g8 = np.frombuffer(buf, np.uint8).reshape(ah, aw)
        sm = cv2.resize(g8, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        ylum.append(float(g8.mean()) / 255.0)
        mu, sd = float(sm.mean()), float(sm.std())
        ring.append(sm)
        if len(ring) > 1:
            ad = np.abs(ring[-1] - ring[-2])
            d1.append(float(ad.mean()))
            d1p.append(float(np.percentile(ad, 98)))
            pm, ps = ystat[-1]
            ds.append(struct_change(ring[-1], ring[-2], mu, sd, pm, ps))
        else:
            d1.append(0.0)
            d1p.append(0.0)
            ds.append(0.0)
        ystat.append((mu, sd))
        near.append(block_change(ring[-1], ring[0]) if len(ring) == lagk + 1 else 0.0)
        tr = cv2.resize(g8, (tw, th), interpolation=cv2.INTER_AREA)
        ringw.append((i, tr))
        ringb.append((i, g8))
        if i in win_set or i <= keep_until:
            win[i] = tr
        if i <= keepb_until:
            big[i] = g8
        k = i - DLY
        c = check(k)
        if c:
            cands.append(c)
            for j, f in ringw:
                if j >= k - WC:
                    win[j] = f
            for j, f in ringb:
                if k - 2 <= j <= k + 1:
                    big[j] = f
            keep_until = max(keep_until, k + WC)
            keepb_until = max(keepb_until, k + 1)
        si = flow_of.get(i)
        if si is not None:
            sharp[i] = sharpness(g8)
            if prev is not None and prev[0] == si:
                st = similarity_step(prev[2], g8, aw, ah)
                if not st['ok'] or st['inl'] < 10:
                    jp = jump_step(prev[2], g8, aw, ah)
                    if jp:
                        st = dict(ok=False, why='jump', jump=jp, n=st.get('n', 0))
                st.update(shot=si, i0=prev[1], i1=i)
                steps.append(st)
            prev = (si, i, g8)
        if si is not None:
            fm, fc = focus_map(g8, fgrid)
            focus.append((i, fm, fc))
            smalls[i] = (sm * 255).astype(np.uint8)
        i += 1
    proc.stdout.close()
    proc.wait()
    for k in range(max(2, i - DLY), i - 1):   # the last frames: neighbourhood is truncated
        c = check(k)
        if c:
            cands.append(c)
            for j, f in ringw:
                if j >= k - WC:
                    win[j] = f
            for j, f in ringb:
                if k - 2 <= j <= k + 1:
                    big[j] = f
    # one event per change: candidates within 3 frames of each other are the same change
    merged = []
    for c in cands:
        if merged and c['k'] - merged[-1]['k'] <= 3:
            merged[-1]['ks'].append(c['k'])
            continue
        c['ks'] = [c['k']]
        merged.append(c)
    log(f'pass A: {i} frames at {fps:.3f} fps, {len(steps)} motion steps, {len(merged)} change candidates')
    return dict(n=i, d1=np.array(d1), d1p=np.array(d1p), near=np.array(near), y=np.array(ylum), ds=np.array(ds),
                steps=steps, sharp=sharp, win=win, big=big, lagk=lagk, cands=merged, WC=WC, focus=focus, fgrid=fgrid,
                smalls=smalls)


# ---------------------------------------------------------------- pass B: 4 fps colour, native size

class OCR:
    """Apple Vision VNRecognizeTextRequest (accurate). Returns [(text, conf, [x, y, w, h] top-left fractions,
    quad [tl, tr, br, bl] as top-left fractions)]; the quad follows tilted text, the box is upright."""

    def __init__(self):
        self.ok = False
        try:
            import objc  # noqa: F401
            import Quartz
            import Vision
            self.Q, self.V, self.objc = Quartz, Vision, sys.modules['objc']
            self.ok = True
        except Exception as e:  # not macOS or pyobjc missing
            self.err = f'{e.__class__.__name__}: {e}'
        self.engine = 'vision' if self.ok else None
        if not self.ok and shutil.which('tesseract'):
            self.ok, self.engine = True, 'tesseract'
        self.calls = 0
        self.seconds = 0.0

    def tesseract(self, rgb):
        """Fallback off macOS: tesseract's TSV, words grouped into lines. Upright boxes only (no quad)."""
        h, w = rgb.shape[:2]
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            Image.fromarray(rgb).save(path)
            p = run(['tesseract', path, 'stdout', '--psm', '11', 'tsv'], timeout=60)
        finally:
            os.unlink(path)
        lines = collections.OrderedDict()
        for row in (p.stdout or '').splitlines()[1:]:
            c = row.split('\t')
            if len(c) < 12 or c[0] != '5' or not c[11].strip():
                continue
            try:
                conf = float(c[10])
            except ValueError:
                continue
            if conf < 0:
                continue
            key = (c[2], c[3], c[4])
            x, y, bw, bh = (int(v) for v in c[6:10])
            lines.setdefault(key, []).append((c[11], conf / 100.0, x, y, bw, bh))
        out = []
        for ws in lines.values():
            x0 = min(v[2] for v in ws); y0 = min(v[3] for v in ws)
            x1 = max(v[2] + v[4] for v in ws); y1 = max(v[3] + v[5] for v in ws)
            out.append((' '.join(v[0] for v in ws), float(np.mean([v[1] for v in ws])),
                        [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h], None))
        return out

    def __call__(self, rgb):
        if not self.ok:
            return []
        if self.engine == 'tesseract':
            t0 = time.time()
            out = self.tesseract(rgb)
            self.calls += 1
            self.seconds += time.time() - t0
            return out
        t0 = time.time()
        Q, V = self.Q, self.V
        h, w = rgb.shape[:2]
        rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)])
        data = rgba.tobytes()
        out = []
        with self.objc.autorelease_pool():
            prov = Q.CGDataProviderCreateWithData(None, data, len(data), None)
            cg = Q.CGImageCreate(w, h, 8, 32, w * 4, Q.CGColorSpaceCreateDeviceRGB(), Q.kCGImageAlphaNoneSkipLast,
                                 prov, None, False, Q.kCGRenderingIntentDefault)
            req = V.VNRecognizeTextRequest.alloc().init()
            req.setRecognitionLevel_(V.VNRequestTextRecognitionLevelAccurate)
            req.setUsesLanguageCorrection_(True)
            req.setMinimumTextHeight_(OCR_MIN_H)
            handler = V.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
            handler.performRequests_error_([req], None)
            for o in req.results() or []:
                c = o.topCandidates_(1)
                if not c:
                    continue
                c = c[0]
                bb = o.boundingBox()
                x, y, bw, bh = bb.origin.x, bb.origin.y, bb.size.width, bb.size.height
                quad = [(float(p_.x), 1.0 - float(p_.y)) for p_ in (o.topLeft(), o.topRight(), o.bottomRight(), o.bottomLeft())]
                out.append((str(c.string()), float(c.confidence()), [x, 1.0 - y - bh, bw, bh], quad))
        self.calls += 1
        self.seconds += time.time() - t0
        return out


def pass_b(video, rate, dur, bw, bh, threads, sheet_step, ocr_step, tile_wh, ocr, review, log):
    fps = float(rate)
    q = fps * B_STEP
    sel = f",select='lt(mod(n\\,{q:.6f})\\,1)'" if q > 1.0 else ''
    cmd = ['ffmpeg', '-v', 'error', '-threads', str(threads), '-i', str(video), '-map', '0:v:0', '-an',
           '-t', f'{dur:.3f}', '-vf', f'fps={rate.numerator}/{rate.denominator}{sel},scale={bw}:{bh}:flags=area,'
           'format=rgb24', '-fps_mode', 'passthrough', '-f', 'rawvideo', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = bw * bh * 3
    lw, lh = fit_long(bw, bh, 320)
    pw, ph = fit_long(bw, bh, 64)
    s_every = max(1, int(round(sheet_step / B_STEP)))
    o_every = max(1, int(round(ocr_step / B_STEP)))
    lum_m, lum_p, thumbs, tiles, obs, rthumbs, frames_b, hl = [], [], [], [], [], [], [], []
    last_g, last_o = None, []
    k = 0
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        nk = int(math.ceil(k * q - 1e-6)) if q > 1.0 else k
        t = nk / fps
        frames_b.append(nk)
        rgb = np.frombuffer(buf, np.uint8).reshape(bh, bw, 3)
        sm = cv2.resize(rgb, (lw, lh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        Y = sm[..., 0] * 0.2126 + sm[..., 1] * 0.7152 + sm[..., 2] * 0.0722
        lum_m.append(float(Y.mean()))
        lum_p.append(float(np.percentile(Y, 95)))
        thr = max(float(np.percentile(Y, 98)), 0.06)
        hp = sm[Y >= thr]
        if len(hp) > 600:
            hp = hp[np.linspace(0, len(hp) - 1, 600).astype(int)]
        # accents: clearly coloured pixels that are not dark (small UI dots, rims, logos the average hides)
        labf = cv2.cvtColor(sm, cv2.COLOR_RGB2Lab)
        chroma_ = np.hypot(labf[..., 1], labf[..., 2])
        am = (chroma_ >= 25) & (Y >= 0.06)
        ap = sm[am]
        n_acc = int(am.sum())
        if len(ap) > 300:
            ap = ap[np.linspace(0, len(ap) - 1, 300).astype(int)]
        hl.append((np.clip(hp * 255, 0, 255).astype(np.uint8), thr, np.clip(ap * 255, 0, 255).astype(np.uint8),
                   n_acc / float(Y.size)))
        thumbs.append(cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA))
        if review is not None:
            rthumbs.append(cv2.resize(rgb, fit_long(bw, bh, 240), interpolation=cv2.INTER_AREA))
        if k % s_every == 0:
            tiles.append((t, nk, Image.fromarray(cv2.resize(rgb, tile_wh, interpolation=cv2.INTER_AREA))))
        if k % o_every == 0 and ocr.ok:
            g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (128, 72), interpolation=cv2.INTER_AREA)
            g = g.astype(np.float32) / 255.0
            if last_g is not None and np.abs(g - last_g).reshape(9, 8, 16, 8).mean(axis=(1, 3)).max() < 0.012:
                o = last_o   # same picture as the last read: reuse it
            else:
                o = ocr(rgb)
                last_g, last_o = g, o
            obs.append((t, o))
        k += 1
    proc.stdout.close()
    proc.wait()
    log(f'pass B: {k} colour frames, {len(tiles)} sheet tiles, OCR {ocr.calls} reads in {ocr.seconds:.1f} s')
    return dict(n=k, lum_m=np.array(lum_m), lum_p=np.array(lum_p), thumbs=thumbs, tiles=tiles, obs=obs,
                rthumbs=rthumbs, t=np.array(frames_b, float) / fps, frames=frames_b, hl=hl)


# ---------------------------------------------------------------- shots, transitions, camera

def shot_index(shots, t):
    for j, s in enumerate(shots):
        if s['t0'] <= t < s['t1']:
            return j
    return len(shots) - 1 if t >= shots[-1]['t0'] else 0


def phase_shift(a, b):
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (sx, sy), resp = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), win)
    return sx, sy, resp


def runs_around(mask, j):
    """Length of the run of True values in mask that contains (or is nearest to) index j."""
    idx = [i for i, m in enumerate(mask) if m]
    if not idx:
        return 0, None, None
    k = min(idx, key=lambda i: abs(i - j))
    a = b = k
    while a - 1 >= 0 and mask[a - 1]:
        a -= 1
    while b + 1 < len(mask) and mask[b + 1]:
        b += 1
    return b - a + 1, a, b


def analyse_boundary(b, wl, wr, A, fps, tw, th):
    """Classify the transition into the shot that starts at frame b. Uses native-rate frames b-wl .. b+wr-1.
    Order: dip to black, blend (dissolve/wipe), blur + fast shift (whip), luma spike (flash / fade from white),
    no visible change (continuous motion), scale flow (zoom), else a cut. On a hard cut (one-frame change), a whip or
    zoom needs its blur/motion on the outgoing side, so the look of the incoming shot is not taken for a transition."""
    idx = list(range(b - wl, b + wr))
    if any(i not in A['win'] for i in idx) or len(idx) < 4:
        return dict(type='cut', frames=0, confidence=0.3, cut_frame=b, why='no frames around the boundary')
    F = [A['win'][i].astype(np.float32) / 255.0 for i in idx]
    Y = np.array([A['y'][i] for i in idx])
    n, jb = len(idx), wl
    d = np.zeros(n)
    ds = np.zeros(n)   # structural change: 1 - correlation of consecutive frames (exposure changes barely move it)
    for j in range(1, n):
        d[j] = np.abs(F[j] - F[j - 1]).mean()
        a_, b_ = F[j - 1] - F[j - 1].mean(), F[j] - F[j].mean()
        sa, sb = float(np.sqrt((a_ * a_).mean())), float(np.sqrt((b_ * b_).mean()))
        if sa < 0.01 or sb < 0.01:
            ds[j] = min(1.0, abs(float(F[j].mean() - F[j - 1].mean())) * 3 + (0.5 if (sa < 0.01) != (sb < 0.01) else 0.0))
        else:
            ds[j] = 1.0 - float((a_ * b_).mean()) / (sa * sb)
    rr = max(2, int(round(0.1 * fps)))
    lo, hi = max(1, jb - rr), min(n - 1, jb + rr)
    jc = lo + int(np.argmax(ds[lo:hi + 1]))
    if not (ds[jc] >= 1.1 * ds[jb] and d[jc] >= 1.2 * d[jb]):
        jc = jb   # move off v1's frame only when both change measures say the cut is elsewhere
    rest = [ds[j] for j in range(1, n) if abs(j - jc) > 1]
    base = float(np.median(rest)) if rest else 0.0
    spike = float(ds[jc] / (base + 0.01))
    rest_d = [d[j] for j in range(1, n) if abs(j - jc) > 1]
    spike_d = float(d[jc] / ((float(np.median(rest_d)) if rest_d else 0.0) + 2e-3))
    hard = spike >= 6
    info = dict(cut_frame=idx[jc], spike=round(spike, 1))

    # 1. dip to black
    pre, post = float(np.median(Y[:3])), float(np.median(Y[-3:]))
    zl, zh = max(0, jc - 10), min(n, jc + 10)
    jmin = zl + int(np.argmin(Y[zl:zh]))
    jmax = zl + int(np.argmax(Y[zl:zh]))
    lowp, highp = min(pre, post), max(pre, post)
    if (Y[jmin] < 0.08 and lowp - Y[jmin] > 0.08) or (lowp < 0.15 and Y[jmin] < 0.5 * lowp and lowp - Y[jmin] > 0.006):
        half = Y[jmin] + 0.5 * (lowp - Y[jmin])
        width, _, _ = runs_around(list(Y < half), jmin)
        return dict(type='fade_from_black', frames=int(width),
                    confidence=r(clip01(max((lowp - Y[jmin]) / 0.25, 1 - Y[jmin] / max(lowp, 1e-4))), 2),
                    **info, dip_luma=r(Y[jmin]))

    # 3. whip: frames near the change much blurrier than their own side's far frames, smeared in one direction
    #    or shifted fast
    sh = [sharpness((f * 255).astype(np.uint8)) for f in F]
    side = lambda lst: [v[0] for v in lst if v[0] is not None]
    pre_b, post_b = side(sh[:3]), side(sh[-3:])
    pre_b = float(np.median(pre_b)) if pre_b else None
    post_b = float(np.median(post_b)) if post_b else None
    blurred, aniso, fast, shifts = [], [], [], {}
    for j in range(max(1, jc - 4), min(n, jc + 4)):
        s_, ex, ey = sh[j]
        bs = pre_b if j < jc else post_b
        if bs and s_ is not None and s_ < 0.5 * bs:
            blurred.append(j)
            if max(ex, ey) / max(1e-6, min(ex, ey)) > 2.5:
                aniso.append(j)
        if j != jc:
            sx, sy, resp = phase_shift(F[j - 1], F[j])
            shifts[j] = (sx, sy)
            if math.hypot(sx / tw, sy / th) > 0.06 and resp > 0.05:
                fast.append(j)
    ev = set(blurred) & (set(aniso) | set(fast))
    if hard:
        ev = {j for j in ev if j < jc}   # on a hard cut the whip must happen before it
    if len(ev) >= 2 or (len(blurred) >= 3 and len(set(aniso) | set(fast)) >= 2 and not hard):
        frames = len(set(blurred) | set(fast))
        if fast:
            mx = np.mean([shifts[j][0] for j in fast])
            my = np.mean([shifts[j][1] for j in fast])
            direction = ('right' if mx > 0 else 'left') if abs(mx / tw) >= abs(my / th) else ('down' if my > 0 else 'up')
        else:
            ex_, ey_ = np.mean([sh[j][1] for j in aniso]), np.mean([sh[j][2] for j in aniso])
            direction = 'horizontal' if ex_ < ey_ else 'vertical'
        return dict(type='whip', frames=int(frames), confidence=r(clip01(0.45 + 0.1 * len(ev)), 2), **info,
                    direction=direction)

    # 4. brightness spike: flash, or a fade from white
    if Y[jmax] - highp > 0.12 or (highp < 0.15 and Y[jmax] > 1.6 * highp and Y[jmax] - highp > 0.02):
        half = highp + 0.5 * (Y[jmax] - highp)
        width, _, _ = runs_around(list(Y > half), jmax)
        kind = 'fade_from_white' if Y[jmax] > 0.85 and width >= 4 else 'flash'
        return dict(type=kind, frames=int(width), confidence=r(clip01((Y[jmax] - highp) / 0.3), 2), **info,
                    peak_luma=r(Y[jmax]))

    # 2. gradual blend between the two shots: dissolve, or a moving edge (wipe)
    A0, B0 = (F[0] + F[1]) / 2, (F[-1] + F[-2]) / 2
    AB = B0 - A0
    nab = float((AB * AB).sum())
    gradual = None
    if not hard and nab / AB.size > 1e-5:
        alpha = np.array([float(((f - A0) * AB).sum()) / nab for f in F])
        resid = np.array([math.sqrt(float(((f - A0 - a * AB) ** 2).sum()) / nab) for f, a in zip(F, alpha)])
        width, a0, b0 = runs_around(list((alpha > 0.1) & (alpha < 0.9)), jc)
        if width >= 2:
            seg = alpha[a0:b0 + 1]
            mono = all(seg[k + 1] >= seg[k] - 0.08 for k in range(len(seg) - 1))
            res = float(resid[a0:b0 + 1].mean())
            moving = 0
            for j in range(max(1, a0), min(n, b0 + 1)):
                st = similarity_step((F[j - 1] * 255).astype(np.uint8), (F[j] * 255).astype(np.uint8), tw, th,
                                     max_corners=150, fb_max=1.5, ransac=1.5)
                if st['ok'] and st['ratio'] >= 0.5 and st.get('cover', 0) >= 0.3 and \
                        (abs(st['ls']) > 0.01 or math.hypot(st['dx'], st['dy']) > 2.0):
                    moving += 1
            if mono and res < 0.45 and moving <= max(1, width // 3):
                return dict(type='dissolve', frames=int(width), confidence=r(clip01(1.1 - res * 1.5), 2), **info)
            colp, rowp = [], []
            for f in F[a0:b0 + 1]:
                m = (np.abs(f - B0) < np.abs(f - A0)).astype(np.float32)
                colp.append(m.mean(0))
                rowp.append(m.mean(1))
            stepc = np.mean([((p_ < 0.2) | (p_ > 0.8)).mean() for p_ in colp])
            stepr = np.mean([((p_ < 0.2) | (p_ > 0.8)).mean() for p_ in rowp])
            if mono and max(stepc, stepr) > 0.8:
                return dict(type='wipe', frames=int(width), confidence=r(clip01(max(stepc, stepr) - 0.3), 2), **info,
                            axis='horizontal' if stepc >= stepr else 'vertical')
            gradual = int(width)

    # 5. nothing changes more at the boundary than around it: continuous motion across a v1 split
    if spike < 2.0 and spike_d < 3.0 and abs(pre - post) < 0.2 and not gradual:
        return dict(type='unknown', frames=0, confidence=0.3, **info,
                    why='no visible cut here: the picture changes no more than in the frames around it '
                        '(continuous motion across the v1 boundary?)')

    # 6. zoom through: strong same-sign scale flow over consecutive frames (before the cut, if it is a hard cut)
    zs = []
    for j in range(max(1, jc - 4), min(n, jc + 5)):
        if j == jc or (hard and j > jc):
            continue
        st = similarity_step((F[j - 1] * 255).astype(np.uint8), (F[j] * 255).astype(np.uint8), tw, th,
                             max_corners=150, fb_max=1.5, ransac=1.5)
        if st['ok'] and st['ratio'] >= 0.5 and abs(st['ls']) > 0.035 and st.get('cover', 0) >= 0.3:
            zs.append((j, st['ls']))
    if len(zs) >= 2 and (all(z[1] > 0 for z in zs) or all(z[1] < 0 for z in zs)):
        return dict(type='zoom', frames=len(zs), confidence=r(clip01(0.4 + 0.15 * len(zs)), 2), **info,
                    direction='in' if zs[0][1] > 0 else 'out')

    if gradual:
        return dict(type='unknown', frames=gradual, confidence=0.3, **info,
                    why='gradual change that is not a clean blend (morph, push or element transition?)')
    if spike >= LIM['cut_spike']:
        return dict(type='cut', frames=0, confidence=r(clip01(0.55 + (spike - LIM['cut_spike']) / 20), 2), **info)
    return dict(type='cut', frames=0, confidence=0.3, **info,
                why='weak change at the boundary (a step change or element swap rather than a clean cut?)')


def classify_change(c, A, fps, aw, ah, nfr):
    """Hidden change inside a v1 shot -> {t, frame, type, frames, confidence, what} or None (an element popping on,
    which is not a change of shot). Types: cut | dip_to_dark | reframe | flash | unknown."""
    k = c['k']
    WC = A['WC']
    y = A['y']
    pre_s = y[max(0, k - WC):max(1, k - 2)]
    post_s = y[min(nfr - 1, k + 4):min(nfr, k + WC + 1)]
    pre = float(np.median(pre_s)) if len(pre_s) else float(y[max(0, k - 1)])
    post = float(np.median(post_s)) if len(post_s) else float(y[min(nfr - 1, k)])
    low, high = min(pre, post), max(pre, post)
    base = dict(t=r(k / fps, 3), frame=int(k), spike=c.get('spike'))
    # dip to dark: the level falls well below both sides, then the next picture comes up
    seg = y[max(0, k - 1):min(nfr, k + 10)]
    j0 = max(0, k - 1)
    ymin, jmin = float(seg.min()), j0 + int(seg.argmin())
    if ymin < 0.5 * low and low - ymin >= 0.004:
        a_ = jmin
        while a_ - 1 >= max(0, k - WC) and y[a_ - 1] < pre - 0.1 * (pre - ymin):
            a_ -= 1
        b_ = jmin
        while b_ + 1 < min(nfr, k + WC + 1) and y[b_ + 1] < ymin + 0.9 * (post - ymin):
            b_ += 1
        tail = y[k:]
        if post < 0.3 * pre and len(tail) and float(np.max(tail[min(len(tail) - 1, WC):])) < max(0.004, 0.5 * pre) \
                and nfr - k < 4 * WC:
            return None   # the film's own fade-out at the end (transition_out)
        return dict(base, type='dip_to_dark', frame=int(a_), t=r(a_ / fps, 3), frames=int(b_ - a_ + 1),
                    confidence=r(clip01(0.5 + (low - ymin) / max(low, 1e-3) * 0.5), 2),
                    what=f'drops to {ymin / max(pre, 1e-4) * 100:.0f}% of the light, then the next picture comes up')
    # flash: a short burst of light that comes back down
    seg2 = y[max(0, k - 1):min(nfr, k + 6)]
    ymax, jmax = float(seg2.max()), max(0, k - 1) + int(seg2.argmax())
    if ymax > 1.5 * high + 0.01 and ymax - high >= 0.02:
        half = high + 0.5 * (ymax - high)
        width = int(sum(1 for v in y[max(0, k - 1):min(nfr, k + 10)] if v > half))
        if width <= 6:
            return dict(base, type='flash', frames=width, confidence=r(clip01((ymax - high) / max(high, 0.05) / 2), 2),
                        what=f'light burst to {ymax:.2f} luma')
    # reframe: one whole-frame transform explains the change (punch-in, jump)
    if (k - 1) in A['big'] and k in A['big']:
        jp = jump_step(A['big'][k - 1], A['big'][k], aw, ah)
        if jp:
            return dict(base, type='reframe', frames=0, confidence=r(clip01(0.4 + 0.6 * jp['ratio']), 2),
                        scale=r(jp['scale'], 3), dx_pct=r(jp['dx'], 1), dy_pct=r(jp['dy'], 1), roll_deg=r(jp['rot'], 1),
                        what=f"jump x{jp['scale']:.2f}")
    # a few-frame fade or dissolve: the change is spread over several frames and the frames between are blends
    dsa = A['ds']
    thr_ = max(0.08, 0.15 * float(dsa[k]))
    a_ = k
    while a_ - 1 > max(0, k - WC) and dsa[a_ - 1] >= thr_:
        a_ -= 1
    b_ = k
    while b_ + 1 < min(nfr, k + WC) and dsa[b_ + 1] >= thr_:
        b_ += 1
    if b_ - a_ + 1 >= 3 and all(f in A['win'] for f in range(a_ - 1, b_ + 1)):
        F0 = A['win'][a_ - 1].astype(np.float32) / 255.0
        F1 = A['win'][b_].astype(np.float32) / 255.0
        AB = F1 - F0
        nab = float((AB * AB).sum())
        if nab / AB.size > 1e-5:
            al = [float(((A['win'][f].astype(np.float32) / 255.0 - F0) * AB).sum()) / nab for f in range(a_, b_)]
            res = [math.sqrt(float(((A['win'][f].astype(np.float32) / 255.0 - F0 - a_l * AB) ** 2).sum()) / nab)
                   for f, a_l in zip(range(a_, b_), al)]
            mono = all(al[q + 1] >= al[q] - 0.08 for q in range(len(al) - 1))
            if mono and al and al[0] < 0.6 and al[-1] > 0.4 and float(np.mean(res)) < 0.45:
                kind = 'fade' if (post < 0.6 * pre or pre < 0.6 * post) else 'dissolve'
                return dict(base, type='unknown', frame=int(a_), t=r(a_ / fps, 3), frames=int(b_ - a_ + 1),
                            confidence=r(clip01(1.0 - float(np.mean(res))), 2), transition=kind,
                            what=f'{b_ - a_ + 1}-frame gradual change between two pictures ({kind}-like blend: fade, '
                                 f'dissolve or blur) that v1 did not split on')
    # cut: the old picture's detail does not survive into the new frame (normalised for brightness and contrast)
    if (k - 1) in A['win'] and k in A['win']:
        fa = A['win'][k - 1].astype(np.float32) / 255.0
        fb = A['win'][k].astype(np.float32) / 255.0
        za = (fa - fa.mean()) / max(float(fa.std()), 0.004)
        zb = (fb - fb.mean()) / max(float(fb.std()), 0.004)
        gh, gw = (6, 10) if fa.shape[1] >= fa.shape[0] else (10, 6)
        h_, w_ = fa.shape
        act = persist = 0
        for by in range(gh):
            for bx in range(gw):
                sl = (slice(by * h_ // gh, (by + 1) * h_ // gh), slice(bx * w_ // gw, (bx + 1) * w_ // gw))
                a_blk = za[sl]
                if a_blk.std() > 0.3 or abs(a_blk.mean()) > 0.5:
                    act += 1
                    if np.abs(a_blk - zb[sl]).mean() < 0.5:
                        persist += 1
        share = persist / max(1, act)
        dsk = c.get('ds') or 0
        if act >= 3 and (share < 0.4 or (dsk >= 0.7 and share < 0.6) or ((c.get('spike') or 0) >= 20 and share < 0.65)):
            return dict(base, type='cut', frames=0, confidence=r(clip01(0.5 + (c.get('spike') or 0) / 40 + (0.4 - share)), 2),
                        what='hard cut that v1 did not split on' + (' (dark frames)' if high < 0.12 else ''))
        fading_out = post < 0.6 * pre and nfr - k < 1.5 * fps and float(np.max(y[k:])) <= pre
        if c.get('rb', 0) >= 0.35 and c.get('ds', 0) < 0.3 and abs(post - pre) / max(low, 0.006) >= 0.35 and not fading_out:
            return dict(base, type='unknown', frames=0, confidence=0.4,
                        what=f'brightness step {pre:.3f} -> {post:.3f} luma (light change?)')
    return None


def same_content(A, f0, f1):
    """Do two frames show the same picture apart from focus? Correlation after a strong blur (sigma 3 px at 160 px)
    removes the detail that focus changes, but not new or moved content."""
    a_, b_ = A['smalls'].get(f0), A['smalls'].get(f1)
    if a_ is None or b_ is None:
        return None
    ga = cv2.GaussianBlur(a_.astype(np.float32), (0, 0), 3)
    gb = cv2.GaussianBlur(b_.astype(np.float32), (0, 0), 3)
    sa, sb = float(ga.std()), float(gb.std())
    if sa < 1.0 or sb < 1.0:
        return None
    return float(((ga - ga.mean()) * (gb - gb.mean())).mean() / (sa * sb))


def focus_evidence(A, f0, f1, sharper):
    """Is a change in overall sharpness a focus change? Yes if detail present at both ends gets sharper (or softer)
    in at least 2 blocks, or if blocks that gain (lose) detail already held the same blurred shape before (after);
    no if the new detail is new content (an element appearing)."""
    fmap = getattr(A, '_fmap', None) or {i_: (fm_, fc_) for i_, fm_, fc_ in A['focus']}
    if f0 not in fmap or f1 not in fmap:
        return False
    (fa, ca), (fb, cb) = fmap[f0], fmap[f1]
    both = ~np.isnan(fa) & ~np.isnan(fb) & (ca >= 12) & (cb >= 12) & (np.maximum(ca, cb) <= 2.0 * np.minimum(ca, cb))
    ratio = (fb + 0.3) / (fa + 0.3)
    if int((both & ((ratio <= 1 / 1.4) if sharper else (ratio >= 1.4))).sum()) >= 2:
        return True
    tex_a = ~np.isnan(fa) & (ca >= 12)
    tex_b = ~np.isnan(fb) & (cb >= 12)
    gained = (tex_b & ~tex_a) if sharper else (tex_a & ~tex_b)
    sa, sb = A['smalls'].get(f0), A['smalls'].get(f1)
    if sa is None or sb is None or not gained.any():
        return False
    ga = cv2.GaussianBlur(sa.astype(np.float32), (0, 0), 3)
    gb = cv2.GaussianBlur(sb.astype(np.float32), (0, 0), 3)
    gh, gw = gained.shape
    h, w = ga.shape
    for by, bx in zip(*np.where(gained)):
        y0, y1 = max(0, (by * h) // gh - 3), min(h, ((by + 1) * h) // gh + 3)
        x0, x1 = max(0, (bx * w) // gw - 3), min(w, ((bx + 1) * w) // gw + 3)
        pa, pb = ga[y0:y1, x0:x1], gb[y0:y1, x0:x1]
        if min(float(pa.std()), float(pb.std())) < 2.0:
            continue   # nothing there at one end: new content, or it left
        cc = float(((pa - pa.mean()) * (pb - pb.mean())).mean() / (pa.std() * pb.std()))
        if cc >= 0.8:
            return True
    return False


def focus_pulls(sh, A, fps, cut_frames, aw=480, ah=270):
    """Focus pulls inside a shot: edge sharpness changes >= 1.5x within 0.15-2 s, gradually (the middle of the change
    lies between its ends), in one direction, without a cut, with steady light, without fast motion, and with the same
    picture at both ends (strongly blurred, they still correlate >= 0.9), so new or moving content does not count."""
    y = A['y']
    lag = max(1, int(round(0.1 * fps)))
    pts = sorted((i, v[0]) for i, v in A['sharp'].items() if sh['sf'] <= i < sh['ef'] and v[0] and y[i] >= 0.012
                 and abs(y[i] - y[max(0, i - lag)]) / max(y[i], 0.01) < 0.15)
    if len(pts) < 6:
        return []
    out = []
    pieces, cur = [], []
    for i, v in pts:
        if cur and any(cur[-1][0] < cf <= i for cf in cut_frames):
            pieces.append(cur)
            cur = []
        cur.append((i, math.log(v)))
    pieces.append(cur)
    ds = A['ds']
    for pc in pieces:
        if len(pc) < 6:
            continue
        idx = [i for i, _ in pc]
        lv = np.array([v for _, v in pc])
        lv = np.array([np.median(lv[max(0, j - 1):j + 2]) for j in range(len(lv))])
        j = 0
        while j < len(pc) - 2:
            best = None
            for e in range(j + 2, len(pc)):
                dt = (idx[e] - idx[j]) / fps
                if dt > 2.0:
                    break
                dv = lv[e] - lv[j]
                if dt < 0.25 or abs(dv) < math.log(1.6):
                    continue
                if float(ds[idx[j] + 1:idx[e] + 1].max()) >= 0.3:
                    break
                steps_ = np.sign(np.diff(lv[j:e + 1]))
                # a pull takes time: the change reaches 90 % only after at least one sample in between (a single blurred
                # frame that snaps back is a blur flash, not a focus pull)
                reach = next(q for q in range(j + 1, e + 1) if abs(lv[q] - lv[j]) >= 0.9 * abs(dv))
                gradual = reach - j >= 2 and (idx[reach] - idx[j]) / fps >= 0.12
                if np.mean(steps_ == np.sign(dv)) < 0.7 or not gradual:
                    continue
                sc = same_content(A, idx[j], idx[e])
                if sc is None or sc < 0.9:
                    continue
                if not focus_evidence(A, idx[j], idx[e], dv > 0):
                    continue
                fast = [vel(st_, fps, aw, ah)[0] for st_ in A['steps'] if st_.get('ok') and idx[j] <= st_['i0'] and st_['i1'] <= idx[e]]
                if fast and float(np.median([np.linalg.norm(v_) for v_ in fast])) >= 15.0:
                    continue
                # motion blur is not focus: no fast move or failed (blurred) tracking just before or at the blurred
                # end, and the blurred end is not smeared in one direction
                bl = idx[j] if dv > 0 else idx[e]
                near_ = [st_ for st_ in A['steps'] if bl - 0.3 * fps <= st_['i1'] and st_['i0'] <= bl + 0.2 * fps]
                if any((st_.get('ok') and float(np.linalg.norm(vel(st_, fps, aw, ah)[0])) >= 30.0) or st_.get('jump')
                       for st_ in near_):
                    continue
                sb_, ss_ = A['sharp'].get(bl), A['sharp'].get(idx[e] if dv > 0 else idx[j])
                if sb_ and ss_ and min(sb_[1], sb_[2], ss_[1], ss_[2]) > 0:
                    an_b = max(sb_[1], sb_[2]) / min(sb_[1], sb_[2])
                    an_s = max(ss_[1], ss_[2]) / min(ss_[1], ss_[2])
                    if an_b > 1.5 * an_s:   # the blurred end is smeared one way more than the sharp end: motion blur
                        continue
                best = e
                break
            if best is None:
                j += 1
                continue
            e = best
            while e + 1 < len(pc) and np.sign(lv[e + 1] - lv[e]) == np.sign(lv[best] - lv[j]) and \
                    (idx[e + 1] - idx[j]) / fps <= 2.5:
                e += 1
            f0, f1 = idx[j], idx[e]
            ratio = math.exp(lv[e] - lv[j])
            out.append(dict(t=r(f0 / fps, 3), frame=int(f0), type='focus_pull', frames=int(f1 - f0),
                            confidence=r(min(0.6, clip01(0.3 + abs(math.log(ratio)) / 3)), 2),
                            what=('comes into focus' if ratio > 1 else 'goes out of focus') + f' (edge sharpness x{ratio:.1f})'))
            j = e + 1
    return out


def focus_for(sh, A, ah, cut_frames, fps=None, top=True):
    """In-focus region (box) and the blur outside it, from the per-block defocus maps sampled every 0.5 s. When hidden
    cuts split the shot, each part gets its own entry in by_part (the shot-level numbers mix the parts)."""
    if top and fps and cut_frames:
        edges = [sh['sf']] + [f for f in cut_frames if sh['sf'] < f < sh['ef']] + [sh['ef']]
        parts = []
        for a_, b_ in zip(edges, edges[1:]):
            if b_ - a_ < 0.5 * fps:
                continue
            fp = focus_for(dict(sf=a_, ef=b_), A, ah, [], fps, top=False)
            if fp:
                parts.append(dict(start=r(a_ / fps, 3), end=r(b_ / fps, 3), **fp))
        res = focus_for(sh, A, ah, [], fps, top=False)
        if res is not None and len(parts) >= 2:
            res['by_part'] = parts
        return res
    samples = [fm for i, fm, fc in A['focus'] if sh['sf'] <= i < sh['ef']]
    if len(samples) > 40:
        samples = [samples[k] for k in np.linspace(0, len(samples) - 1, 40).round().astype(int)]
    if not samples:
        return None
    st = np.stack(samples)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            med = np.nanmedian(st, axis=0)
    tex = ~np.isnan(med)
    if tex.sum() < 4:
        return dict(box=None, blur_px_1080=None, confidence=0.0, why='too little detail to judge focus')
    vals = med[tex]
    s_in = float(np.percentile(vals, 10))
    inf = tex & (med <= s_in + 0.35)
    outb = tex & (med >= s_in + 0.8)
    gh, gw = med.shape
    # the in-focus region = the biggest connected group of sharp blocks (scattered sharp blocks are not a region)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(inf.astype(np.uint8), connectivity=8)
    big_ = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if nlab > 1 else 0
    x0_, y0_ = stats[big_, cv2.CC_STAT_LEFT], stats[big_, cv2.CC_STAT_TOP]
    bw_, bh_ = stats[big_, cv2.CC_STAT_WIDTH], stats[big_, cv2.CC_STAT_HEIGHT]
    box = [r(x0_ / gw, 2), r(y0_ / gh, 2), r(bw_ / gw, 2), r(bh_ / gh, 2)]
    k = 1080.0 / ah
    res = dict(box=box, sharp_sigma_px_1080=r(s_in * k, 1), in_focus_share=r(inf.sum() / tex.sum(), 2))
    in_box = np.zeros_like(inf)
    in_box[y0_:y0_ + bh_, x0_:x0_ + bw_] = True
    outside = outb & ~in_box
    if outside.sum() >= 2:
        s_out = float(np.median(med[outside]))
        res['blur_px_1080'] = r(math.sqrt(max(0.0, s_out ** 2 - s_in ** 2)) * k, 1)
        res['out_of_focus_share'] = r(outside.sum() / tex.sum(), 2)
        sep = s_out - s_in
        res['confidence'] = r(min(0.8, clip01(0.2 + 0.25 * min(2.0, sep) + 0.04 * min(5, outside.sum())
                                              - (0.2 if len(samples) < 2 else 0))), 2)
    else:
        res['blur_px_1080'] = None
        res['confidence'] = r(min(0.6, clip01(0.2 + 0.03 * tex.sum())), 2)
        res['why'] = 'no defocused detail: everything with edges is about equally sharp (deep focus, or flat backgrounds)'
    return res


NAMES = {'pan+': 'pan_left', 'pan-': 'pan_right', 'tilt+': 'tilt_up', 'tilt-': 'tilt_down', 'zoom+': 'push_in',
         'zoom-': 'pull_out', 'roll+': 'roll', 'roll-': 'roll'}
ENAMES = {'pan+': 'slide_right', 'pan-': 'slide_left', 'tilt+': 'slide_down', 'tilt-': 'slide_up', 'zoom+': 'scale_up',
          'zoom-': 'scale_down', 'roll+': 'rotate', 'roll-': 'rotate'}


def vel(s, fps, aw, ah):
    """Step velocity as edge displacement in % of frame width per s: [pan, tilt, zoom, roll]."""
    dt = (s['i1'] - s['i0']) / fps
    return np.array([s['dx'] / dt, s['dy'] / dt * ah / aw, s['ls'] / dt * 50.0, s['rot'] / dt * 0.87]), dt


def smooth_vel(rows, half=0.12):
    """rows: [(t_mid, dt, v)] -> velocity per row averaged over steps whose midpoints are within +-half s."""
    out = []
    for t, _, _ in rows:
        sel = [(dt, v) for tt, dt, v in rows if abs(tt - t) <= half + 1e-9]
        T = sum(dt for dt, _ in sel)
        out.append(sum(v * dt for dt, v in sel) / max(T, 1e-6))
    return out


def classify(totals, rates, jitter, names):
    order = sorted(totals, key=lambda c: -totals[c][0])
    top, second = order[0], order[1]
    sig = totals[top][0] >= LIM['sig_total'] and rates[top] >= LIM['sig_rate']
    if not sig:
        return ('handheld' if jitter >= LIM['jitter'] else 'static'), False, top, second
    if totals[top][0] >= 2.0 * totals[second][0]:
        return names[top + ('+' if totals[top][1] > 0 else '-')], True, top, second
    return 'complex', True, top, second


def edge_disp(t, aw, ah):
    """Size of a move as displacement at the frame edge, in % of frame width (pan, tilt, zoom and roll comparable)."""
    return math.sqrt(t['dx'] ** 2 + (t['dy'] * ah / aw) ** 2 + (t['ls'] * 50.0) ** 2 + (t['rot'] * 0.87) ** 2)


def tot_of(sts):
    return dict(dx=float(sum(x['dx'] for x in sts)), dy=float(sum(x['dy'] for x in sts)),
                ls=float(sum(x['ls'] for x in sts)), rot=float(sum(x['rot'] for x in sts)))


def name_of(t, aw, ah, names):
    comps = dict(pan=(abs(t['dx']), t['dx']), tilt=(abs(t['dy']) * ah / aw, t['dy']), zoom=(abs(t['ls']) * 50.0, t['ls']),
                 roll=(abs(t['rot']) * 0.87, t['rot']))
    order = sorted(comps, key=lambda c: -comps[c][0])
    top, second = order[0], order[1]
    if comps[top][0] >= 2.0 * comps[second][0]:
        return names[top + ('+' if comps[top][1] > 0 else '-')]
    return 'complex'


def total_dict(t, span=None):
    d = dict(dx_pct=r(t['dx'], 2), dy_pct=r(t['dy'], 2), scale=r(math.exp(t['ls']), 4), roll_deg=r(t['rot'], 2))
    if span is not None:
        d['seconds'] = r(span, 3)
    return d


def build_segments(pieces, fps, aw, ah, names):
    """pieces: lists of consecutive steps between cuts / reframes (segments never cross them). Each piece is cut
    into runs of one kind of move; a run of >= 0.25 s without motion stays its own 'static' segment. Each step belongs
    to exactly one segment, so the segment totals add up (scales multiply) to the shot total."""
    segs = []
    for pc in pieces:
        if not pc:
            continue
        rows = []
        for st in pc:
            v, dt = vel(st, fps, aw, ah)
            rows.append((((st['i0'] + st['i1']) / 2) / fps, dt, v))
        sv = smooth_vel(rows)
        runs = []
        for st, v in zip(pc, sv):
            sp = float(np.linalg.norm(v))
            if sp < 1.5:
                lab = 'still'
            else:
                c = int(np.argmax(np.abs(v)))
                lab = ['pan', 'tilt', 'zoom', 'roll'][c] + ('+' if v[c] > 0 else '-') if abs(v[c]) >= 0.6 * sp else 'complex'
            if runs and runs[-1]['lab'] == lab:
                runs[-1]['steps'].append(st)
                runs[-1]['sp'].append(sp)
            else:
                runs.append(dict(lab=lab, steps=[st], sp=[sp]))
        span = lambda rn: (rn['steps'][-1]['i1'] - rn['steps'][0]['i0']) / fps
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for i, rn in enumerate(runs):
                if span(rn) >= 0.25:
                    continue
                nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
                same = [j for j in nbrs if runs[j]['lab'] == rn['lab']]
                cand = same or [j for j in nbrs if not (runs[j]['lab'] == 'still' and span(runs[j]) >= 0.25 and rn['lab'] != 'still')] or nbrs
                j = max(cand, key=lambda q: span(runs[q]))
                tgt = runs[j]
                if j < i:
                    tgt['steps'] += rn['steps']
                    tgt['sp'] += rn['sp']
                else:
                    tgt['steps'] = rn['steps'] + tgt['steps']
                    tgt['sp'] = rn['sp'] + tgt['sp']
                runs.pop(i)
                changed = True
                break
            merged = [runs[0]]
            for rn in runs[1:]:
                if rn['lab'] == merged[-1]['lab']:
                    merged[-1]['steps'] += rn['steps']
                    merged[-1]['sp'] += rn['sp']
                else:
                    merged.append(rn)
            runs = merged
        for rn in runs:
            t = tot_of(rn['steps'])
            sp_ = span(rn)
            ed = edge_disp(t, aw, ah)
            moving = ed >= 1.0 and ed / max(sp_, 1e-3) >= 0.4   # judged on the run's total, so slow drifts count
            if not moving:
                t = dict(dx=0.0, dy=0.0, ls=0.0, rot=0.0)
            move = name_of(t, aw, ah, names) if moving else 'static'
            spd = np.array(rn['sp'])
            ease = (ease_of(spd / spd.max(), move) if len(spd) >= 4 and spd.max() > 0 else ('hold' if move == 'static' else 'linear'))
            segs.append(dict(_t=t, _span=sp_, _moving=moving, start=r(rn['steps'][0]['i0'] / fps, 3),
                             end=r(rn['steps'][-1]['i1'] / fps, 3), move=move,
                             pan_x_pct_s=r(t['dx'] / sp_, 2) if sp_ > 0 else 0.0,
                             pan_y_pct_s=r(t['dy'] / sp_, 2) if sp_ > 0 else 0.0,
                             zoom_per_s=r(math.exp(t['ls'] / sp_), 4) if sp_ > 0 else 1.0,
                             roll_deg_s=r(t['rot'] / sp_, 2) if sp_ > 0 else 0.0, ease=ease,
                             total=total_dict(t, sp_)))
    return segs


def speed_curve(sts, sh, fps, aw, ah):
    rows = []
    for st in sts:
        v, dt = vel(st, fps, aw, ah)
        rows.append((((st['i0'] + st['i1']) / 2) / fps, dt, v))
    if not rows:
        return np.zeros(max(1, int(round((sh['t1'] - sh['t0']) / 0.1)))), 0.0, rows, []
    sv = smooth_vel(rows)
    tm = np.array([t for t, _, _ in rows])
    sp = np.array([float(np.linalg.norm(v)) for v in sv])
    dur = sh['t1'] - sh['t0']
    nb = max(1, int(round(dur / 0.1)))
    grid = sh['t0'] + (np.arange(nb) + 0.5) * dur / nb
    curve = np.interp(grid, tm, sp) if len(tm) > 1 else np.full(nb, sp[0])
    peak = float(curve.max()) if len(curve) else 0.0
    return (curve / peak if peak > 1e-6 else np.zeros_like(curve)), peak, rows, sv


def element_step(st):
    """The element's own motion in one step of an element shot: the dominant layer when it moves, else the second
    layer (a smaller thing moving over a still background), else nothing moved."""
    ident = abs(st['dx']) < 0.05 and abs(st['dy']) < 0.05 and abs(st['ls']) < 0.002 and abs(st['rot']) < 0.05
    if not ident:
        return dict(st, el_box=st['box'], el_moving=True)
    l2 = st.get('layer2')
    if l2 and (abs(l2['dx']) >= 0.05 or abs(l2['dy']) >= 0.05 or abs(l2['ls']) >= 0.002 or abs(l2['rot']) >= 0.05):
        return dict(st, dx=l2['dx'], dy=l2['dy'], ls=l2['ls'], rot=l2['rot'], el_box=l2['box'], el_moving=True)
    return dict(st, dx=0.0, dy=0.0, ls=0.0, rot=0.0, el_box=None, el_moving=False)


def motion_for(sh, steps, fps, aw, ah, d1seg, cut_frames):
    """Camera (frame-wide coherent motion), element motion (things moving over a still frame), reframes (jumps).
    cut_frames: hidden cuts / dips / reframes inside the shot; no step and no segment crosses them."""
    S = [s for s in steps if s['shot'] == sh['j'] and sh['sf'] <= s['i0'] and s['i1'] < sh['ef']
         and not any(s['i0'] < cf <= s['i1'] for cf in cut_frames)]
    dur = sh['t1'] - sh['t0']
    nb = max(1, int(round(dur / 0.1)))
    still_cam = dict(move='static', pan_x_pct_s=0.0, pan_y_pct_s=0.0, zoom_per_s=1.0, roll_deg_s=0.0, ease='hold',
                     speed=[0.0] * nb, peak_speed_pct_s=0.0)
    mine = [s for s in steps if s['shot'] == sh['j'] and sh['sf'] <= s['i0'] and s['i1'] < sh['ef']]

    def part_of_fast_move(k):
        """A jump is a reframe only if the picture holds around it; next to fast motion (or more failed steps) it
        is the fastest part of a whip, zoom or scroll."""
        jp = mine[k]['jump']
        jd = edge_disp(dict(dx=jp['dx'], dy=jp['dy'], ls=math.log(jp['scale']), rot=jp['rot']), aw, ah)
        for q in (k - 1, k + 1):
            if 0 <= q < len(mine):
                nb_ = mine[q]
                if nb_.get('jump') or (not nb_['ok'] and nb_.get('why') == 'track'):
                    return True
                if nb_['ok'] and edge_disp(nb_, aw, ah) >= 0.1 * jd:
                    return True
        return False
    reframes, fastjumps = [], []
    for k, s in enumerate(mine):
        if s.get('jump'):
            (fastjumps if part_of_fast_move(k) else reframes).append(s)
    reframes = [dict(t=r(s['i1'] / fps, 3), frame=s['i1'], scale=r(s['jump']['scale'], 3), dx_pct=r(s['jump']['dx'], 1),
                     dy_pct=r(s['jump']['dy'], 1), roll_deg=r(s['jump']['rot'], 1)) for s in reframes]
    fast_ids = {s['i0'] for s in fastjumps}

    def as_step(s):
        jp = s['jump']
        return dict(s, ok=True, inl=max(10, jp['inl']), ratio=jp['ratio'], dx=jp['dx'], dy=jp['dy'],
                    ls=math.log(jp['scale']), rot=jp['rot'], still=0.0, cover=1.0, box=[0.0, 0.0, 1.0, 1.0], second=0.0,
                    layer2=None)
    S = [as_step(s) if s.get('jump') and s['i0'] in fast_ids else s for s in S]
    ok = [s for s in S if s['ok'] and s['inl'] >= 10]
    quiet = len(d1seg) and float(np.mean(d1seg)) < 0.004
    if not S or not ok:
        if quiet or (S and not ok and all(s.get('why') == 'features' for s in S)):
            cam = dict(still_cam, confidence=0.4, total=total_dict(dict(dx=0, dy=0, ls=0, rot=0)))
            if reframes:
                cam['reframes'] = reframes
            return cam, None, None
        return None, None, 'too short to track' if not S else 'no trackable features (flat, blurred or fully changing frames)'
    cuts_all = sorted(set(cut_frames) | {x['frame'] for x in reframes})

    def pieces_of(sts):
        out, cur = [], []
        for st in sts:
            if cur and any(cur[-1]['i1'] < cf <= st['i1'] for cf in cuts_all):
                out.append(cur)
                cur = []
            cur.append(st)
        if cur:
            out.append(cur)
        return out
    ident = [abs(s['dx']) < 0.05 and abs(s['dy']) < 0.05 and abs(s['ls']) < 0.002 and abs(s['rot']) < 0.05 for s in ok]
    moving = [s for s, idn in zip(ok, ident) if not idn]
    med_still = float(np.median([s['still'] for s in moving])) if moving else 1.0
    med_cover = float(np.median([s['cover'] for s in moving])) if moving else 1.0
    kind = 'element' if moving and (med_still >= 0.3 or med_cover < 0.2) else 'camera'
    cov = len(ok) / len(S)
    inl = float(np.median([s['ratio'] for s in ok]))
    layered = float(np.mean([s['second'] >= 0.25 for s in ok]))
    element = None
    if kind == 'camera':
        segs = build_segments(pieces_of(ok), fps, aw, ah, NAMES)
        tot = dict(dx=sum(g['_t']['dx'] for g in segs), dy=sum(g['_t']['dy'] for g in segs),
                   ls=sum(g['_t']['ls'] for g in segs), rot=sum(g['_t']['rot'] for g in segs))
        norm_, peak, rows, sv = speed_curve(ok, sh, fps, aw, ah)
        raw = np.array([v[:2] for _, _, v in rows])
        jit = np.array([v[:2] for v in sv])
        jitter = float(np.sqrt(np.mean(np.sum((raw - jit) ** 2, axis=1)))) if len(rows) >= 3 else 0.0
        totals = dict(pan=(abs(tot['dx']), tot['dx']), tilt=(abs(tot['dy']) * ah / aw, tot['dy']),
                      zoom=(abs(tot['ls']) * 50.0, tot['ls']), roll=(abs(tot['rot']) * 0.87, tot['rot']))
        rates = {k_: v_[0] / dur for k_, v_ in totals.items()}
        move, sig, top, second = classify(totals, rates, jitter, NAMES)
        conf = cov * (0.3 + 0.7 * inl) * (1 - 0.4 * layered)
        if sig and totals[top][0] < 2.5 * totals[second][0]:
            conf *= 0.85
        if dur < 0.25:
            conf *= 0.7
        cam = dict(move=move, pan_x_pct_s=r(tot['dx'] / dur, 2), pan_y_pct_s=r(tot['dy'] / dur, 2),
                   zoom_per_s=r(math.exp(tot['ls'] / dur), 4), roll_deg_s=r(tot['rot'] / dur, 2),
                   ease=ease_of(norm_, move), speed=[r(v, 2) for v in norm_], peak_speed_pct_s=r(peak, 2),
                   confidence=r(clip01(conf), 2), total=total_dict(tot, dur))
        if jitter >= LIM['jitter']:
            cam['shake_pct_s'] = r(jitter, 2)
        if layered >= 0.3:
            cam['layers'] = 'a second motion layer moves differently (elements over a moving background?)'
        if move not in ('static', 'handheld') and (len(segs) >= 2 or len(cuts_all)):
            cam['segments'] = [{k_: v_ for k_, v_ in g.items() if not k_.startswith('_')} for g in segs]
    else:
        cam = dict(still_cam, confidence=r(clip01(0.35 + 0.5 * min(1.0, med_still / 0.6) * cov), 2),
                   total=total_dict(dict(dx=0, dy=0, ls=0, rot=0), dur))
        es = [element_step(s) for s in ok]
        segs = build_segments(pieces_of(es), fps, aw, ah, ENAMES)
        mv = [g for g in segs if g['_moving']]
        if mv:
            tot = dict(dx=sum(g['_t']['dx'] for g in mv), dy=sum(g['_t']['dy'] for g in mv),
                       ls=sum(g['_t']['ls'] for g in mv), rot=sum(g['_t']['rot'] for g in mv))
            T = sum(g['_span'] for g in mv)
            norm_, peak, _, _ = speed_curve(es, sh, fps, aw, ah)
            boxes = [s['el_box'] for s in es if s['el_moving'] and s['el_box']]
            box = np.median(np.array(boxes), axis=0) if boxes else np.array([0, 0, 1, 1])
            move = name_of(tot, aw, ah, ENAMES) if edge_disp(tot, aw, ah) >= 1.0 else 'jitter'
            element = dict(move=move, dx_pct_s=r(tot['dx'] / T, 2), dy_pct_s=r(tot['dy'] / T, 2),
                           zoom_per_s=r(math.exp(tot['ls'] / T), 4), roll_deg_s=r(tot['rot'] / T, 2),
                           ease=ease_of(norm_, move if move != 'jitter' else 'complex'),
                           speed=[r(v, 2) for v in norm_], peak_speed_pct_s=r(peak, 2),
                           region=[r(v, 2) for v in box], moving_share=r(min(1.0, T / dur), 2),
                           confidence=r(clip01(cov * (0.3 + 0.7 * inl) * 0.8), 2), total=total_dict(tot, T))
            element['segments'] = [{k_: v_ for k_, v_ in g.items() if not k_.startswith('_')} for g in segs]
            element['track'] = element_track(es, sh, fps, cuts_all)
            # self-check: the segments compose to the total (translations add, scales multiply)
            cx_ = sum(g['total']['dx_pct'] for g in element['segments'])
            cs_ = math.prod(g['total']['scale'] for g in element['segments'])
            if abs(cx_ - element['total']['dx_pct']) > 0.05 * max(1.0, abs(element['total']['dx_pct'])) or \
                    abs(math.log(cs_ / element['total']['scale'])) > 0.01:
                element['check'] = f'segments sum to dx {cx_:.2f} / x{cs_:.4f} vs total {element["total"]}'
    if reframes:
        cam['reframes'] = reframes
    return cam, element, None


def element_track(es, sh, fps, cuts_all):
    """The moving element's box every 0.5 s (fractions x, y, w, h): from the points that move with it; carried
    through holds, back-filled to the start of each piece, never across a cut."""
    out = []
    t = sh['t0']
    while t < sh['t1'] - 1e-6:
        near_ = [s['el_box'] for s in es if s['el_moving'] and s['el_box'] and abs((s['i0'] + s['i1']) / 2 / fps - t) <= 0.25]
        out.append(dict(t=r(t, 2), box=[r(v, 3) for v in np.median(np.array(near_), axis=0)] if near_ else None,
                        moving=bool(near_)))
        t += 0.5
    # carry boxes through holds and back-fill, within each piece between cuts
    piece = lambda tt: sum(1 for cf in cuts_all if cf / fps <= tt)
    for i in range(1, len(out)):
        if out[i]['box'] is None and out[i - 1]['box'] is not None and piece(out[i]['t']) == piece(out[i - 1]['t']):
            out[i]['box'] = out[i - 1]['box']
    for i in range(len(out) - 2, -1, -1):
        if out[i]['box'] is None and out[i + 1]['box'] is not None and piece(out[i]['t']) == piece(out[i + 1]['t']):
            out[i]['box'] = out[i + 1]['box']
    return out


def ease_of(c, move):
    if move in (None, 'static'):
        return 'hold'
    if move == 'handheld':
        return 'irregular'
    n = len(c)
    if n < 4:
        return 'linear'
    peaks = [i for i in range(1, n - 1) if c[i] >= 0.6 and c[i] >= c[i - 1] and c[i] >= c[i + 1]]
    big = []
    for p in peaks:
        if not big or (p - big[-1] > 2 and min(c[big[-1]:p + 1]) < 0.35):
            big.append(p)
    if len(big) >= 2:
        return 'irregular'
    q = max(1, n // 4)
    a = float(np.mean(c[:q]))
    m = float(np.mean(c[q:n - q])) if n - 2 * q > 0 else float(np.mean(c))
    z = float(np.mean(c[n - q:]))
    hi = max(a, m, z)
    if min(a, m, z) >= 0.6 * hi:
        return 'linear'
    if a < 0.6 * m and z < 0.6 * m:
        return 'ease_in_out'
    if a >= z and a >= 0.9 * m:
        return 'ease_out'
    if z > a and z >= 0.9 * m:
        return 'ease_in'
    return 'irregular'


def cadence_of(d1seg, fps, floor=0.0012, d2seg=None, floor2=None):
    """Hold share inside moving stretches: ~0 = new picture every frame ('ones'), 0.5 = on twos, 0.375 = 15 fps in 24.
    Measured on the mean frame difference and on the local (98th percentile) difference; the larger hold share
    wins, because a small moving element hides in the mean and a busy background hides in the local measure."""
    if d2seg is not None:
        a = cadence_of(d1seg, fps, floor)
        b = cadence_of(d2seg, fps, floor2)
        cands = [c for c in (a, b) if c and c.get('hold_share') is not None]
        if not cands:
            return a or b
        return max(cands, key=lambda c: c['hold_share'])
    if len(d1seg) < 8:
        return None
    mov, hold = 0, 0
    hv, mvv = [], []
    for j in range(1, len(d1seg) - 1):
        nb = max(d1seg[j - 1], d1seg[j + 1])
        if max(nb, d1seg[j]) < 4 * floor:
            continue
        mov += 1
        if d1seg[j] < 0.3 * nb:
            hold += 1
            hv.append(d1seg[j] / max(nb, 1e-9))
        else:
            mvv.append(d1seg[j])
    if mov < 6:
        return dict(moving_frames=mov, unique_fps=None, hold_share=None, label='still', confidence=0.0, optional=True)
    hs = hold / mov
    ufps = fps * (1 - hs)
    # confidence: how many moving frames it rests on, and how clean the holds are (a held frame changes ~0)
    clean = 1.0 - min(1.0, float(np.median(hv)) / 0.3) if hv else 1.0
    conf = clip01((mov - 12) / 60.0) * (0.6 + 0.4 * clean)
    if hs < 0.12:
        label = 'ones'
    elif 0.42 <= hs <= 0.58:
        label = 'twos'
    elif 0.62 <= hs <= 0.72:
        label = 'threes'
    else:
        label = f'~{ufps:.0f} fps'
    return dict(moving_frames=mov, unique_fps=r(ufps, 1), hold_share=r(hs, 2), label=label, confidence=r(conf, 2),
                optional=bool(mov < 30 or conf < 0.4))


# ---------------------------------------------------------------- palette and text

def palette_of(px, k=5):
    if len(px) == 0:
        return []
    rng = np.random.default_rng(7)
    if len(px) > 12000:
        px = px[rng.choice(len(px), 12000, replace=False)]
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab).reshape(-1, 3)
    uniq = len(np.unique(np.round(lab / 2.0), axis=0))
    kk = max(1, min(k, uniq))
    cv2.setRNGSeed(7)
    _, labels, centers = cv2.kmeans(lab, kk, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2), 2,
                                    cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=kk).astype(float) / len(labels)
    cl = [[centers[i].astype(float), shares[i]] for i in range(kk)]
    merged = True
    while merged and len(cl) > 1:   # fold near-duplicates (dE76 < 8) into the bigger swatch
        merged = False
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                if np.linalg.norm(cl[i][0] - cl[j][0]) < 8:
                    a, b = (i, j) if cl[i][1] >= cl[j][1] else (j, i)
                    tot = cl[a][1] + cl[b][1]
                    cl[a][0] = (cl[a][0] * cl[a][1] + cl[b][0] * cl[b][1]) / tot
                    cl[a][1] = tot
                    cl.pop(b)
                    merged = True
                    break
            if merged:
                break
    cl.sort(key=lambda c: -c[1])
    out = []
    for c, s in cl[:5]:
        rgb = cv2.cvtColor(np.array(c, np.float32).reshape(1, 1, 3), cv2.COLOR_Lab2RGB).reshape(3)
        rgb = np.clip(np.round(rgb * 255), 0, 255).astype(int)
        out.append(dict(hex='#%02X%02X%02X' % tuple(rgb), share=r(s, 2)))
    return out


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def cct_of(rgb01):
    """Correlated colour temperature (McCamy) and distance from the Planckian locus (Duv, CIE 1960 uv; Krystek) of an
    sRGB colour. Returns (kelvin or None, label warm|neutral|cool|tinted)."""
    lin = srgb_to_linear(rgb01)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    X, Y_, Z = M @ lin
    sXYZ = X + Y_ + Z
    if sXYZ <= 1e-6 or Y_ < 1e-4:
        return None, None
    x, y = X / sXYZ, Y_ / sXYZ
    nmc = (x - 0.3320) / (0.1858 - y)
    T = 449 * nmc ** 3 + 3525 * nmc ** 2 + 6823.3 * nmc + 5520.33
    u, v = 4 * X / (X + 15 * Y_ + 3 * Z), 6 * Y_ / (X + 15 * Y_ + 3 * Z)
    if not (1500 <= T <= 15000):
        return None, 'tinted'
    up = (0.860117757 + 1.54118254e-4 * T + 1.28641212e-7 * T * T) / (1 + 8.42420235e-4 * T + 7.08145163e-7 * T * T)
    vp = (0.317398726 + 4.22806245e-5 * T + 4.20481691e-8 * T * T) / (1 - 2.89741816e-5 * T + 1.61456053e-7 * T * T)
    if math.hypot(u - up, v - vp) > 0.02:
        return None, 'tinted'
    return int(round(T, -1)), ('warm' if T < 4000 else 'neutral' if T <= 6500 else 'cool')


def highlights_of(px):
    """Colours of the brightest pixels (top 2 % of luma, at least 0.06), clustered in CIELAB: the rim, icon and accent
    colours a near-black average hides. Each: hex, share of the highlight pixels, hue, luma, temperature."""
    if len(px) < 20:
        return None
    rng = np.random.default_rng(11)
    if len(px) > 12000:
        px = px[rng.choice(len(px), 12000, replace=False)]
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab).reshape(-1, 3)
    kk = max(1, min(3, len(np.unique(np.round(lab / 3.0), axis=0))))
    cv2.setRNGSeed(11)
    _, labels, centers = cv2.kmeans(lab, kk, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2), 2,
                                    cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    out = []
    for i in range(kk):
        m = labels == i
        if m.mean() < 0.05:
            continue
        rgb = px[m].astype(float).mean(axis=0) / 255.0
        L_, A_, B_ = centers[i]
        hue = (math.degrees(math.atan2(B_, A_)) + 360) % 360
        chroma = math.hypot(A_, B_)
        kelvin, temp = cct_of(rgb)
        rec = dict(hex='#%02X%02X%02X' % tuple(int(round(v * 255)) for v in rgb), share=r(m.mean(), 2),
                   hue_deg=r(hue, 0) if chroma >= 8 else None, chroma=r(chroma, 1),
                   luma=r(float(rgb @ np.array([0.2126, 0.7152, 0.0722])), 3), temperature=temp)
        if kelvin:
            rec['kelvin'] = kelvin
        out.append(rec)
    out.sort(key=lambda d: -d['share'])
    return out


def accents_of(px, frame_share):
    """Clearly coloured, not-dark pixels (CIELAB chroma >= 25, luma >= 0.06) clustered into up to 3 accent colours;
    share = that colour's share of the whole frame (accents are usually small)."""
    if len(px) < 20:
        return None
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab).reshape(-1, 3)
    kk = max(1, min(3, len(np.unique(np.round(lab / 4.0), axis=0))))
    cv2.setRNGSeed(13)
    _, labels, centers = cv2.kmeans(lab, kk, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2), 2,
                                    cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    out = []
    for i in range(kk):
        m = labels == i
        if m.mean() < 0.1:
            continue
        rgb = px[m].astype(float).mean(axis=0) / 255.0
        L_, A_, B_ = centers[i]
        out.append(dict(hex='#%02X%02X%02X' % tuple(int(round(v * 255)) for v in rgb),
                        hue_deg=r((math.degrees(math.atan2(B_, A_)) + 360) % 360, 0), chroma=r(math.hypot(A_, B_), 1),
                        share=r(m.mean() * frame_share, 4)))
    out.sort(key=lambda d: -d['share'])
    return out or None


ASC = set('bdfhklt')
DESC = set('gjpqy,;()[]{}/|$@Q')


def text_extent(s):
    """Line ink extent in cap heights, from the characters present (see CAP_A/CAP_B)."""
    if any(c in ASC for c in s):
        top = 1.04
    elif any(c.isupper() or c.isdigit() or c in '$%&#@!?/\\|()[]{}' for c in s):
        top = 1.0
    else:
        top = 0.72
    return top + (0.30 if any(c in DESC for c in s) else 0.0)


def norm_text(s):
    return re.sub(r'[^0-9a-z]+', '', s.lower())


NARROW = set("iljtfr!.,:;'|I1")
WIDE = set('mwMW')
ANGLE_K = 0.009   # Vision's quad height grows ~0.9 % per degree of tilt (rotated-text calibration, 180 reads)


def em_width(s):
    """Advance width of a string in em for a normal-width sans (Helvetica / Inter class)."""
    t = 0.0
    for c in s:
        if c == ' ':
            t += 0.27
        elif c in NARROW:
            t += 0.28
        elif c in WIDE:
            t += 0.85
        elif c.isupper():
            t += 0.68
        elif c.isdigit() or c in '$%':
            t += 0.58
        elif c.isalpha():
            t += 0.54
        else:
            t += 0.35
    return t


def quad_geom(q, W, H):
    """Quad [tl, tr, br, bl] (fractions, y down) -> (line height perpendicular to the baseline px, baseline angle deg,
    + = rising to the right, baseline length px)."""
    P = [np.array([x * W, y * H]) for x, y in q]
    tl, tr, br, bl = P
    base = (br - bl) + (tr - tl)
    L = float(np.linalg.norm(base)) / 2
    if L < 1e-6:
        return None, None, None
    u = base / (2 * L)
    nrm = np.array([-u[1], u[0]])
    hgt = (abs(float(np.dot(tl - bl, nrm))) + abs(float(np.dot(tr - br, nrm)))) / 2
    return hgt, math.degrees(math.atan2(-u[1], u[0])), L


def textlike(text):
    t = text.strip()
    letters = sum(ch.isalpha() for ch in t)
    nonspace = sum(not ch.isspace() for ch in t)
    if letters >= 2 and letters >= 0.5 * nonspace:
        return True
    return bool(re.fullmatch(r'[$€£]?[-+]?\d[\d,.:]*%?', t)) and sum(ch.isdigit() for ch in t) >= 2


def merge_text(obs, ocr_step, frame_w, frame_h):
    """Group per-frame reads into items with first/last times, a median box and a size estimate. Sizes come from the
    line height perpendicular to the baseline (Vision's quad), so tilted text is not inflated by its upright box; a
    height that the line's own width cannot hold is replaced by the width-based size."""
    gap = max(1.0, 2.5 * ocr_step)
    items = []
    for t, reads in obs:
        for rd in reads:
            text, conf, box = rd[0], rd[1], rd[2]
            quad = rd[3] if len(rd) > 3 else None
            key = norm_text(text)
            if conf < 0.3 or len(key) < 2 or len(key) < 0.4 * len(text.strip()) or not textlike(text):
                continue
            best, bscore = None, 0.0
            for it in items:
                if t - it['last'] > gap or t <= it['last'] and it['seen'].get(t):
                    continue
                lb = it['boxes'][-1]
                sim = difflib.SequenceMatcher(None, key, it['key']).ratio()
                pref = (key.startswith(it['key']) or it['key'].startswith(key)) and min(len(key), len(it['key'])) >= 3
                near = abs(box[0] - lb[0]) < 0.04 and abs(box[1] - lb[1]) < 0.04
                ov = not (box[0] > lb[0] + lb[2] or lb[0] > box[0] + box[2] or box[1] > lb[1] + lb[3] or lb[1] > box[1] + box[3])
                score = sim + (0.3 if ov else 0) + (0.2 if near else 0)
                if (sim >= 0.8 and (ov or near or sim >= 0.92)) or (pref and near):
                    if score > bscore:
                        best, bscore = it, score
            if best is None:
                best = dict(key=key, first=t, last=t, reads=[], boxes=[], seen={})
                items.append(best)
            best['last'] = t
            best['seen'][t] = True
            best['reads'].append((text.strip(), conf, box, quad))
            best['boxes'].append(box)
            if len(key) > len(best['key']):
                best['key'] = key
    out = []
    for it in items:
        confs = [c for _, c, _, _ in it['reads']]
        if not (np.mean(confs) >= 0.8 or (len(confs) >= 3 and np.mean(confs) >= 0.5)):
            continue   # one-off, low-confidence reads are mostly icons or blur read as words
        variants = collections.defaultdict(list)
        for text, conf, box, quad in it['reads']:
            variants[text].append(conf)
        texts = sorted(variants, key=lambda v: (-len(variants[v]) * np.mean(variants[v]), -len(v)))
        best = texts[0]
        longer = [v for v in texts if norm_text(v).startswith(norm_text(best)) and len(v) > len(best)
                  and np.mean(variants[v]) >= 0.5]
        if longer:   # typed-on text: keep the finished line
            best = max(longer, key=len)
        mine = [rd for rd in it['reads'] if rd[0] == best] or it['reads']
        box = np.median(np.array([rd[2] for rd in mine]), axis=0)
        geo = [quad_geom(rd[3], frame_w, frame_h) for rd in mine if rd[3]]
        geo = [g for g in geo if g[0]]
        if geo:
            h_px = float(np.median([g[0] for g in geo]))
            ang = float(np.median([g[1] for g in geo]))
            L = float(np.median([g[2] for g in geo]))
        else:
            h_px, ang, L = float(box[3]) * frame_h, 0.0, float(box[2]) * frame_w
        h_eff = h_px / (1 + ANGLE_K * min(25.0, abs(ang)))
        cap = max(1.0, (h_eff - CAP_B * frame_h / 720.0) / (CAP_A * text_extent(best)))
        rec = dict(text=best, first=r(it['first'], 2), last=r(it['last'], 2), box=[r(v, 3) for v in box],
                   line_h_pct=r(h_px / frame_h * 100, 2), cap_h_px_1080=int(round(cap * 1080.0 / frame_h)),
                   angle_deg=r(ang, 1), conf=r(np.mean(confs), 2))
        ew = em_width(best)
        if len(best.replace(' ', '')) >= 3 and ew > 0:
            cap_w = 0.72 * L / ew
            if cap > 1.6 * cap_w:   # taller than its own width allows even for condensed type: the height is inflated
                rec['cap_h_px_1080'] = int(round(cap_w * 1.1 * 1080.0 / frame_h))
                rec['size_from'] = 'width'
        out.append(rec)
    out.sort(key=lambda x: (x['first'], x['box'][1]))
    return out


# ---------------------------------------------------------------- audio

KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def zs(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def analyse_audio(video, work, dur, fps, shots, bounds, words, bed, log, review):
    import librosa
    import soundfile as sf
    from scipy.signal import find_peaks
    notes = []
    wav = work / 'a.wav'
    p = run(['ffmpeg', '-v', 'error', '-y', '-i', str(video), '-map', '0:a:0', '-vn', '-t', f'{dur:.3f}', '-ac', '1',
             '-ar', '22050', '-c:a', 'pcm_f32le', str(wav)])
    if p.returncode or not wav.exists() or wav.stat().st_size < 8000:
        return None, ['audio track could not be decoded'], None
    y, sr = sf.read(str(wav), dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.nan_to_num(y[:int(dur * sr)])
    blk = sr // 2
    nblk = int(math.ceil(len(y) / blk))
    rms_db = []
    for i in range(nblk):
        seg = y[i * blk:(i + 1) * blk]
        v = 20 * math.log10(math.sqrt(float(np.mean(seg ** 2))) + 1e-9) if len(seg) else -90.0
        rms_db.append(round(max(-90.0, v), 1))
    out = dict(has_audio=True, tempo=None, key=None, loudness=None, energy=dict(step_s=0.5, rms_db=rms_db),
               sections=[], cues=[])
    if not rms_db or max(rms_db) < -55:
        notes.append('audio track is silent or nearly silent (max 0.5 s RMS below -55 dBFS)')
        return out, notes, None

    HOP = 512
    fr = sr / HOP
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    Hm, Pm = librosa.decompose.hpss(D, margin=1.0)
    eH, eP = float((Hm ** 2).sum()), float((Pm ** 2).sum())
    harm_share = eH / (eH + eP + 1e-12)
    mel = librosa.feature.melspectrogram(S=D ** 2, sr=sr, n_mels=64)
    melP = librosa.feature.melspectrogram(S=Pm ** 2, sr=sr, n_mels=64)
    mdb = librosa.power_to_db(mel, ref=np.max)
    pdb = librosa.power_to_db(melP, ref=np.max)
    oenv = librosa.onset.onset_strength(S=mdb, sr=sr, hop_length=HOP)
    oenv_p = librosa.onset.onset_strength(S=pdb, sr=sr, hop_length=HOP)
    mf = librosa.mel_frequencies(n_mels=66, fmax=sr / 2)[1:-1]
    oenv_low = librosa.onset.onset_strength(S=mdb[mf < 160], sr=sr, hop_length=HOP)
    nT = len(oenv)
    tt = librosa.frames_to_time(np.arange(nT), sr=sr, hop_length=HOP)
    chroma = librosa.feature.chroma_stft(S=Hm ** 2, sr=sr, hop_length=HOP)
    music_bed = not (bed and bed.get('speech') and bed.get('bed_present') is False)

    # ---- tempo, beats, downbeats
    tempo = None
    beats_t, beat_frames = np.array([]), np.array([], int)
    if dur >= 4:
        # tempo = the local tempo that holds for most of the film (tempogram, octave-folded), at the metrical level
        # the global tempogram supports best (weighted towards ~110 BPM); the beat tracker then runs at that tempo.
        nT_ = len(oenv_p)
        win = int(min(384, max(64, nT_ // 2)))
        tg = librosa.feature.tempogram(onset_envelope=oenv_p, sr=sr, hop_length=HOP, win_length=win)
        tf = librosa.tempo_frequencies(tg.shape[0], sr=sr, hop_length=HOP)
        okr = np.where((tf >= 40) & (tf <= 240))[0]

        def fold(b_):
            while b_ >= 120:
                b_ /= 2
            while b_ < 60:
                b_ *= 2
            return b_
        local = []
        for c in range(0, tg.shape[1], max(1, int(fr))):
            col = tg[okr, c]
            if col.max() > 0:
                local.append(float(tf[okr][int(np.argmax(col))]))
        tg_mean = tg.mean(axis=1)

        def tg_at(b_):
            return float(tg_mean[okr[int(np.argmin(np.abs(tf[okr] - b_)))]])
        bpm = None
        if local:
            fl = np.array([fold(v) for v in local])
            bins = np.round(np.log2(fl) * 50)
            vals, cnts = np.unique(bins, return_counts=True)
            m0 = float(np.median(fl[bins == vals[int(np.argmax(cnts))]]))
            cands = [m0 * k for k in (0.5, 1, 2, 4) if 50 <= m0 * k <= 200]
            if cands:
                bpm = max(cands, key=lambda c: tg_at(c) * math.exp(-0.5 * (math.log2(c / 110.0) / 0.9) ** 2))
        if bpm:
            tmp, bfr = librosa.beat.beat_track(onset_envelope=oenv_p, sr=sr, hop_length=HOP, bpm=bpm, tightness=100,
                                               units='frames')
        else:
            tmp, bfr = librosa.beat.beat_track(onset_envelope=oenv_p, sr=sr, hop_length=HOP, start_bpm=110.0,
                                               tightness=100, units='frames')
            bpm = float(np.atleast_1d(tmp)[0])
        bfr = np.asarray(bfr, int)
        if bpm > 0 and len(bfr) >= 4:
            x = oenv_p - oenv_p.mean()
            f = np.fft.rfft(x, 2 * len(x))
            ac = np.fft.irfft(f * np.conj(f))[:len(x)]
            ac = ac / (ac[0] + 1e-12)

            def ac_at(b):
                lag = 60 * fr / b
                lo, hi = int(lag * 0.97), int(math.ceil(lag * 1.03))
                return float(ac[lo:hi + 1].max()) if 1 <= lo and hi < len(ac) else 0.0
            acb = max(ac_at(bpm), ac_at(bpm / 2) if bpm / 2 >= 30 else 0.0)
            agree = [abs(fold(v) / fold(bpm) - 1) < 0.04 for v in local]
            stability = float(np.mean(agree)) if agree else 0.0
            bt = librosa.frames_to_time(bfr, sr=sr, hop_length=HOP)
            # refine: snap each beat to the strongest onset within +-40 ms on a 5.8 ms envelope, then fit one
            # constant grid (robust line fit); keep the grid when the beats sit on it, else the snapped beats
            fe = librosa.onset.onset_strength(y=y, sr=sr, hop_length=128, n_fft=1024)
            ft = librosa.frames_to_time(np.arange(len(fe)), sr=sr, hop_length=128, n_fft=1024)
            snapped = []
            for t_ in bt:
                a_, b_ = np.searchsorted(ft, t_ - 0.04), np.searchsorted(ft, t_ + 0.04)
                snapped.append(float(ft[a_ + int(np.argmax(fe[a_:b_]))]) if b_ > a_ else float(t_))
            snapped = np.array(snapped)
            kk = np.arange(len(snapped))
            keep_ = np.ones(len(snapped), bool)
            for _ in range(2):
                per, ph = np.polyfit(kk[keep_], snapped[keep_], 1)
                res = snapped - (ph + per * kk)
                mad = float(np.median(np.abs(res[keep_]))) + 1e-3
                keep_ = np.abs(res) <= max(0.02, 3 * mad)
            rstd = float(np.std(res[keep_]))
            grid = bool(rstd <= 0.03 and keep_.mean() >= 0.7 and stability >= 0.5)
            if grid:
                bt = ph + per * kk
                bpm = 60.0 / per
                acb = max(acb, ac_at(bpm))
            else:
                bt = snapped
            ibi = np.diff(bt)
            cv = float(ibi.std() / ibi.mean()) if len(ibi) >= 3 and ibi.mean() > 0 else 1.0
            conf = clip01((acb - 0.05) / 0.30) ** 0.7 * (0.35 + 0.65 * stability)
            if not grid and cv > 0.12:
                conf *= 0.7
            beats_t, beat_frames = bt, bfr
            tempo = dict(bpm=r(bpm, 2), confidence=r(conf, 2), first_downbeat_s=None, beats=[], downbeats=[],
                         grid='constant (fitted)' if grid else 'tracked',
                         pulse=dict(autocorr=r(acb, 3), stability=r(stability, 2), ibi_cv=r(cv, 3),
                                    grid_residual_ms=r(rstd * 1000, 1)))
            # downbeat by accents: kick (low onsets) + harmonic change + loudness, phase that wins over 4/4 bars
            if len(bfr) >= 8:
                kick = np.array([oenv_low[max(0, b - 2):b + 3].max() for b in bfr])
                seg = librosa.util.sync(chroma, bfr, aggregate=np.mean)
                hc = []
                for k_ in range(len(bfr)):
                    a_, b_ = seg[:, k_], seg[:, k_ + 1] if k_ + 1 < seg.shape[1] else seg[:, k_]
                    hc.append(1 - float(a_ @ b_) / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-9))
                loud = np.array([mdb[:, max(0, b - 1):b + 2].mean() for b in bfr])
                acc = zs(kick) + zs(hc) + 0.5 * zs(loud)
                sc = [float(np.mean(acc[p::4])) for p in range(4)]
                order = np.argsort(sc)[::-1]
                p0 = int(order[0])
                dconf = clip01((sc[order[0]] - sc[order[1]]) / 0.8)
                tempo['first_downbeat_s'] = r(bt[p0], 3)
                tempo['downbeat_confidence'] = r(dconf, 2)
                tempo['downbeats'] = [r(v, 3) for v in bt[p0::4]]
            tempo['beats'] = [r(v, 3) for v in bt]
            tempo['beat_period_s'] = r(60.0 / bpm, 4)
            if conf < 0.25:
                notes.append(f'tempo low confidence ({conf:.2f}): no steady pulse; beats are given but should not '
                             'drive edits')
            elif not music_bed:
                notes.append('beat grid measured on a speech-led track; check it before cutting to it')
    out['tempo'] = tempo

    # ---- key
    energy = (Hm ** 2).sum(axis=0)
    cvec = (chroma * energy[None, :]).sum(axis=1)
    if harm_share >= 0.25 and cvec.sum() > 0 and music_bed:
        rs = []
        for t_ in range(12):
            rs.append((float(np.corrcoef(cvec, np.roll(KK_MAJ, t_))[0, 1]), f'{NOTES[t_]} major'))
            rs.append((float(np.corrcoef(cvec, np.roll(KK_MIN, t_))[0, 1]), f'{NOTES[t_]} minor'))
        rs.sort(reverse=True)
        kc = clip01((rs[0][0] - rs[1][0]) / 0.12) * clip01((rs[0][0] - 0.4) / 0.4)
        out['key'] = dict(name=rs[0][1], confidence=r(kc, 2), runner_up=rs[1][1], r=r(rs[0][0], 3))
        if kc < 0.25:
            notes.append(f'key low confidence ({kc:.2f}): {rs[0][1]} vs {rs[1][1]}')
    else:
        notes.append('key not measured: ' + ('no music under the voice' if not music_bed else
                                             f'little tonal content (harmonic share {harm_share:.2f})'))

    # ---- sections: novelty on 0.5 s blocks of timbre + harmony + level, labels by relative energy
    nb = len(rms_db)
    sections = []
    if nb >= 12:
        X = []
        for i in range(nb):
            a_, b_ = int(i * 0.5 * fr), int((i + 1) * 0.5 * fr)
            b_ = max(a_ + 1, min(b_, nT))
            a_ = min(a_, nT - 1)
            X.append(np.concatenate([mdb[:, a_:b_].mean(axis=1) / 10.0, chroma[:, a_:b_].mean(axis=1) * 2.0,
                                     [rms_db[i] / 6.0]]))
        X = np.array(X)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
        Xs = np.array([X[max(0, i - 1):i + 2].mean(axis=0) for i in range(nb)])
        Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)
        SSM = Xn @ Xn.T
        K = int(min(12, max(4, round(nb / 8))))
        g = np.outer(np.r_[-np.ones(K), np.ones(K)], np.r_[-np.ones(K), np.ones(K)])
        nov = np.zeros(nb)
        for i in range(K, nb - K):
            nov[i] = float((SSM[i - K:i + K, i - K:i + K] * g).sum())
        nov = np.maximum(0, nov)
        pk, _ = find_peaks(nov, distance=max(4, nb // 12), height=nov.mean() + 0.5 * nov.std() if nov.max() > 0 else 1e9)
        minb = 4 if dur <= 30 else 6
        maxs = max(3, int(round(dur / 10)))
        order_ = sorted((int(v) for v in pk), key=lambda v: -nov[v])
        rd = np.array(rms_db)
        rs_ = np.convolve(rd, np.ones(4) / 4, mode='same')
        jumps = [i for i in range(4, nb - 4) if abs(rs_[i + 2] - rs_[i - 2]) >= 10]
        cuts_ = []
        for c in order_ + jumps:
            if len(cuts_) >= maxs - 1:
                break
            if minb <= c <= nb - minb and all(abs(c - x) >= minb for x in cuts_):
                cuts_.append(c)
        merged_ = [0] + sorted(cuts_) + [nb]
        for a_, b_ in zip(merged_, merged_[1:]):
            seg = rd[a_:b_]
            e_db = 10 * math.log10(float(np.mean(10 ** (seg / 10))) + 1e-12)
            slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0]) * len(seg) if len(seg) >= 3 else 0.0
            sections.append(dict(start=r(a_ * 0.5, 2), end=r(min(dur, b_ * 0.5), 2), label='steady', rms_db=r(e_db, 1),
                                 _slope=slope))
        top = max(s['rms_db'] for s in sections)
        spread = top - min(s['rms_db'] for s in sections)
        for i, s in enumerate(sections):
            rel = s['rms_db'] - top
            prev = sections[i - 1]['rms_db'] if i > 0 else None
            nxt = sections[i + 1]['rms_db'] if i + 1 < len(sections) else None
            if spread < 3:
                s['label'] = 'steady'
            elif i == 0 and rel <= -4 and len(sections) > 1:
                s['label'] = 'intro'
            elif i == len(sections) - 1 and len(sections) > 1 and (rel <= -4 or s['_slope'] <= -6):
                s['label'] = 'outro'
            elif prev is not None and rel >= -2.5 and s['rms_db'] - prev >= 3:
                s['label'] = 'drop'
            elif prev is not None and nxt is not None and rel <= -5 and prev - s['rms_db'] >= 3 and nxt - s['rms_db'] >= 3:
                s['label'] = 'break'
            elif s['_slope'] >= 4 or (nxt is not None and nxt - s['rms_db'] >= 3 and rel <= -2.5):
                s['label'] = 'build'
        for s in sections:
            s.pop('_slope', None)
    elif nb:
        sections = [dict(start=0.0, end=r(dur, 2), label='steady',
                         rms_db=r(10 * math.log10(float(np.mean(10 ** (np.array(rms_db) / 10))) + 1e-12), 1))]
    out['sections'] = sections

    # ---- cues: onsets + rising events, with attack, length, centroid, percussive share
    eh = 128
    e = librosa.feature.rms(y=y, frame_length=512, hop_length=eh)[0]
    e_db = 20 * np.log10(e + 1e-7)
    e_s = np.convolve(e_db, np.ones(5) / 5, mode='same')
    et = (np.arange(len(e_db)) * eh + 256) / sr
    cent = librosa.feature.spectral_centroid(S=D, sr=sr)[0]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    lowmask = freqs < 150
    pw = D ** 2
    low_share_t = pw[lowmask].sum(axis=0) / (pw.sum(axis=0) + 1e-12)
    perc_t = (Pm ** 2).sum(axis=0) / ((Hm ** 2).sum(axis=0) + (Pm ** 2).sum(axis=0) + 1e-12)
    wmask = [(w[0] - 0.05, w[1]) for w in words]

    def in_words(t):
        return any(a <= t <= b for a, b in wmask)

    def feats(t, span=0.1):
        i0 = int(t * fr)
        i1 = max(i0 + 1, int((t + span) * fr) + 1)
        return (float(cent[i0:i1].mean()), float(low_share_t[i0:i1].mean()), float(perc_t[i0:i1].mean()))

    def env_at(t0, t1):
        a_, b_ = np.searchsorted(et, t0), np.searchsorted(et, t1)
        return e_db[a_:max(a_ + 1, b_)], et[a_:max(a_ + 1, b_)]

    p99 = float(np.percentile(oenv, 99)) + 1e-9
    onf = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=HOP, units='frames', wait=3)
    cues = []
    for f_ in onf:
        t = float(tt[f_])
        strength = clip01(oenv[f_] / p99)
        pre, _ = env_at(t - 0.25, t - 0.03)
        post, ptimes = env_at(t, t + 0.15)
        if not len(pre) or not len(post):
            continue
        pre_db, pk = float(np.median(pre)), int(np.argmax(post))
        peak_db, t_pk = float(post[pk]), float(ptimes[pk])
        rise = peak_db - pre_db
        if rise < 4 and strength < 0.5:
            continue
        tail, ttail = env_at(t_pk, t_pk + 3.0)
        below = np.where(tail < max(peak_db - 12, pre_db + 2))[0]
        length = max(0.01, float(ttail[below[0]] - t) if len(below) else float(ttail[-1] - t))
        c_hz, low_s, perc = feats(t)
        if in_words(t) and not (low_s >= 0.4 or (rise >= 12 and strength >= 0.6)):
            continue
        attack = (t_pk - t) * 1000
        if perc >= 0.5:
            if low_s >= 0.4 and rise >= 9 and length >= 0.25:
                kind = 'impact'
            elif length <= 0.08 or (c_hz >= 3500 and length <= 0.15):
                kind = 'click'
            else:
                kind = 'hit'
        elif perc <= 0.3:
            kind = 'tonal'
        else:
            kind = 'hit' if attack <= 40 and rise >= 10 else 'unknown'
        cues.append(dict(t=r(t, 3), kind=kind, strength=r(strength, 2), dur_s=r(max(0.0, length), 2),
                         _rise=rise, _src='onset'))
    # rising events (whoosh / riser): the level climbs steadily into a peak
    pk_idx, props = find_peaks(e_s, prominence=8, distance=max(1, int(0.15 * sr / eh)))
    for p_ in pk_idx:
        s_ = p_
        while s_ > 0 and e_s[s_ - 1] <= e_s[s_] + 1.0 and (et[p_] - et[s_ - 1]) < 4.0:
            s_ -= 1
        rise_t, rise_db = float(et[p_] - et[s_]), float(e_s[p_] - e_s[s_])
        if rise_t < 0.15 or rise_db < 8:
            continue
        seg = np.diff(e_s[s_:p_ + 1])
        if len(seg) == 0 or float((seg > 0).mean()) < 0.6:
            continue
        a_, b_ = int(et[s_] * fr), max(int(et[s_] * fr) + 1, int(et[p_] * fr))
        flat = float(np.mean(librosa.feature.spectral_flatness(S=D[:, a_:b_ + 1])))
        if rise_t >= 1.0:
            kind = 'riser'
        elif flat >= 0.05:
            kind = 'whoosh'
        else:
            kind = 'riser' if rise_t >= 0.6 else 'unknown'
        t = float(et[p_])
        if in_words(t) and rise_t < 0.6:
            continue
        cues = [c for c in cues if not (et[s_] - 0.02 <= c['t'] <= t + 0.03)]
        cues.append(dict(t=r(t, 3), kind=kind, strength=r(clip01(rise_db / 24), 2), dur_s=r(rise_t, 2),
                         rise_s=r(rise_t, 2), _rise=rise_db, _src='rise'))
    beat_ok = tempo is not None and tempo['confidence'] >= 0.25 and len(beats_t)
    btol = 0.06
    for c in cues:
        c['near_cut'] = bool(any(abs(c['t'] - b) <= 2.0 / fps + 1e-6 for b in bounds))
        c['on_beat'] = bool(beat_ok and float(np.min(np.abs(beats_t - c['t']))) <= btol)
        c['shot'] = shots[shot_index(shots, c['t'])]['n']
    cap = max(12, int(round(dur * 1.0)))
    keep = [c for c in cues if c['near_cut']]
    rest = sorted([c for c in cues if not c['near_cut']], key=lambda c: -c['strength'])
    keep += rest[:max(0, cap - len(keep))]
    keep.sort(key=lambda c: c['t'])
    for c in keep:
        c.pop('_rise', None)
        c.pop('_src', None)
    out['cues'] = keep
    if len(cues) > len(keep):
        notes.append(f'cue sheet keeps {len(keep)} of {len(cues)} sound events (all near cuts, then the strongest)')
    rv = None
    if review is not None:
        rv = dict(oenv=oenv_p / (oenv_p.max() + 1e-9), tt=tt, beats=beats_t,
                  downbeats=np.array(tempo['downbeats']) if tempo and tempo.get('downbeats') else np.array([]))
    log(f'audio: tempo {tempo and tempo["bpm"]} (conf {tempo and tempo["confidence"]}), key '
        f'{out["key"] and out["key"]["name"]}, {len(sections)} sections, {len(keep)} cues, harmonic share {harm_share:.2f}')
    return out, notes, rv


# ---------------------------------------------------------------- sheets and review images

def sheet_layout(w, h):
    r_ = h / w if w else 1
    if r_ > 1.25:
        cols, rows, tw = 6, 4, 250
    elif r_ > 0.8:
        cols, rows, tw = 6, 4, 300
    else:
        cols, rows, tw = 4, 6, 380
    th = int(round(tw * h / w / 2) * 2) if w else tw
    return dict(cols=cols, rows=rows, tw=tw, th=th)


def draw_sheets(tiles, lay, out_dir, stem, fontf):
    """tiles: [(label, PIL image)] -> pages in the v1 sheet look (dark grid, black label box, yellow label)."""
    tw, th, cols = lay['tw'], lay['th'], lay['cols']
    per, gap = cols * lay['rows'], 4
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f'{stem}-*.jpg'):
        old.unlink()
    paths = []
    for k in range(0, len(tiles), per):
        chunk = tiles[k:k + per]
        nrows = math.ceil(len(chunk) / cols)
        sheet = Image.new('RGB', (cols * tw + (cols + 1) * gap, nrows * th + (nrows + 1) * gap), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        for j, (label, im) in enumerate(chunk):
            x, y = gap + (j % cols) * (tw + gap), gap + (j // cols) * (th + gap)
            sheet.paste(ImageOps.pad(im.convert('RGB'), (tw, th), color=(16, 16, 16)), (x, y))
            box = d.textbbox((0, 0), label, font=fontf)
            d.rectangle([x, y, x + box[2] - box[0] + 12, y + box[3] - box[1] + 10], fill=(0, 0, 0))
            d.text((x + 6, y + 4 - box[1]), label, font=fontf, fill=(255, 235, 59))
        p = out_dir / f'{stem}-{k // per + 1:02d}.jpg'
        sheet.save(p, quality=82)
        paths.append(p)
    return paths


def ts3(t):
    """75.25 -> '1:15.250'"""
    t = max(0.0, float(t))
    m = int(t // 60)
    return f'{m}:{t - 60 * m:06.3f}'


def closeup_sheets(video, rate, nfr, W, H, out_shots, cues, out_dir, fontf, small_font, threads, cap=12):
    """Close-up strips around transitions and hidden changes: 14 consecutive native frames per event (6 before the
    change, the change frame with a yellow frame, 7 after); longer transitions are sampled across their length.
    The 12 most telling events per film: hidden changes and non-cut transitions first, then cuts with a sound hit."""
    fps = float(rate)
    ev = []
    for s in out_shots[1:]:
        ti = s['transition_in']
        typ = ti['type']
        score = (4.0 if typ != 'cut' else 2.0) + (ti.get('confidence') or 0)
        if typ == 'cut' and any(c.get('near_cut') and abs(c['t'] - s['start']) < 0.1 for c in cues or []):
            score += 0.5
        ev.append(dict(shot=s['n'], frame=s['start_frame'], type=typ, frames=int(ti.get('frames') or 0), score=score,
                       ref=ti, title=f"S{s['n']} in: {typ}" + (f" {ti.get('frames')}f" if ti.get('frames') else '')))
    for s in out_shots:
        for ic in s.get('internal_changes') or []:
            score = (5.0 if ic['type'] in ('cut', 'dip_to_dark', 'reframe') else 3.5) + (ic.get('confidence') or 0)
            ev.append(dict(shot=s['n'], frame=ic['frame'], type=ic['type'], frames=int(ic.get('frames') or 0), score=score,
                           ref=ic, title=f"S{s['n']} inside: {ic['type']}" + (f" {ic['frames']}f" if ic.get('frames') else '')
                           + (f" - {ic['what']}" if ic.get('what') else '')))
    for old in out_dir.glob('transitions-*.jpg'):
        old.unlink()
    if not ev:
        return [], []
    ev.sort(key=lambda e: -e['score'])
    ev = sorted(ev[:cap], key=lambda e: e['frame'])
    for e in ev:
        f0, n_ = e['frame'], e['frames']
        if n_ <= 8:
            fr_ = list(range(f0 - 6, f0 + 8))
        else:
            fr_ = [int(v) for v in np.linspace(f0 - 4, f0 + n_ + 4, 14).round()]
        e['show'] = [min(nfr - 1, max(0, f)) for f in fr_]
    r_ = H / W
    tw = 180 if r_ > 1.25 else 220 if r_ > 0.8 else 300
    th_ = int(round(tw * H / W / 2) * 2)
    need = sorted({f for e in ev for f in e['show']})
    ranges = []
    for f in need:
        if ranges and f <= ranges[-1][1] + 1:
            ranges[-1][1] = f
        else:
            ranges.append([f, f])
    expr = '+'.join(f'between(n\\,{a}\\,{b})' for a, b in ranges)
    cmd = ['ffmpeg', '-v', 'error', '-threads', str(threads), '-i', str(video), '-map', '0:v:0', '-an', '-vf',
           f"fps={rate.numerator}/{rate.denominator},select='{expr}',scale={tw}:{th_}:flags=area,format=rgb24",
           '-fps_mode', 'passthrough', '-f', 'rawvideo', '-']
    proc = subprocess.run(cmd, capture_output=True)
    n1 = tw * th_ * 3
    order = [f for a, b in ranges for f in range(a, b + 1)]
    frames = {}
    for i, f in enumerate(order):
        buf = proc.stdout[i * n1:(i + 1) * n1]
        if len(buf) == n1:
            frames[f] = Image.fromarray(np.frombuffer(buf, np.uint8).reshape(th_, tw, 3))
    shot_of = lambda f: next((s['n'] for s in out_shots if s['start_frame'] <= f < s['end_frame']), out_shots[-1]['n'])
    per_page = 2 if r_ > 0.8 else 3
    cols, gap, head = 7, 4, 26
    bw_ = cols * tw + (cols + 1) * gap
    bh_ = head + 2 * th_ + 3 * gap
    paths, index = [], []
    for pg in range(0, len(ev), per_page):
        chunk = ev[pg:pg + per_page]
        sheet = Image.new('RGB', (bw_, len(chunk) * bh_), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        for bi, e in enumerate(chunk):
            y0 = bi * bh_
            d.text((gap + 2, y0 + 5), e['title'][:120], font=small_font, fill=(255, 235, 59))
            for k, f in enumerate(e['show']):
                x, y = gap + (k % cols) * (tw + gap), y0 + head + gap + (k // cols) * (th_ + gap)
                im = frames.get(f)
                if im is not None:
                    sheet.paste(im, (x, y))
                if f == e['frame']:
                    d.rectangle([x - 2, y - 2, x + tw + 1, y + th_ + 1], outline=(255, 235, 59), width=3)
                lab = f'S{shot_of(f)} {ts3(f / fps)} f{f}'
                box = d.textbbox((0, 0), lab, font=fontf)
                d.rectangle([x, y, x + box[2] - box[0] + 10, y + box[3] - box[1] + 8], fill=(0, 0, 0))
                d.text((x + 5, y + 3 - box[1]), lab, font=fontf, fill=(255, 235, 59))
            index.append(dict(shot=e['shot'], frame=e['frame'], type=e['type'], page=pg // per_page + 1, block=bi + 1))
            e['ref']['closeup'] = dict(page=pg // per_page + 1, block=bi + 1)
        pth = out_dir / f'transitions-{pg // per_page + 1:02d}.jpg'
        sheet.save(pth, quality=85)
        paths.append(pth)
    return paths, index


# ---------------------------------------------------------------- main measurement

# ---------------------------------------------------------------- main measurement

def measure_timeline(vid, meta, v1_shots, words, bed, sheets_dir, work, log, font, ts, loud=None, source=None,
                     depth='standard', threads=2, max_s=None):
    """Measure one video. v1_shots: dissect's shots [{n, start, end}] (the boundaries); words: [[start, end, word,
    prob]]; bed: dissect's sound-bed dict. Writes sheets/timeline-NN.jpg and sheets/transitions-NN.jpg and returns the
    timeline dict (schema: timeline v2.1)."""
    cv2.setNumThreads(max(1, threads))
    notes = []
    rate = probe_rate(vid)
    if rate is None:
        raise Fail(7, 'could not read the frame rate')
    fps = float(rate)
    W, H = int(meta['width']), int(meta['height'])
    dur = float(meta['duration'])
    if max_s and dur > max_s + 1:
        notes.append(f'timeline measured over the first {max_s:.0f} s of {dur:.0f} s (--timeline-minutes)')
        dur = float(max_s)
    log(f'timeline: {W}x{H} {fps:.3f} fps, measuring {dur:.2f} s')
    words = [w for w in words or [] if len(w) >= 4 and w[3] >= 0.4]

    nfr_est = int(math.floor(dur * fps + 1e-6))
    shots = []
    for s in v1_shots:
        if s['start'] >= dur - 0.5 / fps:
            continue
        shots.append(dict(n=s['n'], v1_start=s['start'], v1_end=s['end'], sf=int(round(s['start'] * fps)),
                          ef=int(round(min(s['end'], dur) * fps))))
    shots = [s for s in shots if s['ef'] > s['sf']] or [dict(n=1, v1_start=0.0, v1_end=dur, sf=0, ef=nfr_est)]
    shots[-1]['ef'] = max(shots[-1]['ef'], nfr_est)
    for j, s in enumerate(shots):
        s['j'] = j
    bounds_f = [s['sf'] for s in shots[1:]]

    aw, ah = fit_long(W, H, AN_LONG)
    tw, th = fit_long(W, H, TR_LONG)
    kflow = max(1, int(round(fps / 12.5)))
    flow_of = {}
    for s in shots:
        step = 1 if (s['ef'] - s['sf']) < 1.2 * fps else kflow
        for i in range(s['sf'], s['ef'], step):
            flow_of[i] = s['j']
    Wwin = int(min(30, max(8, round(0.5 * fps))))
    win_set, wins = set(), {}
    allb = [0] + bounds_f + [nfr_est]
    for k, b in enumerate(bounds_f, start=1):
        wl = max(2, min(Wwin, (allb[k] - allb[k - 1]) // 2 + 2))
        wr = max(2, min(Wwin, (allb[k + 1] - allb[k]) // 2 + 2))
        wins[b] = (wl, wr)
        win_set.update(range(max(0, b - wl), b + wr))
    win_set.update(range(0, min(nfr_est, Wwin)))
    win_set.update(range(max(0, nfr_est - Wwin), nfr_est))

    t_a = time.time()
    A = pass_a(vid, rate, dur, aw, ah, flow_of, win_set, tw, th, threads, bounds_f, log)
    nfr = A['n']
    if nfr < 2:
        raise Fail(7, 'could not decode frames')
    shots[-1]['ef'] = nfr
    log(f'pass A took {time.time() - t_a:.1f} s')

    bound_info = []
    for k, b in enumerate(bounds_f, start=1):
        wl, wr = wins[b]
        wl, wr = min(wl, b), min(wr, nfr - b)
        info = analyse_boundary(b, wl, wr, A, fps, tw, th)
        bound_info.append(dict(n=shots[k]['n'], b=b, info=info))
        if info['type'] == 'unknown' and 'no visible cut' in info.get('why', ''):
            notes.append(f"S{shots[k]['n']} (boundary {shots[k]['v1_start']:.3f} s, f{b}): no visible cut; the "
                         f"picture keeps moving across it (one continuous shot split by the cut detector?)")
        cf = info.get('cut_frame', b)
        if info['type'] == 'cut' and cf != b and info.get('spike', 0) >= LIM['cut_spike'] and \
                shots[k - 1]['sf'] < cf < shots[k]['ef'] - 1:
            notes.append(f"S{shots[k]['n']} start moved from {shots[k]['v1_start']:.3f} s (f{b}) to "
                         f"{cf / fps:.3f} s (f{cf}): the frame difference peaks there")
            shots[k]['sf'] = cf
            shots[k - 1]['ef'] = cf
    for s in shots:
        s['t0'], s['t1'] = s['sf'] / fps, s['ef'] / fps
    y = A['y']
    first = dict(type='start', frames=0, confidence=1.0)
    n3 = max(4, int(3 * fps))
    if len(y) > 3 and y[0] < 0.06:
        top = float(np.percentile(y[:min(len(y), n3, shots[0]['ef'])], 90))
        up = np.where(y[:min(len(y), n3)] >= y[0] + 0.9 * (top - y[0]))[0]
        if len(up) and up[0] >= 2 and (top - y[0] > 0.08 or (y[0] < 0.3 * top and top - y[0] > 0.005)):
            first.update(frames=int(up[0]), fade='from_black')
    last_out = None
    if len(y) > 6 and y[-1] < 0.06:
        tail = y[-min(len(y), n3, nfr - shots[-1]['sf']):]
        lev = float(np.percentile(tail, 90))
        above = np.where(tail >= y[-1] + 0.9 * (lev - y[-1]))[0]
        if len(above) and len(tail) - 1 - above[-1] >= 2 and (lev - y[-1] > 0.08 or (y[-1] < 0.3 * lev and lev - y[-1] > 0.005)):
            last_out = dict(type='fade_to_black', frames=int(len(tail) - 1 - above[-1]), confidence=0.7)

    d1a, dp = A['d1'], A['d1p']
    floor = max(0.004, float(np.percentile(dp[1:], 10)) * 1.5) if len(dp) > 10 else 0.004
    for s in shots:
        s['cadence'] = cadence_of(d1a[s['sf'] + 1:s['ef']], fps, 0.0012, dp[s['sf'] + 1:s['ef']], floor)
    edges_ = set()
    for s in shots[1:]:
        edges_.update(range(s['sf'] - 3, s['sf'] + 4))
    internal = {s['j']: [] for s in shots}
    for c in A['cands']:
        if c['k'] in edges_:
            continue
        ic = classify_change(c, A, fps, aw, ah, nfr)
        if not ic:
            continue
        j = next((s['j'] for s in shots if s['sf'] <= ic['frame'] < s['ef']), None)
        if j is not None and not any(abs(o['frame'] - ic['frame']) <= 3 for o in internal[j]):
            internal[j].append(ic)
    for s in shots:
        cf = []
        for ic in internal[s['j']]:
            cf.append(ic['frame'])
            if ic.get('frames'):
                cf.append(ic['frame'] + ic['frames'])
        s['cut_frames'] = sorted(set(cf))
        for fpull in focus_pulls(s, A, fps, s['cut_frames'], aw, ah):
            if not any(abs(o['frame'] - fpull['frame']) <= 3 and o['type'] == 'focus_pull' for o in internal[s['j']]):
                internal[s['j']].append(fpull)
    nhid = sum(1 for v in internal.values() for ic in v if ic['type'] in ('cut', 'dip_to_dark', 'reframe'))
    if nhid:
        notes.append(f"{nhid} hidden cuts / dips / reframes inside shots (luma-normalised frame difference): "
                     + ', '.join(f"S{shots[j]['n']} {ic['type']} at {ic['t']:.3f} s (f{ic['frame']})"
                                 for j, v in internal.items() for ic in v if ic['type'] in ('cut', 'dip_to_dark', 'reframe'))[:600]
                     + '; listed in shots[].internal_changes')

    # colour pass: brightness, palette, text, timeline sheet tiles
    bw, bh = (W, H) if max(W, H) <= 1280 else fit_long(W, H, 1280)
    lay = sheet_layout(W, H)
    cap_tiles = {'quick': 48, 'standard': 96, 'deep': 192}.get(depth, 96)
    sheet_step = max(0.5 if dur <= 60 else 1.0 if dur <= 180 else 2.0, math.ceil(dur / cap_tiles / B_STEP) * B_STEP)
    ocr_step = (0.25 if dur <= 30 else 0.5) * (2 if depth == 'quick' else 1)
    if dur > 600:
        ocr_step = max(ocr_step, 1.0)
    ocr = OCR()
    if not ocr.ok:
        notes.append(f'text not measured: no OCR engine (Apple Vision on macOS, or install tesseract) '
                     f'({getattr(ocr, "err", "?")})')
    elif ocr.engine == 'tesseract':
        notes.append('text read with tesseract (less accurate than Apple Vision: upright boxes only, misses stylised type)')
    t_b = time.time()
    B = pass_b(vid, rate, dur, bw, bh, threads, sheet_step, ocr_step, (lay['tw'], lay['th']), ocr, None, log)
    log(f'pass B took {time.time() - t_b:.1f} s')

    sh_vals = [v[0] for v in A['sharp'].values() if v[0] is not None and v[0] > 0]
    lo_s, hi_s = (np.percentile(np.log(sh_vals), [5, 95]) if len(sh_vals) >= 5 else (None, None))
    texts = merge_text(B['obs'], ocr_step, bw, bh) if ocr.ok else []
    tB = B['t']
    out_shots = []
    for s in shots:
        dseg = d1a[s['sf'] + 1:s['ef']]
        cam, elem, why = motion_for(s, A['steps'], fps, aw, ah, dseg, s['cut_frames'])
        if cam is None:
            notes.append(f"S{s['n']}: camera not measured ({why})")
        frames_in = [i for i in range(s['sf'], s['ef']) if i in A['sharp'] and A['sharp'][i][0]]
        blur = None
        if frames_in and lo_s is not None and hi_s - lo_s > 1e-6:
            v = float(np.median(np.log([A['sharp'][i][0] for i in frames_in])))
            blur = r(clip01((hi_s - v) / (hi_s - lo_s)), 2)
        a_ = s['sf'] + A['lagk']
        me = A['near'][a_:s['ef']] if s['ef'] > a_ else A['near'][s['sf']:s['ef']]
        sel = np.where((tB >= s['t0'] - 1e-6) & (tB < s['t1'] - 1e-6))[0]
        if not len(sel):
            sel = np.array([int(np.argmin(np.abs(tB - (s['t0'] + s['t1']) / 2)))])
        luma = dict(mean=r(B['lum_m'][sel].mean()), p95=r(B['lum_p'][sel].mean()))
        pick = sel if len(sel) <= 12 else sel[np.linspace(0, len(sel) - 1, 12).round().astype(int)]
        pal = palette_of(np.concatenate([B['thumbs'][i].reshape(-1, 3) for i in pick]))
        pick2 = sel if len(sel) <= 24 else sel[np.linspace(0, len(sel) - 1, 24).round().astype(int)]
        hpx = [B['hl'][i][0] for i in pick2 if len(B['hl'][i][0])]
        hil = highlights_of(np.concatenate(hpx)) if hpx else None
        apx = [B['hl'][i][2] for i in pick2 if len(B['hl'][i][2])]
        acc_share = float(np.mean([B['hl'][i][3] for i in pick2])) if len(pick2) else 0.0
        acc = accents_of(np.concatenate(apx), acc_share) if apx and acc_share >= 0.0005 else None
        foc = focus_for(s, A, ah, s['cut_frames'], fps)
        st = []
        for it in texts:
            if it['first'] < s['t1'] - 1e-6 and it['last'] >= s['t0'] - 1e-6:
                e = dict(it)
                e['first'] = r(max(it['first'], s['t0']), 2)
                e['last'] = r(min(it['last'], s['t1']), 2)
                st.append(e)
        if len(st) > 40:
            st.sort(key=lambda e: (-e['line_h_pct'], e['first']))
            notes.append(f"S{s['n']}: {len(st)} text items read; the 40 largest are kept")
            st = sorted(st[:40], key=lambda e: (e['first'], e['box'][1]))
        if s['j'] == 0:
            tin = first
        else:
            bi = next(x for x in bound_info if x['n'] == s['n'])['info']
            tin = dict(type=bi['type'], frames=int(bi.get('frames') or 0), confidence=bi.get('confidence'))
            for k_ in ('direction', 'axis'):
                if bi.get(k_):
                    tin[k_] = bi[k_]
        rec = dict(n=s['n'], start=r(s['t0']), end=r(s['t1']), dur=r(s['t1'] - s['t0']), start_frame=s['sf'],
                   end_frame=s['ef'], transition_in=tin, camera=cam,
                   motion_energy=r(float(me.mean()) if len(me) else 0.0), blur=blur, luma=luma, palette=pal, text=st)
        ics = sorted(internal[s['j']], key=lambda x: x['frame'])
        for x in (cam or {}).get('reframes') or []:
            if not any(abs(o['frame'] - x['frame']) <= 3 for o in ics):
                ics.append(dict(t=x['t'], frame=x['frame'], type='reframe', frames=0, confidence=0.6, scale=x['scale'],
                                dx_pct=x['dx_pct'], dy_pct=x['dy_pct'], roll_deg=x['roll_deg'], what=f"jump x{x['scale']:.2f}"))
        ics.sort(key=lambda x: x['frame'])
        for x in ics:
            x.pop('spike', None)
        rec['internal_changes'] = ics
        if hil:
            rec['highlights'] = hil
        if acc:
            rec['accents'] = acc
        if foc:
            rec['focus'] = foc
        if elem:
            rec['element_motion'] = elem
        if s.get('cadence'):
            rec['cadence'] = s['cadence']
        if abs(s['sf'] / fps - s['v1_start']) > 0.5 / fps and s['j'] > 0:
            rec['v1_start'] = s['v1_start']
        out_shots.append(rec)
    if last_out:
        out_shots[-1]['transition_out'] = last_out

    cads = [s['cadence'] for s in shots if s.get('cadence') and s['cadence'].get('hold_share') is not None]
    cad = None
    if cads:
        mv = sum(c['moving_frames'] for c in cads)
        hs = sum(c['hold_share'] * c['moving_frames'] for c in cads) / max(1, mv)
        ufps = fps * (1 - hs)
        label = 'ones' if hs < 0.12 else 'twos' if 0.42 <= hs <= 0.58 else 'threes' if 0.62 <= hs <= 0.72 else f'~{ufps:.0f} fps'
        cf_ = sum((c.get('confidence') or 0) * c['moving_frames'] for c in cads) / max(1, mv)
        cad = dict(unique_fps=r(ufps, 1), hold_share=r(hs, 2), label=label, moving_frames=int(mv),
                   confidence=r(min(cf_, clip01((mv - 12) / 60.0)), 2), optional=bool(mv < 30 or cf_ < 0.4))

    fontf = font(max(14, lay['tw'] // 14))
    labels = []
    for t, nk, im in B['tiles']:
        j = shot_index(shots, t + 1e-6)
        labels.append((f"S{shots[j]['n']} {ts(t)} f{nk}", im))
    sheet_paths = draw_sheets(labels, lay, sheets_dir, 'timeline', fontf)
    log(f'{len(sheet_paths)} timeline sheet pages at {sheet_step} s steps')
    brightness = dict(step_s=B_STEP, mean=[r(v) for v in B['lum_m']], p95=[r(v) for v in B['lum_p']])
    del B

    audio = dict(has_audio=False, tempo=None, key=None, loudness=None, energy=None, sections=[], cues=[])
    if meta.get('has_audio'):
        t_au = time.time()
        bounds_t = [s['t0'] for s in shots[1:]]
        try:
            audio_, anotes, _ = analyse_audio(vid, work, dur, fps, [dict(n=s['n'], t0=s['t0'], t1=s['t1']) for s in shots],
                                              bounds_t, words, bed, log, None)
        except ImportError as e:
            audio_, anotes = None, [f'sound grid not measured: {e} (run with uv so librosa installs)']
        notes += anotes
        audio['has_audio'] = True
        if audio_:
            audio = audio_
        if loud:
            audio['loudness'] = dict(integrated_lufs=loud.get('integrated_lufs'), true_peak_dbfs=loud.get('true_peak_dbfs'),
                                     range_lu=loud.get('range_lu'))
        log(f'audio took {time.time() - t_au:.1f} s')
    else:
        notes.append('no audio track')

    tr_paths, tr_index = closeup_sheets(vid, rate, nfr, W, H, out_shots, audio.get('cues'), sheets_dir,
                                        font(15 if lay['tw'] < 300 else 16), font(16), threads,
                                        cap={'quick': 8, 'standard': 12, 'deep': 24}.get(depth, 12))
    log(f'{len(tr_paths)} transition close-up pages ({len(tr_index)} events)')
    if ocr.ok:
        notes.append('cap_h_px_1080 is estimated from the text line height (perpendicular to the baseline with Apple '
                     'Vision): median error about 8-11 %; checked against the line width (size_from = width)')

    return dict(
        version=TIMELINE_VERSION, source=source, measured_at=dt.datetime.now().isoformat(timespec='seconds'),
        video=dict(width=W, height=H, fps=r(fps, 3), duration_s=r(dur), aspect=meta.get('aspect'), frames=nfr,
                   cadence=cad),
        shots=out_shots,
        brightness=brightness,
        audio=audio,
        sheets=dict(timeline=[p.name for p in sheet_paths], transitions=[p.name for p in tr_paths],
                    transitions_index=tr_index, timeline_step_s=sheet_step),
        notes=notes,
        method=dict(
            version=f'dissect timeline {VERSION}',
            frames='start_frame/end_frame at video.fps, end exclusive; shot boundaries are the cut detector\'s cuts '
                   'snapped to the frame where the picture difference peaks; cuts it missed are not split into new '
                   'shots, so shot numbers match facts.md: see shots[].internal_changes',
            camera=f'corners + pyramidal Lucas-Kanade + RANSAC similarity on a {aw}x{ah} grey copy; pan_x/pan_y = motion '
                   'of the frame centre in % of width/height per s (+x = content moves right, +y = content moves down; '
                   'pan_left = content moves right); zoom_per_s > 1 = content grows; speed = 0.1 s bins normalised '
                   'to the shot peak',
            blur='edge sharpness, 0 = sharpest 5 % of this film, 1 = blurriest 5 %',
            motion_energy='share of the frame changing per 0.1 s, mean over the shot, 0-1',
            transitions='native-fps frames around each boundary: luma dip/spike (fade/flash), directional blur + fast '
                        'shift (whip), scale flow (zoom), blend fit (dissolve), moving edge (wipe); frames = length',
            text=f'{ocr.engine or "no OCR"} every {ocr_step} s on the {bw}x{bh} frame; repeats merged into items',
            palette='k-means (k=5) in CIELAB per shot', brightness='Rec.709 luma 0-1 every 0.25 s',
            cadence='share of held frames inside moving stretches: ones = every frame new, twos = 0.5',
            tempo='librosa beat tracker on the percussive part; confidence from onset autocorrelation and tempo '
                  'stability; downbeat = 4/4 phase with the strongest accents',
            key='Krumhansl-Kessler profiles', sections='novelty on 0.5 s blocks; labels from relative energy',
            cues='onsets and rising events; near_cut = within 2 frames of a boundary; on_beat = within 60 ms of a beat'),
    )
