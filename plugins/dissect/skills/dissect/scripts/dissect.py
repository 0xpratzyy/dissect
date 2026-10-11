#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "numpy", "pillow", "faster-whisper>=1.0", "praat-parselmouth",
#   "opencv-python-headless", "librosa", "soundfile", "scipy",
#   "pyobjc-framework-Vision; sys_platform == 'darwin'",
#   "pyobjc-framework-Quartz; sys_platform == 'darwin'",
# ]
# ///
"""dissect.py - measure a video so it can be turned into a reusable blueprint.

Give it a link (YouTube, Shorts, TikTok, Instagram, X/Twitter, Facebook, Vimeo, Reddit, ... anything
yt-dlp supports) or a local video file. It writes <out>/<date>-<platform>-<slug>/ containing:

  facts.md            every measurement in readable form (start here)
  measured.md         the measured blueprint sections (shots in frames, camera, transitions, type, palette,
                      brightness, sound grid), ready to paste into the blueprint template
  sheets/*.jpg        labelled frame sheets: hook (first 3 s at 0.25 s), shots (one frame per shot), timeline (the
                      whole film at fixed steps), transitions (every frame around each cut and change), plus
                      most-replayed and cover
  data/timeline.json  the frame-level timeline (v2.1): shots with in/out frames, transitions, camera moves, text
                      events, palette, brightness, beat grid, cues, loudness
  data/analysis.json  the v1 measurements, machine-readable
  data/transcript.*   the words with timings (.json) and as timestamped lines (.txt)
  data/info.json      trimmed platform metadata; data/comments.txt top comments (YouTube)

The downloaded video and scratch files are deleted at the end unless --keep-video.
Run it with uv so the Python dependencies install themselves:  uv run dissect.py <url-or-file> [options]

Two more commands:
  uv run dissect.py check <your-render.mp4> --ref <dissection folder>   score a rebuild against the reference
  uv run dissect.py assemble <dissection folder>                        build the final blueprint from the draft

Exit codes: 0 ok, 2 download failed, 3 login needed (rerun with cookies, ask the user first),
4 no video in that post, 5 ffmpeg/ffprobe missing.
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

VERSION = '2.0.0'
DEPTHS = {  # frames on the shot sheets, dense hook window, whisper model
    'quick':    dict(max_frames=36,  hook_s=3.0, hook_fps=4, model='base'),
    'standard': dict(max_frames=72,  hook_s=3.0, hook_fps=4, model='small'),
    'deep':     dict(max_frames=160, hook_s=5.0, hook_fps=6, model='medium'),
}
MIN_SHOT = 0.25   # cuts closer together than this are one transition
GAP = 0.25        # silence between words that counts as a pause (s)
HOP = 0.05        # loudness frame (s)

STOP = set('''a an the and or but of to in on at for with is are was were be been being it its this that these those
i me my we our you your he she they them their his her as by from so if then than not no yes do does did have has had
just very really there here what which who whom how why when where all any some can could would should will shall
about into over also too only own same such more most other each both few up down out off again once'''.split())
NUMWORDS = set('''zero one two three four five six seven eight nine ten eleven twelve thirteen fifteen twenty thirty
forty fifty sixty seventy eighty ninety hundred thousand million billion trillion lakh crore percent half double triple'''.split())
FILLERS = {'um', 'uh', 'erm', 'umm', 'uhh', 'hmm', 'mm', 'uhm'}
YOU = {'you', 'your', "you're", 'youre', 'yours', 'yourself', "you've", "you'll", "you'd"}

NOT_VIDEO = re.compile(r"No video formats found|There's no video|does not contain (a )?video|No media found|"
                       r"no video could be found|This post does not have", re.I)
LOGIN = re.compile(r"log ?in|sign ?in|cookies|private|rate.?limit|authenticat|not a bot|age.?restrict|"
                   r"members.?only|account|confirm your age", re.I)


# ---------------------------------------------------------------- small helpers

def log(msg):
    print(f'[dissect] {msg}', file=sys.stderr, flush=True)


def run(cmd, timeout=None):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired as e:
        dec = lambda b: b.decode('utf-8', 'replace') if isinstance(b, bytes) else (b or '')
        return subprocess.CompletedProcess(cmd, 124, dec(e.stdout), dec(e.stderr) + '\nERROR: timed out')
    except FileNotFoundError as e:
        return subprocess.CompletedProcess(cmd, 127, '', f'ERROR: {e}')


def ts(s):
    """75.25 -> '1:15.3'"""
    tenths = int(round(max(0.0, float(s)) * 10))
    m, r = divmod(tenths, 600)
    return f'{m}:{r / 10:04.1f}'


def spark(vals):
    bars = '▁▂▃▄▅▆▇█'
    vals = [float(v) for v in vals]
    hi = max(vals) if vals else 0
    return ''.join(bars[min(7, int(v / hi * 7.999))] if hi > 0 else bars[0] for v in vals)


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def r1(x, n=1):
    return None if x is None else round(float(x), n)


def slugify(s, n=48):
    s = re.sub(r'[^a-z0-9]+', '-', (s or '').lower()).strip('-')
    return s[:n].rstrip('-')


def md_table(head, rows):
    esc = lambda c: str('' if c is None else c).replace('|', '\\|').replace('\n', ' ')
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join(esc(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def need(tool):
    if shutil.which(tool):
        return
    fix = {'darwin': 'brew install ffmpeg', 'win32': 'winget install Gyan.FFmpeg'}.get(
        sys.platform, 'sudo apt install ffmpeg   (or your distro\'s package manager)')
    print(json.dumps({'ok': False, 'error': f'{tool} not found on PATH', 'fix': fix}))
    sys.exit(5)


# ---------------------------------------------------------------- download

def ytdlp_runners():
    """Ways to run yt-dlp, best first: $DISSECT_YTDLP, yt-dlp on PATH, an importable yt_dlp, then the newest
    release through uvx (also the fallback when an installed copy is too old for a site)."""
    out, seen = [], set()

    def add(cmd):
        if ' '.join(cmd) not in seen:
            seen.add(' '.join(cmd))
            out.append(cmd)
    if os.environ.get('DISSECT_YTDLP'):
        add(shlex.split(os.environ['DISSECT_YTDLP']))
    if shutil.which('yt-dlp'):
        add([shutil.which('yt-dlp')])
    try:
        import yt_dlp  # noqa: F401
        add([sys.executable, '-m', 'yt_dlp'])
    except Exception:
        pass
    if shutil.which('uvx'):
        add([shutil.which('uvx'), '--from', 'yt-dlp[default,curl-cffi]@latest', 'yt-dlp'])
    return out


def err_line(text):
    lines = [l for l in text.splitlines() if 'ERROR' in l] or text.strip().splitlines()[-1:]
    return (lines[-1] if lines else 'no output').strip()[:400]


def find_video(work):
    exts = ('.mp4', '.mkv', '.webm', '.mov', '.m4v', '.flv', '.avi', '.ts', '.3gp')
    vids = [p for p in work.glob('video.*') if p.suffix.lower() in exts and p.stat().st_size > 0]
    return max(vids, key=lambda p: p.stat().st_size) if vids else None


def cookie_args(args):
    out = []
    if args.cookies:
        out += ['--cookies', str(Path(args.cookies).expanduser())]
    if args.cookies_from_browser:
        out += ['--cookies-from-browser', args.cookies_from_browser]
    return out


def download(url, work, args):
    base = ['--no-playlist', '--playlist-items', '1', '--no-write-playlist-metafiles', '--no-progress', '--no-mtime',
            '-S', 'res:720,ext:mp4:m4a', '-f', 'bv*+ba/b', '--merge-output-format', 'mp4',
            '--write-info-json', '--write-thumbnail', '--convert-thumbnails', 'jpg',
            '-o', str(work / 'video.%(ext)s')]
    if args.max_minutes:
        base += ['--download-sections', f'*0-{int(args.max_minutes * 60)}']
    runners = ytdlp_runners()
    if not runners:
        return None, 'failed', 'yt-dlp not found (install it, or install uv so the newest yt-dlp can be fetched)', None
    errors = []
    for runner in runners:
        for extra in ([], ['--impersonate', 'chrome']):
            for f in work.glob('video.*'):
                f.unlink()
            log(f'downloading with {Path(runner[0]).name}{" " + " ".join(extra) if extra else ""} ...')
            r = run(runner + base + cookie_args(args) + extra + [url], timeout=3600)
            vid = find_video(work)
            if r.returncode == 0 and vid:
                return vid, 'ok', '', runner
            e = err_line(r.stderr + '\n' + r.stdout)
            errors.append(e)
            log(f'  failed: {e}')
            if NOT_VIDEO.search(e):
                return None, 'not_video', e, None
            if not re.search(r'403|impersonat|cloudflare|blocked|forbidden|captcha', e, re.I):
                break  # browser impersonation would not help this error
    detail = errors[-1] if errors else 'unknown error'
    return None, ('login' if any(LOGIN.search(e) for e in errors) else 'failed'), detail, None


def fetch_comments(runner, url, work, args):
    cmd = runner + ['--skip-download', '--no-playlist', '--playlist-items', '1', '--no-write-playlist-metafiles',
                    '--no-progress', '--write-comments', '--write-info-json',
                    '--extractor-args', 'youtube:comment_sort=top;max_comments=60,60,0,0',
                    '-o', str(work / 'comments.%(ext)s')] + cookie_args(args) + [url]
    run(cmd, timeout=180)
    f = work / 'comments.info.json'
    if not f.exists():
        return []
    cs = json.loads(f.read_text(encoding='utf-8')).get('comments') or []
    cs = [c for c in cs if c.get('parent', 'root') == 'root' and c.get('text')]
    cs.sort(key=lambda c: -(c.get('like_count') or 0))
    # text and likes only: no names or handles of commenters
    return [dict(likes=c.get('like_count') or 0, text=re.sub(r'\s+', ' ', c['text']).strip()[:300],
                 by_creator=bool(c.get('author_is_uploader')), pinned=bool(c.get('is_pinned'))) for c in cs[:40]]


# ---------------------------------------------------------------- video

def aspect_label(w, h):
    r = w / h if h else 1
    for name, v in (('9:16', 9 / 16), ('3:4', 3 / 4), ('4:5', 4 / 5), ('1:1', 1), ('4:3', 4 / 3), ('16:9', 16 / 9), ('21:9', 21 / 9)):
        if abs(r - v) / v < 0.03:
            return name
    g = math.gcd(w, h) or 1
    return f'{w // g}:{h // g}'


def probe(path):
    r = run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', str(path)])
    j = json.loads(r.stdout or '{}')
    v = next((s for s in j.get('streams', []) if s.get('codec_type') == 'video' and
              not (s.get('disposition') or {}).get('attached_pic')), None)
    a = next((s for s in j.get('streams', []) if s.get('codec_type') == 'audio'), None)
    if not v:
        return None
    w, h = int(v.get('width') or 0), int(v.get('height') or 0)
    sar = v.get('sample_aspect_ratio') or '1:1'
    try:
        n, d = (int(x) for x in sar.split(':'))
        if n and d and n != d:
            w = int(round(w * n / d))
    except ValueError:
        pass
    rot = 0
    for sd in v.get('side_data_list') or []:
        if 'rotation' in sd:
            rot = int(float(sd['rotation']))
    rot = int((v.get('tags') or {}).get('rotate', rot))
    if abs(rot) % 180 == 90:
        w, h = h, w

    def rate(x):
        try:
            n, d = x.split('/')
            return float(n) / float(d) if float(d) else 0.0
        except Exception:
            return 0.0
    fps = rate(v.get('avg_frame_rate', '0/0')) or rate(v.get('r_frame_rate', '0/0'))
    dur = float((j.get('format') or {}).get('duration') or v.get('duration') or 0)
    return dict(width=w, height=h, aspect=aspect_label(w, h), fps=round(fps, 2), duration=round(dur, 3),
                video_codec=v.get('codec_name'), audio_codec=a.get('codec_name') if a else None, has_audio=bool(a))


def scene_pass(video):
    """Per-frame scene-change score (0..1) from ffmpeg, on a small copy of the picture."""
    vf = ("scale=w=192:h=192:force_original_aspect_ratio=decrease,"
          "select='gte(scene,0)',metadata=print:key=lavfi.scene_score")
    r = run(['ffmpeg', '-hide_banner', '-nostats', '-threads', '4', '-i', str(video), '-map', '0:v:0', '-an',
             '-vf', vf, '-f', 'null', '-'])
    times, scores, t = [], [], None
    for line in r.stderr.splitlines():
        m = re.search(r'pts_time:([0-9.]+)', line)
        if m:
            t = float(m.group(1))
            continue
        m = re.search(r'lavfi\.scene_score=([0-9.]+)', line)
        if m and t is not None:
            times.append(t)
            scores.append(float(m.group(1)))
            t = None
    return np.array(times), np.array(scores)


def find_cuts(times, scores, dur, thr):
    cuts = []
    for t, s in zip(times, scores):
        if s >= thr and 0.05 < t < dur - 0.05 and (not cuts or t - cuts[-1] >= MIN_SHOT):
            cuts.append(round(float(t), 3))
    return cuts


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def aligned_similarity(a, b, reach=8):
    """Best correlation between two frames over small shifts: camera moves line back up, real cuts don't."""
    s = a.shape[0]
    return max(ncc(a[max(0, dy):s + min(0, dy), max(0, dx):s + min(0, dx)],
                   b[max(0, -dy):s + min(0, -dy), max(0, -dx):s + min(0, -dx)])
               for dy in range(-reach, reach + 1, 2) for dx in range(-reach, reach + 1, 2))


def change_pass(video, fps=10, size=96, block=12, thr=0.035):
    """A second look at the picture, 10 times a second on a 96x96 grey copy: the share of the frame's 12x12 blocks
    that changed in 0.1 s ('near') and in 0.5 s ('far'). Where nearly everything changed at once, also how well the
    two frames line up after a small shift. Streams frames so long videos stay light on memory."""
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(video), '-map', '0:v:0', '-vf',
                             f'fps={fps},scale={size}:{size},format=gray', '-f', 'rawvideo', '-'],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n, k = size * size, size // block
    hist, near, far, align = [], [], [], {}

    def frac(a, b):
        return float((np.abs(a - b).reshape(k, block, k, block).mean(axis=(1, 3)) > thr).mean())
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        f = np.frombuffer(buf, np.uint8).reshape(size, size).astype(np.float32) / 255
        near.append(frac(f, hist[-1]) if hist else 0.0)
        far.append(frac(f, hist[-5]) if len(hist) >= 5 else 0.0)
        if near[-1] >= 0.9:
            align[len(near) - 1] = aligned_similarity(hist[-1], f)
        hist = (hist + [f])[-5:]
    proc.wait()
    return np.array(near), np.array(far), fps, align


def picture_changes(near, far, fps, align, cuts, dur):
    """Instant changes the scene score missed, split into likely cuts (punch-ins, glitch transitions: a lone spike
    that no small shift lines back up) and uncertain ones (camera shake, smoke, flashes, fast motion: they come in
    bursts or line up after a shift). Plus step changes inside shots: a still picture turning into another still
    picture (a caption or graphic popping on, a new card, a dissolve). Continuous motion is not a step change."""
    t = np.arange(len(near)) / fps

    def clear_of(x, marks, d):
        return not len(marks) or min(abs(m - x) for m in marks) > d
    extra, unsure = [], []
    for i in np.where(near >= 0.9)[0]:
        x = round(float(t[i] - 0.05), 2)
        if not (0.1 < x < dur - 0.1) or not clear_of(x, cuts + extra + unsure, 0.3):
            continue
        burst = int((near[max(0, i - 3):i + 4] >= 0.9).sum())
        (extra if align.get(int(i), 1.0) < 0.6 and burst <= 2 else unsure).append(x)
    allcuts = sorted(cuts + extra)
    steps = []
    for i in range(10, len(far) - 6):
        if far[i] < 0.15 or far[i] < far[i - 1] or far[i] < far[i + 1]:
            continue
        if near[i - 10:i - 5].mean() >= 0.08 or near[i + 1:i + 6].mean() >= 0.08:
            continue  # the picture was moving before or after: motion, not a step
        j = i - 4 + int(np.argmax(near[i - 4:i + 1]))
        x = round(float(t[j] - 0.05), 2)
        if clear_of(x, allcuts, 0.6) and (not steps or x - steps[-1] >= 1.0):
            steps.append(x)
    return extra, unsure, steps


def window_for(dur):
    return 5 if dur <= 90 else 10 if dur <= 600 else 30


def pacing(shots, cuts, steps, extra, unsure, near, fps, dur):
    durs = np.array([s['dur'] for s in shots])
    win = window_for(dur)
    nwin = max(1, math.ceil(dur / win))
    per, per_changes, motion = [0] * nwin, [0] * nwin, []
    for c in cuts:
        per[min(nwin - 1, int(c // win))] += 1
    for c in cuts + steps:
        per_changes[min(nwin - 1, int(c // win))] += 1
    t = np.arange(len(near)) / fps
    for i in range(nwin):
        m = (t >= i * win) & (t < (i + 1) * win)
        motion.append(round(float(near[m].mean()) * 100, 1) if m.any() else 0.0)
    mins = dur / 60 if dur else 1
    return dict(
        shots=len(shots), cuts=len(cuts), cuts_per_min=r1(len(cuts) / mins),
        cuts_found_by_second_pass=len(extra),
        uncertain_changes=len(unsure), uncertain_at=unsure,
        cuts_per_min_if_all_uncertain_are_cuts=r1((len(cuts) + len(unsure)) / mins),
        step_changes_inside_shots=len(steps), picture_changes_per_min=r1((len(cuts) + len(steps)) / mins),
        shot_len=dict(mean=r1(durs.mean(), 2), median=r1(np.median(durs), 2), min=r1(durs.min(), 2),
                      max=r1(durs.max(), 2), sd=r1(durs.std(), 2)),
        first_cut_at=cuts[0] if cuts else None,
        first_picture_change_at=min(cuts + steps) if cuts or steps else None,
        cuts_first_3s=sum(c < 3 for c in cuts), cuts_first_10s=sum(c < 10 for c in cuts),
        changes_first_3s=sum(c < 3 for c in cuts + steps),
        window_s=win, cuts_per_window=per, cuts_spark=spark(per), changes_per_window=per_changes,
        changes_spark=spark(per_changes), motion_per_window=motion, motion_spark=spark(motion))


# ---------------------------------------------------------------- frames and sheets

def font(size):
    for f in ('/System/Library/Fonts/Supplemental/Arial Bold.ttf', '/System/Library/Fonts/Helvetica.ttc',
              '/Library/Fonts/Arial Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf', '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf',
              'C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/segoeuib.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def layout(w, h):
    r = h / w if w else 1
    if r > 1.25:
        cols, rows, tw = 6, 3, 250      # vertical: 18 tiles a sheet
    elif r > 0.8:
        cols, rows, tw = 5, 4, 300      # square-ish: 20
    else:
        cols, rows, tw = 4, 5, 380      # landscape: 20
    th = int(round(tw * h / w / 2) * 2) if w else tw
    return dict(cols=cols, rows=rows, tw=tw, th=th)


def grab(video, t, out, tw):
    vf = f'scale=trunc(iw*sar/2)*2:ih,setsar=1,scale={tw}:-2'
    for tt in (t, max(0.0, t - 0.2), max(0.0, t - 1.0)):
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{tt:.3f}', '-i', str(video),
             '-map', '0:v:0', '-frames:v', '1', '-vf', vf, '-q:v', '3', str(out)])
        if out.exists() and out.stat().st_size > 0:
            return True
    return False


def make_sheets(video, items, lay, raw_dir, out_dir, stem):
    """items: [(label, t)] -> sheets of labelled tiles. Returns [(sheet path, first label, last label)]."""
    tw, th, cols, rows = lay['tw'], lay['th'], lay['cols'], lay['rows']
    per_sheet, gap = cols * rows, 4
    f = font(max(14, tw // 14))
    tiles = []
    for i, (label, t) in enumerate(items):
        p = raw_dir / f'{stem}_{i:04d}.jpg'
        if grab(video, t, p, tw):
            tiles.append((label, p))
    sheets = []
    for k in range(0, len(tiles), per_sheet):
        chunk = tiles[k:k + per_sheet]
        nrows = math.ceil(len(chunk) / cols)
        sheet = Image.new('RGB', (cols * tw + (cols + 1) * gap, nrows * th + (nrows + 1) * gap), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        for j, (label, p) in enumerate(chunk):
            x, y = gap + (j % cols) * (tw + gap), gap + (j // cols) * (th + gap)
            with Image.open(p) as im:
                sheet.paste(ImageOps.pad(im.convert('RGB'), (tw, th), color=(16, 16, 16)), (x, y))
            box = d.textbbox((0, 0), label, font=f)
            d.rectangle([x, y, x + box[2] - box[0] + 12, y + box[3] - box[1] + 10], fill=(0, 0, 0))
            d.text((x + 6, y + 4 - box[1]), label, font=f, fill=(255, 235, 59))
        name = out_dir / (f'{stem}.jpg' if len(tiles) <= per_sheet else f'{stem}-{k // per_sheet + 1:02d}.jpg')
        sheet.save(name, quality=82)
        sheets.append((name, chunk[0][0], chunk[-1][0]))
    return sheets


def pick_frames(shots, steps, unsure, max_frames, fps=30.0):
    """Frames for the shot sheets: the middle of every shot, just after every step change, every ~4 s through long
    shots, then just after uncertain changes if there's room. Thinned evenly when there are too many, keeping shot
    middles longest. Returns [(label, time)]."""
    def shot_of(x):
        return next((s['n'] for s in shots if s['start'] <= x < s['end']), shots[-1]['n'])
    mids = [(s['n'], s['start'] + s['dur'] / 2, '') for s in shots]
    after = [(shot_of(x + 0.35), x + 0.35, '+') for x in steps]
    maybe = [(shot_of(x + 0.25), x + 0.25, '?') for x in unsure]
    fill = []
    for s in shots:
        if s['dur'] > 6:
            k = int(s['dur'] // 4)
            for i in range(1, k + 1):
                x = s['start'] + i * s['dur'] / (k + 1)
                if all(abs(x - y[1]) > 1.5 for y in mids + after):
                    fill.append((s['n'], x, ''))

    def thin(items, n):
        if n <= 0:
            return []
        if len(items) <= n:
            return items
        return [items[i] for i in sorted(set(np.linspace(0, len(items) - 1, n).round().astype(int)))]
    keep = thin(mids, max(max_frames, 240))   # one frame per shot, always (up to 240 shots)
    for group in (after, fill, maybe):  # later groups only add frames that show something new
        new = [p for p in group if all(abs(p[1] - q[1]) > 0.5 for q in keep)]
        keep += thin(new, max_frames - len(keep))
    return [(f'S{n}{mark} {ts(x)} f{int(x * fps)}', x) for n, x, mark in sorted(keep, key=lambda p: p[1])]


# ---------------------------------------------------------------- audio

def extract_wav(video, out):
    r = run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video), '-map', '0:a:0', '-vn',
             '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(out)])
    return out if r.returncode == 0 and out.exists() and out.stat().st_size > 2000 else None


def read_wav(path):
    with wave.open(str(path), 'rb') as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def rms_series(x, sr):
    n = int(sr * HOP)
    m = len(x) // n
    if m == 0:
        return np.array([]), np.array([])
    fr = x[:m * n].reshape(m, n)
    return (np.arange(m) + 0.5) * HOP, 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)


def loudness(video):
    r = run(['ffmpeg', '-hide_banner', '-nostats', '-i', str(video), '-map', '0:a:0', '-vn',
             '-af', 'ebur128=peak=true:framelog=verbose', '-f', 'null', '-'])
    s = r.stderr[r.stderr.rfind('Summary:'):] if 'Summary:' in r.stderr else ''

    def g(p):
        m = re.search(p, s)
        return float(m.group(1)) if m else None
    return dict(integrated_lufs=g(r'I:\s+(-?[\d.]+) LUFS'), range_lu=g(r'LRA:\s+(-?[\d.]+) LU'),
                true_peak_dbfs=g(r'Peak:\s+(-?[\d.]+) dBFS'))


def transcribe(audio, model_name, lang):
    """audio: 16 kHz mono float32 samples (decoded here by ffmpeg, so whisper's own decoder is never needed)."""
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        return None, f'no transcript: faster-whisper unavailable ({e.__class__.__name__}); run with uv'
    device, ctype = 'cpu', 'int8'
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            device, ctype = 'cuda', 'float16'
    except Exception:
        pass
    log(f'transcribing with whisper "{model_name}" on {device} (the first run downloads the model) ...')
    model = WhisperModel(model_name, device=device, compute_type=ctype, cpu_threads=min(4, os.cpu_count() or 4))
    segs, info = model.transcribe(audio, language=lang, word_timestamps=True, vad_filter=False,
                                  condition_on_previous_text=False, beam_size=5)
    out = []
    for s in segs:
        if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
            continue  # music or noise that whisper tried to read as words
        words = [[round(w.start, 3), round(w.end, 3), w.word.strip(), round(w.probability, 3)]
                 for w in (s.words or []) if w.word.strip()]
        if words:
            out.append(dict(start=round(s.start, 3), end=round(s.end, 3), text=s.text.strip(), words=words))
    return dict(model=model_name, language=info.language, language_probability=round(info.language_probability, 3),
                segments=out), None


def clean(w):
    return re.sub(r"[^\w']", '', w.lower())


def speech_stats(tr, dur):
    words = [w for s in tr['segments'] for w in s['words']]
    if len(words) < 3:
        return None
    first, last = words[0][0], words[-1][1]
    span = max(0.5, last - first)
    gaps = [(words[i][1], round(words[i + 1][0] - words[i][1], 3), words[i + 1][2]) for i in range(len(words) - 1)]
    pauses = [g for g in gaps if g[1] >= GAP]
    talk = max(0.5, span - sum(g[1] for g in pauses))
    cw = [clean(w[2]) for w in words]
    text = ' '.join(s['text'] for s in tr['segments'])
    sentences = [x.strip() for x in re.split(r'(?<=[.!?…])\s+', text) if x.strip()]
    win = window_for(dur)
    nwin = max(1, math.ceil(dur / win))
    per = [0] * nwin
    for w in words:
        per[min(nwin - 1, int(w[0] // win))] += 1
    wpm_w = [round(c * 60 / win) for c in per]
    pv = np.array([g[1] for g in pauses]) if pauses else np.array([])
    return dict(
        words=len(words), first_word_at=round(first, 2), last_word_end=round(last, 2),
        speech_share_of_video=round(span / dur, 2) if dur else None,
        wpm=round(len(words) / span * 60), wpm_excluding_pauses=round(len(words) / talk * 60),
        window_s=win, wpm_per_window=wpm_w, wpm_spark=spark(wpm_w),
        pauses=dict(count=len(pauses), per_min=round(len(pauses) / span * 60, 1), median_s=r1(pct(pv, 50), 2),
                    p90_s=r1(pct(pv, 90), 2), max_s=r1(pv.max() if len(pv) else None, 2),
                    longest=[dict(at=round(t, 2), length_s=g, next_word=nxt)
                             for t, g, nxt in sorted(pauses, key=lambda p: -p[1])[:8]]),
        sentences=len(sentences), words_per_sentence=round(len(words) / max(1, len(sentences)), 1),
        questions=text.count('?'), you_per_100_words=round(100 * sum(c in YOU for c in cw) / len(words), 1),
        number_mentions=sum(bool(re.search(r'\d', w[2])) or c in NUMWORDS for w, c in zip(words, cw)),
        fillers=sum(c in FILLERS for c in cw), low_confidence_words=sum(1 for w in words if w[3] < 0.5),
        first_sentence=sentences[0] if sentences else '')


def pitch_track(wav):
    try:
        import parselmouth
    except Exception:
        return None
    p = parselmouth.Sound(str(wav)).to_pitch_ac(time_step=0.01, pitch_floor=65.0, pitch_ceiling=550.0)
    return np.array(p.xs()), np.array(p.selected_array['frequency'])


def word_mask(t, words):
    m = np.zeros(len(t), dtype=bool)
    for w in words:
        m[np.searchsorted(t, w[0]):np.searchsorted(t, w[1])] = True
    return m


def pitch_stats(track, words):
    if track is None or not words:
        return None
    t, f = track
    v = f[word_mask(t, words) & (f > 0)]
    if len(v) < 50:
        return None
    med = float(np.median(v))
    p10, p90 = pct(v, 10), pct(v, 90)
    q25, q75 = np.percentile(12 * np.log2(v / med), [25, 75])
    sd = float((q75 - q25) / 1.349)  # robust spread: octave slips in the tracker don't inflate it
    ends = {'rise': 0, 'fall': 0, 'level': 0}
    for w in words:  # sentence endings: last ~0.35 s against the 0.6 s before it
        if not re.search(r'[.!?…]$', w[2]):
            continue
        a = f[(t >= w[1] - 0.35) & (t <= w[1]) & (f > 0)]
        b = f[(t >= w[1] - 0.95) & (t < w[1] - 0.35) & (f > 0)]
        if len(a) >= 5 and len(b) >= 5:
            st = 12 * math.log2(float(np.median(a)) / float(np.median(b)))
            ends['rise' if st > 1.5 else 'fall' if st < -1.5 else 'level'] += 1
    return dict(median_hz=round(med), p10_hz=round(p10), p90_hz=round(p90),
                range_semitones=r1(12 * math.log2(p90 / p10)), variability_semitones=r1(sd),
                register='low' if med < 130 else 'medium' if med < 200 else 'high',
                expressiveness='flat' if sd < 2.0 else 'moderate' if sd < 3.5 else 'very animated',
                sentence_endings=ends)


def emphasis_words(words, rms_t, rms_db, track, top=12):
    """Words said louder and/or higher than the words around them."""
    if len(words) < 12 or not len(rms_db):
        return []
    loud, pitch = [], []
    for w in words:
        seg = rms_db[np.searchsorted(rms_t, w[0]):max(np.searchsorted(rms_t, w[1]), np.searchsorted(rms_t, w[0]) + 1)]
        loud.append(float(seg.max()) if len(seg) else np.nan)
        if track is not None:
            t, f = track
            fv = f[(t >= w[0]) & (t <= w[1]) & (f > 0)]
            pitch.append(12 * math.log2(float(np.median(fv)) / 100.0) if len(fv) >= 3 else np.nan)
        else:
            pitch.append(np.nan)
    loud, pitch = np.array(loud), np.array(pitch)

    def z(a, i, k=8):
        nb = a[max(0, i - k):i + k + 1]
        nb = nb[~np.isnan(nb)]
        if len(nb) < 5 or np.isnan(a[i]) or nb.std() < 1e-6:
            return 0.0
        return float((a[i] - nb.mean()) / nb.std())
    scored = []
    for i, w in enumerate(words):
        c = clean(w[2])
        if len(c) < 3 or c in STOP:
            continue
        zl, zp = z(loud, i), z(pitch, i)
        scored.append((zl + 0.7 * zp, w, zl, zp))
    scored.sort(key=lambda s: -s[0])
    best = sorted(scored[:top], key=lambda s: s[1][0])
    return [dict(at=round(w[0], 2), word=w[2], loud_z=round(zl, 1), pitch_z=round(zp, 1)) for _, w, zl, zp in best]


def flatness(x, sr, t_list):
    """Median spectral flatness over 50 ms frames: low = tonal (music), high = noise-like."""
    n = int(sr * HOP)
    vals = []
    for t in t_list[:400]:
        i = int((t - HOP / 2) * sr)
        fr = x[i:i + n]
        if len(fr) < n:
            continue
        p = np.abs(np.fft.rfft(fr * np.hanning(n), 1024)) ** 2 + 1e-12
        vals.append(float(np.exp(np.mean(np.log(p))) / np.mean(p)))
    return float(np.median(vals)) if vals else None


def sound_bed(x, sr, rms_t, rms_db, words, dur):
    if not len(rms_db):
        return None
    if not words:
        fl = flatness(x, sr, list(rms_t[::10]))
        return dict(speech=False, level_dbfs=r1(np.median(rms_db)),
                    texture='tonal (music-like)' if fl is not None and fl < 0.15 else 'noise-like', flatness=r1(fl, 3))
    inw = word_mask(rms_t, words)
    gm = np.zeros(len(rms_t), dtype=bool)
    for i in range(len(words) - 1):
        a, b = words[i][1] + 0.08, words[i + 1][0] - 0.08
        if b - a >= 0.2:
            gm[np.searchsorted(rms_t, a):np.searchsorted(rms_t, b)] = True
    lead, tail = rms_t < words[0][0] - 0.15, rms_t > words[-1][1] + 0.15
    speech_db = float(np.median(rms_db[inw])) if inw.any() else None
    gap_db = float(np.median(rms_db[gm])) if gm.sum() >= 6 else None
    out = dict(speech=True, voice_level_dbfs=r1(speech_db), level_in_pauses_dbfs=r1(gap_db),
               intro_level_dbfs=r1(np.median(rms_db[lead])) if lead.sum() >= 10 else None,
               outro_level_dbfs=r1(np.median(rms_db[tail])) if tail.sum() >= 10 else None)
    if gap_db is not None:
        out['bed_present'] = gap_db > -45
        out['bed_below_voice_db'] = r1(speech_db - gap_db) if speech_db is not None else None
        if out['bed_present']:
            fl = flatness(x, sr, list(rms_t[gm]))
            out['bed_texture'] = 'tonal (music-like)' if fl is not None and fl < 0.15 else 'noise-like (room/ambience)'
            out['bed_flatness'] = r1(fl, 3)
    return out


def rough_tempo(x, sr, words, dur):
    """BPM of the longest stretch without speech (>= 6 s), or of the whole track if nobody talks."""
    spans = []
    if not words:
        spans.append((0.0, dur))
    else:
        edges = [0.0] + [v for w in words for v in (w[0], w[1])] + [dur]
        spans = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]
    spans = [s for s in spans if s[1] - s[0] >= 6]
    if not spans:
        return None
    a, b = max(spans, key=lambda s: s[1] - s[0])
    mid = (a + b) / 2
    a, b = max(a, mid - 30), min(b, mid + 30)
    seg = x[int(a * sr):int(b * sr)]
    n_fft, hop = 1024, 256
    if len(seg) < n_fft * 8:
        return None
    idx = np.arange(0, len(seg) - n_fft, hop)
    frames = np.stack([seg[i:i + n_fft] for i in idx]) * np.hanning(n_fft)
    mag = np.log1p(10 * np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(0, np.diff(mag, axis=0)).sum(axis=1)
    flux -= flux.mean()
    ac = np.correlate(flux, flux, 'full')[len(flux) - 1:]
    fps = sr / hop
    lags = np.arange(1, len(ac))
    bpm = 60 * fps / lags
    ok = (bpm >= 70) & (bpm <= 180)
    if not ok.any() or ac[0] <= 0:
        return None
    best = lags[ok][np.argmax(ac[1:][ok])]
    return dict(bpm=round(60 * fps / best), confidence=r1(ac[best] / ac[0], 2), measured_from_s=r1(a), to_s=r1(b))


def cut_accents(cuts, rms_t, rms_db, dur):
    """Do cuts carry a sound accent (whoosh, hit, pop)? Compare loudness jumps at cuts with jumps elsewhere."""
    if len(cuts) < 4 or not len(rms_db):
        return None

    def jump(c):
        pre = rms_db[(rms_t >= c - 1.0) & (rms_t < c - 0.2)]
        post = rms_db[(rms_t >= c - 0.1) & (rms_t <= c + 0.3)]
        return float(post.max() - np.median(pre)) if len(pre) >= 5 and len(post) >= 3 else None
    cj = [j for j in map(jump, cuts) if j is not None]
    ca = np.array(cuts)
    base = [t for t in np.arange(1.0, dur - 0.5, 0.37) if np.min(np.abs(ca - t)) > 0.6]
    bj = [j for j in map(jump, base) if j is not None]
    if len(cj) < 4 or len(bj) < 8:
        return None
    p80 = np.percentile(bj, 80)
    share = float(np.mean(np.array(cj) > p80))
    lift = float(np.median(cj) - np.median(bj))
    return dict(cuts_checked=len(cj), median_jump_at_cuts_db=r1(np.median(cj)), median_jump_elsewhere_db=r1(np.median(bj)),
                share_of_cuts_louder_than_usual=r1(share, 2), likely_sfx_on_cuts=bool(share >= 0.45 and lift >= 2.0))


# ---------------------------------------------------------------- platform metadata

KEEP = ['id', 'title', 'description', 'uploader', 'uploader_id', 'channel', 'channel_follower_count', 'channel_is_verified',
        'upload_date', 'timestamp', 'duration', 'view_count', 'like_count', 'comment_count', 'repost_count', 'tags',
        'categories', 'chapters', 'track', 'artist', 'artists', 'album', 'alt_title', 'webpage_url', 'extractor_key',
        'width', 'height', 'fps', 'language', 'age_limit', 'availability', 'live_status', 'media_type']


def trim_info(info):
    return {k: info[k] for k in KEEP if info.get(k) not in (None, '', [], {})}


def fix_counts(info):
    """Some platforms (TikTok) report counts above ~2.1 billion as negative 32-bit numbers; undo the wrap-around."""
    for k in ('view_count', 'like_count', 'comment_count', 'repost_count', 'channel_follower_count'):
        v = info.get(k)
        if isinstance(v, int) and v < 0 < v + 2 ** 32:
            info[k] = v + 2 ** 32
    return info


def engagement(info):
    v, l, c, f = (info.get(k) for k in ('view_count', 'like_count', 'comment_count', 'channel_follower_count'))
    out = {}
    if v:
        if l is not None:
            out['likes_per_1k_views'] = round(1000 * l / v, 1)
        if c is not None:
            out['comments_per_1k_views'] = round(1000 * c / v, 2)
        if f:
            out['views_per_follower'] = round(v / f, 2)
        ud = info.get('upload_date')
        if ud and re.fullmatch(r'\d{8}', ud):
            age = (dt.date.today() - dt.date(int(ud[:4]), int(ud[4:6]), int(ud[6:]))).days
            out['age_days'] = age
            out['views_per_day'] = round(v / max(1, age))
    return out


def replay_peaks(hm, dur, full_dur=None):
    """Peaks of YouTube's most-replayed curve inside the analysed window, plus the whole video's top moment when it
    falls outside that window (e.g. with --max-minutes)."""
    every = [(h.get('start_time', 0), h.get('end_time', 0), h.get('value', 0)) for h in hm or []]
    pts = [p for p in every if p[0] < dur]
    if len(pts) < 6 or not dur:
        return None
    outside = None
    if full_dur and full_dur > dur + 1:
        top = max(every[max(1, len(every) // 20):], key=lambda p: p[2])
        if top[0] >= dur:
            outside = dict(start=r1(top[0]), end=r1(top[1]), value=r1(top[2], 2))
    vals = np.array([p[2] for p in pts])
    cand = [i for i in range(1, len(pts) - 1)
            if pts[i][0] > 0.05 * dur and vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]]
    cand.sort(key=lambda i: -vals[i])
    picked = []
    for i in cand:
        if all(abs(pts[i][0] - pts[j][0]) > max(5, 0.06 * dur) for j in picked):
            picked.append(i)
        if len(picked) == 3:
            break
    later = [i for i in range(len(pts)) if pts[i][0] > 0.1 * dur]
    lo = min(later, key=lambda i: vals[i]) if later else None
    step = max(1, len(vals) // 40)
    return dict(peaks=[dict(start=r1(pts[i][0]), end=r1(pts[i][1]), value=r1(vals[i], 2)) for i in picked],
                lowest=dict(start=r1(pts[lo][0]), end=r1(pts[lo][1]), value=r1(vals[lo], 2)) if lo is not None else None,
                curve_spark=spark(vals[::step]), top_moment_outside_window=outside)


# ---------------------------------------------------------------- report

def write_facts(path, A, tr, shot_words, sheet_list, T=None):
    m, p, pl = A['video'], A['pacing'], A['platform']
    L = [f"# Facts: {A['title']}", '',
         f"_Measured by dissect.py {VERSION} on {A['measured_at']}. Numbers are measurements; words in (brackets) are "
         "rough heuristics to check against the sheets._", '', '## Source']
    L.append(f"- Link: {A['source']}")
    L.append(f"- Platform: {A['platform_name']}" + (f" · creator: {pl.get('uploader') or pl.get('channel')}" if
                                                       pl.get('uploader') or pl.get('channel') else ''))
    if pl.get('upload_date'):
        L.append(f"- Uploaded: {pl['upload_date'][:4]}-{pl['upload_date'][4:6]}-{pl['upload_date'][6:]}")
    stats = [f"{k.replace('_count', 's')}: {pl[k]:,}" for k in ('view_count', 'like_count', 'comment_count', 'repost_count')
             if isinstance(pl.get(k), int)]
    if isinstance(pl.get('channel_follower_count'), int):
        stats.append(f"followers: {pl['channel_follower_count']:,}")
    if stats:
        L.append('- Stats at capture: ' + ' · '.join(stats))
    if A['engagement']:
        L.append('- Engagement: ' + ' · '.join(f'{k.replace("_", " ")} {v}' for k, v in A['engagement'].items()))
    if pl.get('track') or pl.get('artist'):
        L.append(f"- Music credited by the platform: {pl.get('track', '?')} by {pl.get('artist') or ', '.join(pl.get('artists') or []) or '?'}")
    if pl.get('tags'):
        L.append('- Tags: ' + ', '.join(pl['tags'][:25]))
    if pl.get('chapters'):
        L.append('- Chapters: ' + ' · '.join(f"{ts(c.get('start_time', 0))} {c.get('title', '')}" for c in pl['chapters']))
    if pl.get('description'):
        L += ['', '**Description** (first 800 characters):', '', '> ' + pl['description'][:800].replace('\n', '\n> ')]

    L += ['', '## Format',
          f"- {m['width']}×{m['height']} ({m['aspect']}), {m['fps']} fps, {ts(m['duration'])} ({m['duration']:.1f} s)"
          f"{', no audio' if not m['has_audio'] else ''}"]
    if A.get('analysed_seconds') and A['analysed_seconds'] < m['duration'] - 1:
        L.append(f"- Only the first {ts(A['analysed_seconds'])} were analysed (--max-minutes).")

    L += ['', '## Pacing',
          f"- {p['shots']} shots, {p['cuts']} cuts = {p['cuts_per_min']} cuts/min"
          + (f" ({p['cuts_found_by_second_pass']} of them instant changes such as punch-in zooms)" if p['cuts_found_by_second_pass'] else '')]
    if p['uncertain_changes']:
        L.append(f"- Plus {p['uncertain_changes']} uncertain changes: camera shake, smoke, flashes, fast motion or caption swaps "
                 f"that can look like cuts. If all were cuts: {p['cuts_per_min_if_all_uncertain_are_cuts']} cuts/min, so report "
                 f"the pace as {p['cuts_per_min']}–{p['cuts_per_min_if_all_uncertain_are_cuts']} cuts/min. They're marked \"?\" on the "
                 f"shot sheets when there was room: " + ', '.join(ts(x) for x in p['uncertain_at'][:40])
                 + (' ...' if len(p['uncertain_at']) > 40 else ''))
    L += [
          f"- {p['step_changes_inside_shots']} step changes inside shots (a still picture turning into another: text or a "
          f"graphic popping on, a new card, a dissolve) → {p['picture_changes_per_min']} picture changes/min including cuts",
          f"- Shot length: mean {p['shot_len']['mean']} s, median {p['shot_len']['median']} s, "
          f"shortest {p['shot_len']['min']} s, longest {p['shot_len']['max']} s",
          f"- First cut at {p['first_cut_at']} s · first picture change of any kind at {p['first_picture_change_at']} s · "
          f"in the first 3 s: {p['cuts_first_3s']} cuts, {p['changes_first_3s']} changes · first 10 s: {p['cuts_first_10s']} cuts",
          f"- Cuts per {p['window_s']} s: `{p['cuts_spark']}`  {p['cuts_per_window']}",
          f"- All picture changes per {p['window_s']} s: `{p['changes_spark']}`  {p['changes_per_window']}",
          f"- Motion (% of the frame changing every 0.1 s) per {p['window_s']} s: `{p['motion_spark']}`  {p['motion_per_window']}"]

    s = A.get('speech')
    L += ['', '## Speech']
    if not s:
        L.append(f"- {A.get('transcript_note') or 'No speech detected.'}")
    else:
        L += [f"- Language: {A['transcript']['language']} (p={A['transcript']['language_probability']}), whisper {A['transcript']['model']}",
              f"- {s['words']} words · {s['wpm']} wpm overall · {s['wpm_excluding_pauses']} wpm while talking",
              f"- First word at {s['first_word_at']} s · speech covers {int(100 * (s['speech_share_of_video'] or 0))}% of the video",
              f"- Words per minute per {s['window_s']} s: `{s['wpm_spark']}`  {s['wpm_per_window']}",
              f"- Pauses ≥ {GAP} s: {s['pauses']['count']} ({s['pauses']['per_min']}/min), median {s['pauses']['median_s']} s, "
              f"90th pct {s['pauses']['p90_s']} s, longest {s['pauses']['max_s']} s",
              f"- {s['sentences']} sentences, {s['words_per_sentence']} words each on average · questions: {s['questions']} · "
              f"'you' per 100 words: {s['you_per_100_words']} · number mentions: {s['number_mentions']} · fillers: {s['fillers']}",
              f"- First sentence: \"{s['first_sentence'][:200]}\""]
        if s['pauses']['longest']:
            L.append('- Longest pauses (what comes right after): ' +
                     ' · '.join(f"{ts(x['at'])} {x['length_s']} s → \"{x['next_word']}\"" for x in s['pauses']['longest']))
        if A.get('emphasis'):
            L.append('- Emphasised words (louder/higher than their neighbours): ' +
                     ' · '.join(f"{ts(e['at'])} {e['word']}" for e in A['emphasis']))
        if s['low_confidence_words'] > 0.15 * s['words']:
            L.append(f"- Note: {s['low_confidence_words']} words have low confidence (music, noise, accents or mixed languages); "
                     "check the transcript against the picture.")

    v = A.get('voice')
    if v:
        e = v['sentence_endings']
        L += ['', '## Voice',
              f"- Pitch: median {v['median_hz']} Hz ({v['register']} register), 10–90% range {v['p10_hz']}–{v['p90_hz']} Hz "
              f"= {v['range_semitones']} semitones",
              f"- Pitch movement: {v['variability_semitones']} semitones s.d. ({v['expressiveness']})",
              f"- Sentence endings: {e['fall']} fall, {e['rise']} rise, {e['level']} level"]

    snd, bed, tempo, acc = A.get('loudness') or {}, A.get('bed'), A.get('tempo'), A.get('cut_accents')
    L += ['', '## Sound']
    if not m['has_audio']:
        L.append('- No audio track.')
    else:
        L.append(f"- Loudness: {snd.get('integrated_lufs')} LUFS integrated, range {snd.get('range_lu')} LU, "
                 f"true peak {snd.get('true_peak_dbfs')} dBFS")
        if bed and bed.get('speech'):
            if bed.get('level_in_pauses_dbfs') is None:
                L.append('- Background between words: not enough clean pauses to measure.')
            elif bed.get('bed_present'):
                L.append(f"- A bed keeps playing between words: {bed['level_in_pauses_dbfs']} dBFS, about "
                         f"{bed['bed_below_voice_db']} dB under the voice ({bed.get('bed_texture')})")
            else:
                L.append(f"- Pauses are nearly silent ({bed['level_in_pauses_dbfs']} dBFS): no music bed under the voice.")
            for k, lab in (('intro_level_dbfs', 'Before the first word'), ('outro_level_dbfs', 'After the last word')):
                if bed.get(k) is not None:
                    L.append(f"- {lab}: {bed[k]} dBFS")
        elif bed:
            L.append(f"- No speech. Audio level {bed['level_dbfs']} dBFS, {bed['texture']}.")
        if tempo:
            L.append(f"- Rough tempo of the music: {tempo['bpm']} BPM (or half/double), confidence {tempo['confidence']}, "
                     f"from {ts(tempo['measured_from_s'])}–{ts(tempo['to_s'])}")
        if acc:
            L.append(f"- Sound accents on cuts: {int(100 * acc['share_of_cuts_louder_than_usual'])}% of cuts jump louder than "
                     f"usual (median +{acc['median_jump_at_cuts_db']} dB vs +{acc['median_jump_elsewhere_db']} dB elsewhere) "
                     f"→ {'(likely whooshes/hits on cuts)' if acc['likely_sfx_on_cuts'] else '(no clear SFX on cuts)'}")

    rp, com = A.get('replays'), A.get('comments') or []
    if rp or com:
        L += ['', '## Audience signals']
        if rp:
            L.append(f"- Most replayed (YouTube): " + ' · '.join(f"{ts(x['start'])}–{ts(x['end'])} ({x['value']})" for x in rp['peaks']))
            if rp.get('lowest'):
                L.append(f"- Least replayed after the opening: {ts(rp['lowest']['start'])}–{ts(rp['lowest']['end'])}")
            L.append(f"- Replay curve over the analysed part: `{rp['curve_spark']}`")
            if rp.get('top_moment_outside_window'):
                o = rp['top_moment_outside_window']
                L.append(f"- The whole video's most-replayed moment is {ts(o['start'])}–{ts(o['end'])}, outside the analysed "
                         "part (not shown on the sheets)")
        if com:
            L.append(f"- Top comments by likes ({len(com)} kept in data/comments.txt):")
            L += [f"  - [{c['likes']}]{' (creator)' if c['by_creator'] else ''} {c['text'][:200]}" for c in com[:12]]

    L += ['', '## Shots and what is said during each', '']
    rows = [[sh['n'], ts(sh['start']), sh['dur'], ', '.join([ts(x) for x in sh['changes_inside']] +
                                                         [ts(x) + '?' for x in sh['uncertain_inside']]),
             shot_words.get(sh['n'], '')[:110]] for sh in A['shots'][:250]]
    L.append(md_table(['#', 'in', 'dur s', 'changes inside', 'words spoken during the shot'], rows))
    if len(A['shots']) > 250:
        L.append(f"\n({len(A['shots']) - 250} more shots in data/analysis.json)")

    if tr and tr['segments']:
        hook_end = A['hook_window_s'] + 2
        hw = [w for s_ in tr['segments'] for w in s_['words'] if w[0] < hook_end]
        L += ['', f"## Hook, word by word (first {hook_end:.0f} s)", '', ' · '.join(f"`{w[0]:.2f}` {w[2]}" for w in hw)]
        L += ['', '## Transcript', ''] + [f"[{ts(s_['start'])}] {s_['text']}" for s_ in tr['segments']]

    if T:
        S_ = T['shots']
        kinds = {}
        for s_ in S_[1:]:
            k_ = (s_.get('transition_in') or {}).get('type') or 'cut'
            kinds[k_] = kinds.get(k_, 0) + 1
        hid = sum(1 for s_ in S_ for c_ in s_.get('internal_changes') or [] if c_.get('type') in ('cut', 'dip_to_dark', 'reframe'))
        tp = (T.get('audio') or {}).get('tempo') or {}
        moving = sum(1 for s_ in S_ if (s_.get('camera') or {}).get('move') not in (None, 'static'))
        texts = sum(len(s_.get('text') or []) for s_ in S_)
        L += ['', '## Timeline (frame level, v2.1)', '',
              f"- {T['video']['frames']} frames at {T['video']['fps']} fps; transitions: "
              + (', '.join(f'{v} {k}' for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])) or 'none')
              + (f'; {hid} hidden cuts/dips/reframes inside shots' if hid else ''),
              f"- Camera or element moves in {moving} of {len(S_)} shots; {texts} on-screen text items read"
              + (f"; cadence {T['video']['cadence']['label']}" if T['video'].get('cadence') else ''),
              f"- Tempo: " + (f"{tp.get('bpm')} BPM, confidence {tp.get('confidence')}" if tp.get('bpm') else 'no steady pulse'),
              '- Every number is in `measured.md` (ready-made blueprint sections) and `data/timeline.json`.']
        if T.get('notes'):
            L.append('- Notes: ' + ' · '.join(str(n)[:200] for n in T['notes'][:8]))
    L += ['', '## Sheets to look at', '']
    L += [f"- `sheets/{p_.name}`: {desc}" for p_, desc in sheet_list]
    path.write_text('\n'.join(L) + '\n', encoding='utf-8')


# ---------------------------------------------------------------- main

def selftest():
    out = {'dissect': VERSION, 'python': sys.version.split()[0]}
    for mod in ('numpy', 'PIL', 'faster_whisper', 'parselmouth'):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, '__version__', getattr(m, 'VERSION', 'ok'))
        except Exception as e:
            out[mod] = f'missing ({e.__class__.__name__})'
    out['ffmpeg'] = shutil.which('ffmpeg') or 'missing'
    out['ffprobe'] = shutil.which('ffprobe') or 'missing'
    out['yt-dlp runners'] = [' '.join(r) for r in ytdlp_runners()] or 'none'
    print(json.dumps(out, indent=1))


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('check', 'assemble'):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        if sys.argv[1] == 'check':
            import check
            check.main(sys.argv[2:])
        else:
            import assemble
            assemble.main(sys.argv[2:])
        return
    ap = argparse.ArgumentParser(description='Measure a video for the dissect skill. Also: "dissect.py check <render> '
                                             '--ref <folder>" and "dissect.py assemble <folder>".')
    ap.add_argument('source', nargs='?', help='video link or local file')
    ap.add_argument('--depth', choices=sorted(DEPTHS), default='standard')
    ap.add_argument('--out', help='output folder (default: $DISSECT_OUT or ./dissections)')
    ap.add_argument('--lang', help='spoken language code, e.g. en, hi, es (default: auto-detect)')
    ap.add_argument('--model', help='whisper model: tiny, base, small, medium, large-v3, turbo ... (default by depth)')
    ap.add_argument('--cookies-from-browser', metavar='BROWSER', help='let yt-dlp use this browser\'s login (ask the user first)')
    ap.add_argument('--cookies', metavar='FILE', help='Netscape cookies.txt for yt-dlp (ask the user first)')
    ap.add_argument('--max-minutes', type=float, help='only analyse the first N minutes')
    ap.add_argument('--cut-threshold', type=float, default=0.30, help='scene-change score that counts as a cut (default 0.30)')
    ap.add_argument('--keep-video', action='store_true', help='keep the downloaded video as source.<ext>')
    ap.add_argument('--no-comments', action='store_true', help='skip fetching YouTube comments')
    ap.add_argument('--no-timeline', action='store_true', help='skip the frame-level timeline (v1 output only, faster)')
    ap.add_argument('--timeline-minutes', type=float, default=10.0,
                    help='measure the frame-level timeline over at most the first N minutes (default 10)')
    ap.add_argument('--threads', type=int, default=2, help='threads for ffmpeg and OpenCV in the timeline pass (default 2)')
    ap.add_argument('--selftest', action='store_true', help='check dependencies and exit')
    ap.add_argument('--version', action='version', version=f'dissect {VERSION}')
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.source:
        ap.error('give a video link or file')
    need('ffmpeg')
    need('ffprobe')
    cfg = dict(DEPTHS[args.depth])
    if args.model:
        cfg['model'] = args.model

    out_root = Path(args.out or os.environ.get('DISSECT_OUT') or 'dissections').expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.work-', dir=out_root))
    folder, result = None, None
    try:
        src, info, runner = args.source, {}, None
        local = Path(src).expanduser()
        if local.is_file():
            vid = work / ('video' + local.suffix.lower())
            try:
                os.symlink(local.resolve(), vid)
            except OSError:
                shutil.copy2(local, vid)
            info = {'title': local.stem, 'extractor_key': 'Local', 'webpage_url': str(local.resolve())}
        elif re.match(r'^https?://', src, re.I):
            vid, status, detail, runner = download(src, work, args)
            if vid is None:
                code = {'login': 3, 'not_video': 4}.get(status, 2)
                fix = {3: 'This post needs a logged-in session. Ask the user, then rerun with --cookies-from-browser '
                          '<chrome|firefox|safari|edge|brave|...> or --cookies cookies.txt.',
                       4: 'The link has no video (image or text post?).',
                       2: 'Update yt-dlp (brew upgrade yt-dlp / pip install -U yt-dlp), retry, or download the video '
                          'another way and pass the file path.'}[code]
                print(json.dumps({'ok': False, 'status': status, 'error': detail, 'fix': fix}, indent=1))
                sys.exit(code)
            j = work / 'video.info.json'
            info = fix_counts(json.loads(j.read_text(encoding='utf-8'))) if j.exists() else {}
        else:
            print(json.dumps({'ok': False, 'status': 'bad_input', 'error': f'not a link or an existing file: {src}'}))
            sys.exit(2)

        meta = probe(vid)
        if not meta:
            print(json.dumps({'ok': False, 'status': 'not_video', 'error': 'no video stream in the file'}))
            sys.exit(4)
        if args.max_minutes and meta['duration'] > args.max_minutes * 60 + 1:
            cut = work / f'trim{vid.suffix}'
            run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(vid), '-t', str(args.max_minutes * 60),
                 '-c', 'copy', str(cut)])
            if cut.exists():
                vid = cut
                meta = dict(probe(vid), duration_full=meta['duration'])
        dur = meta['duration']

        title = info.get('title') or info.get('fulltitle') or local.stem
        platform = (info.get('extractor_key') or 'video').lower().replace(':', '-')
        day = dt.date.today().isoformat()
        base = f"{day}-{slugify(platform, 16)}-{slugify(title) or slugify(info.get('id') or '') or 'video'}"
        folder = out_root / base
        k = 2
        while folder.exists():
            folder, k = out_root / f'{base}-{k}', k + 1
        (folder / 'sheets').mkdir(parents=True)
        (folder / 'data').mkdir()
        log(f'{title} · {platform} · {ts(dur)} · {meta["width"]}x{meta["height"]}')

        log('finding cuts and on-screen changes ...')
        times, scores = scene_pass(vid)
        cuts = find_cuts(times, scores, dur, args.cut_threshold)
        near, far, cfps, align = change_pass(vid)
        extra, unsure, steps = picture_changes(near, far, cfps, align, cuts, dur)
        cuts = sorted(cuts + extra)
        bounds = [0.0] + cuts + [dur]
        shots = [dict(n=i + 1, start=round(a, 3), end=round(b, 3), dur=round(b - a, 2))
                 for i, (a, b) in enumerate(zip(bounds, bounds[1:])) if b - a > 0.01]
        for sh in shots:
            sh['changes_inside'] = [x for x in steps if sh['start'] <= x < sh['end']]
            sh['uncertain_inside'] = [x for x in unsure if sh['start'] <= x < sh['end']]
        pace = pacing(shots, cuts, steps, extra, unsure, near, cfps, dur)

        tr, tr_note, words, x, sr = None, None, [], None, 16000
        rms_t, rms_db, track = np.array([]), np.array([]), None
        loud = bed = tempo = acc = None
        if meta['has_audio']:
            log('measuring sound ...')
            wav = extract_wav(vid, work / 'audio.wav')
            if wav:
                loud = loudness(vid)
                x, sr = read_wav(wav)
                rms_t, rms_db = rms_series(x, sr)
                tr, tr_note = transcribe(x, cfg['model'], args.lang)
                words = [w for s in (tr or {}).get('segments', []) for w in s['words']]
                if words:
                    log('measuring the voice ...')
                    track = pitch_track(wav)
                bed = sound_bed(x, sr, rms_t, rms_db, words, dur)
                tempo = rough_tempo(x, sr, words, dur)
                acc = cut_accents(cuts, rms_t, rms_db, dur)
            else:
                tr_note = 'audio track could not be decoded'
        else:
            tr_note = 'the video has no audio track'

        comments = []
        if runner and not args.no_comments and (info.get('extractor_key') or '').lower().startswith('youtube'):
            log('reading top comments ...')
            try:
                comments = fetch_comments(runner, args.source, work, args)
            except Exception as e:
                log(f'  comments skipped: {e}')

        log('making contact sheets ...')
        lay = layout(meta['width'], meta['height'])
        raw = work / 'frames'
        raw.mkdir()
        sheet_list = []
        hook_n = int(min(cfg['hook_s'], max(0.5, dur - 0.1)) * cfg['hook_fps'])
        hook = [(f'{i / cfg["hook_fps"]:.2f}s f{int(round(i / cfg["hook_fps"] * (meta["fps"] or 30)))}', i / cfg['hook_fps'])
                for i in range(hook_n)]
        for p_, a, b in make_sheets(vid, hook, lay, raw, folder / 'sheets', 'hook'):
            sheet_list.append((p_, f"the opening at {cfg['hook_fps']} frames/s ({a} to {b})"))
        for p_, a, b in make_sheets(vid, pick_frames(shots, steps, unsure, cfg['max_frames'], meta['fps'] or 30.0), lay, raw,
                                    folder / 'sheets', 'shots'):
            sheet_list.append((p_, f'the middle of each shot (S = shot number), "+" = just after a change inside the '
                                   f'shot, "?" = just after an uncertain change, extra frames through long shots: {a} to {b}'))
        replays = replay_peaks(info.get('heatmap'), dur, meta.get('duration_full') or info.get('duration'))
        if replays and replays['peaks']:
            pk = replays['peaks'][0]
            a, b = max(0, pk['start'] - 1), min(dur - 0.05, pk['end'] + 1)
            items = [(ts(t), t) for t in np.linspace(a, b, lay['cols'] * 2)]
            for p_, _, _ in make_sheets(vid, items, lay, raw, folder / 'sheets', 'most-replayed'):
                sheet_list.append((p_, f"around the most-replayed moment {ts(pk['start'])}–{ts(pk['end'])}"))
        thumb = next((p_ for p_ in work.glob('video.*') if p_.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')), None)
        if thumb:
            try:
                with Image.open(thumb) as im:
                    im.convert('RGB').save(folder / 'sheets' / 'cover.jpg', quality=88)
                sheet_list.append((folder / 'sheets' / 'cover.jpg', 'the cover / thumbnail the platform shows'))
            except Exception:
                pass

        shot_words = {}
        for sh in shots:
            ws = [w[2] for w in words if sh['start'] <= w[0] < sh['end']]
            shot_words[sh['n']] = ' '.join(ws)

        T, tl_note = None, None
        if not args.no_timeline:
            log('measuring the frame-level timeline (transitions, camera, type, colour, sound grid) ...')
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                import timeline as tlm
                T = tlm.measure_timeline(vid, meta, shots, words, bed, folder / 'sheets', work, log, font, ts, loud=loud,
                                         source=info.get('webpage_url') or args.source, depth=args.depth,
                                         threads=max(1, args.threads), max_s=args.timeline_minutes * 60)
            except ImportError as e:
                tl_note = f'timeline not measured: {e} (run with uv so opencv and librosa install)'
            except Exception as e:  # noqa: BLE001 - the v1 outputs are still worth writing
                import traceback
                traceback.print_exc(file=sys.stderr)
                tl_note = f'timeline not measured: {e.__class__.__name__}: {e}'
            if tl_note:
                log(tl_note)
        if T:
            T['sheets'].update(hook=next((p_.name for p_, d_ in sheet_list if p_.name.startswith('hook')), None),
                               hook_step=f"{1 / cfg['hook_fps']:.2f} s",
                               shots=[p_.name for p_, _ in sheet_list if p_.name.startswith('shots')])
            import plates
            (folder / 'data' / 'timeline.json').write_text(json.dumps(T, indent=1, ensure_ascii=False), encoding='utf-8')
            extra = re.findall(r'[A-Za-z]{3,}', ' '.join(str(info.get(k) or '') for k in ('title', 'uploader', 'channel', 'description')))
            (folder / 'measured.md').write_text(plates.render(T, words, extra), encoding='utf-8')
            tls = [folder / 'sheets' / n for n in T['sheets']['timeline']]
            trs = [folder / 'sheets' / n for n in T['sheets']['transitions']]
            sheet_list += [(p_, f"the whole film at {T['sheets']['timeline_step_s']} s steps (S = shot, f = frame)") for p_ in tls]
            sheet_list += [(p_, '14 consecutive frames around each transition and change inside a shot, the change frame '
                                'outlined') for p_ in trs]

        pl = trim_info(info)
        A = dict(dissect_version=VERSION, measured_at=dt.datetime.now().isoformat(timespec='seconds'),
                 source=info.get('webpage_url') or args.source, title=title, platform_name=info.get('extractor_key') or 'Local',
                 depth=args.depth, cut_threshold=args.cut_threshold, hook_window_s=cfg['hook_s'],
                 analysed_seconds=dur, video=meta, pacing=pace, shots=shots, platform=pl, engagement=engagement(info),
                 replays=replays, comments=comments, loudness=loud, transcript_note=tr_note,
                 transcript={k: tr[k] for k in ('model', 'language', 'language_probability')} if tr else None,
                 speech=speech_stats(tr, dur) if tr else None, voice=pitch_stats(track, words),
                 emphasis=emphasis_words(words, rms_t, rms_db, track), bed=bed, tempo=tempo, cut_accents=acc,
                 sheets=[str(p_.relative_to(folder)) for p_, _ in sheet_list], timeline_note=tl_note)

        (folder / 'data' / 'analysis.json').write_text(json.dumps(A, indent=1, ensure_ascii=False), encoding='utf-8')
        (folder / 'data' / 'info.json').write_text(json.dumps(pl, indent=1, ensure_ascii=False), encoding='utf-8')
        if tr:
            (folder / 'data' / 'transcript.json').write_text(json.dumps(tr, ensure_ascii=False), encoding='utf-8')
            (folder / 'data' / 'transcript.txt').write_text(
                '\n'.join(f"[{ts(s['start'])}] {s['text']}" for s in tr['segments']) + '\n', encoding='utf-8')
        if comments:
            (folder / 'data' / 'comments.txt').write_text(
                '\n'.join(f"[{c['likes']}]{' (creator)' if c['by_creator'] else ''} {c['text']}" for c in comments) + '\n',
                encoding='utf-8')
        write_facts(folder / 'facts.md', A, tr, shot_words, sheet_list, T)

        if args.keep_video:
            shutil.move(str(vid), str(folder / f'source{vid.suffix}'))
        order = ('hook', 'shots', 'transitions', 'timeline', 'most-replayed', 'cover')
        ordered = sorted((p_ for p_, _ in sheet_list), key=lambda p_: (next((i for i, o in enumerate(order) if p_.name.startswith(o)), 9), p_.name))
        result = dict(ok=True, folder=str(folder), facts=str(folder / 'facts.md'),
                      measured=str(folder / 'measured.md') if T else None,
                      timeline=str(folder / 'data' / 'timeline.json') if T else None,
                      sheets_in_reading_order=[str(p_) for p_ in ordered],
                      draft=str(folder / f"{slugify(title, 40) or 'video'}.blueprint.draft.md") if T else None,
                      blueprint=str(folder / f"{slugify(title, 40) or 'video'}.blueprint.md"),
                      title=title, platform=A['platform_name'], duration_s=dur, shots=len(shots),
                      words=(A['speech'] or {}).get('words', 0),
                      notes=[n for n in (tr_note, tl_note) if n])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
