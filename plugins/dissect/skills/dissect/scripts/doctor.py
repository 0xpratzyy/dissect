#!/usr/bin/env python3
"""doctor.py - check that this computer can run dissect, and print the install command for anything missing.

    python3 doctor.py            # check
    python3 doctor.py --warmup   # also let uv install dissect's Python packages now, so the first real run is quick

Standard library only, so it runs before anything else is installed. It never installs anything itself.
Exit code 0 when ready, 1 when something required is missing.
"""
import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def os_key():
    if sys.platform == 'darwin':
        return 'mac'
    if sys.platform.startswith('win'):
        return 'win'
    try:
        txt = Path('/etc/os-release').read_text().lower()
    except OSError:
        txt = ''
    if re.search(r'debian|ubuntu|mint|pop', txt):
        return 'apt'
    if re.search(r'fedora|rhel|centos|rocky|alma', txt):
        return 'dnf'
    if 'arch' in txt or 'manjaro' in txt:
        return 'pacman'
    return 'linux'


INSTALL = {
    'uv': {'mac': 'brew install uv   (or: curl -LsSf https://astral.sh/uv/install.sh | sh)',
           'win': 'winget install --id=astral-sh.uv -e',
           '*': 'curl -LsSf https://astral.sh/uv/install.sh | sh'},
    'ffmpeg': {'mac': 'brew install ffmpeg', 'win': 'winget install Gyan.FFmpeg', 'apt': 'sudo apt install ffmpeg',
               'dnf': 'sudo dnf install ffmpeg   (enable RPM Fusion first)', 'pacman': 'sudo pacman -S ffmpeg',
               '*': 'install ffmpeg with your package manager (https://ffmpeg.org/download.html)'},
    'yt-dlp': {'mac': 'brew install yt-dlp', 'win': 'winget install yt-dlp.yt-dlp',
               '*': 'uv tool install "yt-dlp[default,curl-cffi]"   (or: pipx install yt-dlp)'},
    'deno': {'mac': 'brew install deno', 'win': 'winget install DenoLand.Deno',
             '*': 'curl -fsSL https://deno.land/install.sh | sh'},
}


def how(tool):
    k = os_key()
    return INSTALL[tool].get(k, INSTALL[tool]['*'])


def version(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        out = (r.stdout or r.stderr).strip().splitlines()
        return out[0] if out else ''
    except Exception:
        return ''


def main():
    warmup = '--warmup' in sys.argv
    rows, missing = [], []

    def row(ok, name, detail, fix=None, required=True):
        status = 'ok' if ok else ('MISSING' if required else 'optional')
        rows.append(f'{status:<8} {name:<11} {detail}')
        if not ok and fix:
            if required:
                missing.append((name, fix))
            else:
                rows.append(f'{"":<8} {"":<11} -> {fix}')

    py_ok = sys.version_info >= (3, 9)
    row(py_ok, 'python', platform.python_version() + ('' if py_ok else ' (need 3.9+)'),
        'install Python 3.9+ from https://www.python.org or your package manager')

    uv = shutil.which('uv')
    row(bool(uv), 'uv', version([uv, '--version']) if uv else 'not found (installs the Python packages automatically)',
        how('uv'))

    for tool in ('ffmpeg', 'ffprobe'):
        p = shutil.which(tool)
        row(bool(p), tool, version([p, '-version'])[:60] if p else 'not found', how('ffmpeg'))

    y = shutil.which('yt-dlp')
    if y:
        v = version([y, '--version'])
        detail = v
        m = re.match(r'(\d{4})\.(\d{2})\.(\d{2})', v)
        if m:
            age = (datetime.date.today() - datetime.date(*map(int, m.groups()))).days
            if age > 90:
                detail += f'  ({age} days old; sites change often, consider updating)'
        row(True, 'yt-dlp', detail)
    elif shutil.which('uvx'):
        row(True, 'yt-dlp', 'not installed, but uvx can fetch the newest release on demand')
    else:
        row(False, 'yt-dlp', 'not found', how('yt-dlp'))

    js = next((n for n in ('deno', 'node', 'bun') if shutil.which(n)), None)
    row(bool(js), 'js runtime', js or 'none (YouTube may refuse some formats without one)', how('deno'), required=False)

    print('dissect doctor\n')
    print('\n'.join(rows))
    if missing:
        print('\nInstall these first, then run doctor again:')
        for name, fix in missing:
            print(f'  {name}: {fix}')
        sys.exit(1)

    if warmup:
        if not uv:
            print('\n--warmup needs uv.')
            sys.exit(1)
        print('\nInstalling dissect\'s Python packages with uv (one time) ...')
        r = subprocess.run([uv, 'run', str(HERE / 'dissect.py'), '--selftest'], text=True, capture_output=True)
        print(r.stdout.strip() or r.stderr.strip()[-2000:])
        if r.returncode:
            sys.exit(1)
    print('\nReady. Whisper downloads its speech model the first time you dissect something.')


if __name__ == '__main__':
    main()
