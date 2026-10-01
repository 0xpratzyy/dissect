# Downloading: how it works and what to do when it doesn't

`dissect.py` downloads with [yt-dlp](https://github.com/yt-dlp/yt-dlp), which supports YouTube, TikTok, Instagram, X/Twitter, Facebook, Vimeo, Reddit, Twitch, LinkedIn, Bilibili, Dailymotion and well over a thousand other sites, plus many pages with an embedded player.

## What the script already tries

1. The yt-dlp it can find, in this order:
   - `$DISSECT_YTDLP` (a full command)
   - `yt-dlp` on PATH
   - an importable `yt_dlp` module
   - the newest release via `uvx --from "yt-dlp[default,curl-cffi]@latest" yt-dlp`
   That last one also rescues an outdated installed copy, since sites change often and old yt-dlp versions break.
2. For each, a plain download, then a retry with `--impersonate chrome` if the site blocked it (HTTP 403, Cloudflare, "impersonation").
3. A format no larger than 720p on the short side (plenty for analysis, quick to fetch), merged to MP4.
4. With `--max-minutes N`, only the first N minutes are downloaded (`--download-sections`).
5. Cookies only when `--cookies-from-browser` or `--cookies` is passed.

Exit codes: **2** download failed · **3** login needed · **4** no video in the post · **5** ffmpeg missing.

## Platform notes

- **YouTube and Shorts:**
  - Public videos work without login.
  - Recent yt-dlp versions want a JavaScript runtime (Deno recommended; Node or Bun also work) to unlock all formats. `doctor.py` checks for it.
  - Age-restricted, members-only and private videos need cookies.
  - YouTube is the only platform that exposes the most-replayed heatmap and comments to the script.
- **TikTok:**
  - Usually works. If blocked, impersonation (built into the retry) normally fixes it.
  - Photo-mode posts (slideshows) have no video stream, so they exit with code 4.
  - The metadata often names the music track.
- **Instagram Reels:**
  - Public reels often work without login, but Instagram rate-limits and sometimes demands login. Then use cookies (with permission).
  - Private accounts need the user to follow the account and pass cookies.
  - Carousels download the first video only.
- **X / Twitter:**
  - Public video posts usually work.
  - Sensitive-media and protected posts need cookies.
  - For posts with several videos, add `/video/2` (etc.) to the post URL to pick one.
- **Facebook:** public videos and reels usually work; some need cookies.
- **Anything else:** try it. yt-dlp's generic extractor finds many embedded players. If it says "Unsupported URL", ask the user for the direct video page.

## Errors and fixes

| Message contains | Meaning | Do this |
|---|---|---|
| `Sign in`, `login`, `cookies`, `private`, `rate-limit`, `not a bot` | The site wants a logged-in session | Ask the user, then rerun with `--cookies-from-browser <browser>` |
| `HTTP Error 403`, `Cloudflare`, `impersonat` | Bot protection | The script retries with impersonation. Otherwise update yt-dlp, or have the user download it manually |
| `Unsupported URL` | No extractor for that page | Ask for the post's own URL, not a feed or search page |
| `No video formats found`, `There's no video` | Image or text post | Exit code 4. Tell the user there's no video |
| `Requested format is not available` | Unusual formats | Rerun; the newest yt-dlp via uvx usually fixes it |
| `n challenge`, `JavaScript`, `JS runtime` | YouTube needs a JS runtime | Install Deno (see `doctor.py`), then rerun |
| `timed out` | Slow or huge download | Use `--max-minutes`, or retry later |

## Cookies, with permission

Some posts only download for a logged-in user. yt-dlp can borrow the login from a browser on this computer. The cookies are read locally, used for that one request, and never saved by dissect. Always ask first, name the browser, and use it only for that download.

```bash
uv run dissect.py "<url>" --cookies-from-browser chrome        # or firefox, safari, edge, brave, opera, vivaldi, chromium
uv run dissect.py "<url>" --cookies-from-browser "chrome:Profile 2"   # a specific browser profile
uv run dissect.py "<url>" --cookies ~/Downloads/cookies.txt    # a Netscape cookies.txt exported by the user
```

- **macOS:** a Keychain prompt may ask to allow access to the browser's safe storage. Safari's cookies need the terminal to have Full Disk Access.
- **Windows:** newer Chrome and Edge versions encrypt cookies so other programs can't read them. Firefox, or a `cookies.txt` exported with a browser extension, is the reliable route.
- **Linux:** works with Chrome, Chromium and Firefox profiles. A keyring prompt may appear.

## Manual fallback

If nothing works, the user can download the video with any tool they trust and pass the file path instead of the link:

```bash
uv run dissect.py ~/Downloads/video.mp4
```

Platform stats and comments won't be available for a local file. Everything else works the same.

## Keeping yt-dlp current

- Homebrew: `brew upgrade yt-dlp`
- pip / pipx: `pipx upgrade yt-dlp` or `pip install -U yt-dlp`
- uv tool: `uv tool upgrade yt-dlp`
- Standalone binary: `yt-dlp -U`
- winget: `winget upgrade yt-dlp.yt-dlp`

## Be a good citizen

Download only to analyse, keep it private, and let the script delete the file afterwards (the default). Respect each platform's terms and the creator's copyright: the blueprint should teach a format, not reproduce the video.
