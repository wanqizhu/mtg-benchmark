#!/usr/bin/env python3
"""Build the numbered puzzle dataset from possibilitystorm.com + Patreon solutions.

Puzzle images and catalog come from the public WordPress API.
Solution text is behind Patreon; gallery-dl reads your existing browser session
(--cookies-from-browser) rather than a username/password.

Difficulty in metadata.json is copied from the WordPress category. Those categories
are often leftover from the previous post — after each pull, read DIFFICULTY on the
new puzzle.jpg footer and correct metadata.json before transcribing.

Requires: curl, gallery-dl, and a logged-in Patreon session in the chosen browser.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import time
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
API_BASE = "https://www.possibilitystorm.com/wp-json/wp/v2"


class HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip = True
        if tag in {"p", "br", "li", "ol", "ul", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip = False
        if tag in {"p", "li", "ol", "ul", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def html_to_text(raw: str) -> str:
    if not raw or not raw.strip():
        return ""
    if raw.strip().startswith("<"):
        parser = HTMLTextExtractor()
        parser.feed(raw)
        text = html.unescape(" ".join(parser.parts))
    else:
        text = html.unescape(raw)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def curl_json(url: str) -> list | dict:
    result = subprocess.run(
        ["curl", "-sfL", url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return json.loads(result.stdout)


def curl_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-sfL", url, "-o", str(dest)],
        check=True,
    )


def gallery_dl_bin() -> str:
    path = shutil.which("gallery-dl")
    if not path:
        raise RuntimeError("gallery-dl not found on PATH. Install with: pip install gallery-dl")
    return path


def fetch_patreon_content(post_url: str, *, browser: str) -> str:
    result = subprocess.run(
        [
            gallery_dl_bin(),
            "--cookies-from-browser",
            browser,
            "--user-agent",
            "browser",
            "--print",
            "{content}",
            post_url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout).strip()
        print(f"  patreon fetch failed: {err[:300]}")
        return ""
    return html_to_text(result.stdout.strip())


def parse_post_links(content_html: str) -> tuple[str | None, str | None]:
    img_match = re.search(
        r'href="(https://i\d\.wp\.com/www\.possibilitystorm\.com/wp-content/uploads/[^"]+)"'
        r"[^>]*>\s*(?:<img[^>]*>\s*)?View Hi-res Image",
        content_html,
        re.I | re.S,
    )
    if not img_match:
        img_match = re.search(
            r'wcp-caption-image"\s+src="(https://i\d\.wp\.com/www\.possibilitystorm\.com/wp-content/uploads/[^"?]+(?:\?[^"]*)?)"',
            content_html,
            re.I,
        )
    pat_match = re.search(
        r'href="(https://www\.patreon\.com/(?:c/)?(?:[\w.-]+/)?posts/[^"]+)"',
        content_html,
        re.I,
    )
    solution_url = pat_match.group(1).rstrip("/") if pat_match else None
    if solution_url:
        # gallery-dl's post extractor accepts /posts/<id>; creator-scoped
        # /mtgpuzzles/posts/<id> links are equivalent.
        m = re.search(r"/posts/(\d+)", solution_url)
        if m:
            solution_url = f"https://www.patreon.com/posts/{m.group(1)}"
    image_url = img_match.group(1) if img_match else None
    if image_url:
        image_url = re.sub(r"\?w=\d+.*", "", image_url)
        if "ssl=1" not in image_url:
            image_url += ("&" if "?" in image_url else "?") + "ssl=1"
    return image_url, solution_url


def parse_puzzle_number(title: str) -> str | None:
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*[:\.]", title)
    if not m:
        return None
    num = m.group(1)
    return num.rstrip("0").rstrip(".") if "." in num else num


def season_from_title(title: str) -> str | None:
    """Prefer the set name in the title; WP tags are often leftover from the previous post.

    Examples: '302: Marvel Super Heroes #1' -> 'Marvel Super Heroes'
              '306: The Hobbit #1' -> 'The Hobbit'
    """
    m = re.match(r"\s*\d+(?:\.\d+)?\s*[:.]\s*(.+?)\s+#\s*\d+\s*$", title.strip())
    if m:
        return m.group(1).strip() or None
    return None


def folder_name_for(puzzle_id: str) -> str:
    return puzzle_id.replace(".", "_")


def public_page_url(post: dict) -> str | None:
    link = str(post.get("link") or "").strip()
    return link or None


def persist_source_url(meta_path: Path, meta: dict, post: dict) -> None:
    url = public_page_url(post)
    if not url or meta.get("source_url") == url:
        return
    meta["source_url"] = url
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def image_extension(url: str) -> str:
    path = url.split("?")[0].lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        if path.endswith(ext):
            return ext
    return ".jpg"


def load_taxonomies() -> tuple[dict[int, str], dict[int, str]]:
    categories = {}
    for page in range(1, 5):
        data = curl_json(f"{API_BASE}/categories?per_page=100&page={page}")
        if not data:
            break
        for item in data:
            slug = item["slug"]
            if slug != "uncategorized":
                categories[item["id"]] = item["name"]
    tags = {}
    for page in range(1, 5):
        data = curl_json(f"{API_BASE}/tags?per_page=100&page={page}")
        if not data:
            break
        for item in data:
            tags[item["id"]] = item["name"]
    return categories, tags


def is_complete(folder: Path) -> bool:
    meta_path = folder / "metadata.json"
    if not meta_path.exists():
        return False
    if not any(folder.glob("puzzle.*")):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if meta.get("solution_url") and not meta.get("solution_text", "").strip():
        return False
    return True


def rebuild_index(out_root: Path) -> list[dict]:
    index = []
    for folder in out_root.iterdir():
        if not folder.is_dir():
            continue
        meta_path = folder / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index.append({k: v for k, v in meta.items() if k != "solution_text"})
    index.sort(key=lambda m: float(str(m.get("puzzle_id", "0")).replace("_", ".")))
    return index


def fetch_all_posts() -> list[dict]:
    posts = []
    page = 1
    while True:
        batch = curl_json(f"{API_BASE}/posts?per_page=100&page={page}&orderby=date&order=desc")
        if not batch:
            break
        posts.extend(batch)
        page += 1
    return posts


def build_dataset(*, out_root: Path, browser: str) -> dict:
    categories, tags = load_taxonomies()
    posts = fetch_all_posts()
    out_root.mkdir(parents=True, exist_ok=True)

    stats = {
        "downloaded": 0,
        "skipped": 0,
        "missing_image": 0,
        "missing_solution": 0,
        "missing_patreon": 0,
    }
    # WP categories are often stale; remind after writing new metadata.
    check_difficulty: list[str] = []

    for post in sorted(posts, key=lambda p: float(parse_puzzle_number(p["title"]["rendered"]) or 0)):
        title = html.unescape(post["title"]["rendered"])
        puzzle_id = parse_puzzle_number(title)
        if puzzle_id is None:
            continue
        # "2017: A Year of Puzzling in Review" matches the number parser.
        try:
            if float(puzzle_id.replace("_", ".")) >= 1000:
                continue
        except ValueError:
            continue

        folder_name = folder_name_for(puzzle_id)
        folder = out_root / folder_name
        meta_path = folder / "metadata.json"

        content_html = post["content"]["rendered"]
        image_url, solution_url = parse_post_links(content_html)

        existing_meta = None
        if meta_path.exists():
            try:
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_meta = None

        if existing_meta and existing_meta.get("solution_text", "").strip():
            persist_source_url(meta_path, existing_meta, post)
            stats["skipped"] += 1
            print(f"[{folder_name}] {title[:50]}  skip=ok")
            continue

        if existing_meta and is_complete(folder) and not solution_url:
            persist_source_url(meta_path, existing_meta, post)
            stats["skipped"] += 1
            print(f"[{folder_name}] {title[:50]}  skip=ok (no patreon link)")
            continue

        difficulty = [categories[cid] for cid in post.get("categories", []) if cid in categories]
        title_season = season_from_title(title)
        seasons = [title_season] if title_season else [tags[tid] for tid in post.get("tags", []) if tid in tags]
        folder.mkdir(parents=True, exist_ok=True)

        if existing_meta and any(folder.glob("puzzle.*")):
            solution_text = ""
            if solution_url:
                solution_text = fetch_patreon_content(solution_url, browser=browser)
                if not solution_text:
                    stats["missing_solution"] += 1
                time.sleep(0.5)
            elif not existing_meta.get("solution_url"):
                stats["missing_patreon"] += 1
                stats["missing_solution"] += 1

            meta = {
                **existing_meta,
                "source_url": public_page_url(post) or existing_meta.get("source_url"),
                "solution_url": solution_url or existing_meta.get("solution_url"),
                "solution_text": solution_text or existing_meta.get("solution_text", ""),
            }
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            stats["downloaded"] += 1
            print(f"[{folder_name}] {title[:50]}  backfill  sol={'ok' if meta['solution_text'] else 'MISS'}")
            continue

        if is_complete(folder):
            if existing_meta:
                persist_source_url(meta_path, existing_meta, post)
            stats["skipped"] += 1
            print(f"[{folder_name}] {title[:50]}  skip=ok")
            continue

        image_file = None
        if image_url:
            ext = image_extension(image_url)
            image_file = f"puzzle{ext}"
            if not (folder / image_file).exists():
                try:
                    curl_download(image_url, folder / image_file)
                except subprocess.CalledProcessError:
                    stats["missing_image"] += 1
                    image_file = None
            else:
                image_file = next(folder.glob("puzzle.*")).name
        else:
            stats["missing_image"] += 1

        solution_text = ""
        if solution_url:
            solution_text = fetch_patreon_content(solution_url, browser=browser)
            if not solution_text:
                stats["missing_solution"] += 1
            time.sleep(0.5)
        else:
            stats["missing_patreon"] += 1
            stats["missing_solution"] += 1

        wp_difficulty = difficulty[0] if len(difficulty) == 1 else difficulty
        meta = {
            "puzzle_id": puzzle_id,
            "difficulty": wp_difficulty,
            "seasons": seasons,
            "image_url": image_url,
            "source_url": public_page_url(post),
            "solution_url": solution_url,
            "solution_text": solution_text,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        stats["downloaded"] += 1
        check_difficulty.append(folder_name)
        print(
            f"[{folder_name}] {title[:50]}  img={'ok' if image_file else 'MISS'}  "
            f"sol={'ok' if solution_text else 'MISS'}  "
            f"difficulty={wp_difficulty} (WP category — check image footer)"
        )

    index = rebuild_index(out_root)
    manifest = {"puzzle_count": len(index), "stats": stats, "puzzles": index}
    (out_root / "index.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "check_difficulty": check_difficulty}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "datasets" / "mtg",
        help="Dataset output directory (gitignored)",
    )
    parser.add_argument(
        "--browser",
        default="chrome",
        help="Browser whose cookies gallery-dl should read (chrome, firefox, safari, …)",
    )
    args = parser.parse_args()
    manifest = build_dataset(out_root=args.out, browser=args.browser)
    print(json.dumps(manifest["stats"], indent=2))
    check = manifest.get("check_difficulty") or []
    if check:
        print(
            "\nCheck difficulty on the image footer (WP categories are often leftover "
            "from the previous post). Fix metadata.json for: "
            + ", ".join(check)
            + "\nThen transcribe problem_gold.md: datasets/TRANSCRIBING.md"
        )


if __name__ == "__main__":
    main()
