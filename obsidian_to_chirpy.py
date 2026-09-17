#!/usr/bin/env python3
"""
Obsidian -> Jekyll Chirpy converter

Features
--------
1. Obsidian callouts -> Chirpy prompts
   > [!note] Title
   > content
   ->
   > **Title**
   >
   > content
   {: .prompt-info }

2. Obsidian images
   ![[figure.png]]
   ![[figure.png|600]]
   ![[figure.png|caption]]
   -> copies the file into the Jekyll assets directory and rewrites the link.

3. Obsidian wikilinks
   [[My Note]] -> [My Note](/posts/my-note/)
   [[My Note|label]] -> [label](/posts/my-note/)
   [[#Heading]] -> [Heading](#heading)

4. Jekyll-aware wikilinks
   Existing target posts become links; missing targets become plain text,
   so HTML-Proofer does not fail on invented /posts/... URLs.

5. Obsidian-style hard line breaks
   Plain Enter-separated prose is rewritten with two trailing spaces so
   Chirpy/Jekyll renders the line breaks. Code fences, math blocks, tables,
   headings, lists, HTML blocks, and prompt attribute lines are excluded.

6. Existing YAML front matter is preserved.
   Missing title/date are added, and math:true / mermaid:true are inferred.

Usage
-----
python obsidian_to_chirpy.py NOTE.md \
  --vault "C:/path/to/ObsidianVault" \
  --jekyll "C:/path/to/username.github.io"

Optional:
  --image-dest assets/img/posts
  --no-hard-breaks
  --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import unicodedata
from datetime import datetime
from pathlib import Path


IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".bmp", ".tif", ".tiff", ".avif"
}

CALLOUT_MAP = {
    # info
    "note": "info",
    "info": "info",
    "abstract": "info",
    "summary": "info",
    "tldr": "info",
    "example": "info",
    "quote": "info",
    "cite": "info",

    # tip
    "tip": "tip",
    "hint": "tip",
    "important": "tip",
    "success": "tip",
    "check": "tip",
    "done": "tip",

    # warning
    "warning": "warning",
    "caution": "warning",
    "attention": "warning",
    "question": "warning",
    "help": "warning",
    "faq": "warning",

    # danger
    "danger": "danger",
    "error": "danger",
    "failure": "danger",
    "fail": "danger",
    "bug": "danger",
    "missing": "danger",
}


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.strip())
    text = re.sub(r"\.(md|markdown)$", "", text, flags=re.I)
    text = text.lower()
    text = re.sub(r"[^\w가-힣]+", "-", text, flags=re.UNICODE)
    text = re.sub(r"-{2,}", "-", text).strip("-_")
    return text or "post"


def split_front_matter(text: str):
    if not text.startswith("---"):
        return "", text

    m = re.match(r"^---[ \t]*\n(.*?)\n---[ \t]*\n?", text, flags=re.S)
    if not m:
        return "", text

    return m.group(1), text[m.end():]


def yaml_has_key(front: str, key: str) -> bool:
    return re.search(rf"(?m)^\s*{re.escape(key)}\s*:", front) is not None


def yaml_value(front: str, key: str):
    m = re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*(.*?)\s*$", front)
    if not m:
        return None
    value = m.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def ensure_front_matter(front: str, body: str, fallback_title: str):
    now = datetime.now().astimezone()

    title = yaml_value(front, "title") or fallback_title
    title = title.strip() or fallback_title

    date_value = yaml_value(front, "date")
    if date_value:
        m = re.search(r"\d{4}-\d{2}-\d{2}", date_value)
        post_date = m.group(0) if m else now.strftime("%Y-%m-%d")
    else:
        post_date = now.strftime("%Y-%m-%d")

    additions = []

    if not yaml_has_key(front, "title"):
        safe_title = title.replace('"', '\\"')
        additions.append(f'title: "{safe_title}"')

    if not yaml_has_key(front, "date"):
        additions.append(f"date: {now.strftime('%Y-%m-%d %H:%M:%S %z')}")

    has_math = (
        "$$" in body
        or re.search(r"(?<!\\)\$[^$\n]+\$", body) is not None
        or "\\[" in body
        or "\\(" in body
    )
    if has_math and not yaml_has_key(front, "math"):
        additions.append("math: true")

    if re.search(r"(?m)^```mermaid\s*$", body) and not yaml_has_key(front, "mermaid"):
        additions.append("mermaid: true")

    new_front = front.rstrip()
    if additions:
        if new_front:
            new_front += "\n"
        new_front += "\n".join(additions)

    return new_front, title, post_date


def strip_one_quote(line: str) -> str:
    return re.sub(r"^\s*>\s?", "", line, count=1)


def transform_callouts(body: str) -> str:
    """
    Convert top-level Obsidian callouts to Chirpy prompt blockquotes.
    Fold markers +/- are intentionally discarded because Chirpy prompts do
    not provide Obsidian's fold behavior by default.
    """
    lines = body.splitlines()
    out = []
    i = 0

    header_re = re.compile(
        r"^\s*>\s*\[!([A-Za-z0-9_-]+)\]([+-])?(?:\s+(.*?))?\s*$"
    )

    while i < len(lines):
        m = header_re.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue

        callout_type = m.group(1).lower()
        title = (m.group(3) or "").strip()
        prompt = CALLOUT_MAP.get(callout_type, "info")

        content = []
        i += 1

        # Consume consecutive blockquote lines belonging to this callout.
        while i < len(lines):
            line = lines[i]
            if re.match(r"^\s*>", line):
                # A new callout header starts a new block, not part of this one.
                if header_re.match(line):
                    break
                content.append(strip_one_quote(line))
                i += 1
            else:
                break

        if title:
            out.append(f"> **{title}**")
            if content and (not content[0].strip()):
                pass
            else:
                out.append(">")

        for c in content:
            out.append(">" if c == "" else f"> {c}")

        # Empty callout: still create a valid prompt block.
        if not title and not content:
            out.append(">")

        out.append(f"{{: .prompt-{prompt} }}")

    return "\n".join(out)


def find_vault_file(vault: Path, target: str) -> Path | None:
    """
    Resolve an Obsidian embed target.
    First try it as a vault-relative path, then search by filename.
    """
    normalized = target.replace("\\", "/").lstrip("/")
    exact = vault / normalized
    if exact.is_file():
        return exact

    name = Path(normalized).name
    matches = [p for p in vault.rglob(name) if p.is_file()]
    if not matches:
        return None

    # Prefer shortest relative path to keep resolution deterministic.
    matches.sort(key=lambda p: (len(p.relative_to(vault).parts), str(p).lower()))
    if len(matches) > 1:
        print(
            f"[warning] Multiple attachments named '{name}'. "
            f"Using: {matches[0]}",
            file=sys.stderr,
        )
    return matches[0]


def unique_destination(directory: Path, filename: str, source: Path) -> Path:
    """
    Avoid overwriting a different image with the same filename.
    """
    candidate = directory / filename
    if not candidate.exists():
        return candidate

    try:
        if candidate.resolve() == source.resolve():
            return candidate
        if candidate.read_bytes() == source.read_bytes():
            return candidate
    except OSError:
        pass

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    n = 2
    while True:
        candidate = directory / f"{stem}-{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1



def safe_asset_filename(source: Path) -> str:
    """
    Make URLs clean and stable.
    Example:
      Pasted image 20260902225444.png
      -> pasted-image-20260902225444.png
    """
    stem = slugify(source.stem)
    suffix = source.suffix.lower()
    return f"{stem}{suffix}"

def transform_images(
    body: str,
    vault: Path,
    jekyll: Path,
    image_dest_rel: Path,
    dry_run: bool = False,
) -> str:
    """
    ![[img.png]]
    ![[img.png|600]]
    ![[img.png|Caption]]
    """
    pattern = re.compile(r"!\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")

    def replace(match: re.Match) -> str:
        raw_target = match.group(1).strip()
        modifier = (match.group(2) or "").strip()

        suffix = Path(raw_target).suffix.lower()
        if suffix not in IMAGE_EXTS:
            # Not an image embed. Leave untouched instead of guessing.
            return match.group(0)

        source = find_vault_file(vault, raw_target)
        if source is None:
            print(f"[warning] Image not found: {raw_target}", file=sys.stderr)
            return match.group(0)

        dest_dir = jekyll / image_dest_rel
        dest = unique_destination(dest_dir, safe_asset_filename(source), source)

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or dest.read_bytes() != source.read_bytes():
                shutil.copy2(source, dest)

        web_path = "/" + dest.relative_to(jekyll).as_posix()

        if modifier.isdigit():
            # Chirpy/kramdown image width attribute.
            return f"![{source.stem}]({web_path}){{: width=\"{modifier}\" }}"

        alt = modifier if modifier else source.stem
        alt = alt.replace("[", "").replace("]", "")
        return f"![{alt}]({web_path})"

    return pattern.sub(replace, body)


def heading_anchor(text: str) -> str:
    """
    Close approximation for common Markdown heading IDs.
    """
    text = re.sub(r"[*_`~]", "", text.strip().lower())
    text = re.sub(r"[^\w가-힣\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text


def _post_keys(file: Path, front: str):
    """
    Return names by which an Obsidian wikilink may refer to a Jekyll post.
    """
    stem = file.stem
    stem_without_date = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", stem)
    title = yaml_value(front, "title")
    explicit_slug = yaml_value(front, "slug")

    candidates = {
        stem,
        stem_without_date,
        slugify(stem_without_date),
    }

    if title:
        candidates.add(title)
        candidates.add(slugify(title))

    if explicit_slug:
        candidates.add(explicit_slug)
        candidates.add(slugify(explicit_slug))

    return {c.strip().lower() for c in candidates if c and c.strip()}


def build_post_index(jekyll: Path):
    """
    Build a lookup of existing Jekyll posts.

    This prevents Obsidian [[wikilinks]] from becoming links to pages that
    do not actually exist, which otherwise makes htmlproofer fail.
    """
    index = {}
    posts_dir = jekyll / "_posts"
    if not posts_dir.is_dir():
        return index

    files = list(posts_dir.rglob("*.md")) + list(posts_dir.rglob("*.markdown"))
    for file in files:
        try:
            raw = file.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue

        front, _ = split_front_matter(raw)
        stem_without_date = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", file.stem)

        # Chirpy's normal post URL is /posts/:title/.
        explicit_permalink = yaml_value(front, "permalink")
        explicit_slug = yaml_value(front, "slug")
        url_slug = explicit_slug or stem_without_date

        if explicit_permalink:
            url = explicit_permalink
            if not url.startswith("/"):
                url = "/" + url
            if not url.endswith("/") and "." not in Path(url).name:
                url += "/"
        else:
            url = f"/posts/{slugify(url_slug)}/"

        for key in _post_keys(file, front):
            index[key] = url

    return index


def transform_wikilinks(body: str, jekyll: Path) -> str:
    """
    Convert Obsidian wikilinks only when the linked Jekyll post already exists.

    Existing post:
      [[My Note]] -> [My Note](/posts/my-note/)

    Missing post:
      [[My Future Note]] -> My Future Note

    This intentionally avoids emitting a broken internal URL that would fail
    HTML-Proofer.
    """
    pattern = re.compile(r"(?<!!)\[\[([^\]]+)\]\]")
    post_index = build_post_index(jekyll)

    def resolve_note(note: str):
        raw = note.strip()
        keys = [
            raw.lower(),
            re.sub(r"\.(md|markdown)$", "", raw, flags=re.I).lower(),
            slugify(raw).lower(),
        ]
        for key in keys:
            if key in post_index:
                return post_index[key]
        return None

    def replace(match: re.Match) -> str:
        raw = match.group(1).strip()

        if "|" in raw:
            target, label = raw.split("|", 1)
            target = target.strip()
            label = label.strip()
        else:
            target = raw
            label = ""

        # Same-page heading link.
        if target.startswith("#"):
            heading = target[1:].strip()
            return f"[{label or heading}](#{heading_anchor(heading)})"

        if "#" in target:
            note, heading = target.split("#", 1)
            note = note.strip()
            heading = heading.strip()
            text = label or heading or note
            base = resolve_note(note)
            if base:
                return f"[{text}]({base}#{heading_anchor(heading)})"

            print(
                f"[warning] Jekyll post not found for wikilink: [[{raw}]]. "
                f"Leaving it as plain text.",
                file=sys.stderr,
            )
            return text

        note = target.strip()
        text = label or re.sub(r"\.(md|markdown)$", "", note, flags=re.I)
        base = resolve_note(note)

        if base:
            return f"[{text}]({base})"

        print(
            f"[warning] Jekyll post not found for wikilink: [[{raw}]]. "
            f"Leaving it as plain text.",
            file=sys.stderr,
        )
        return text

    return pattern.sub(replace, body)


def is_fence(line: str) -> bool:
    return bool(re.match(r"^\s*(```+|~~~+)", line))


def blockquote_payload(line: str):
    """
    Returns (prefix, payload). Example:
    '> hello' -> ('> ', 'hello')
    '>> hi'   -> ('>> ', 'hi')
    """
    m = re.match(r"^(\s*(?:>\s*)+)(.*)$", line)
    if not m:
        return "", line
    return m.group(1), m.group(2)


def is_structural_payload(payload: str) -> bool:
    s = payload.lstrip()

    if not s:
        return True
    if re.match(r"^#{1,6}\s+", s):
        return True
    if re.match(r"^([-+*]|\d+[.)])\s+", s):
        return True
    if re.match(r"^[-*_]{3,}\s*$", s):
        return True
    if s.startswith("{:"):
        return True
    if s.startswith("|"):
        return True
    if re.match(r"^\[[^\]]+\]:\s+", s):
        return True
    if re.match(r"^<[/!?A-Za-z]", s):
        return True
    if s.startswith("$$"):
        return True
    if s.startswith("```") or s.startswith("~~~"):
        return True
    return False


def is_tableish(line: str) -> bool:
    _, payload = blockquote_payload(line)
    s = payload.strip()
    if not s:
        return False
    if s.startswith("|") and s.endswith("|"):
        return True
    if re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", s):
        return True
    return False


def add_obsidian_hard_breaks(body: str) -> str:
    """
    Add Markdown's two-space hard break to prose lines where Obsidian users
    commonly expect a single Enter to remain visually visible.

    Intentionally skips:
    - fenced code blocks
    - $$ math blocks
    - tables
    - headings
    - list items
    - HTML
    - Chirpy IAL/prompt attribute lines
    - blank lines

    For blockquotes (including converted Chirpy prompts), plain content lines
    can receive the two-space break safely.
    """
    lines = body.splitlines()
    out = []

    in_fence = False
    fence_token = None
    in_math = False

    for idx, line in enumerate(lines):
        stripped = line.strip()

        # Fenced code tracking
        fence_match = re.match(r"^\s*(```+|~~~+)", line)
        if fence_match:
            token = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_token = token[0]
            elif token[0] == fence_token:
                in_fence = False
                fence_token = None
            out.append(line)
            continue

        if in_fence:
            out.append(line)
            continue

        # Display math tracking for lines beginning/ending with $$
        if stripped.startswith("$$"):
            if stripped.count("$$") == 1:
                in_math = not in_math
            out.append(line)
            continue

        if in_math:
            out.append(line)
            continue

        if not stripped or line.endswith("  "):
            out.append(line)
            continue

        # Last line never needs a forced break.
        if idx + 1 >= len(lines):
            out.append(line)
            continue

        next_line = lines[idx + 1]
        if not next_line.strip():
            out.append(line)
            continue

        # Tables rely on physical lines as rows.
        if is_tableish(line) or is_tableish(next_line):
            out.append(line)
            continue

        prefix, payload = blockquote_payload(line)
        next_prefix, next_payload = blockquote_payload(next_line)

        if is_structural_payload(payload):
            out.append(line)
            continue

        # If the next line begins a new Markdown block, don't force a <br>.
        if is_structural_payload(next_payload):
            out.append(line)
            continue

        # Avoid interfering with indented code / nested structural blocks.
        if re.match(r"^\s{4,}\S", line) and not prefix:
            out.append(line)
            continue

        out.append(line + "  ")

    return "\n".join(out)


def build_output(
    note: Path,
    vault: Path,
    jekyll: Path,
    image_dest_rel: Path,
    hard_breaks: bool,
    dry_run: bool,
):
    raw = note.read_text(encoding="utf-8-sig")
    front, body = split_front_matter(raw)

    body = transform_callouts(body)
    body = transform_images(body, vault, jekyll, image_dest_rel, dry_run=dry_run)
    body = transform_wikilinks(body, jekyll)

    if hard_breaks:
        body = add_obsidian_hard_breaks(body)

    fallback_title = note.stem
    front, title, post_date = ensure_front_matter(front, body, fallback_title)

    result = f"---\n{front}\n---\n\n{body.rstrip()}\n"
    output_name = f"{post_date}-{slugify(title)}.md"
    output_path = jekyll / "_posts" / output_name

    return result, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert an Obsidian note to a Jekyll Chirpy post."
    )
    parser.add_argument("note", type=Path, help="Obsidian .md note")
    parser.add_argument("--vault", type=Path, required=True, help="Obsidian vault root")
    parser.add_argument("--jekyll", type=Path, required=True, help="Jekyll/Chirpy repo root")
    parser.add_argument(
        "--image-dest",
        type=Path,
        default=Path("assets/img"),
        help="Image destination relative to Jekyll repo (default: assets/img)",
    )
    parser.add_argument(
        "--no-hard-breaks",
        action="store_true",
        help="Do not convert Obsidian Enter line breaks to Markdown two-space hard breaks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print converted Markdown without writing files",
    )

    args = parser.parse_args()

    note = args.note.expanduser().resolve()
    vault = args.vault.expanduser().resolve()
    jekyll = args.jekyll.expanduser().resolve()

    if not note.is_file():
        parser.error(f"Note not found: {note}")
    if not vault.is_dir():
        parser.error(f"Vault not found: {vault}")
    if not jekyll.is_dir():
        parser.error(f"Jekyll repo not found: {jekyll}")

    result, output_path = build_output(
        note=note,
        vault=vault,
        jekyll=jekyll,
        image_dest_rel=args.image_dest,
        hard_breaks=not args.no_hard_breaks,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(result)
        print(f"\n[dry-run target] {output_path}", file=sys.stderr)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding="utf-8")
    print(f"Created: {output_path}")


if __name__ == "__main__":
    main()
