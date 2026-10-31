"""Downloads 5 FastAPI doc pages (from GitHub raw) and cleans them for RAG.

Run:  python -m scripts.fetch_docs

Why is cleaning needed?
  In FastAPI's markdown the code examples come in as placeholders:
      {* ../../docs_src/first_steps/tutorial001_py310.py *}
  If we download the pages as they are, all code snippets disappear. So we replace every
  placeholder with the real code file, inlined. We also remove the special mkdocs syntax
  (/// tip, { #anchor }, <div class="termy">).
"""
import re
import time
from pathlib import Path

import httpx

RAW = "https://raw.githubusercontent.com/fastapi/fastapi/master"
SITE = "https://fastapi.tiangolo.com"
OUT_DIR = Path("data/docs")

# (output file name, path in repo under docs/en/docs, public URL path)
PAGES = [
    ("fastapi-first-steps.md", "tutorial/first-steps.md", "/tutorial/first-steps/"),
    ("fastapi-path-parameters.md", "tutorial/path-params.md", "/tutorial/path-params/"),
    ("fastapi-query-parameters.md", "tutorial/query-params.md", "/tutorial/query-params/"),
    ("fastapi-request-body.md", "tutorial/body.md", "/tutorial/body/"),
    ("fastapi-security-first-steps.md", "tutorial/security/first-steps.md", "/tutorial/security/first-steps/"),
]

INCLUDE_RE = re.compile(r"\{\*\s*(\S+?)(?:\s+(?:hl|ln|title)\[[^\]]*\])*\s*\*\}")
ANCHOR_RE = re.compile(r"\s*\{\s*#[\w-]+\s*\}")
HTML_TAG_RE = re.compile(r"</?(?:div|span|font|img|details|summary|br|u|b|dfn|abbr)\b[^>]*>", re.IGNORECASE)
ADMONITION_RE = re.compile(r"^///\s*(\w+)?\s*(?:\|\s*(.*))?$")


def get(client: httpx.Client, url: str, retries: int = 4) -> str:
    """Download with retries, for slow networks."""
    for attempt in range(1, retries + 1):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as exc:
            if attempt == retries:
                raise RuntimeError(f"Download failed: {url} ({exc})") from exc
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def inline_code(client: httpx.Client, text: str) -> str:
    """Replace {* ../../docs_src/x.py *} with the real code block."""

    def repl(match: re.Match) -> str:
        path = match.group(1)
        if "docs_src/" not in path:
            return ""
        rel = "docs_src/" + path.split("docs_src/", 1)[1]
        try:
            code = get(client, f"{RAW}/{rel}").rstrip()
        except RuntimeError:
            return ""  # if one snippet is missing, the whole page should not fail
        return f"```python\n{code}\n```"

    return INCLUDE_RE.sub(repl, text)


def convert_admonitions(text: str) -> str:
    """'/// tip' -> '**Tip:**', and drop the closing '///'. Nothing inside a code fence is touched."""
    out, in_fence = [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not in_fence:
            m = ADMONITION_RE.match(line.strip())
            if m:
                kind, title = m.group(1), m.group(2)
                if kind:  # opening line
                    label = kind.capitalize() + (f" - {title.strip()}" if title else "")
                    out.append(f"**{label}:**")
                continue  # skip both the opening and the closing '///' line
        out.append(line)
    return "\n".join(out)


def clean(client: httpx.Client, text: str) -> str:
    text = inline_code(client, text)
    text = ANCHOR_RE.sub("", text)
    text = convert_admonitions(text)
    text = HTML_TAG_RE.sub("", text)
    # "**Tip:**\n\nbody" -> "**Tip:** body"  (the label and its text stay in the same chunk)
    text = re.sub(r"^(\*\*[A-Z][^*\n]*:\*\*)\n\n", r"\1 ", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for filename, repo_path, url_path in PAGES:
            print(f"Downloading {repo_path} ...")
            raw = get(client, f"{RAW}/docs/en/docs/{repo_path}")
            body = clean(client, raw)
            header = f"<!-- source: {SITE}{url_path} -->\n"
            (OUT_DIR / filename).write_text(header + body, encoding="utf-8")
            print(f"  saved {OUT_DIR / filename}  ({len(body):,} chars)")
    print(f"\nDone. {len(PAGES)} docs ready in {OUT_DIR}/")


if __name__ == "__main__":
    main()