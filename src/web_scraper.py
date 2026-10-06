from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

SCORE_RE = re.compile(
    r"(?:overall\\s*band|band\\s*score|score)\\s*[:\\-]?\\s*(\\d+(?:\\.5)?)",
    re.I,
)

COLUMNS = ["essay", "score", "prompt", "source_url", "source_name"]


def robots_allowed(url: str, user_agent: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        rp = RobotFileParser(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return False


def select_text(soup: BeautifulSoup, selector: str | None) -> str:
    if not selector:
        return ""
    node = soup.select_one(selector)
    return node.get_text(" ", strip=True) if node else ""


def extract_score(text: str) -> float | None:
    match = SCORE_RE.search(text)
    return float(match.group(1)) if match else None


def scrape(
    urls: list[str],
    essay_selector: str,
    score_selector: str,
    prompt_selector: str | None,
    source_name: str,
    delay: float,
    timeout: int,
    user_agent: str,
) -> pd.DataFrame:
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    rows: list[dict[str, object]] = []

    for index, url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] {url}")

        if not robots_allowed(url, user_agent):
            print("  SKIP: robots.txt does not permit this URL.")
            continue

        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"  SKIP: request failed: {exc}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        essay = select_text(soup, essay_selector)
        score_text = select_text(soup, score_selector)
        prompt = select_text(soup, prompt_selector)
        score = extract_score(score_text)

        if not essay:
            print("  SKIP: essay selector returned no text.")
        elif score is None:
            print("  SKIP: no score could be extracted.")
        else:
            rows.append(
                {
                    "essay": essay,
                    "score": score,
                    "prompt": prompt,
                    "source_url": url,
                    "source_name": source_name,
                }
            )
            print(f"  OK: score={score:g}, words={len(essay.split()):,}")

        if index < len(urls):
            time.sleep(max(delay, 0.0))

    return pd.DataFrame(rows, columns=COLUMNS).drop_duplicates(
        subset=["essay"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect scored essays from a source you are authorized to retrieve."
    )
    parser.add_argument("--urls", required=True, help="Text file with one URL per line.")
    parser.add_argument("--essay-selector", required=True)
    parser.add_argument("--score-selector", required=True)
    parser.add_argument("--prompt-selector")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--user-agent",
        default="AESResearchBot/1.0",
    )
    parser.add_argument(
        "--confirm-permission",
        action="store_true",
        help="Confirm that the source permits your intended automated retrieval and use.",
    )
    args = parser.parse_args()

    if not args.confirm_permission:
        raise SystemExit(
            "Refusing to scrape without --confirm-permission. "
            "Use only a source that permits your intended retrieval and use."
        )

    urls = [
        line.strip()
        for line in Path(args.urls).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not urls:
        raise SystemExit("No URLs found in the URL file.")

    df = scrape(
        urls=urls,
        essay_selector=args.essay_selector,
        score_selector=args.score_selector,
        prompt_selector=args.prompt_selector,
        source_name=args.source_name,
        delay=args.delay,
        timeout=args.timeout,
        user_agent=args.user_agent,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\\nSaved {len(df):,} scored essays to {output}")
    if not df.empty:
        print(f"Score range: {df['score'].min():g}-{df['score'].max():g}")
    else:
        print("No rows were collected. Check selectors, permissions, and URLs.")


if __name__ == "__main__":
    main()
