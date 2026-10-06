from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup


SCORE_RE = re.compile(r"(?:overall\s*band|band\s*score|score)\s*[:\-]?\s*(\d+(?:\.5)?)", re.I)


def robots_allowed(url: str, user_agent: str = "AESResearchBot/1.0") -> bool:
    parsed = requests.utils.urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser(robots_url)
    try:
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return False


def text_from_selector(soup: BeautifulSoup, selector: str) -> str:
    node = soup.select_one(selector)
    return node.get_text(" ", strip=True) if node else ""


def score_from_text(text: str) -> float | None:
    match = SCORE_RE.search(text)
    return float(match.group(1)) if match else None


def scrape_urls(
    urls: list[str],
    essay_selector: str,
    score_selector: str,
    prompt_selector: str | None = None,
    source_name: str = "",
    delay: float = 1.5,
    timeout: int = 20,
    user_agent: str = "AESResearchBot/1.0",
) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    rows: list[dict[str, object]] = []

    for url in urls:
        if not robots_allowed(url, user_agent):
            raise PermissionError(
                f"robots.txt does not permit fetching {url} with user agent {user_agent!r}."
            )

        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        essay = text_from_selector(soup, essay_selector)
        score_text = text_from_selector(soup, score_selector)
        prompt = text_from_selector(soup, prompt_selector) if prompt_selector else ""

        score = score_from_text(score_text)
        if essay and score is not None:
            rows.append(
                {
                    "essay": essay,
                    "score": score,
                    "prompt": prompt,
                    "source_url": url,
                    "source_name": source_name,
                }
            )

        time.sleep(delay)

    return pd.DataFrame(rows, columns=["essay", "score", "prompt", "source_url", "source_name"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape scored essays from a source you are authorized to scrape."
    )
    parser.add_argument("--urls", required=True, help="Text file containing one URL per line")
    parser.add_argument("--essay-selector", required=True)
    parser.add_argument("--score-selector", required=True)
    parser.add_argument("--prompt-selector")
    parser.add_argument("--source-name", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument(
        "--confirm-permission",
        action="store_true",
        help="Confirm that you have permission to retrieve and use this source.",
    )
    args = parser.parse_args()

    if not args.confirm_permission:
        raise SystemExit(
            "Refusing to scrape without --confirm-permission. "
            "Only scrape sources that permit your intended use."
        )

    urls = [
        line.strip()
        for line in Path(args.urls).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    df = scrape_urls(
        urls=urls,
        essay_selector=args.essay_selector,
        score_selector=args.score_selector,
        prompt_selector=args.prompt_selector,
        source_name=args.source_name,
        delay=args.delay,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} scored essays to {output}")


if __name__ == "__main__":
    main()
