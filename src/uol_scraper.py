from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

SCORE_RE = re.compile(r"(?:Redação corrigida|NOTA|Nota final)\s*[:]?\s*(\d+(?:[.,]\d+)?)", re.I)
FINAL_SCORE_RE = re.compile(r"Nota final\s*(\d+(?:[.,]\d+)?)", re.I)
COLUMNS = ["essay", "score", "prompt", "source_url", "source_name"]

def allowed_by_robots(url: str, user_agent: str) -> bool:
    p = urlparse(url)
    rp = RobotFileParser(f"{p.scheme}://{p.netloc}/robots.txt")
    try:
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return False

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_score(soup: BeautifulSoup) -> float | None:
    text = clean(soup.get_text(" ", strip=True))
    match = FINAL_SCORE_RE.search(text) or SCORE_RE.search(text)
    return float(match.group(1).replace(",", ".")) if match else None

def extract_prompt(soup: BeautifulSoup) -> str:
    text = clean(soup.get_text(" ", strip=True))
    match = re.search(r"Tema:\s*(.*?)(?:Redação corrigida|NOTA|Inconsistente)", text, re.I)
    return match.group(1).strip(" -:") if match else ""

def extract_essay(soup: BeautifulSoup) -> str:
    for bad in soup.select("script, style, nav, header, footer, aside"):
        bad.decompose()
    candidates = soup.select("main article, article, [role='main']")
    root = max(candidates, key=lambda n: len(n.get_text(" ", strip=True)), default=soup)

    stop_markers = (
        "Comentário geral", "Aspectos pontuais",
        "Competências avaliadas", "Redações corrigidas"
    )
    paragraphs = []
    for node in root.find_all("p"):
        text = clean(node.get_text(" ", strip=True))
        if len(text) < 20:
            continue
        if any(marker.lower() in text.lower() for marker in stop_markers):
            continue
        paragraphs.append(text)

    unique = []
    seen = set()
    for paragraph in paragraphs:
        key = paragraph.lower()
        if key not in seen:
            seen.add(key)
            unique.append(paragraph)
    return "\n\n".join(unique)

def scrape_pages(urls: list[str], delay: float, timeout: int, user_agent: str) -> pd.DataFrame:
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    rows = []

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        if not allowed_by_robots(url, user_agent):
            print("  SKIP: robots.txt disallows this URL")
            continue
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            essay = extract_essay(soup)
            score = extract_score(soup)
            prompt = extract_prompt(soup)
            if essay and score is not None:
                rows.append({
                    "essay": essay,
                    "score": score,
                    "prompt": prompt,
                    "source_url": url,
                    "source_name": "UOL Banco de Redações",
                })
                print(f"  OK: score={score:g}, words={len(essay.split()):,}")
            else:
                print(f"  SKIP: essay={bool(essay)}, score={score}")
        except requests.RequestException as exc:
            print(f"  SKIP: request failed: {exc}")
        if i < len(urls):
            time.sleep(delay)

    return pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("essay").reset_index(drop=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape scored essays from UOL Banco de Redações.")
    parser.add_argument("--urls", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--confirm-educational-use", action="store_true")
    args = parser.parse_args()

    if not args.confirm_educational_use:
        raise SystemExit("Use --confirm-educational-use only for the educational/non-commercial use permitted by the source's stated terms.")

    urls = [
        line.strip()
        for line in Path(args.urls).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    df = scrape_pages(urls, args.delay, args.timeout, "AESResearchBot/1.0")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} rows to {output}")

if __name__ == "__main__":
    main()
