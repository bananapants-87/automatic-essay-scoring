# Web scraping workflow

## Selected source

The UOL Banco de Redações publishes individual student essays together with human-assigned scores. Its essay pages state that reproduction is permitted for school work, without commercial use, with credit to UOL and the authors.

This repository uses a small educational/non-commercial scraping demonstration.

## URLs

See urls.uol.txt.

## Extraction

src/uol_scraper.py:
1. requests the individual HTML page;
2. checks robots.txt;
3. parses HTML with BeautifulSoup;
4. extracts essay paragraphs from the main article;
5. extracts the displayed score;
6. extracts the topic when present;
7. preserves source URL and source name;
8. writes a CSV compatible with the existing AES model pipeline.

The scraper deliberately avoids pretending that generic selectors such as .essay and .score are known to exist on UOL. It uses semantic article structure and text anchors instead.

## Run

PowerShell:

.venv\Scripts\python.exe src\uol_scraper.py --urls urls.uol.txt --output data\scraped\uol_essays.csv --confirm-educational-use

Then:

.venv\Scripts\python.exe src\compare_models.py --data data\scraped\uol_essays.csv

## Important

UOL has used different scoring scales across historical rubric periods. Do not combine pages with incompatible scales without normalization. Start with pages from one rubric period and inspect the resulting CSV before training.

The project therefore keeps the raw displayed score rather than silently transforming it.
