# Ground Truth HTR & Trilingual Search Corpus: *Mat̤laʿ al-ʿajāʾib* (Lucknow: Munshi Naval Kishore Press, 1873)

**Target Repository:** `FID4SA@heiDATA` — Heidelberg University Library (Specialised Information Service for South Asian Studies)  
**Source Shelfmark / Digital Edition:** [`digi.ub.uni-heidelberg.de/diglit/aminaddinhan1873`](https://digi.ub.uni-heidelberg.de/diglit/aminaddinhan1873)  
**Author / Compiler of 1873 Edition:** Amīn al-Dīn Khān  
**Publisher:** Munshi Naval Kishore Press, Lucknow (1873 / 1290 AH)  
**Dataset Author:** Syed Faizyab Mujtaba (Lucknow, India)  
**License:** CC0 1.0 Universal (Public Domain Dedication)

---

## 1. Corpus Overview & Statistics
This dataset provides complete page-level **Urdu *Nastaliq* Ground Truth transcriptions**, **searchable Romanized Urdu transliterations**, **English scholarly summaries**, and **historical Named-Entity indices** for Heidelberg University Library's digitized 1873 Lucknow lithograph *Mat̤laʿ al-ʿajāʾib* (`aminaddinhan1873`).

* **Total IIIF Scans Processed:** 309
* **Verified Lithograph Text Pages (with Urdu Ground Truth `.txt`):** 307
* **Non-Text Paratext Scans Filtered (Covers / Color Charts / Blank Flyleaves):** 2 (Scans: [2, 307])
* **Total Transcribed Urdu *Nastaliq* Lines:** 2,121
* **Extracted Historical & Cosmographical Named Entities:** 1,330

## 2. Directory Structure
* `ground_truth_txt/` — Pure line-by-line UTF-8 Urdu *Nastaliq* transcriptions (`aminaddinhan1873_pXXXX_gt.txt`) ready for HTR model training (Transkribus / PyLaia / Kraken). Non-text flyleaves and color charts are excluded from this folder to prevent HTR training contamination.
* `pages_json/` — Individual structured JSON files for all 309 scans containing `urdu_ground_truth_text`, `roman_urdu_transliteration`, `english_page_summary`, `historical_entities`, and direct Heidelberg IIIF image links.
* `aminaddinhan1873_corpus.jsonl` — Master JSON Lines corpus for bulk NLP and digital humanities analysis.
* `index.html` — Zero-dependency interactive split-screen IIIF + trilingual search portal.
