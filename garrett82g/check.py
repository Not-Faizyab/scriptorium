import json
import zipfile

with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = sorted(json.load(f), key=lambda x: x["page_number"])

# 1. Build JSONL for HuggingFace / NLP pipelines
with open("garrett82g_corpus.jsonl", "w", encoding="utf-8") as jf:
    for rec in corpus:
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

# 2. Build zipped per-folio Ground Truth .txt pack cleanly in memory
with zipfile.ZipFile("garrett82g_gt_txt.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for rec in corpus:
        text = rec.get("persian_ground_truth_text", "").strip()
        if not text:
            continue
        p_num = rec["page_number"]
        header = f"# Scan {p_num:03d} | {rec.get('folio_label', '')} | Princeton Garrett no. 82G\n\n"
        zf.writestr(f"scan_{p_num:03d}.txt", header + text + "\n")

# 3. Build scholarly README.md
readme = """# Project Scriptorium — Volume II: *ʿAjāʾib al-makhlūqāt wa-gharāʾib al-mawjūdāt* (`Garrett no. 82G`)

- **Author:** Zakarīyā ibn Muḥammad al-Qazwīnī (d. 1283)
- **Origin / Date:** 18th-century India (colophon signed *ʿAbd Allāh ibn ʿAlī Bey Damāvandī*)
- **Holding Institution:** Princeton University Library, Special Collections (`Islamic Manuscripts, Garrett no. 82G`, `ark:/88435/h989r3272`)
- **Scope:** 502 IIIF scans (152 illuminated miniatures and astronomical diagrams indexed)
- **License:** CC0 1.0 Public Domain Dedication
- **Live Workstation:** https://project-scriptorium.netlify.app/garrett82g/

## Files Included
- `corpus.json` — Full structured diplomatic, transliteration, summary, entity, and miniature-iconography dataset
- `garrett82g_corpus.jsonl` — Line-delimited JSON for NLP and LLM fine-tuning pipelines
- `garrett82g_complete.txt` — Consolidated single-file UTF-8 diplomatic edition
- `garrett82g_gt_txt.zip` — Page-by-page Persian/Arabic UTF-8 ground-truth text files
"""
with open("README.md", "w", encoding="utf-8") as rf:
    rf.write(readme)

print("[✓] Created garrett82g_corpus.jsonl, garrett82g_gt_txt.zip, and README.md!")