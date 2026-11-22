import asyncio
import json
import os
import re
import time
import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel

MANIFEST_URL = "https://collections.library.yale.edu/manifests/32496252"
CATALOG_URL = "https://collections.library.yale.edu/catalog/32496252"
CORPUS_PATH = "corpus.json"
START_SCAN = 1
END_SCAN = 999
MIN_EXPECTED_SCANS = 20
FLAGSHIP_BATCH_SIZE = 4
SPILLOVER_ON_503 = False

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, application/ld+json, image/*, */*",
    "Referer": "https://collections.library.yale.edu/",
}


def load_all_keys() -> list[str]:
    found = []
    if os.environ.get("GEMINI_API_KEY", "").strip():
        found.append((0, os.environ["GEMINI_API_KEY"].strip()))
    for k, v in os.environ.items():
        m = re.fullmatch(r"GEMINI_API_KEY_(\d+)", k)
        if m and v.strip():
            found.append((int(m.group(1)), v.strip()))
    found.sort(key=lambda x: x[0])
    return list(dict.fromkeys([val for _, val in found]))


API_KEYS = load_all_keys()
if not API_KEYS:
    raise RuntimeError("No API keys found! Set GEMINI_API_KEY_1..9 in your environment.")

TIER1_PATTERNS = [
    "gemini-4-hydrogen",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3-flash",
    "gemini-2.5-flash",
]
TIER2_PATTERNS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]
TIER3_PATTERNS = [
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
]
EXCLUDED_BASE = ("tts", "image", "audio", "live", "transcribe", "embedding")


class FolioRecord(BaseModel):
    page_number: int
    folio_label: str
    page_type: str  # "cover_or_flyleaf", "title_or_colophon", "manuscript_text", "diagram_or_ornament"
    contains_illumination: bool
    artwork_description: str
    original_script_text: str
    roman_transliteration: str
    english_page_summary: str
    historical_entities: list[str]
    yale_shelfmark: str
    image_url: str


class BatchFolioResponse(BaseModel):
    records: list[FolioRecord]


def discover_active_models(api_key: str) -> tuple[list[str], list[str], list[str]]:
    client = genai.Client(api_key=api_key)
    available = []
    for m in client.models.list():
        name = m.name.replace("models/", "")
        actions = getattr(m, "supported_actions", []) or []
        if "generateContent" in actions or not actions:
            available.append(name)

    tier1, tier2, tier3 = [], [], []
    for pat in TIER1_PATTERNS:
        matches = sorted([n for n in available if pat in n and "lite" not in n and not any(b in n for b in EXCLUDED_BASE)])
        if matches:
            tier1.append(matches[0])
    for pat in TIER2_PATTERNS:
        matches = sorted([n for n in available if pat in n and not any(b in n for b in EXCLUDED_BASE)])
        if matches:
            tier2.append(matches[0])
    for pat in TIER3_PATTERNS:
        matches = sorted([n for n in available if pat in n])
        if matches:
            tier3.append(matches[0])

    return (
        tier1 or ["gemini-2.5-flash"],
        tier2 or ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        tier3 or ["gemma-4-31b-it", "gemma-4-26b-a4b-it"],
    )


class QuotaRouter:
    def __init__(self, keys: list[str], tier1: list[str], tier2: list[str], tier3: list[str]):
        self.keys = keys
        self.tiers = {1: tier1, 2: tier2, 3: tier3}
        self.clients = {i: genai.Client(api_key=k) for i, k in enumerate(keys)}
        self.dead_pairs: set[tuple[int, str]] = set()
        self.pair_cooldown: dict[tuple[int, str], float] = {}
        self.model_cooldown: dict[str, float] = {}
        self.rr_cursor = {1: 0, 2: 0, 3: 0}
        self.announced_tier = 1

    def _tier_pairs(self, tier_num: int) -> list[tuple[int, str]]:
        return [(k_idx, m) for m in self.tiers[tier_num] for k_idx in range(len(self.keys))]

    async def acquire_slot(self) -> tuple[int, int, str, genai.Client]:
        while True:
            now = time.time()
            any_alive_globally = False

            for tier_num in (1, 2, 3):
                all_pairs = self._tier_pairs(tier_num)
                live_pairs = [p for p in all_pairs if p not in self.dead_pairs]
                if not live_pairs:
                    continue

                any_alive_globally = True
                if tier_num > self.announced_tier and all(
                    p in self.dead_pairs
                    for prev in range(1, tier_num)
                    for p in self._tier_pairs(prev)
                ):
                    self.announced_tier = tier_num
                    print(f"[!] Previous tier RPD exhausted -> unlocking Tier-{tier_num} pool ({self.tiers[tier_num]})")

                n = len(all_pairs)
                cursor = self.rr_cursor[tier_num]
                for offset in range(n):
                    k_idx, model = all_pairs[(cursor + offset) % n]
                    pair = (k_idx, model)
                    if pair in self.dead_pairs or now < self.model_cooldown.get(model, 0.0):
                        continue
                    if now >= self.pair_cooldown.get(pair, 0.0):
                        self.rr_cursor[tier_num] = (cursor + offset + 1) % n
                        pace = 13.0 if tier_num == 1 else (6.5 if "2.5" in model else 4.5) if tier_num == 2 else 8.0
                        self.pair_cooldown[pair] = now + pace
                        return tier_num, k_idx, model, self.clients[k_idx]

                all_503_paused = all(now < self.model_cooldown.get(m, 0.0) for _, m in live_pairs)
                if not (SPILLOVER_ON_503 and all_503_paused):
                    break

            if not any_alive_globally:
                raise RuntimeError("All keys x all 3 tiers hit daily RPD quota! Resume tomorrow.")

            await asyncio.sleep(1.0)

    def report_error(self, key_idx: int, model: str, err: Exception):
        msg = str(err)
        pair = (key_idx, model)
        if "PerDay" in msg or "RPD" in msg or ("quota" in msg.lower() and "day" in msg.lower()):
            self.dead_pairs.add(pair)
            print(f"[X] Retired Key#{key_idx+1} + {model} (Daily RPD reached)")
        elif "404" in msg or "not found" in msg.lower():
            self.dead_pairs.add(pair)
            print(f"[X] Retired {model} (Model ID unavailable)")
        elif "503" in msg or "UNAVAILABLE" in msg or "high demand" in msg.lower():
            if time.time() >= self.model_cooldown.get(model, 0.0):
                print(f"[!] Model {model} overloaded (503) -> pausing {model} for 45s across all keys")
            self.model_cooldown[model] = time.time() + 45.0
        else:
            self.pair_cooldown[pair] = time.time() + 30.0
            print(f"[!] Cooldown 30s on Key#{key_idx+1} + {model}: {msg[:110]}")


def atomic_save(records: list[dict]):
    tmp = CORPUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CORPUS_PATH)


def extract_label(raw_label, fallback: str) -> str:
    if isinstance(raw_label, str) and raw_label.strip():
        return raw_label.strip()
    if isinstance(raw_label, list) and raw_label:
        first = raw_label[0]
        if isinstance(first, dict):
            return str(first.get("@value") or first.get("value") or fallback).strip()
        return str(first).strip()
    if isinstance(raw_label, dict):
        for vals in raw_label.values():
            if isinstance(vals, list) and vals:
                return str(vals[0]).strip()
            if isinstance(vals, str) and vals.strip():
                return vals.strip()
    return fallback


def extract_manifest_context(manifest: dict) -> tuple[str, str]:
    title = extract_label(manifest.get("label"), "Yale Beinecke Codex (OID 32496252)")
    meta_lines = []
    for entry in manifest.get("metadata", []):
        if isinstance(entry, dict):
            k = extract_label(entry.get("label"), "")
            v = extract_label(entry.get("value"), "")
            if k and v:
                meta_lines.append(f"{k}: {v}")
    return title, " | ".join(meta_lines[:10])


def build_candidate_urls(service_id: str | None, direct_id: str | None) -> list[str]:
    urls = []
    if service_id and service_id.startswith("http"):
        base = service_id.rstrip("/")
        for suffix in (
            "/full/!1500,1500/0/default.jpg",
            "/full/1500,/0/default.jpg",
            "/full/max/0/default.jpg",
            "/full/full/0/default.jpg",
        ):
            u = base + suffix
            if u not in urls:
                urls.append(u)
    if direct_id and direct_id.startswith("http") and direct_id not in urls:
        urls.append(direct_id)
    return urls


def fetch_yale_canvases(manifest_url: str) -> tuple[str, str, list[dict]]:
    resp = httpx.get(manifest_url, headers=HEADERS, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    manifest = resp.json()
    title, meta_summary = extract_manifest_context(manifest)
    canvases = []

    if isinstance(manifest.get("items"), list):
        for idx, canvas in enumerate(manifest["items"], start=1):
            label = extract_label(canvas.get("label"), f"Scan {idx}")
            try:
                body = canvas["items"][0]["items"][0]["body"]
                if isinstance(body, list):
                    body = body[0]
                service = body.get("service", {})
                if isinstance(service, list):
                    service = service[0] if service else {}
                service_id = service.get("@id") or service.get("id")
                direct_id = body.get("id") or body.get("@id")
                cands = build_candidate_urls(service_id, direct_id)
                if cands:
                    canvases.append({
                        "page_number": idx,
                        "folio_label": label,
                        "service_id": service_id,
                        "candidates": cands,
                        "image_url": cands[0],
                    })
            except (KeyError, IndexError, TypeError):
                continue

    elif isinstance(manifest.get("sequences"), list) and manifest["sequences"]:
        for idx, canvas in enumerate(manifest["sequences"][0].get("canvases", []), start=1):
            label = extract_label(canvas.get("label"), f"Scan {idx}")
            try:
                resource = canvas["images"][0]["resource"]
                service = resource.get("service", {})
                if isinstance(service, list):
                    service = service[0] if service else {}
                service_id = service.get("@id") or service.get("id")
                direct_id = resource.get("@id") or resource.get("id")
                cands = build_candidate_urls(service_id, direct_id)
                if cands:
                    canvases.append({
                        "page_number": idx,
                        "folio_label": label,
                        "service_id": service_id,
                        "candidates": cands,
                        "image_url": cands[0],
                    })
            except (KeyError, IndexError, TypeError):
                continue

    if len(canvases) < MIN_EXPECTED_SCANS:
        raise RuntimeError(f"Aborting before API spend: manifest only contains {len(canvases)} scans.")

    return title, meta_summary, canvases


def negotiate_working_url_style(canvases: list[dict]) -> list[dict]:
    first = canvases[0]
    working_suffix = None
    for u in first["candidates"]:
        try:
            r = httpx.get(u, headers=HEADERS, follow_redirects=True, timeout=20.0)
            ctype = r.headers.get("content-type", "")
            if r.status_code == 200 and len(r.content) > 1000 and ("image" in ctype or r.content[:2] == b"\xff\xd8"):
                print(f"[IIIF] Verified working image URL (HTTP 200, {len(r.content)//1024} KB): {u}")
                if first.get("service_id") and u.startswith(first["service_id"].rstrip("/")):
                    working_suffix = u[len(first["service_id"].rstrip("/")):]
                break
        except Exception:
            pass

    if working_suffix:
        for c in canvases:
            if c.get("service_id"):
                c["image_url"] = c["service_id"].rstrip("/") + working_suffix
                c["candidates"] = [c["image_url"]] + [x for x in c["candidates"] if x != c["image_url"]]
    return canvases


def parse_json_safe(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def transcribe_batch_sync(
    client: genai.Client,
    model_name: str,
    codex_title: str,
    codex_meta: str,
    batch_items: list[tuple[dict, bytes]],
) -> dict:
    scan_list = ", ".join(f"Scan #{item['page_number']} ({item['folio_label']})" for item, _ in batch_items)
    prompt = (
        f"You are an expert paleographer and manuscript historian transcribing {len(batch_items)} consecutive scan(s) "
        f"[{scan_list}] of Yale University, Beinecke Rare Book & Manuscript Library codex '{codex_title}' "
        f"(OID 32496252; Catalog metadata: {codex_meta}).\n\n"
        "Instructions:\n"
        f"1. Return a JSON object with `records` containing EXACTLY {len(batch_items)} FolioRecord objects—one for each scan above, "
        "preserving each scan's exact integer `page_number` and `folio_label`.\n"
        "2. Use consecutive scans to preserve textual continuity, catchwords, and marginalia across recto/verso turnovers.\n"
        "3. If a scan is a leather cover, marbled endpaper, blank flyleaf, or library color target, set `page_type` to "
        "'cover_or_flyleaf' and leave `original_script_text` and `roman_transliteration` empty.\n"
        "4. For all inscribed folios, transcribe EVERY visible line of main text, rubricated headings, marginal glosses/notes, "
        "ownership seals, and colophons verbatim in its original script (Arabic, Persian, Ottoman Turkish, or Latin) "
        "with exact '\\n' line breaks in `original_script_text`.\n"
        "5. Provide a COMPLETE line-by-line `roman_transliteration` using standard academic ALA-LC / IJMES diacritics "
        "(ā, ī, ū, ḥ, kh, dh, ṣ, ḍ, ṭ, ẓ, ʿ, gh, ʾ) for non-Latin scripts — never truncate or summarize in brackets.\n"
        "6. If an illuminated headpiece, decorative border, seal, stamp, or diagram appears, set `contains_illumination` to true "
        "(and `page_type` to 'diagram_or_ornament' or 'title_or_colophon') and describe it in `artwork_description`.\n"
        "7. Write a concise scholarly `english_page_summary` explaining the contents of the folio, and list `historical_entities` "
        "(authors, scribes, owners, historical figures, concepts, and places)."
    )

    contents: list = []
    for item, img_bytes in batch_items:
        contents.append(f"=== SCAN #{item['page_number']} | FOLIO LABEL: {item['folio_label']} ===")
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
    contents.append(prompt)

    res = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchFolioResponse,
            temperature=0.1,
        ),
    )
    return parse_json_safe(res.text)


async def main():
    print(f"[BOOT] Loaded {len(API_KEYS)} API keys.")
    tier1, tier2, tier3 = discover_active_models(API_KEYS[0])
    print(f"[MODELS] Tier-1 Flagship   (4-page batch | {len(tier1)*len(API_KEYS)*20*FLAGSHIP_BATCH_SIZE} pages/day): {tier1}")
    print(f"[MODELS] Tier-2 Flash-Lite (1-page mode  | bulk high-RPD): {tier2}")
    print(f"[MODELS] Tier-3 Gemma 4    (1-page mode  | {len(tier3)*len(API_KEYS)*14400} pages/day): {tier3}")

    router = QuotaRouter(API_KEYS, tier1, tier2, tier3)
    save_lock = asyncio.Lock()
    dl_sem = asyncio.Semaphore(6)

    existing = {}
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            for item in json.load(f):
                existing[item["page_number"]] = item

    codex_title, codex_meta, canvases = fetch_yale_canvases(MANIFEST_URL)
    canvases = negotiate_working_url_style(canvases)
    target_canvases = [c for c in canvases if START_SCAN <= c["page_number"] <= END_SCAN]
    todo = [c for c in target_canvases if c["page_number"] not in existing]
    print(f"[YALE] {codex_title}")
    if codex_meta:
        print(f"[META] {codex_meta}")
    print(f"[YALE] OID 32496252 | Total scans: {len(canvases)} | Saved: {len(existing)} | Remaining: {len(todo)}")

    queue: asyncio.Queue = asyncio.Queue()
    for i in range(0, len(todo), FLAGSHIP_BATCH_SIZE):
        queue.put_nowait(todo[i : i + FLAGSHIP_BATCH_SIZE])

    timeout = httpx.Timeout(connect=25.0, read=60.0, write=25.0, pool=30.0)
    async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS, timeout=timeout) as http_client:
        async def download_image(item: dict) -> bytes:
            last_err = ""
            for attempt in range(1, 5):
                for url in item["candidates"]:
                    try:
                        async with dl_sem:
                            resp = await http_client.get(url)
                        if resp.status_code == 200 and len(resp.content) > 1000:
                            item["image_url"] = url
                            return resp.content
                        last_err = f"HTTP {resp.status_code} on {url}"
                    except Exception as e:
                        last_err = f"{type(e).__name__}: {e} on {url}"
                await asyncio.sleep(2 * attempt)
            raise RuntimeError(f"Failed downloading Scan {item['page_number']} | Last error: {last_err}")

        async def worker():
            while True:
                try:
                    batch_meta = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                try:
                    img_bytes_list = await asyncio.gather(*(download_image(c) for c in batch_meta))
                    pending = list(zip(batch_meta, img_bytes_list))

                    while pending:
                        tier_num, k_idx, model_name, client = await router.acquire_slot()
                        chunk = pending[:FLAGSHIP_BATCH_SIZE] if tier_num == 1 else pending[:1]

                        try:
                            parsed = await asyncio.to_thread(
                                transcribe_batch_sync, client, model_name, codex_title, codex_meta, chunk
                            )
                            records = parsed.get("records", [])
                            chunk_by_num = {item["page_number"]: item for item, _ in chunk}
                            saved_nums = set()

                            async with save_lock:
                                for rec in records:
                                    p_num = rec.get("page_number")
                                    if p_num in chunk_by_num:
                                        meta = chunk_by_num[p_num]
                                        rec["page_number"] = p_num
                                        rec["folio_label"] = meta["folio_label"]
                                        rec["yale_shelfmark"] = f"Yale University, Beinecke Rare Book & Manuscript Library (OID 32496252 — {codex_title})"
                                        rec["image_url"] = meta["image_url"]
                                        existing[p_num] = rec
                                        saved_nums.add(p_num)
                                if saved_nums:
                                    atomic_save([existing[k] for k in sorted(existing.keys())])
                                done_count = len(existing)

                            if not saved_nums:
                                raise ValueError(f"Model returned 0 matching page numbers for {sorted(chunk_by_num.keys())}")

                            art_pages = [n for n in sorted(saved_nums) if existing[n].get("contains_illumination")]
                            art_tag = f" [ILLUM/SEAL: {art_pages}]" if art_pages else ""
                            print(
                                f"[✓] T{tier_num} Scans {sorted(saved_nums)} ({done_count}/{len(target_canvases)} saved)"
                                f"{art_tag} | Key#{k_idx+1} -> {model_name}"
                            )
                            pending = [(item, b) for item, b in pending if item["page_number"] not in saved_nums]
                        except Exception as e:
                            router.report_error(k_idx, model_name, e)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(len(API_KEYS))]
        await asyncio.gather(*workers)

    expected_nums = {c["page_number"] for c in target_canvases}
    missing = sorted(expected_nums - set(existing.keys()))
    print(f"[DONE] Saved {len(existing)}/{len(target_canvases)} folios to {CORPUS_PATH} | Missing: {missing}")


if __name__ == "__main__":
    asyncio.run(main())