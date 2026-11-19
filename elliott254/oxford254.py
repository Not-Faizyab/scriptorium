import asyncio
import json
import os
import re
import time
import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel

MANIFEST_URL = "https://iiif.bodleian.ox.ac.uk/iiif/manifest/56911adf-3d32-4a52-a9c6-653eb5bd6cb7.json"
CORPUS_PATH = "corpus.json"
START_SCAN = 1
END_SCAN = 999
FLAGSHIP_BATCH_SIZE = 4   # 4 consecutive scans/call for Tier-1 Flagships
SPILLOVER_ON_503 = False  # Set True to jump into Tier-2 immediately if ALL Tier-1 models hit 503

RAW_KEYS = [
    os.environ.get("GEMINI_API_KEY", ""),
    os.environ.get("GEMINI_API_KEY_1", ""),
    os.environ.get("GEMINI_API_KEY_2", ""),
    os.environ.get("GEMINI_API_KEY_3", ""),
    os.environ.get("GEMINI_API_KEY_4", ""),
    os.environ.get("GEMINI_API_KEY_5", ""),
    os.environ.get("GEMINI_API_KEY_6", ""),
    os.environ.get("GEMINI_API_KEY_7", ""),
    os.environ.get("GEMINI_API_KEY_8", ""),
]
API_KEYS = list(dict.fromkeys([k.strip() for k in RAW_KEYS if k.strip()]))
if not API_KEYS:
    raise RuntimeError("No API keys found! Set GEMINI_API_KEY_1..8 in your environment.")

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
    page_type: str  # "cover_or_flyleaf", "english_preface", "illuminated_frontispiece", "manuscript_text", "mughal_miniature"
    contains_illumination: bool
    artwork_description: str
    persian_ground_truth_text: str
    roman_persian_transliteration: str
    english_page_summary: str
    historical_entities: list[str]
    bodleian_shelfmark: str
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
        matches = sorted([
            n for n in available
            if pat in n and "lite" not in n and not any(bad in n for bad in EXCLUDED_BASE)
        ])
        if matches:
            tier1.append(matches[0])

    for pat in TIER2_PATTERNS:
        matches = sorted([
            n for n in available
            if pat in n and not any(bad in n for bad in EXCLUDED_BASE)
        ])
        if matches:
            tier2.append(matches[0])

    for pat in TIER3_PATTERNS:
        matches = sorted([n for n in available if pat in n])
        if matches:
            tier3.append(matches[0])

    if not tier1:
        tier1 = ["gemini-2.5-flash"]
    if not tier2:
        tier2 = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    if not tier3:
        tier3 = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]

    return tier1, tier2, tier3


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
        """Returns (tier_num, key_idx, model_name, client)."""
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
                    if pair in self.dead_pairs:
                        continue
                    if now < self.model_cooldown.get(model, 0.0):
                        continue
                    if now >= self.pair_cooldown.get(pair, 0.0):
                        self.rr_cursor[tier_num] = (cursor + offset + 1) % n
                        if tier_num == 1:
                            pace = 13.0
                        elif tier_num == 2:
                            pace = 6.5 if "2.5" in model else 4.5
                        else:
                            pace = 8.0
                        self.pair_cooldown[pair] = now + pace
                        return tier_num, k_idx, model, self.clients[k_idx]

                # Stay locked to highest alive tier unless SPILLOVER_ON_503 is enabled and all live models are 503-paused
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
    if isinstance(raw_label, dict):
        for vals in raw_label.values():
            if isinstance(vals, list) and vals:
                return str(vals[0]).strip()
            if isinstance(vals, str) and vals.strip():
                return vals.strip()
    return fallback


def fetch_bodleian_canvases(manifest_url: str, width: int = 1500) -> list[dict]:
    headers = {
        "User-Agent": "Mozilla/5.0 ProjectScriptorium/1.0",
        "Accept": "application/json, application/ld+json",
    }
    resp = httpx.get(manifest_url, headers=headers, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    manifest = resp.json()
    canvases = []

    # IIIF Presentation v2
    if "sequences" in manifest and manifest["sequences"]:
        raw_canvases = manifest["sequences"][0].get("canvases", [])
        for idx, canvas in enumerate(raw_canvases, start=1):
            label = extract_label(canvas.get("label"), f"Scan {idx}")
            try:
                resource = canvas["images"][0]["resource"]
                service = resource.get("service", {})
                if isinstance(service, list):
                    service = service[0] if service else {}
                service_id = service.get("@id") or service.get("id")
                img_url = (
                    f"{service_id.rstrip('/')}/full/{width},/0/default.jpg"
                    if service_id
                    else resource.get("@id") or resource.get("id")
                )
                canvases.append({"page_number": idx, "folio_label": label, "image_url": img_url})
            except (KeyError, IndexError):
                continue

    # IIIF Presentation v3 fallback
    elif "items" in manifest:
        for idx, canvas in enumerate(manifest.get("items", []), start=1):
            label = extract_label(canvas.get("label"), f"Scan {idx}")
            try:
                body = canvas["items"][0]["items"][0]["body"]
                if isinstance(body, list):
                    body = body[0]
                service = body.get("service", {})
                if isinstance(service, list):
                    service = service[0] if service else {}
                service_id = service.get("@id") or service.get("id")
                img_url = (
                    f"{service_id.rstrip('/')}/full/{width},/0/default.jpg"
                    if service_id
                    else body.get("id") or body.get("@id")
                )
                canvases.append({"page_number": idx, "folio_label": label, "image_url": img_url})
            except (KeyError, IndexError):
                continue

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
    batch_items: list[tuple[dict, bytes]],
) -> dict:
    """Transcribes 1 to 4 consecutive scans in a single API call."""
    scan_list = ", ".join(f"Scan #{item['page_number']} ({item['folio_label']})" for item, _ in batch_items)
    prompt = (
        f"You are an expert Mughal Indo-Persian paleographer transcribing {len(batch_items)} consecutive scan(s) "
        f"[{scan_list}] of the Bodleian Library, University of Oxford's royal Mughal manuscript of "
        "Nūr al-Dīn ʿAbd al-Raḥmān Jāmī's 'Bahāristān' (Garden of Spring), shelfmark MS. Elliott 254, "
        "copied in Lahore in 1595 (1003 AH) for Emperor Akbar and later owned by Emperor Shah Jahan "
        "(imperial Nastaʿlīq script with gold-painted animal/floral margins and 6 Mughal court miniatures).\n\n"
        "Instructions:\n"
        f"1. Return a JSON object with `records` containing EXACTLY {len(batch_items)} FolioRecord objects—one for each scan above, "
        "preserving each scan's exact integer `page_number` and `folio_label`.\n"
        "2. Use consecutive pages to resolve catchwords and split sentences accurately.\n"
        "3. If a scan is a leather cover, marbled endpaper, blank flyleaf, or color chart, set `page_type` to 'cover_or_flyleaf' "
        "and leave `persian_ground_truth_text` and `roman_persian_transliteration` empty.\n"
        "4. Early flyleaves contain Sir Gore Ouseley's 1817 handwritten English notes: set `page_type` to 'english_preface', "
        "transcribe his English text accurately into `english_page_summary`, and note any Persian seals/inscriptions in `persian_ground_truth_text`.\n"
        "5. For all manuscript folios, transcribe EVERY visible Persian/Arabic Nastaʿlīq line (including verse hemistichs, headings, "
        "imperial inspection seals, and catchwords) with exact '\\n' line breaks in `persian_ground_truth_text`.\n"
        "6. Provide a COMPLETE line-by-line `roman_persian_transliteration` (never truncate or summarize in brackets).\n"
        "7. If a full Mughal court miniature, illuminated ʿunwān/shamsa frontispiece, or notable gold marginal painting appears, "
        "set `contains_illumination` to true (and `page_type` to 'mughal_miniature' for full paintings) and describe the iconography in `artwork_description`.\n"
        "8. Write a concise scholarly `english_page_summary` (noting which of the 8 Rawḍas/Gardens or stories appears) and list `historical_entities`."
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

    existing = {}
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            for item in json.load(f):
                existing[item["page_number"]] = item

    canvases = fetch_bodleian_canvases(MANIFEST_URL)
    target_canvases = [c for c in canvases if START_SCAN <= c["page_number"] <= END_SCAN]
    todo = [c for c in target_canvases if c["page_number"] not in existing]
    print(f"[MANIFEST] Total scans: {len(canvases)} | Saved: {len(existing)} | Remaining: {len(todo)}")

    queue: asyncio.Queue = asyncio.Queue()
    for i in range(0, len(todo), FLAGSHIP_BATCH_SIZE):
        queue.put_nowait(todo[i : i + FLAGSHIP_BATCH_SIZE])

    headers = {"User-Agent": "Mozilla/5.0 ProjectScriptorium/1.0"}
    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as http_client:
        async def download_image(item: dict) -> bytes:
            for attempt in range(1, 6):
                resp = await http_client.get(item["image_url"], timeout=45.0)
                if resp.status_code == 200:
                    return resp.content
                await asyncio.sleep(2 * attempt)
            raise RuntimeError(f"Failed downloading image for Scan {item['page_number']}")

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
                        # Tier-1 takes up to 4 scans/call; Tier-2 Flash-Lite & Tier-3 Gemma 4 take 1 scan/call
                        chunk = pending[:FLAGSHIP_BATCH_SIZE] if tier_num == 1 else pending[:1]

                        try:
                            parsed = await asyncio.to_thread(transcribe_batch_sync, client, model_name, chunk)
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
                                        rec["bodleian_shelfmark"] = "Oxford, Bodleian Library, MS. Elliott 254"
                                        rec["image_url"] = meta["image_url"]
                                        existing[p_num] = rec
                                        saved_nums.add(p_num)
                                if saved_nums:
                                    atomic_save([existing[k] for k in sorted(existing.keys())])
                                done_count = len(existing)

                            if not saved_nums:
                                raise ValueError(f"Model returned 0 matching page numbers for {sorted(chunk_by_num.keys())}")

                            art_pages = [n for n in sorted(saved_nums) if existing[n].get("contains_illumination")]
                            art_tag = f" [ILLUMINATED: {art_pages}]" if art_pages else ""
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