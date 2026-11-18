import asyncio
import json
import os
import time
import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel

MANIFEST_URL = "https://figgy.princeton.edu/concern/scanned_resources/da45a495-2286-4fa6-99f0-b7c2a4ab6f6f/manifest"
CORPUS_PATH = "corpus.json"
START_SCAN = 1
END_SCAN = 999  # Runs all 502 canvases using your 6,480 RPD pool

RAW_KEYS = [
    os.environ.get("GEMINI_API_KEY", ""),
    os.environ.get("GEMINI_API_KEY_1", ""),
    os.environ.get("GEMINI_API_KEY_2", ""),
    os.environ.get("GEMINI_API_KEY_3", ""),
    os.environ.get("GEMINI_API_KEY_4", ""),
    os.environ.get("GEMINI_API_KEY_5", ""),
    os.environ.get("GEMINI_API_KEY_6", ""),
]
API_KEYS = list(dict.fromkeys([k.strip() for k in RAW_KEYS if k.strip()]))
if not API_KEYS:
    raise RuntimeError("No API keys found! Set GEMINI_API_KEY_1..6 in your environment.")

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
]
EXCLUDED_SUBSTRINGS = ("tts", "image", "audio", "live", "transcribe", "embedding", "lite")

class FolioRecord(BaseModel):
    page_number: int
    folio_label: str
    page_type: str  # "cover_or_flyleaf", "illuminated_frontispiece", "manuscript_text", "illustrated_folio"
    contains_illumination: bool
    artwork_description: str
    persian_ground_truth_text: str
    roman_persian_transliteration: str
    english_page_summary: str
    historical_entities: list[str]
    princeton_shelfmark: str
    image_url: str

def discover_active_models(api_key: str) -> tuple[list[str], list[str]]:
    client = genai.Client(api_key=api_key)
    available = []
    for m in client.models.list():
        name = m.name.replace("models/", "")
        actions = getattr(m, "supported_actions", []) or []
        if "generateContent" in actions or not actions:
            available.append(name)

    tier1, tier2 = [], []
    for pat in TIER1_PATTERNS:
        matches = sorted([
            n for n in available
            if pat in n and not any(bad in n for bad in EXCLUDED_SUBSTRINGS)
        ])
        if matches:
            tier1.append(matches[0])

    for pat in TIER2_PATTERNS:
        matches = sorted([
            n for n in available
            if pat in n and not any(bad in n for bad in ("tts", "image", "live"))
        ])
        if matches:
            tier2.append(matches[0])

    if not tier1:
        tier1 = ["gemini-2.5-flash"]
    if not tier2:
        tier2 = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    return tier1, tier2

class QuotaRouter:
    def __init__(self, keys: list[str], tier1_models: list[str], tier2_models: list[str]):
        self.keys = keys
        self.tier1 = tier1_models
        self.tier2 = tier2_models
        self.clients = {i: genai.Client(api_key=k) for i, k in enumerate(keys)}
        self.dead_pairs: set[tuple[int, str]] = set()
        self.pair_cooldown: dict[tuple[int, str], float] = {}
        self.model_cooldown: dict[str, float] = {}  # Model-wide 503 shield!
        self.rr_cursor = 0

    async def acquire_slot(self, flagship_only: bool = False) -> tuple[int, str, genai.Client]:
        model_pool = self.tier1 if flagship_only else (self.tier1 + self.tier2)
        all_pairs = [(k_idx, m) for m in model_pool for k_idx in range(len(self.keys))]

        while True:
            now = time.time()
            n = len(all_pairs)
            for offset in range(n):
                k_idx, model = all_pairs[(self.rr_cursor + offset) % n]
                pair = (k_idx, model)
                if pair in self.dead_pairs:
                    continue
                if now < self.model_cooldown.get(model, 0.0):
                    continue
                if now >= self.pair_cooldown.get(pair, 0.0):
                    self.rr_cursor = (self.rr_cursor + offset + 1) % n
                    self.pair_cooldown[pair] = now + 13.0
                    return k_idx, model, self.clients[k_idx]

            if flagship_only and all(p in self.dead_pairs for p in all_pairs):
                print("[!] Tier-1 Flagship RPD exhausted for today; unlocking Tier-2 Flash-Lite pool...")
                flagship_only = False
                model_pool = self.tier1 + self.tier2
                all_pairs = [(k_idx, m) for m in model_pool for k_idx in range(len(self.keys))]
                continue

            if all(p in self.dead_pairs for p in all_pairs):
                raise RuntimeError("All keys x all models hit daily RPD quota! Resume tomorrow.")

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
            # Put the ENTIRE model on 45s cooldown across all keys so we switch models immediately
            if time.time() >= self.model_cooldown.get(model, 0.0):
                print(f"[!] Model {model} overloaded (503) -> pausing {model} for 45s across all keys")
            self.model_cooldown[model] = time.time() + 45.0
        else:
            self.pair_cooldown[pair] = time.time() + 30.0
            print(f"[!] RPM cooldown 30s on Key#{key_idx+1} + {model}")

def atomic_save(records: list[dict]):
    tmp = CORPUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CORPUS_PATH)

def fetch_princeton_canvases(manifest_url: str, width: int = 1500) -> list[dict]:
    headers = {
        "User-Agent": "Mozilla/5.0 ProjectScriptorium/1.0",
        "Accept": "application/json, application/ld+json",
    }
    resp = httpx.get(manifest_url, headers=headers, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    manifest = resp.json()
    canvases = []

    if "sequences" in manifest and manifest["sequences"]:
        for idx, canvas in enumerate(manifest["sequences"][0].get("canvases", []), start=1):
            label = str(canvas.get("label", f"Scan {idx}"))
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
    return canvases

def transcribe_sync(client: genai.Client, model_name: str, img_bytes: bytes, scan_num: int, folio_label: str) -> str:
    prompt = (
        f"You are an expert Indo-Persian paleographer transcribing Scan #{scan_num} (Folio: {folio_label}) "
        "of Princeton University Library's 18th-century Indian illuminated manuscript of Zakarīyā al-Qazwīnī's "
        "'ʿAjāʾib al-makhlūqāt wa-gharāʾib al-mawjūdāt' (Garrett no. 82G, Nastaʿlīq script, ~20 lines/page).\n\n"
        "Reference Anchors from Princeton Catalog:\n"
        "- Opening Incipit (fol. 1b/3a): 'العظمة لك والبرهان لجلالك اللهم يا قائم الذات ومفيض الخيرات... اما بعد همی گوید اصغر العباد زکریاء بن محمد بن محمود الکمونی القزوینی'\n"
        "- Colophon Explicit (fol. 240a): 'ابو ریحان گوید که بنده التماس کردم... عبدالله ولد علی بیگ دماوندی'\n\n"
        "Instructions:\n"
        "1. If this scan is a leather binding, blank leaf, or library flyleaf with no manuscript text, set "
        "`page_type` to 'cover_or_flyleaf' and leave `persian_ground_truth_text` and `roman_persian_transliteration` empty.\n"
        "2. Transcribe all visible Persian/Arabic Nastaʿlīq lines accurately with exact '\\n' line breaks in `persian_ground_truth_text`.\n"
        "3. Provide a COMPLETE line-by-line `roman_persian_transliteration` (never truncate or summarize in brackets).\n"
        "4. If an opaque watercolor miniature or gold/red astronomical diagram appears on the page, set "
        "`contains_illumination` to true and describe the iconography in `artwork_description`.\n"
        "5. Write a concise scholarly `english_page_summary` and list `historical_entities`."
    )
    res = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FolioRecord,
            temperature=0.1,
        ),
    )
    return res.text

async def main():
    print(f"[BOOT] Loaded {len(API_KEYS)} API keys.")
    tier1, tier2 = discover_active_models(API_KEYS[0])
    print(f"[MODELS] Tier-1 Flagship ({len(tier1)*len(API_KEYS)*20} RPD): {tier1}")
    print(f"[MODELS] Tier-2 Bulk     ({len(tier2)*len(API_KEYS)*500} RPD): {tier2}")

    router = QuotaRouter(API_KEYS, tier1, tier2)
    save_lock = asyncio.Lock()

    existing = {}
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            for item in json.load(f):
                existing[item["page_number"]] = item

    canvases = fetch_princeton_canvases(MANIFEST_URL)
    target_canvases = [c for c in canvases if START_SCAN <= c["page_number"] <= END_SCAN]
    todo = [c for c in target_canvases if c["page_number"] not in existing]
    print(f"[MANIFEST] Total scans: {len(canvases)} | Saved: {len(existing)} | Remaining: {len(todo)}")

    queue: asyncio.Queue = asyncio.Queue()
    for c in todo:
        queue.put_nowait(c)

    headers = {"User-Agent": "Mozilla/5.0 ProjectScriptorium/1.0"}
    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as http_client:
        async def worker(worker_id: int):
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                p_num = item["page_number"]
                try:
                    # Download image with retry so network blips never skip a scan
                    img_bytes = None
                    for img_try in range(1, 6):
                        resp = await http_client.get(item["image_url"], timeout=45.0)
                        if resp.status_code == 200:
                            img_bytes = resp.content
                            break
                        await asyncio.sleep(2 * img_try)
                    if not img_bytes:
                        raise RuntimeError(f"Failed downloading image for Scan {p_num}")

                    flagship_only = p_num <= 100
                    while True:
                        k_idx, model_name, client = await router.acquire_slot(flagship_only=flagship_only)
                        try:
                            raw = await asyncio.to_thread(
                                transcribe_sync, client, model_name, img_bytes, p_num, item["folio_label"]
                            )
                            rec = json.loads(raw)
                            rec["page_number"] = p_num
                            rec["folio_label"] = item["folio_label"]
                            rec["princeton_shelfmark"] = "Islamic Manuscripts, Garrett no. 82G"
                            rec["image_url"] = item["image_url"]

                            async with save_lock:
                                existing[p_num] = rec
                                atomic_save([existing[k] for k in sorted(existing.keys())])
                                done_count = len(existing)

                            art_tag = " [ILLUMINATED]" if rec.get("contains_illumination") else ""
                            print(f"[✓] Scan {p_num:03d}/{len(canvases)} ({done_count}/{len(target_canvases)} saved){art_tag} | Key#{k_idx+1} -> {model_name}")
                            break
                        except Exception as e:
                            router.report_error(k_idx, model_name, e)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(len(API_KEYS))]
        await asyncio.gather(*workers)

    expected_nums = {c["page_number"] for c in target_canvases}
    missing = sorted(expected_nums - set(existing.keys()))
    print(f"[DONE] Saved {len(existing)}/{len(target_canvases)} folios to {CORPUS_PATH} | Missing: {missing}")

if __name__ == "__main__":
    asyncio.run(main())