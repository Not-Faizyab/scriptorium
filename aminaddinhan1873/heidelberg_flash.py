
import asyncio
import itertools
import json
import os
import random
import httpx

from google import genai
from google.genai import types
from pydantic import BaseModel


API_KEYS = [
    os.environ.get("GEMINI_API_KEY", ""),
    os.environ.get("GEMINI_API_KEY_2", ""),
]

API_KEYS = [key for key in API_KEYS if key]

if not API_KEYS:
    raise RuntimeError(
        "Set GEMINI_API_KEY in your environment first!"
    )

key_cycle = itertools.cycle(API_KEYS)

MODEL_ID = "gemini-3.5-flash-lite"

PAGES_PER_REQUEST = 1
RPM_PER_KEY = 10

DELAY_BETWEEN_CALLS = 60.0 / (
    RPM_PER_KEY * len(API_KEYS)
)

MAX_DOWNLOAD_ATTEMPTS = 5
MAX_GEMINI_ATTEMPTS = 6
MAX_BATCH_ROUNDS = 3

HEIDELBERG_BOOK_ID = "aminaddinhan1873"

MANIFEST_URL = (
    f"https://digi.ub.uni-heidelberg.de/diglit/"
    f"iiif/{HEIDELBERG_BOOK_ID}/manifest.json"
)

OUTPUT_DIR = "heidelberg_aminaddinhan1873_GT"

os.makedirs(OUTPUT_DIR, exist_ok=True)


class SinglePageGT(BaseModel):
    page_number: int
    urdu_ground_truth_text: str
    roman_urdu_transliteration: str
    english_page_summary: str
    historical_entities: list[str]


class BatchPagesGT(BaseModel):
    pages: list[SinglePageGT]


def call_gemini(contents: list) -> str:

    api_key = next(key_cycle)

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchPagesGT,
            temperature=0.1,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text

def is_fatal_error(error):

    message = str(error).upper()

    fatal_markers = (
        "400",
        "INVALID_ARGUMENT",
        "401",
        "UNAUTHENTICATED",
        "403",
        "PERMISSION_DENIED",
    )

    return any(
        marker in message
        for marker in fatal_markers
    )


def retry_delay(attempt):

    base = min(
        90,
        8 * (2 ** (attempt - 1))
    )

    return base + random.uniform(1, 4)

def page_paths(page_number):

    json_path = os.path.join(
        OUTPUT_DIR,
        f"page_{page_number:04d}.json"
    )

    txt_path = os.path.join(
        OUTPUT_DIR,
        f"page_{page_number:04d}_gt.txt"
    )

    return json_path, txt_path


def page_is_saved(page_number):

    json_path, txt_path = page_paths(page_number)

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        required_fields = (
            "page_number",
            "urdu_ground_truth_text",
            "roman_urdu_transliteration",
            "english_page_summary",
            "historical_entities",
        )

        if not all(
            field in data
            for field in required_fields
        ):
            return False

        if data["page_number"] != page_number:
            return False

        urdu_text = data["urdu_ground_truth_text"]

        if not isinstance(urdu_text, str) or not urdu_text.strip():
            return False

        # Repair the text file automatically if necessary.
        if not os.path.exists(txt_path):

            with open(
                txt_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(urdu_text)

        else:

            with open(
                txt_path,
                "r",
                encoding="utf-8"
            ) as f:

                existing_text = f.read()

            if existing_text != urdu_text:

                with open(
                    txt_path,
                    "w",
                    encoding="utf-8"
                ) as f:

                    f.write(urdu_text)

        return True

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        KeyError,
    ):

        return False


def validate_page(raw_page):

    if hasattr(SinglePageGT, "model_validate"):

        validated = SinglePageGT.model_validate(raw_page)

        return validated.model_dump()

    validated = SinglePageGT.parse_obj(raw_page)

    return validated.dict()


def save_page(page_number, page_obj, image_url):

    page_obj["heidelberg_book_id"] = HEIDELBERG_BOOK_ID
    page_obj["heidelberg_image_url"] = image_url

    json_path, txt_path = page_paths(page_number)

    json_tmp = json_path + ".tmp"
    txt_tmp = txt_path + ".tmp"

    with open(
        txt_tmp,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            page_obj["urdu_ground_truth_text"]
        )

    with open(
        json_tmp,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            page_obj,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(txt_tmp, txt_path)
    os.replace(json_tmp, json_path)


async def download_page(
    http_client: httpx.AsyncClient,
    page_number: int,
    canvas: dict,
):

    try:

        img_url = canvas["images"][0]["resource"]["@id"]

        img_url_opt = img_url.replace(
            "/full/full/",
            "/full/1500,/"
        )

    except (KeyError, IndexError, TypeError) as e:

        print(
            f"[!!] Invalid IIIF image metadata "
            f"for page {page_number}: {e}"
        )

        return None

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):

        try:

            response = await http_client.get(
                img_url_opt,
                timeout=45.0
            )

            if response.status_code == 200:

                if response.content:

                    return img_url, response.content

                error = "Empty image response"

            else:

                error = (
                    f"HTTP {response.status_code}"
                )

                if response.status_code not in (
                    408, 425, 429, 500, 502, 503, 504
                ):

                    print(
                        f"[!] Page {page_number}: "
                        f"HTTP {response.status_code}"
                    )

                    return None

        except httpx.TransportError as e:

            error = str(e)

        if attempt < MAX_DOWNLOAD_ATTEMPTS:

            wait = retry_delay(attempt)

            print(
                f"[!] Download failed for page {page_number}: "
                f"{error}. "
                f"Retry {attempt}/{MAX_DOWNLOAD_ATTEMPTS} "
                f"in {wait:.1f}s..."
            )

            await asyncio.sleep(wait)

    print(
        f"[!!] Download permanently failed for "
        f"page {page_number} after "
        f"{MAX_DOWNLOAD_ATTEMPTS} attempts."
    )

    return None



async def process_batch(
    http_client: httpx.AsyncClient,
    batch: list[tuple[int, dict]],
):

    todo = [
        (p_num, canvas)
        for p_num, canvas in batch
        if not page_is_saved(p_num)
    ]

    if not todo:

        print("[✓] All pages in this batch already exist.")

        return []

    print(
        f"[*] Pages requiring processing: "
        f"{[p for p, _ in todo]}"
    )

    downloaded = {}
    failed_downloads = set()

    for page_number, canvas in todo:

        result = await download_page(
            http_client,
            page_number,
            canvas
        )

        if result is None:

            failed_downloads.add(page_number)

            continue

        img_url, image_bytes = result

        downloaded[page_number] = {
            "url": img_url,

            "part": types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        }

        print(
            f"[+] Downloaded image for page {page_number}"
        )

    if not downloaded:

        print("[!!] No images available for Gemini.")

        return sorted(failed_downloads)

    pending = set(downloaded.keys())


    for attempt in range(1, MAX_GEMINI_ATTEMPTS + 1):

        if not pending:
            break

        pages_this_attempt = sorted(pending)

        print(
            f"\n[*] Gemini attempt {attempt}/"
            f"{MAX_GEMINI_ATTEMPTS}"
        )

        print(
            f"[*] Submitting pages: {pages_this_attempt}"
        )

        contents = []

        for page_number in pages_this_attempt:

            contents.append(
                f"--- IMAGE FOR PAGE_NUMBER "
                f"{page_number} ---"
            )

            contents.append(
                downloaded[page_number]["part"]
            )

        contents.append(
            "You are generating Ground Truth HTR data "
            "for Heidelberg University Library's digitized "
            "1873 Lucknow Urdu lithograph "
            "(aminaddinhan1873).\n\n"

            "For EVERY page image supplied, return exactly "
            "one corresponding item with its exact "
            "page_number.\n\n"

            "Include:\n"

            "1. Line-by-line Urdu Nastaliq transcription.\n"

            "2. Full searchable Romanized Urdu transliteration.\n"

            "3. Concise English summary of the page.\n"

            "4. Historical named entities, including "
            "people, locations, dates, and works.\n\n"

            "Do not omit any supplied page. "
            "Do not invent unreadable text. "
            "Preserve the original Urdu wording and "
            "line structure as accurately as possible."
        )

        try:

            raw_json = await asyncio.to_thread(
                call_gemini,
                contents
            )

            parsed = json.loads(raw_json)

            if not isinstance(parsed, dict):

                raise ValueError(
                    "Gemini response is not a JSON object."
                )

            returned_pages = parsed.get("pages", [])

            if not isinstance(returned_pages, list):

                raise ValueError(
                    "Gemini response does not contain "
                    "a valid pages array."
                )

            results_by_number = {}

            for raw_page in returned_pages:

                try:

                    page_obj = validate_page(raw_page)

                    page_number = page_obj["page_number"]

                    if page_number not in pending:

                        continue

                    results_by_number[page_number] = page_obj

                except Exception as e:

                    print(
                        f"[!] Invalid page in Gemini response: {e}"
                    )


            for page_number in pages_this_attempt:

                page_obj = results_by_number.get(
                    page_number
                )

                if page_obj is None:

                    continue

                save_page(
                    page_number,
                    page_obj,
                    downloaded[page_number]["url"]
                )

                pending.remove(page_number)

                print(
                    f"[+] Saved Ground Truth for "
                    f"Page {page_number}/310"
                )
            if not pending:

                print(
                    "[✓] All pages in this batch completed!"
                )

                break
            print(
                f"[!] Missing pages in Gemini response: "
                f"{sorted(pending)}"
            )

        except Exception as e:

            print(
                f"[!] Gemini error on pages "
                f"{pages_this_attempt}: {e}"
            )

            if is_fatal_error(e):

                print(
                    "[!!] Error appears to be a permanent "
                    "API configuration or permission error."
                )

                break


        if pending and attempt < MAX_GEMINI_ATTEMPTS:

            wait = retry_delay(attempt)

            print(
                f"[*] Retrying pages {sorted(pending)} "
                f"in {wait:.1f}s..."
            )

            await asyncio.sleep(wait)

    unresolved = failed_downloads | pending

    if unresolved:

        print(
            f"\n[!!] Unresolved pages in batch: "
            f"{sorted(unresolved)}"
        )

    else:

        print("[✓] Batch fully completed.")

    return sorted(unresolved)


async def main():

    headers = {
        "User-Agent": "Mozilla/5.0 ResearchArchiver/1.0"
    }

    async with httpx.AsyncClient(
        follow_redirects=True,
        headers=headers
    ) as http_client:

        print(
            f"[*] Fetching IIIF Manifest from Heidelberg "
            f"for '{HEIDELBERG_BOOK_ID}'..."
        )

        manifest_response = await http_client.get(
            MANIFEST_URL,
            timeout=30.0
        )

        manifest_response.raise_for_status()

        manifest = manifest_response.json()

        canvases = manifest["sequences"][0]["canvases"]

        indexed_canvases = [
            (i + 1, canvas)
            for i, canvas in enumerate(canvases)
        ]

        total_pages = len(indexed_canvases)

        print(
            f"[*] Found {total_pages} pages."
        )

        print(
            f"[*] Model: {MODEL_ID}"
        )

        print(
            f"[*] Pages per request: {PAGES_PER_REQUEST}"
        )

        print(
            f"[*] API keys: {len(API_KEYS)}"
        )

        already_completed = [
            page_number
            for page_number, _ in indexed_canvases
            if page_is_saved(page_number)
        ]

        print(
            f"[*] Previously completed pages: "
            f"{len(already_completed)}/{total_pages}"
        )

        remaining = [
            (page_number, canvas)
            for page_number, canvas in indexed_canvases
            if not page_is_saved(page_number)
        ]

        print(
            f"[*] Pages remaining: {len(remaining)}"
        )

        for round_no in range(1, MAX_BATCH_ROUNDS + 1):

            if not remaining:

                print(
                    "\n[✓] No remaining pages to process."
                )

                break

            print(
                f"\n{'=' * 60}"
            )

            print(
                f"PROCESSING ROUND "
                f"{round_no}/{MAX_BATCH_ROUNDS}"
            )

            print(
                f"{'=' * 60}"
            )

            batches = [
                remaining[i:i + PAGES_PER_REQUEST]
                for i in range(
                    0,
                    len(remaining),
                    PAGES_PER_REQUEST
                )
            ]

            unresolved_canvases = []

            for batch_index, batch in enumerate(
                batches,
                start=1
            ):

                page_numbers = [
                    page_number
                    for page_number, _ in batch
                ]

                print(
                    f"\n[*] Batch {batch_index}/"
                    f"{len(batches)} "
                    f"(Pages {page_numbers})"
                )

                try:

                    unresolved = await process_batch(
                        http_client,
                        batch
                    )

                except Exception as e:

                    print(
                        f"[!!] Unexpected batch error "
                        f"on pages {page_numbers}: {e}"
                    )

                    unresolved = page_numbers

                unresolved = set(unresolved)

                for page_number, canvas in batch:

                    if page_number in unresolved:

                        unresolved_canvases.append(
                            (page_number, canvas)
                        )

                if batch_index < len(batches):

                    await asyncio.sleep(
                        DELAY_BETWEEN_CALLS
                    )

            remaining = unresolved_canvases

            print(
                f"\n[*] Round {round_no} complete."
            )

            print(
                f"[*] Unresolved pages: "
                f"{[p for p, _ in remaining]}"
            )

            if remaining and round_no < MAX_BATCH_ROUNDS:

                wait = 20 * round_no

                print(
                    f"[*] Starting another round "
                    f"in {wait} seconds..."
                )

                await asyncio.sleep(wait)

        print(
            f"\n{'=' * 60}"
        )

        print(
            "FINAL DATASET AUDIT"
        )

        print(
            f"{'=' * 60}"
        )

        missing_pages = []

        for page_number, _ in indexed_canvases:

            if not page_is_saved(page_number):

                missing_pages.append(page_number)

        completed_count = total_pages - len(missing_pages)

        print(
            f"[*] Total pages: {total_pages}"
        )

        print(
            f"[*] Successfully saved: {completed_count}"
        )

        print(
            f"[*] Missing: {len(missing_pages)}"
        )

        report_path = os.path.join(
            OUTPUT_DIR,
            "failed_pages.txt"
        )

        if missing_pages:

            with open(
                report_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    "\n".join(
                        str(p)
                        for p in missing_pages
                    )
                )

            print(
                "\n[!!] PROCESSING INCOMPLETE"
            )

            print(
                f"[!!] Missing pages: {missing_pages}"
            )

            print(
                f"[!!] Failure report: {report_path}"
            )

            print(
                "[*] Rerun the script to resume "
                "only the missing or invalid pages."
            )

        else:

            if os.path.exists(report_path):

                os.remove(report_path)

            print(
                "\n[✓] ALL PAGES SUCCESSFULLY PROCESSED!"
            )

            print(
                f"[✓] Dataset: ./{OUTPUT_DIR}/"
            )

        print(
            f"\n[*] Dataset contains "
            f"{completed_count}/{total_pages} "
            f"completed pages."
        )

if __name__ == "__main__":

    asyncio.run(main())
