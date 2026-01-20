import cv2
import time
import base64
import requests
import numpy as np

# ============================================================
# CONFIG
# ============================================================
# If Ollama is running on another device, change this to:
# OLLAMA_API_URL = "http://<YOUR_MAC_IP>:11434/api/generate"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

MODEL = "llava:7b"                 # faster than 13b
TIMEOUT = (5, 25)                  # (connect_timeout, read_timeout)
KEEP_ALIVE = "10m"                 # keep model in RAM between runs

# CV stain detector threshold (paper/cardboard only)
STAIN_RATIO_THRESHOLD = 0.012      # tune up/down

# ============================================================
# LEDS (Raspberry Pi) - BCM numbering
# Your wiring:
# pin 11 -> GPIO17 -> RED (TRASH)
# pin 13 -> GPIO27 -> YELLOW (RECYCLING)
# pin 15 -> GPIO22 -> GREEN (COMPOST)
# ============================================================
from gpiozero import LED
from time import sleep

LED_TRASH   = LED(17)  # red
LED_RECYCLE = LED(27)  # yellow
LED_COMPOST = LED(22)  # green

def leds_off():
    LED_TRASH.off()
    LED_RECYCLE.off()
    LED_COMPOST.off()

def show_bin(bin_label: str, hold_seconds: float = 3.0):
    """
    bin_label: 'RECYCLING', 'TRASH', 'COMPOST' (or 'NONE')
    """
    leds_off()
    b = (bin_label or "").strip().upper()

    if b == "RECYCLING":
        LED_RECYCLE.on()
    elif b == "COMPOST":
        LED_COMPOST.on()
    else:
        # TRASH / NONE / unknown => TRASH
        LED_TRASH.on()

    sleep(hold_seconds)
    leds_off()

# ============================================================
# CAPTURE IMAGE (USB CAM / OPENCV)
# ============================================================
def capture_image():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌ Could not access camera.")
        return None

    print("📷 Starting camera... capturing in 1 second.")
    time.sleep(1)

    ret, frame = camera.read()
    camera.release()
    cv2.destroyAllWindows()

    if not ret:
        print("❌ Failed to capture image.")
        return None

    filename = "capture.jpg"
    cv2.imwrite(filename, frame)
    print(f"✅ Image saved as {filename}")
    return filename


# ============================================================
# OPTIONAL: WARMUP (reduces first-call lag)
# ============================================================
def warmup():
    try:
        payload = {
            "model": MODEL,
            "prompt": "Reply with OK.",
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": 0.0, "num_predict": 4},
        }
        requests.post(OLLAMA_API_URL, json=payload, timeout=TIMEOUT).raise_for_status()
    except Exception:
        pass


# ============================================================
# FAST CV STAIN DETECTOR (paper/cardboard)
# ============================================================
def cv_detect_paper_and_stains(image_path, debug=True):
    """
    Detect paper-like area and obvious orange/brown/red-ish stains on it.
    Returns: (paper_like_present, stained, info_dict)
    """
    img = cv2.imread(image_path)
    if img is None:
        return False, False, {"reason": "cv2.imread failed"}

    # Downscale for speed
    h, w = img.shape[:2]
    scale = 700 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Paper-like: low saturation + bright
    paper_mask = cv2.inRange(hsv, (0, 0, 120), (179, 70, 255))
    paper_pixels = int(np.count_nonzero(paper_mask))
    paper_like_present = paper_pixels > 2500

    # Stain-like: orange/brown/red-ish
    stain_mask1 = cv2.inRange(hsv, (0, 60, 50), (25, 255, 255))
    stain_mask2 = cv2.inRange(hsv, (160, 60, 50), (179, 255, 255))
    stain_mask = cv2.bitwise_or(stain_mask1, stain_mask2)

    stain_on_paper = cv2.bitwise_and(stain_mask, paper_mask)

    # remove tiny specks
    kernel = np.ones((3, 3), np.uint8)
    stain_on_paper = cv2.morphologyEx(stain_on_paper, cv2.MORPH_OPEN, kernel, iterations=1)

    stain_pixels = int(np.count_nonzero(stain_on_paper))
    ratio = stain_pixels / max(paper_pixels, 1)

    stained = paper_like_present and (ratio >= STAIN_RATIO_THRESHOLD)

    info = {
        "paper_pixels": paper_pixels,
        "stain_pixels": stain_pixels,
        "stain_ratio": ratio,
        "threshold": STAIN_RATIO_THRESHOLD,
    }

    if debug:
        print(f"🧪 CV paper_pixels={paper_pixels}, stain_pixels={stain_pixels}, "
              f"ratio={ratio:.4f} (thresh={STAIN_RATIO_THRESHOLD})")

    return paper_like_present, stained, info


# ============================================================
# STAGE 1: FOOD-ONLY CHECK
# ============================================================
def call_llava_food_only(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = """
Answer with EXACTLY ONE WORD: YES or NO.

Question: Is there edible food (fruit/vegetables/leftovers) as the main object?

Rules:
- If you clearly see an edible item like an orange/apple/banana -> YES
- Do NOT guess. If unsure -> NO
""".strip()

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.0, "top_p": 0.1, "num_predict": 5},
    }

    r = requests.post(OLLAMA_API_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("response", "").strip().upper()


# ============================================================
# STAGE 2: MATERIAL/CONTAINS FLAGS
# ============================================================
def call_llava_flags(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = """
You are a waste-sorting detector.

Output EXACTLY these 8 lines, nothing else. Use only YES or NO:

FOOD_PRESENT=<YES|NO>
GLASS_PRESENT=<YES|NO>
METAL_PRESENT=<YES|NO>
PAPER_PRESENT=<YES|NO>
PLASTIC_BOTTLE_OR_TUB_PRESENT=<YES|NO>
WRAPPER_OR_FILM_PRESENT=<YES|NO>
SMALL_RIGID_PLASTIC_PRESENT=<YES|NO>
CONTAINS_OTHER_ITEM=<YES|NO>

Definitions (do NOT guess):
- FOOD_PRESENT = visible food scraps (fruit/vegetables/leftovers).
- PAPER_PRESENT = paper/cardboard item (box, paper bag, napkin/tissue/paper towel).
- GLASS_PRESENT = glass cup/bottle/jar.
- METAL_PRESENT = metal can/foil/metal piece.
- PLASTIC_BOTTLE_OR_TUB_PRESENT = plastic bottle/jug/tub/cup (packaging).
- WRAPPER_OR_FILM_PRESENT = plastic wrapper/bag/cling film.
- SMALL_RIGID_PLASTIC_PRESENT = floss pick, utensil, toothbrush, small plastic parts.
- CONTAINS_OTHER_ITEM = YES if a container is holding other discardable items inside it.

Ignore people/hands/background. Focus only on discardable items.
""".strip()

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.0, "top_p": 0.1, "num_predict": 120},
    }

    r = requests.post(OLLAMA_API_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def parse_flags(raw: str):
    flags = {}
    for line in raw.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            key = k.strip().upper()
            val = v.strip().upper().replace("<", "").replace(">", "").strip()
            flags[key] = val

    keys = [
        "FOOD_PRESENT",
        "GLASS_PRESENT",
        "METAL_PRESENT",
        "PAPER_PRESENT",
        "PLASTIC_BOTTLE_OR_TUB_PRESENT",
        "WRAPPER_OR_FILM_PRESENT",
        "SMALL_RIGID_PLASTIC_PRESENT",
        "CONTAINS_OTHER_ITEM",
    ]
    for k in keys:
        flags.setdefault(k, "NO")
    return flags


# ============================================================
# DECISION LOGIC (3-bin)
# ============================================================
def decide_bin(flags, paper_stained=False):
    # packed container rule
    if flags.get("CONTAINS_OTHER_ITEM") == "YES":
        return "TRASH"

    has_recycling = any(flags.get(k) == "YES" for k in [
        "GLASS_PRESENT",
        "METAL_PRESENT",
        "PAPER_PRESENT",
        "PLASTIC_BOTTLE_OR_TUB_PRESENT",
    ])

    has_trash_signal = (flags.get("WRAPPER_OR_FILM_PRESENT") == "YES") or (flags.get("SMALL_RIGID_PLASTIC_PRESENT") == "YES")

    has_compost_signal = (flags.get("FOOD_PRESENT") == "YES") or paper_stained

    categories_present = sum([has_recycling, has_trash_signal, has_compost_signal])

    if categories_present == 0:
        return "NONE"
    if categories_present >= 2:
        return "TRASH"
    if has_trash_signal:
        return "TRASH"
    if has_compost_signal:
        return "COMPOST"
    return "RECYCLING"


def pretty(label):
    return {
        "RECYCLING": "♻️ Recycling",
        "COMPOST": "🌿 Compost",
        "TRASH": "🗑️ Trash",
        "NONE": "🗑️ Trash",
    }.get(label, "🗑️ Trash")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("🚀 Starting camera capture and classification...")
    leds_off()
    warmup()

    image_path = capture_image()
    if not image_path:
        raise SystemExit(0)

    # ---- Stage 1: Food-only
    try:
        food_yesno = call_llava_food_only(image_path)
        print(f"🥕 FOOD_ONLY={food_yesno}")
        if food_yesno == "YES":
            final = "COMPOST"
            print(f"\n🔎 Classification result → {pretty(final)}")
            show_bin(final, hold_seconds=3.0)
            raise SystemExit(0)
    except Exception as e:
        print("❌ Food-only check failed:", e)
        # continue rather than dying

    # ---- CV stain detection
    paper_like, paper_stained, _info = cv_detect_paper_and_stains(image_path, debug=True)

    # ---- Stage 2: General flags
    try:
        raw = call_llava_flags(image_path)
        print(f"🧩 Raw response:\n{raw}")
        flags = parse_flags(raw)
    except Exception as e:
        print("❌ LLaVA error/timeout:", e)
        final = "TRASH"
        print(f"\n🔎 Classification result → {pretty(final)}")
        show_bin(final, hold_seconds=3.0)
        raise SystemExit(0)

    # Only apply stain logic if paper-like OR model says paper present
    use_stain = paper_stained if (paper_like or flags.get("PAPER_PRESENT") == "YES") else False

    final = decide_bin(flags, paper_stained=use_stain)

    print(f"\n🧪 CV_PAPER_STAINED={use_stain}")
    print(f"🔎 Classification result → {pretty(final)}")

    # ✅ LED output
    show_bin(final, hold_seconds=3.0)