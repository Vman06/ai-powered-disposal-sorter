import time
import base64
import requests
import os

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"  # if Ollama is on the Pi
MODEL = "llava"
TIMEOUT = 25  # seconds (keep small for fast response; increase if needed)

# -----------------------------
# LED SETUP (gpiozero)
# -----------------------------
from gpiozero import LED
from time import sleep

# BCM pins
LED_RECYCLE = LED(17)  # YELLOW
LED_TRASH   = LED(27)  # RED
LED_COMPOST = LED(22)  # GREEN

def leds_off():
    LED_RECYCLE.off()
    LED_TRASH.off()
    LED_COMPOST.off()

def show_bin(bin_label: str, hold_seconds: float = 2.0):
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
        # TRASH or NONE or unknown => TRASH
        LED_TRASH.on()

    sleep(hold_seconds)
    leds_off()

# -----------------------------
# IMAGE CAPTURE (Picamera2 preferred)
# -----------------------------
def capture_image(filename="capture.jpg"):
    # Try Picamera2 first (Pi Camera Module)
    try:
        from picamera2 import Picamera2
        print("📷 Using Picamera2...")
        picam2 = Picamera2()
        config = picam2.create_still_configuration(main={"size": (1280, 720)})
        picam2.configure(config)
        picam2.start()
        time.sleep(1)
        picam2.capture_file(filename)
        picam2.stop()
        print(f"✅ Image saved as {filename}")
        return filename
    except Exception as e:
        print(f"⚠️ Picamera2 not available / failed ({e}). Trying OpenCV...")

    # Fallback: OpenCV webcam
    try:
        import cv2
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("❌ Could not access camera.")
            return None

        print("📷 Using OpenCV camera... capturing in 1 second.")
        time.sleep(1)

        ret, frame = camera.read()
        camera.release()
        cv2.destroyAllWindows()

        if not ret:
            print("❌ Failed to capture image.")
            return None

        cv2.imwrite(filename, frame)
        print(f"✅ Image saved as {filename}")
        return filename
    except Exception as e:
        print("❌ OpenCV capture failed:", e)
        return None

# -----------------------------
# LLaVA CALL
# -----------------------------
PROMPT = """
You are a waste-sorting detector.

Output EXACTLY these 9 lines, nothing else. Use only YES or NO:

FOOD_PRESENT=<YES|NO>
GLASS_PRESENT=<YES|NO>
METAL_PRESENT=<YES|NO>
PAPER_PRESENT=<YES|NO>
PLASTIC_BOTTLE_OR_TUB_PRESENT=<YES|NO>
WRAPPER_OR_FILM_PRESENT=<YES|NO>
SMALL_RIGID_PLASTIC_PRESENT=<YES|NO>
OBVIOUS_FOOD_STAINS_ON_RECYCLABLE=<YES|NO>
CONTAINS_OTHER_ITEM=<YES|NO>

Definitions:
- FOOD_PRESENT = actual food scraps (fruit/vegetables/leftovers). Do NOT guess.
- GLASS_PRESENT = glass cup/bottle/jar.
- PLASTIC_BOTTLE_OR_TUB_PRESENT = plastic bottle/jug/tub/cup (packaging).
- WRAPPER_OR_FILM_PRESENT = plastic bag, wrapper, cling film.
- SMALL_RIGID_PLASTIC_PRESENT = floss pick, utensil, toothbrush, small plastic parts.

IMPORTANT (stains):
OBVIOUS_FOOD_STAINS_ON_RECYCLABLE = YES ONLY IF you can clearly see solid food pieces or thick opaque residue.
If it looks like glare/reflection/water/glass distortion or you are unsure → output NO.

IMPORTANT (contains):
CONTAINS_OTHER_ITEM = YES if a container (cup/tub/bowl/bag) is holding other discardable items inside it
(e.g., napkins/tissues/other trash packed inside a tub). If empty → NO.

Ignore people/hands/background.
""".strip()

def call_llava(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.1,
            "num_predict": 120
        },
    }

    r = requests.post(OLLAMA_API_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("response", "").strip()

# -----------------------------
# PARSE + DECIDE BIN
# -----------------------------
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
        "OBVIOUS_FOOD_STAINS_ON_RECYCLABLE",
        "CONTAINS_OTHER_ITEM",
    ]
    for k in keys:
        flags.setdefault(k, "NO")
    return flags

def decide_bin(flags):
    # If a container is holding other discardable items -> TRASH
    if flags["CONTAINS_OTHER_ITEM"] == "YES":
        return "TRASH"

    has_trash = (flags["WRAPPER_OR_FILM_PRESENT"] == "YES") or (flags["SMALL_RIGID_PLASTIC_PRESENT"] == "YES")
    has_compost = (flags["FOOD_PRESENT"] == "YES")
    has_recycling = any(flags[k] == "YES" for k in [
        "GLASS_PRESENT",
        "METAL_PRESENT",
        "PAPER_PRESENT",
        "PLASTIC_BOTTLE_OR_TUB_PRESENT"
    ])

    # If recyclable but clearly stained -> treat as TRASH
    stained = (flags["OBVIOUS_FOOD_STAINS_ON_RECYCLABLE"] == "YES")
    if stained and has_recycling:
        has_trash = True

    # Your rule: if 2+ categories appear together => TRASH
    categories_present = sum([has_trash, has_compost, has_recycling])
    if categories_present == 0:
        return "NONE"
    if categories_present >= 2:
        return "TRASH"

    if has_trash:
        return "TRASH"
    if has_compost:
        return "COMPOST"
    return "RECYCLING"

# -----------------------------
# MAIN LOOP (automatic)
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting automatic sorter (Ctrl+C to stop)")
    leds_off()

    try:
        while True:
            image_path = capture_image("capture.jpg")
            if not image_path:
                print("❌ No image captured. Retrying...")
                show_bin("TRASH", 1.0)
                continue

            try:
                raw = call_llava(image_path)
                print("\n🧩 Raw response:\n" + raw)
                flags = parse_flags(raw)
                final = decide_bin(flags)
            except Exception as e:
                print("❌ LLaVA/Ollama error:", e)
                final = "TRASH"

            print(f"\n🔎 Final bin → {final}")
            show_bin(final, hold_seconds=2.0)

            # small pause between cycles (optional)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Stopping.")
        leds_off()
