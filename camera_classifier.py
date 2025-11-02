import cv2
import time
import base64
import requests

# ----------------------------------
# CONFIGURATION
# ----------------------------------
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL = "llava"

# ----------------------------------
# CAPTURE IMAGE FROM MAC CAMERA
# ----------------------------------
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

# ----------------------------------
# CLASSIFY IMAGE WITH LLAVA
# ----------------------------------
def classify_image(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "You are a waste classification assistant. "
        "Decide whether the item in this image belongs in 'trash', 'recycling', or 'compost'. "
        "Rules:\n"
        "- Recycling: clean plastics, glass, metals, cans, or paper.\n"
        "- Compost: food scraps, fruit/vegetable peels, napkins, or organic material.\n"
        "- Trash: everything else or items with mixed materials.\n\n"
        "Try to analyze the whole image and see if there is any item fitting the above rules. The person might be holding a water bottle or a trash.\n"
        "If you cannot confidently decide, respond only with 'none'.\n"
        "Output just one word: trash, recycling, compost, or none .\n"
        "If the output is none try saying what you see in one line"
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        data = response.json()
        raw = data.get("response", "").strip().lower()
        print(f"🧩 Raw response: {raw}")
    except Exception as e:
        print("❌ Error communicating with Ollama:", e)
        return "none"

    if "recycl" in raw:
        return "♻️ Recycling"
    elif "compost" in raw:
        return "🌿 Compost"
    elif "trash" in raw:
        return "🗑️ Trash"
    else:
        return "None"

# ----------------------------------
# MAIN LOGIC
# ----------------------------------
if __name__ == "__main__":
    print("🚀 Starting camera capture and classification...")
    image_path = capture_image()

    if image_path:
        result = classify_image(image_path)
        print(f"\n🔎 Classification result → {result}")