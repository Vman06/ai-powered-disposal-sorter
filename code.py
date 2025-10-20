import requests
import base64
import os

# -------------------------------------
# CONFIGURATION
# -------------------------------------
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL = "llava"

# -------------------------------------
# IMAGE ENCODER
# -------------------------------------
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# -------------------------------------
# CLASSIFIER
# -------------------------------------
def classify_item(image_path):
    img_b64 = encode_image_to_base64(image_path)

    # 🧠 Strong prompt: prioritize compost for food items
    prompt = (
        "You are an expert in waste management and sustainability. "
        "Classify the item in the image as one of these categories: 'trash', 'recycling', or 'compost'.\n\n"
        "- 'compost' → all fruits, vegetable scraps, peels, coffee grounds, napkins, or other organic matter.\n"
        "- 'recycling' → clean plastics, metals, glass, paper, cardboard.\n"
        "- 'trash' → everything else that cannot be recycled or composted.\n\n"
        "**Rule:** If the item is clearly a fruit or vegetable (like a banana, apple, carrot, peel, etc.), classify as 'compost', regardless of packaging.\n"
        "Return ONLY strict JSON in this format:\n"
        "{ \"category\": \"trash/recycling/compost\", \"reason\": \"brief explanation\" }"
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False
    }

    # Send request to Ollama
    response = requests.post(OLLAMA_API_URL, json=payload)

    try:
        raw_response = response.json()["response"].strip()
    except Exception as e:
        print("❌ Error:", e)
        print("Full response:", response.text)
        return "Error"

    print(f"\n🧩 Raw model response for {os.path.basename(image_path)}:")
    print(raw_response)

    # Extract classification
    result = raw_response.lower()
    if '"compost"' in result:
        category = "🌿 Compost"
    elif '"recycling"' in result:
        category = "♻️ Recycling"
    else:
        category = "🗑️ Trash"

    return category

# -------------------------------------
# MAIN TEST
# -------------------------------------
if __name__ == "__main__":
    test_folder = "images"
    print("\n🚀 Starting Waste Classifier using LLaVA...\n")

    for filename in os.listdir(test_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(test_folder, filename)
            category = classify_item(path)
            print(f"{filename:<25} → {category}")

    print("\n✅ Classification complete!")