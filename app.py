import os
import uuid
from io import BytesIO

from flask import Flask, render_template, request, jsonify
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}

# Output sizes mapped to Gemini aspect ratio strings
ASPECT_RATIOS = {
    "1080x1080": "1:1",
    "1080x1920": "9:16",
    "1200x628": "16:9",
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def describe_image(client, img):
    """Send image to Gemini and get a detailed text description."""
    prompt = (
        "Describe this image in detail. Include the main subject, composition, "
        "lighting, colors, background, and any text or branding visible. "
        "Be specific enough that someone could recreate the scene accurately."
    )

    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=[prompt, img],
    )
    return response.text or ""


def ai_resize(client, img, description, ratio_str):
    """Use Gemini image generation to resize an image to a target aspect ratio."""
    prompt = (
        f"Act as a professional Art Director using high-end photo editing software.\n"
        f"Resize the attached marketing asset to {ratio_str} aspect ratio.\n\n"
        f"Here is a description of the image:\n{description}\n\n"
        f"CRITICAL INSTRUCTION - STRICT SUBJECT PRESERVATION:\n"
        f"The input image contains a SPECIFIC REAL-WORLD OBJECT (e.g., a specific house, building, car, or consumer product).\n"
        f"You MUST preserve the identity of this central subject exactly.\n\n"
        f"DO NOT:\n"
        f"- Do NOT generate a different house, building, or product.\n"
        f"- Do NOT change the architectural details (windows, doors, roof).\n"
        f"- Do NOT change the product packaging or logo.\n"
        f"- Do NOT hallucinate a 'similar' looking object. It must represent the EXACT object in the source image.\n\n"
        f"DO:\n"
        f"1. Keep the main subject visually unchanged. Treat the pixels of the product/house as immutable.\n"
        f"2. Create a NEW layout composition by extending the background (outpainting) or repositioning text elements if they exist.\n"
        f"3. Ensure the lighting on the extended background matches the original scene perfectly.\n"
        f"4. Render in photorealistic 4k quality."
    )

    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        #model="gemini-2.5-flash-image",
        contents=[prompt, img],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=ratio_str),
        ),
    )

    # Extract generated image from response parts
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return Image.open(BytesIO(part.inline_data.data))

    return None


def center_crop_fallback(img, ratio_str):
    """Pillow center-crop fallback if AI resize fails."""
    w, h = img.size
    rw, rh = [int(x) for x in ratio_str.split(":")]
    target_aspect = rw / rh
    img_aspect = w / h

    if img_aspect > target_aspect:
        crop_h = h
        crop_w = int(h * target_aspect)
    else:
        crop_w = w
        crop_h = int(w / target_aspect)

    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return img.crop((left, top, left + crop_w, top + crop_h))


# Gemini-supported aspect ratios for ImageConfig
GEMINI_RATIOS = ["1:1", "3:4", "4:3", "9:16", "16:9"]


def _nearest_aspect_ratio(size_str):
    """Find the closest Gemini-supported aspect ratio for a custom WxH size."""
    w, h = [int(x) for x in size_str.split("x")]
    target = w / h
    best = None
    best_diff = float("inf")
    for r in GEMINI_RATIOS:
        rw, rh = [int(x) for x in r.split(":")]
        diff = abs(target - rw / rh)
        if diff < best_diff:
            best_diff = diff
            best = r
    return best


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    # Save original
    file_id = uuid.uuid4().hex[:8]
    ext = file.filename.rsplit(".", 1)[1].lower()
    original_filename = f"{file_id}_original.{ext}"
    original_path = os.path.join(app.config["UPLOAD_FOLDER"], original_filename)
    file.save(original_path)

    img = Image.open(original_path)
    if img.mode == "RGBA":
        img = img.convert("RGB")

    results = {
        "original": f"/static/uploads/{original_filename}",
        "original_description": "",
        "crops": {},
    }

    # Filter to requested sizes (default: all presets)
    requested_sizes = request.form.getlist("sizes")
    if requested_sizes:
        selected = {}
        for size_str in requested_sizes:
            if size_str in ASPECT_RATIOS:
                selected[size_str] = ASPECT_RATIOS[size_str]
            else:
                # Custom size — compute nearest Gemini-supported aspect ratio
                selected[size_str] = _nearest_aspect_ratio(size_str)
    else:
        selected = ASPECT_RATIOS

    # Initialize Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        # No API key — fall back to center crops for all ratios
        for size_name, ratio_str in selected.items():
            target_w, target_h = [int(x) for x in size_name.split("x")]
            cropped = center_crop_fallback(img, ratio_str)
            cropped = cropped.resize((target_w, target_h), Image.LANCZOS)
            crop_filename = f"{file_id}_{size_name}.jpg"
            crop_path = os.path.join(app.config["UPLOAD_FOLDER"], crop_filename)
            cropped.save(crop_path, "JPEG", quality=90)
            results["crops"][size_name] = f"/static/uploads/{crop_filename}"
        return jsonify(results)

    client = genai.Client(api_key=api_key)

    # Step 1: Describe the image
    description = describe_image(client, img)
    results["original_description"] = description

    # Step 2: AI resize for each aspect ratio
    for size_name, ratio_str in selected.items():
        target_w, target_h = [int(x) for x in size_name.split("x")]

        try:
            resized = ai_resize(client, img, description, ratio_str)
        except Exception:
            resized = None

        if resized is None:
            resized = center_crop_fallback(img, ratio_str)

        # Resize to exact target pixel dimensions
        resized = resized.resize((target_w, target_h), Image.LANCZOS)

        crop_filename = f"{file_id}_{size_name}.jpg"
        crop_path = os.path.join(app.config["UPLOAD_FOLDER"], crop_filename)
        if resized.mode == "RGBA":
            resized = resized.convert("RGB")
        resized.save(crop_path, "JPEG", quality=90)
        results["crops"][size_name] = f"/static/uploads/{crop_filename}"

    return jsonify(results)


@app.route("/refine", methods=["POST"])
def refine():
    generated_url = request.form.get("generated_url", "")
    original_url = request.form.get("original_url", "")
    original_description = request.form.get("original_description", "")
    feedback = request.form.get("feedback", "")
    size_name = request.form.get("size", "")

    if not generated_url or not original_url or not feedback or not size_name:
        return jsonify({"error": "Missing required fields"}), 400

    # Load images
    generated_path = os.path.join(".", generated_url.lstrip("/"))
    original_path = os.path.join(".", original_url.lstrip("/"))

    if not os.path.exists(generated_path) or not os.path.exists(original_path):
        return jsonify({"error": "Image not found"}), 404

    generated_img = Image.open(generated_path)
    original_img = Image.open(original_path)
    if original_img.mode == "RGBA":
        original_img = original_img.convert("RGB")
    if generated_img.mode == "RGBA":
        generated_img = generated_img.convert("RGB")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        return jsonify({"error": "No API key configured"}), 500

    client = genai.Client(api_key=api_key)

    # Step 1: Describe the generated image
    generated_description = describe_image(client, generated_img)

    # Step 2: Compute aspect ratio
    if size_name in ASPECT_RATIOS:
        ratio_str = ASPECT_RATIOS[size_name]
    else:
        ratio_str = _nearest_aspect_ratio(size_name)

    target_w, target_h = [int(x) for x in size_name.split("x")]

    # Step 3: Generate refined image using all context
    prompt = (
        f"Act as a professional Art Director using high-end photo editing software.\n"
        f"You previously resized a marketing asset to {ratio_str} aspect ratio, but the client has feedback.\n\n"
        f"ORIGINAL IMAGE DESCRIPTION:\n{original_description}\n\n"
        f"CURRENT GENERATED IMAGE DESCRIPTION:\n{generated_description}\n\n"
        f"CLIENT FEEDBACK:\n{feedback}\n\n"
        f"Please regenerate the image at {ratio_str} aspect ratio, addressing the client's feedback.\n\n"
        f"CRITICAL INSTRUCTION - STRICT SUBJECT PRESERVATION:\n"
        f"The input image contains a SPECIFIC REAL-WORLD OBJECT (e.g., a specific house, building, car, or consumer product).\n"
        f"You MUST preserve the identity of this central subject exactly.\n\n"
        f"DO NOT:\n"
        f"- Do NOT generate a different house, building, or product.\n"
        f"- Do NOT change the architectural details (windows, doors, roof).\n"
        f"- Do NOT change the product packaging or logo.\n"
        f"- Do NOT hallucinate a 'similar' looking object. It must represent the EXACT object in the source image.\n\n"
        f"DO:\n"
        f"1. Keep the main subject visually unchanged. Treat the pixels of the product/house as immutable.\n"
        f"2. Create a NEW layout composition by extending the background (outpainting) or repositioning text elements if they exist.\n"
        f"3. Ensure the lighting on the extended background matches the original scene perfectly.\n"
        f"4. Render in photorealistic 4k quality."
    )

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            #model="gemini-2.5-flash-image",
            contents=[prompt, original_img],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=ratio_str),
            ),
        )

        resized = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                resized = Image.open(BytesIO(part.inline_data.data))
                break

        if resized is None:
            return jsonify({"error": "AI failed to generate image"}), 500
    except Exception:
        return jsonify({"error": "AI generation failed"}), 500

    # Resize to exact target dimensions
    resized = resized.resize((target_w, target_h), Image.LANCZOS)
    if resized.mode == "RGBA":
        resized = resized.convert("RGB")

    file_id = uuid.uuid4().hex[:8]
    crop_filename = f"{file_id}_{size_name}.jpg"
    crop_path = os.path.join(app.config["UPLOAD_FOLDER"], crop_filename)
    resized.save(crop_path, "JPEG", quality=90)

    return jsonify({"url": f"/static/uploads/{crop_filename}"})


if __name__ == "__main__":
    app.run(debug=True, port=5000,host="0.0.0.0")
