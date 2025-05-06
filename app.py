import os
from flask import Flask, render_template, request
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Carrega o modelo BLIP e processador
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Carrega o tradutor de inglês para português
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-pt")


@app.route("/", methods=["GET", "POST"])
def index():
    caption = ""
    image_path = ""
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            raw_image = Image.open(image_path).convert("RGB")

            # Gera legenda em inglês
            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs)
            caption_en = processor.decode(out[0], skip_special_tokens=True)

            # Traduz para português
            caption_pt = translator(caption_en, max_length=60)[0]["translation_text"]
            caption = caption_pt

    return render_template("index.html", caption=caption, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
