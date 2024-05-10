import base64
import glob
import os
from iptcinfo3 import IPTCInfo
from langchain_community.llms import Ollama
from io import BytesIO
from PIL import Image
from pathlib import Path

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
SOURCE_DIR = "."
TARGET_DIR = ""

def convert_to_base64(pil_image):
    buffered = BytesIO()
    rgb_im = pil_image.convert('RGB')
    rgb_im.save(buffered, format="JPEG") 
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def main():
    # Verbindung zu LlaVA 1.6 
    llava_model = Ollama(model="llava:v1.6", base_url=OLLAMA_BASE_URL, temperature=0)
    prompt = "Please find very precise keywords and separate them with commas."

    # Über alle Bilder des Quellordners
    for img_filename in glob.glob(f"{SOURCE_DIR}/*.JPG"):  
        # Lese Bild ein...
        print(f"Image '{img_filename}' processing...")
        info = IPTCInfo(img_filename, force=True, inp_charset='utf8')
        pil_image = Image.open(img_filename)

        # Bild auf 672 Pixel beschneiden
        base_width= 672
        wpercent = (base_width / float(pil_image.size[0]))
        hsize = int((float(pil_image.size[1]) * float(wpercent)))
        pil_image = pil_image.resize((base_width, hsize), Image.Resampling.LANCZOS)

        # Bild nach Base64 konvertieren und dem Modell zusammen mit dem Prompt übergeben
        image_b64 = convert_to_base64(pil_image)
        llm_with_image_context = llava_model.bind(images=[image_b64])
        response = llm_with_image_context.invoke(prompt)
		 
        # Antwort des Modells aufbereiten
        response = response.replace(" ", "")
        keywords = response.split(',')
        print(response)
        
        # Neues Bild im Ordner Zielbilder erzeugen
        output_filename = Path(os.path.basename(img_filename)).stem
        output_filename = f"{output_filename}_meta.jpg"
        output_file = os.path.join(TARGET_DIR, output_filename)

        # IPTC Felder ausfüllen
        info['writer/editor'] = Path(__file__).name
        print(Path(__file__).name)
        info['object name'] = output_filename
        info['keywords'] = keywords
        
        # Neue Bilddatei speichern
        print(f"save to '{output_file}'")
        info.save_as(output_file)

if __name__ == "__main__":
    main()

