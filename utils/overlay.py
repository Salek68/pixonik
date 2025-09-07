from PIL import Image, ImageDraw, ImageFont

def annotate_image(image_path, text, output_path):
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Draw arrow (simple line)
    draw.line((width*0.1, height*0.9, width*0.9, height*0.1), fill="red", width=5)

    # Add text
    try:
        # مسیر فونت فارسی رو درست بده
        font = ImageFont.truetype("/usr/share/fonts/truetype/vazir/Vazir-Regular.ttf", 40)
    except:
        font = ImageFont.load_default()

    draw.text((20, 20), text, font=font, fill="yellow")
    image.save(output_path)
    return output_path
