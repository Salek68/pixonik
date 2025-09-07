from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image

def export_storyboard_pdf(shots, annotated_images, croqui_images, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    for idx, shot in enumerate(shots):
        # نوشتن عنوان و توضیح شات
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height-50, f"{shot['title']} - {shot['description']}")

        # تصویر Annotated
        if idx < len(annotated_images):
            img = Image.open(annotated_images[idx])
            img_w, img_h = img.size
            ratio = min((width-100)/img_w, (height/2-80)/img_h)
            img_w, img_h = int(img_w*ratio), int(img_h*ratio)
            img = img.resize((img_w, img_h))
            img_path_tmp = f"tmp_annotated_{idx}.png"
            img.save(img_path_tmp)
            c.drawImage(img_path_tmp, 50, height/2, width=img_w, height=img_h)

        # تصویر Croqui
        if idx < len(croqui_images):
            img = Image.open(croqui_images[idx])
            img_w, img_h = img.size
            ratio = min((width-100)/img_w, (height/2-80)/img_h)
            img_w, img_h = int(img_w*ratio), int(img_h*ratio)
            img = img.resize((img_w, img_h))
            img_path_tmp = f"tmp_croqui_{idx}.png"
            img.save(img_path_tmp)
            c.drawImage(img_path_tmp, 50, 50, width=img_w, height=img_h)

        c.showPage()  # شروع صفحه جدید

    c.save()
