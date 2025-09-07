import cv2
import numpy as np

def draw_scene_croqui(objects, output_path):
    # ایجاد یک بوم سفید
    canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255

    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        # مقیاس ساده برای کروکی
        x1_c, y1_c = int(x1 / 2), int(y1 / 2)
        x2_c, y2_c = int(x2 / 2), int(y2 / 2)
        cv2.rectangle(canvas, (x1_c, y1_c), (x2_c, y2_c), (0, 0, 255), 2)
        cv2.putText(canvas, obj['type'], (x1_c, y1_c - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # ذخیره تصویر کروکی
    cv2.imwrite(output_path, canvas)
