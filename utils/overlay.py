import cv2


def annotate_image(image_path, text, output_path):
    image = cv2.imread(image_path)
    cv2.putText(image, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imwrite(output_path, image)
