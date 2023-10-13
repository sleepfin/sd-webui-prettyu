import io
import glob
import base64

import cv2
import numpy as np
from PIL import Image


def read_img_rgb(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def base64_to_img(image_base64):
    if isinstance(image_base64, str):
        image_base64 = image_base64.encode('utf-8')

    if not isinstance(image_base64, bytes):
        raise ValueError(f'input image_base64 should be bytes or str, get {type(image_base64)}')
    
    base64_decoded = base64.b64decode(image_base64)
    img = np.array(Image.open(io.BytesIO(base64_decoded)))
    return img


def img_to_sdwebui_base64(img):
    buffer = io.BytesIO()
    _, img_encoded = cv2.imencode('.png', img)
    buffer.write(img_encoded.tobytes())
    image_base64 = "data:image/png;base64," + str(base64.b64encode(buffer.getvalue()), 'utf-8')
    return image_base64
    


def save_img(img, path):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def grid_images(images_pattern, save_path, size=512, grid_size=(6, 4), spacing=10):
    images = [Image.open(img) for img in glob.glob(images_pattern)]
    # Create a new image with white background
    width, height = size, size
    new_width = grid_size[0] * width + (grid_size[0] - 1) * spacing
    new_height = grid_size[1] * height + (grid_size[1] - 1) * spacing   
    new_image = Image.new("RGB", (new_width, new_height), "white")

    # Paste each image into the new image
    for i in range(len(images)):
        w, h = images[i].size
        if w > 512:
            left = (w - 512) // 2
            right = left + 512
            print(left, right)
            images[i] = images[i].crop((left, 0, right, height))

        row = i // grid_size[0]
        col = i % grid_size[0]
        x = col * (width + spacing)
        y = row * (height + spacing)
        new_image.paste(images[i], (x, y))

    # Save the new image
    new_image.save(save_path)
