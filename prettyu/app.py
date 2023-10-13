from flask import Flask, request, render_template, jsonify
from io import BytesIO
import base64
from werkzeug.security import generate_password_hash, check_password_hash
import threading
import logging
import os
from werkzeug.utils import secure_filename

from sd import interfaces as sd_api

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 配置uploads目录为静态目录，前端就可以访问了
app = Flask(__name__, static_folder=UPLOAD_FOLDER, static_url_path='/'+UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

image_cache = {}

def gen_img(task_id, prompt='', negative_prompt='', batch_size=1):
    global image_cache
    images_base64 = sd_api.txt2img(prompt=prompt, negative_prompt=negative_prompt, batch_size=batch_size)
    image_cache[task_id] = ["data:image/jpeg;base64," + im for im in images_base64]
    return


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    task_id = request.form.get('task_id')
    positive_prompt = request.form.get('positive_prompt')
    negative_prompt = request.form.get('negative_prompt')
    batch_size = request.form.get('batch_size')
    
    # images_base64 = sd_api.txt2img(prompt=positive_prompt, negative_prompt=negative_prompt)
    # return jsonify({'image_base64': images_base64[0]})
    t = threading.Thread(target=gen_img, kwargs={"task_id": task_id, "prompt": positive_prompt, "negative_prompt": negative_prompt, "batch_size": batch_size})
    t.start()
    return jsonify({"status": "started"})
    

@app.route('/getProgress', methods=['GET'])
def get_progress():
    global image_cache

    task_id = request.args.get("task_id")
    p = sd_api.progress()
    p = float(p*100)
    if (p >= 100 or p <= 0) and task_id in image_cache:
        p = 100
        images_base64 = image_cache.pop(task_id)
        return jsonify({"progress": p, "status": "complete", "images": images_base64})
    else:
        if p >= 100:
            p = 99

        return jsonify({"progress": p, "status": "waiting"})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'images' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    files = request.files.getlist('images')
    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({'success': True})

@app.route('/get-images', methods=['GET'])
def get_images():
    images = [f"/{UPLOAD_FOLDER}/{image}" for image in os.listdir(UPLOAD_FOLDER)]
    return jsonify({'images': images})

@app.route('/delete', methods=['POST'])
def delete_image():
    image_path = request.form.get('image', '')[1:]  # Remove leading slash
    if os.path.exists(image_path):
        os.remove(image_path)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run()
