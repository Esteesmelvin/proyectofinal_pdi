from flask import Flask, request, send_file, render_template, jsonify, redirect, url_for
from rembg import remove
from PIL import Image, ImageEnhance
import tensorflow as tf
import numpy as np
import io
import os

app = Flask(__name__)

# Inicializar la carpeta para las imágenes procesadas
processed_images_folder = 'static/processed_images'
os.makedirs(processed_images_folder, exist_ok=True)

# Cargar el modelo HDF5 (.h5)
model = tf.keras.models.load_model('nsfw_mobilenet2.224x224.h5', compile=False)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

@app.route('/')
def index():
    return render_template('index.html')

def es_inapropiada(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    resultados = model.predict(img_array)
    probabilidad_nsfw = resultados[0][3] + resultados[0][1] + resultados[0][4]
    
    if probabilidad_nsfw > 0.5:
        return True
    else:
        print("La imagen es apropiada para su uso.")
        return False

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_bytes = file.read()

    if es_inapropiada(image_bytes):
        return jsonify({"error": "La imagen contiene contenido inapropiado y no puede ser procesada."}), 400

    input_image = Image.open(io.BytesIO(image_bytes)).convert("L")
    enhancer = ImageEnhance.Contrast(input_image)
    high_contrast_image = enhancer.enhance(4)
    rgba_image = high_contrast_image.convert("RGBA")
    input_array = np.array(rgba_image)
    output_array = remove(input_array, alpha_matting=False)
    output_image = Image.fromarray(output_array)

    # Guardar la imagen en un buffer de bytes para enviarla como respuesta
    img_io = io.BytesIO()
    output_image.save(img_io, 'PNG')  # Asegúrate de que el formato sea PNG
    img_io.seek(0)

    # Guardar la imagen en la carpeta de imágenes procesadas
    image_count = len(os.listdir(processed_images_folder)) + 1
    image_path = os.path.join(processed_images_folder, f"user_{image_count}.png")
    output_image.save(image_path)

    return send_file(img_io, mimetype='image/png')

# Ruta para la página de collage
@app.route('/collage')
def collage():
    # Obtener solo los nombres de archivos en lugar de rutas completas
    image_paths = [os.path.basename(filename) for filename in os.listdir(processed_images_folder)]
    image_count = len(image_paths)
    return render_template('collage.html', image_count=image_count, image_paths=image_paths)

@app.route('/log', methods=['POST'])
def log():
    data = request.json
    # Imprime el mensaje de depuración en la consola de Python
    print(f"Debug log: {data.get('message')}")
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(debug=True)
