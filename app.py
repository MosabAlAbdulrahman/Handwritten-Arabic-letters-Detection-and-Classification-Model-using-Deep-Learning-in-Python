from flask import Flask, request, jsonify
import h5py
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Paths to the models
model_paths = [
    'alif.h5', 'baa.h5', 'taa.h5', 'tha.h5', 'gim.h5', 'haa.h5',
    'kha.h5', 'dal.h5', 'thal.h5', 'raa.h5', 'zay.h5', 'sin.h5',
    'shin.h5', 'sad.h5', 'dad.h5', 'da.h5', 'za.h5', 'ayn.h5',
    'gayn.h5', 'fa.h5', 'qaf.h5', 'kaf.h5', 'lam.h5', 'mim.h5',
    'non.h5', 'ha.h5', 'waw.h5', 'ya.h5'
]

# Character mapping
chars_map = [
                        # /* ALEF */
                        ['0x0627', '0xfe8d', None, None, '0xfe8e'],
                        # /* BEH ب */
                        ['0x0628', '0xfe8f', '0xfe91', '0xfe92', '0xfe90'],
                        # /* TEH ت */
                        ['0x062a', '0xfe95', '0xfe97', '0xfe98', '0xfe96'],
                        # /* THEH ث */
                        ['0x062b', '0xfe99', '0xfe9b', '0xfe9c', '0xfe9a'],
                        # /* JEEM ج */
                        ['0x062c', '0xfe9d', '0xfe9f', '0xfea0', '0xfe9e'],
                        # /* HAH ح*/
                        ['0x062d', '0xfea1', '0xfea3', '0xfea4', '0xfea2'],
                        # /* KHAH خ */
                        ['0x062e', '0xfea5', '0xfea7', '0xfea8', '0xfea6'],
                        # /* DAL د */
                        ['0x062f', '0xfea9', None, None, '0xfeaa'],
                        # /* THAL ذ */
                        ['0x0630', '0xfeab', None, None, '0xfeac'],
                        # /* REH ر */
                        ['0x0631', '0xfead', None, None, '0xfeae'],
                        # /* ZAIN ز */
                        ['0x0632', '0xfeaf', None, None, '0xfeb0'],
                        # /* SEEN س */
                        ['0x0633', '0xfeb1', '0xfeb3', '0xfeb4', '0xfeb2'],
                        # /* SHEEN ش */
                        ['0x0634', '0xfeb5', '0xfeb7', '0xfeb8', '0xfeb6'],
                        # /* SAD ص */
                        ['0x0635', '0xfeb9', '0xfebb', '0xfebc', '0xfeba'],
                        # /* DAD ض */
                        ['0x0636', '0xfebd', '0xfebf', '0xfec0', '0xfebe'],
                        # /* TAH ط */
                        ['0x0637', '0xfec1', '0xfec3', '0xfec4', '0xfec2'],
                        # /* ZAH ظ */
                        ['0x0638', '0xfec5', '0xfec7', '0xfec8', '0xfec6'],
                        # /* AIN ع */
                        ['0x0639', '0xfec9', '0xfecb', '0xfecc', '0xfeca'],
                        # /* GHAIN غ */
                        ['0x063a', '0xfecd', '0xfecf', '0xfed0', '0xfece'],
                        # /* FEH ف */
                        ['0x0641', '0xfed1', '0xfed3', '0xfed4', '0xfed2'],
                        # /* QAF ق */
                        ['0x0642', '0xfed5', '0xfed7', '0xfed8', '0xfed6'],
                        # /* KAF ك */
                        ['0x0643', '0xfed9', '0xfedb', '0xfedc', '0xfeda'],
                        # /* LAM ل */
                        ['0x0644', '0xfedd', '0xfedf', '0xfee0', '0xfede'],
                        # /* MEEM م */
                        ['0x0645', '0xfee1', '0xfee3', '0xfee4', '0xfee2'],
                        # /* NOON ن */
                        ['0x0646', '0xfee5', '0xfee7', '0xfee8', '0xfee6'],
                        # /* HEH ه */
                        ['0x0647', '0xfee9', '0xfeeb', '0xfeec', '0xfeea'],
                        # /* WAW و */
                        ['0x0648', '0xfeed', None, None, '0xfeee'],
                        # /* YEH ي */
                        ['0x064a', '0xfef1', '0xfef3', '0xfef4', '0xfef2'],
                    ]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the class number and image data from the request
        class_num = int(request.form['class_num'])
        image_data = request.files['image'].read()

        # Ensure class_num is within valid range
        if class_num < 0 or class_num >= len(model_paths):
            raise ValueError(f"Invalid class number: {class_num}")

        # Load model for the specified class
        model = load_model(model_paths[class_num])

        # Preprocess the image
        image = Image.open(io.BytesIO(image_data)).convert('RGB').resize((224, 224))
        image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the class
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])

        # Get the corresponding Unicode character
        prediction_unicode = chars_map[class_num][predicted_class]

        response = {
            'class_num': class_num,
            'prediction': prediction_unicode,
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
