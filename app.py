from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import Model, load_model
from utils_tourism import preprocess_image, get_tourism_prediction, get_tourism_data, get_tourism_classes
from utils_hieroglyphs import get_hieroglyphs_classes, predict_hieroglyphs

app = Flask(__name__)

# Load model_tourism model
model_tourism = load_model('model_tourism.h5')
# Load model_hieroglyphs model
model_hieroglyphs = load_model('model_hieroglyphs.h5')


# Define a function to preprocess an image


@app.route('/')
def index():
    return render_template('index.html', prediction=None)


@app.route('/predictTourism', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        selected_language = request.form['language']
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        # Load tourism data
        data_dict = get_tourism_data()

        # load classes names
        class_names = get_tourism_classes()

        # Load and preprocess the image
        processed_image = preprocess_image(image_path)

        # Make a prediction
        translated_text, predicted_class_name = get_tourism_prediction(processed_image, model_tourism, data_dict,
                                                                       class_names, selected_language)
        if selected_language != 'en':
            translated_text = translated_text.text
        else:
            translated_text = translated_text
        return render_template('index.html', prediction=translated_text)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predictTourismAPI', methods=['POST'])
def predictapi():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        selected_language = request.form['language']
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        # Load tourism data
        data_dict = get_tourism_data()

        # load classes names
        class_names = get_tourism_classes()

        # Load and preprocess the image
        processed_image = preprocess_image(image_path)

        # Make a prediction
        translated_text, predicted_class_name = get_tourism_prediction(processed_image, model_tourism, data_dict,
                                                                       class_names, selected_language)
        if selected_language != 'en':
            translated_text = translated_text.text
        else:
            translated_text = translated_text
        return jsonify({
            "information": translated_text,
            "name": predicted_class_name
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predictHieroglyphs', methods=['POST'])
def predict2():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        # load classes names
        class_names = get_hieroglyphs_classes()

        # Load and preprocess the image
        processed_image = preprocess_image(image_path)

        # Make a prediction
        prediction = predict_hieroglyphs(model_hieroglyphs, processed_image, class_names)

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predictHieroglyphsAPI', methods=['POST'])
def predictapi2():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        # load classes names
        class_names = get_hieroglyphs_classes()

        # Load and preprocess the image
        processed_image = preprocess_image(image_path)

        # Make a prediction
        prediction = predict_hieroglyphs(model_hieroglyphs, processed_image, class_names)

        return jsonify({"class": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
