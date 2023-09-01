from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array

app = Flask(__name__)

model = tf.keras.models.load_model("predection_001.h5")
model.make_predict_function()
print("Model loaded successfullt")

def model_predict(image_path, model):
    img = load_img(image_path,target_size = (224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    preds = model.predict(img)
    return preds

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        if file.filename == '':
            print = " "
            return render_template("prediction.html", print = print)
        else:
            file.save(os.path.join(app.config['UPLOAD'], filename))
            img = os.path.join(app.config['UPLOAD'], filename)
        
        preds = model_predict(img, model)

        if preds[0][0] > preds[0][1]:
            predict = "COVID19"
        else:
            predict = "Non-COVID19"
        return render_template('result.html',prediction = predict, img=img)
    return render_template('prediction.html')


if __name__ == '__main__':
    app.run(debug=True, port=8001)
