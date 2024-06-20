from flask import Flask, render_template, request,redirect
import matlab.engine
import scipy.io
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'edf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
eng = matlab.engine.start_matlab()
###################################################################################
gpus = tf.config.list_physical_devices('GPU')
if gpus: 
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3800)]
    )

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
#######################################################################################

#model = tf.keras.models.load_model("model_lstm_max_accuracy_bin.keras")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        m = request.form['modelSelect']
        global model
        global wordlist
        global selected_model
        if m == "model1":
            model = tf.keras.models.load_model("model_lstm_max_accuracy_bin.keras")
            wordlist = ["No","Toilet"]
            selected_model = "Binary Classification Model"
        elif m == "model2":
            model = tf.keras.models.load_model("model_lstm_max_accuracy_3.keras")
            wordlist = ["No","Toilet","Television"]
            selected_model = "Ternary Classification Model"
        return redirect('/file-upload')
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/file-upload')
def file_upload():
    global selected_model
    return render_template('index.html',selected_model=selected_model)

@app.route('/model')
def Model_page():
    return render_template('models.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'input.edf'))
        return redirect('/output')

    return 'Invalid file type'

@app.route('/output')
def output():
    global word
    eng.edf_single_read(nargout=0)
    data = scipy.io.loadmat('static/data.mat')
    x = data['data']
    x = np.expand_dims(x, axis=0)
    yhat= model.predict(x)
    y = np.argmax(yhat)
    word = wordlist[y]
    return redirect('/output_w')

@app.route('/output_w')
def output_w():
    global word
    return render_template('output.html',predicted_word=word)


    




if __name__ == "__main__":
    app.run(host='0.0.0.0',port='5000',debug=True)
