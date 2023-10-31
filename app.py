from flask import Flask, render_template, request, redirect, url_for, flash 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


#api request  
@app.route('/predict', methods=['POST'])  
def predict():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        return render_template('index.html', label=label)