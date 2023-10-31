from flask import Flask, render_template, request, redirect, url_for, flash 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


#api request  


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)