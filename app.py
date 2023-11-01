from flask import Flask, render_template, request, redirect, flash
import torch 
from src.DronVid.components.utils.common import read_yaml 
from PIL import Image
import numpy as np
import json
import albumentations as A 
import matplotlib.pyplot as plt

# t = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.GaussNoise(),
#              A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5), p=0.4),A.Resize(512, 512)])

t= A.Resize(512, 512)


config = read_yaml('./config/config.yaml')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

model = torch.load(r'./artifacts\model_ckpt\best_model.pth').cpu()
@app.route('/', methods=['POST'])

def predict():
    count = 0
    if request.method == 'POST':
        file = request.files['imagefile']
        if not file:
            return render_template('index.html', label="No file")
        file.save('./static/image.png')
        img = Image.open(file)
        img = np.array(img).astype(np.float32)
        img = t(image=img)['image']
        #save the image to sataic folder
        img = np.transpose(img,(2,0,1))
        #save the image to sataic folder 


        img = torch.tensor(img).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            prediction = model(img).cpu()   
            prediction = np.argmax(prediction[0], axis=0)
            # prediction = np.array(prediction).astype(np.float32)
            # prediction = Image.fromarray(prediction)
            # prediction.save('./static/mask.png')

            plt.close()
            prediction = np.array(prediction)
            plt.imshow(prediction, cmap='cividis')
            plt.axis('off')
            plt.savefig('./static/mask.png')
            plt.close()
            count += 1

    

        return render_template('index.html', label="Uploaded", image = './static/image.png', mask = './static/mask.png', count = count)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)