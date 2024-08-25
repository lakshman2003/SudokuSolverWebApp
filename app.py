from flask import Flask, render_template,request
from Model import model
from urllib.request import urlopen
from SudokuSolver import Solve
import numpy as np
import base64
import io
import cv2
from PIL import Image
import requests
from io import BytesIO
 

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/info')
def INFO():
    return render_template("info.html")
    

@app.route("/", methods=["POST"])
def predict():
    path = request.form['yoo']

    if(path==''):
        return render_template("index.html", img_data = "", error = "URL cannot be empty!!!")
    
    req = urlopen(path)
    print(req)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1) 
    
    img,sol = Solve(img,model)

    if(sol == False):
        return render_template("index.html", img_data = "", error="Cannot find a solution:(")
    

    img = cv2.resize(img,(400,400))
    img = Image.fromarray(img)

    data = io.BytesIO()
    img.save(data,"JPEG")
    encodeed_img_data = base64.b64encode(data.getvalue())

    return render_template("index.html", img_data = encodeed_img_data.decode('utf-8'), error="Successful!!")


if __name__ == '__main__':
    app.run(debug=True)