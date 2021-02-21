#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 23:57:21 2021

@author: shiro
"""

from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename
from app.forms import PredictionForm
import os
from app.process import Preprocess, load_npz, to_device
import sys
sys.path.append("/workspace/UNITER")
from model.itm import UniterForImageTextRetrieval
import torch
## Parameters model
device = 0
IMG_DIM = 2048
model_config = "/workspace/UNITER/config/uniter-base.json"
proto = '/workspace/BUTD-UNITER-NLVR2/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'
netDir = "/workspace/data/faster_rcnn_models/"
yaml = "/workspace/BUTD-UNITER-NLVR2/experiments/cfgs/faster_rcnn_end2end_resnet.yml"

save_folder = "app/static/temp"
checkpoints= "model_step_16000.pt"
prefix = "nvlr2"
conf_th=0.2
max_bb=100
min_bb=10
num_bb=36
processor = Preprocess(tokenizer_name='bert-base-cased')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


weights = torch.load(checkpoints)
model = UniterForImageTextRetrieval.from_pretrained(model_config, state_dict=weights, img_dim=IMG_DIM, margin=0.2).to(device)
model.eval()
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/')
def upload_form():
    form = PredictionForm()
    return render_template('upload.html', form=form)

@app.route('/', methods=['POST'])
def upload_image():   
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        session["text"] = request.form["text"]
        session["filename"] = filename
        return redirect('results')
    else:
        print('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)
@app.route('/results', methods=['GET', 'POST'])
def results_prediction():
    filename = session["filename"]
    text = session["text"] if "text" in session else "No input text"
    
    ## compute the score
    # 1) extract image features

    os.system(f"python app/extract_features.py --gpu={device} --netDir={netDir} --def={proto} --filename={'app'+app.config['UPLOAD_FOLDER'] + filename} --cfg={yaml} --prefix={prefix} --out={save_folder}")
    if not os.path.exists('app'+app.config['UPLOAD_FOLDER'] + filename) :
        return redirect(request.url)
    score =  -1
    npz_filename = os.path.join(save_folder, prefix + "_" + os.path.splitext(os.path.split(filename)[-1])[0] + ".npz")
    
    if not os.path.exists(npz_filename) :
        return redirect(request.url)
    
    fname, features, nbb = load_npz(conf_th=conf_th, max_bb=max_bb, min_bb=min_bb, num_bb=num_bb, fname=npz_filename, keep_all=False)
    os.system(f"rm {npz_filename}")
    # create batch
    batch = processor(text, features)
    batch = to_device(batch, device)
    # prediction
    with torch.no_grad():
        score = torch.sigmoid(model(batch, False)).cpu().numpy()[0][0]
    return render_template('results.html' , text=text, score=score, filename=app.config["UPLOAD_FOLDER"] + filename)

