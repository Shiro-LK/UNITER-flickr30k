#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:38:59 2021

@author: shiro
"""
import os
import sys
sys.path.append("/workspace/UNITER")
import numpy as np
import torch
from torch.cuda.amp import autocast
from model.itm import UniterForImageTextRetrieval
from pytorch_pretrained_bert import BertTokenizer
def _compute_nbb(img_dump, conf_th, max_bb, min_bb, num_bb):
    num_bb = max(min_bb, (img_dump['conf'] > conf_th).sum())
    num_bb = min(max_bb, num_bb)
    return int(num_bb)

def load_npz(conf_th, max_bb, min_bb, num_bb, fname, keep_all=False):
    try:
        img_dump = np.load(fname, allow_pickle=True)
        if keep_all:
            nbb = None
        else:
            nbb = _compute_nbb(img_dump, conf_th, max_bb, min_bb, num_bb)
        dump = {}
        for key, arr in img_dump.items():
            if arr.dtype == np.float32:
                arr = arr.astype(np.float16)
            if arr.ndim == 2:
                dump[key] = arr[:nbb, :]
            elif arr.ndim == 1:
                dump[key] = arr[:nbb]
            else:
                raise ValueError('wrong ndim')
    except Exception as e:
        # corrupted file
        print(f'corrupted file {fname}', e)
        dump = {}
        nbb = 0

    name = os.path.basename(fname)
    return name, dump, nbb


class Preprocess(object):
    def __init__(self, tokenizer_name='bert-base-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.cls_ = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.sep_ = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.unk_ = self.tokenizer.convert_tokens_to_ids(['[UNK]'])[0]
        self.mask_ = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    def bert_tokenize(self, text):
        ids = []
        for word in text.strip().split():
            ws = self.tokenizer.tokenize(word)
            if not ws:
                # some special char
                continue
            ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
        return [self.cls_] + ids + [self.sep_]
    def _get_img_feat(self, image_npz):
        img_feat, bb = torch.as_tensor(image_npz["features"]).float(), torch.as_tensor(image_npz["norm_bb"]).float()
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        num_bb = img_feat.size(0)
        return img_feat.unsqueeze(0), img_bb.unsqueeze(0), num_bb
    
    def get_gather_index(self, txt_lens, num_bbs, batch_size, max_len, out_size):
        assert len(txt_lens) == len(num_bbs) == batch_size
        gather_index = torch.arange(0, out_size, dtype=torch.long,
                                    ).unsqueeze(0).repeat(batch_size, 1)
    
        for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
            gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                           dtype=torch.long).data
        return gather_index
    def __call__(self, text, image_npz):
        
        # text 
        input_ids = torch.as_tensor(self.bert_tokenize(text)).unsqueeze(0)
        tl = input_ids.size(1)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
        
        # images
        img_feats, img_pos_feats, num_bbs = self._get_img_feat(image_npz)
        
        # attn and gather index
        attn_masks = torch.ones(len(input_ids), num_bbs + tl).long() # no padding because batch 1
        out_size = attn_masks.size(1)
        gather_index = self.get_gather_index([tl]*len(input_ids), [num_bbs],
                                        len(input_ids), tl, out_size)
        #print(gather_index)
        batch = {"input_ids": input_ids, "position_ids": position_ids, "attn_masks":attn_masks, 
                 "img_feat":img_feats, "img_pos_feat": img_pos_feats, "gather_index":gather_index}
        
        return batch
    
def to_device(batches, device):
    for k, v in batches.items():
        batches[k] = batches[k].to(device)
    return batches
if __name__ == "__main__": 
    device = "cuda"
    IMG_DIM = 2048
    model_config = "/workspace/UNITER/config/uniter-base.json"
    checkpoints= "model_step_16000.pt"
    save_to_folder = ""
    prefix = "nlvr2_"
    image_id = "1001773457"
    output_file = os.path.join(save_to_folder, prefix+image_id+".npz")
    # convert in similar format lmb for inference
    conf_th=0.2
    max_bb=100
    min_bb=10
    num_bb=36
    
    fname, features, nbb = load_npz(conf_th=conf_th, max_bb=max_bb, min_bb=min_bb, num_bb=num_bb, fname=output_file, keep_all=False)
    processor = Preprocess(tokenizer_name='bert-base-cased')
    batch = processor("there are two dogs", features)
    batch = to_device(batch, device)
    #print(batch)
    # load model 
    weights = torch.load(checkpoints)
    model = UniterForImageTextRetrieval.from_pretrained(model_config, state_dict=weights, img_dim=IMG_DIM, margin=0.2).to(device)
    model.eval()
    print(model)
    with torch.no_grad():
    
        print(torch.sigmoid(model(batch, False)))