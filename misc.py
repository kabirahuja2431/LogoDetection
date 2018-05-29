
# coding: utf-8

# In[25]:


import selectivesearch
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

def scale(img):
    return (img.astype(np.float32) - (255/2))/255

def get_region_proposals(img,scale = 500, sigma = 0.9,min_size = 10):
    img_lbl,regions = selectivesearch.selective_search(img,scale,sigma,min_size)
    proposals = set()
    for region in regions:
        if(region['rect'] in proposals):
            continue
        x,y,w,h = region['rect']
        if(region['size'] < 2000 or w > 0.95*img.shape[1] or h > 0.95*img.shape[0]): 
            continue
        if(x+w > img.shape[1] or y+h >img.shape[0]):
            continue
        if(w == 0 or h == 0):
            continue
        if(w/h >5 or h/w > 5):
            continue
        proposals.add(region['rect'])
    return proposals

def iou(box1, box2):
    x11,y11,x21,y21 = box1
    x12,y12,x22,y22 = box2
    x1 = max(x11,x12)
    y1 = max(y11,y12)
    x2 = min(x21,x22)
    y2 = min(y21,y22)
    inter_area = (x2-x1)*(y2-y1)
    box1_area = (x21-x11)*(y21-y11)
    box2_area = (x22-x12)*(y22-y12)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area/union_area
    return iou

def get_sorted_idx(results):
    prob = np.array([result['prob'] for result in results])
    idx = np.argsort(prob)[::-1]
    return idx

def nms(results, prob_thresh, min_iou):
    thres_results = []
    for i in range(len(results)):
        if(results[i]['prob'] > prob_thresh):
            thres_results.append(results[i])

    nms_results = []
    if(len(thres_results) == 0):
        return thres_results
    thres_results = np.array(thres_results)
    idx = get_sorted_idx(thres_results)
    thres_results = thres_results[idx]
    max_prob_result = thres_results[0]
    nms_results.append(max_prob_result)
    thres_results = thres_results[1:]

    idx = get_sorted_idx(thres_results)
    thres_results = thres_results[idx]

    while(len(thres_results) > 0):
        candidates = []
        for i in range(len(thres_results)):
            x1,y1,w1,h1 = thres_results[i]['region']
            x2,y2,w2,h2 = max_prob_result['region']
            if(iou((x1,y1,x1+w1,y1+h1),(x2,y2,x2+w2,y2+h2)) < min_iou):
                candidates.append(thres_results[i])

        if(len(candidates) == 0):
            break

        candidates = np.array(candidates)
        idx = get_sorted_idx(candidates)
        candidates = candidates[idx]
        max_prob_result = candidates[0]
        thres_results = candidates[1:]
        if(len(thres_results) ==0):
            break
        nms_results.append(max_prob_result)

    return nms_results


def get_bg(img,annot):
    proposals = get_region_proposals(img)
    bg = None
    for region in proposals:
        x,y,w,h = region
        box2 = [x,y,w+x,h+y]
        if iou(annot,box2) < 0.2:
            bg = box2
            break
    return bg



