import numpy as np
from PIL import Image

eps = 2.2204e-16

def ssim(pred, gt):
    size = pred.size
    
    x = np.mean(pred)
    y = np.mean(gt)
    
    sig_x = np.sum((pred - x) ** 2) / (size - 1 + eps)
    sig_y = np.sum((gt - y) ** 2) / (size - 1 + eps)
    sig_xy = np.sum((pred - x) * (gt - y)) / (size - 1 + eps)
    
    alpha = 4 * x * y * sig_xy
    beta = (x**2 + y**2) * (sig_x + sig_y)
    
    #print(x, y, sig_x, sig_y, sig_xy, alpha, beta)
    
    if alpha != 0:
        Q = alpha / (beta + eps)
    elif alpha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q
    

def divide(img, x, y):
    h, w = img.shape
    size = w * h
    
    lt = img[:y, :x]
    rt = img[:y, x:]
    lb = img[y:, :x]
    rb = img[y:, x:]
    
    w1 = x * y / size
    w2 = (w - x) * y / size
    w3 = (x * (h - y)) / size
    w4 = 1 - w1 - w2 - w3
    
    return lt, rt, lb, rb, w1, w2, w3, w4
    

def centroid(img):
    h, w = img.shape
    
    total = np.sum(img)
    if total == 0:
        X = w // 2
        Y = h // 2
    else:
        i = range(1, w + 1)
        j = range(1, h + 1)
        X = int(np.sum(np.sum(img, axis=0) * i) / total + 0.5)
        Y = int(np.sum(np.sum(img, axis=1) * j) / total + 0.5)
    
    return X, Y

def obj(pred, gt):
    x = np.mean(pred[gt == 1])
    sigma_x = np.std(pred[gt == 1])
    #print(np.max(gt))

    score = 2.0 * x / (x**2 + 1.0 + sigma_x + eps)
    return score

def S_object(pred, gt):
    p_fg = pred * gt
    O_FG = obj(p_fg, gt)
    
    p_bg = (1 - pred) * (1 - gt)
    O_BG = obj(p_bg, 1 - gt)
    
    u = np.mean(gt)
    Q = u * O_FG + (1 - u) * O_BG
    #print(O_FG, O_BG)
    return Q

def S_region(pred, gt):
    X, Y = centroid(gt)
    
    ltg, rtg, lbg, rbg, w1, w2, w3, w4 = divide(gt, X, Y)
    ltp, rtp, lbp, rbp, _, _, _, _ = divide(pred, X, Y)
    
    Q1 = ssim(ltp, ltg)
    Q2 = ssim(rtp, rtg)
    Q3 = ssim(lbp, lbg)
    Q4 = ssim(rbp, rbg)
    
    #print(X, Y)
    
    #print(w1, w2, w3, w4, Q1, Q2, Q3, Q4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q


def SMeasure(pred, gt):
    pred = pred / 255.
    gt = gt / 255.
    gt = gt > 0.5

    y = np.mean(gt)
    #print(y)
    if y == 0:
        x = np.mean(pred)
        Q = 1 - x
    elif y == 1:
        x = np.mean(pred)
        Q = x
    else:
        alpha = 0.5
        so = S_object(pred, gt)
        sr = S_region(pred, gt)
        #print(so, sr)
        Q = alpha * so + (1 - alpha) * sr
    Q = 0 if Q < 0 else Q

    return Q