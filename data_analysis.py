import os
import numpy as np
import torch
import skimage.io
import skimage.exposure
import matplotlib.pyplot as plt
import torchvision.ops
import matplotlib.patches
from matplotlib.backends.backend_pdf import PdfPages
import pickle


def get_best_boxes(pred, th = 0.5):
    superimposition = torchvision.ops.box_iou(pred[0]['boxes'], pred[0]['boxes'])
    thr = torch.nn.Threshold(th, 0)
    keep = []
    toss = []
    for iss, s in enumerate(superimposition):
        if iss not in toss:
            sup = torch.nonzero(thr(s).to(torch.float))
            scores = pred[0]['scores'][sup]
            if torch.max(scores) >= 0.8:
                for sss, ss in enumerate(sup):
                    if not sss == torch.argmax(scores):
                        toss.append(ss)
                    else:
                        keep.append(ss)
    return keep

def get_cell_num(lst, idx):
    out = 0
    for l in lst:
        if l == idx:
            out+=1
    return out

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
filename = 'trained_model_mesothelial_all_2cls_new.pth'
output = '' #ADD THE PATH AND FILE NAME OF THE OUTPUT .PKL FILE
output_imgs = PdfPages() #ADD THE PATH AND FILE NAME OF THE OUTPUT .PDF FILE WHERE TO  SAVE THE IMAGES
model = torch.load(filename, map_location=device)
model.eval()

images_folder = '' #ADD THE PATH OF THE FOLDER CONTAINING THE INPUT IMAGES

rescale = 1 # set to 0 to skip intensity rescaling (line 170)

images = os.listdir(images_folder)
labels = {}
boxes = {}

cancer_annotations = []
non_cancer_annotations = []
cancer_predictions = []
non_cancer_predictions = []
colors = {1: (0/255, 90/255, 181/255), 2: (220/255, 50/255, 32/255)}
intervals = {0: [0, 1/4, 0, 1/4], 1: [0, 1/4, 1/4, 1/2], 2: [0, 1/4, 1/2, 3/4], 3: [0, 1/4, 3/4, 1],
             4: [1/4, 1/2, 0, 1/4], 5: [1/4, 1/2,  1/4, 1/2], 6: [1/4, 1/2, 1/2, 3/4], 7: [1/4, 1/2, 3/4, 1],
             8: [1/2, 3/4, 0, 1/4], 9: [1/2, 3/4, 1/4, 1/2], 10: [1/2, 3/4, 1/2, 3/4], 11: [1/2, 3/4, 3/4, 1],
             12: [3/4, 1, 0, 1/4], 13: [3/4, 1, 1/4, 1/2], 14: [3/4, 1, 1/2, 3/4], 15: [3/4, 1, 3/4, 1]}
for i in images:
    print(i)
    if not i.startswith('.'):
        img = skimage.io.imread(images_folder+i)
        h, w, _ = img.shape
        img = img[:, :, 2]
        f, ax = plt.subplots()
        ax.imshow(img, cmap='bone')
        if rescale:
            p2, p98 = np.percentile(img, (2, 98))
            img = skimage.exposure.rescale_intensity(img, in_range=(p2, p98))

        labels[i] = {}
        boxes[i] = {}
        for s in intervals:
            labels[i][s] = {'cells':[]}
            sub_img = img[int(intervals[s][0] * h):int(intervals[s][1] * h),
                      int(intervals[s][2] * w):int(intervals[s][3] * w)]
            try:
                hi, wi = sub_img.shape
            except:
                hi,wi,di = sub_img.shape

            try:
                image_3d_plot = np.zeros((hi, wi, 3))
                image_3d = np.zeros((3, hi, wi))
                image_3d[0, :, :] = sub_img
                image_3d[1, :, :] = sub_img
                image_3d[2, :, :] = sub_img
                image_3d_plot[:, :, 2] = sub_img
            except ValueError:
                image_3d = np.zeros((3, hi, wi))
                image_3d_plot = np.zeros((hi, wi, 3))
                image_3d[0, :, :] = sub_img[:, :]
                image_3d[1, :, :] = sub_img[:, :]
                image_3d[2, :, :] = sub_img[:, :]
                image_3d_plot[:, :, 2] = sub_img[:, :, 2]
            img3 = [torch.from_numpy(image_3d)]
            imgs = list(im.to(device, dtype=torch.float) for im in img3)
            ypred = model(imgs)
            best_boxes = get_best_boxes(ypred)

            for b in best_boxes:
                labels[i][s]['cells'].append(ypred[0]['labels'][b].cpu().detach().numpy()[0])
            non_cancer_predictions.append(get_cell_num(labels[i][s]['cells'], 1))
            cancer_predictions.append(get_cell_num(labels[i][s]['cells'], 2))
            annotated_bb = []
            annotated_labels = []
            boxes[i][s] = ypred[0]['boxes'][best_boxes, :].cpu().detach().numpy()
            for bb, b in enumerate(boxes[i][s]):
                anchor = (int(b[0] + w * intervals[s][2]), int(b[1] + h * intervals[s][0]))
                width = int(b[2] + w * intervals[s][2]) - anchor[0]
                height = int(b[3] + h * intervals[s][0]) - anchor[1]
                patch = matplotlib.patches.Rectangle(anchor, width, height, fill=False, linewidth=2,
                                                     edgecolor=colors[labels[i][s]['cells'][bb]])
                ax.add_patch(patch)
                ax.set_title(i + '_'+str(s))
        output_imgs.savefig()

output_imgs.close()

out_var = {'cancer_predictions': cancer_predictions, 'non_cancer_predictions': non_cancer_predictions,
           'labels': labels, 'boxes': boxes}
with open(output, 'wb') as F:
   pickle.dump(out_var, F)
