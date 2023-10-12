import os
import numpy as np
import torch
from PIL import Image
import skimage.io
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import torchvision,transforms as T
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F



class CellsDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PATH_TO_TRAINING IMAGES"))))
        self.masks = os.path.join(root, "PATH_TO_FOLDER_WITH_MASKS")


    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PATH_TO_FOLDER_WITH_MASKS", self.imgs[idx])
        img_array = skimage.io.imread(img_path)

        img = Image.fromarray(img_array.astype(float))# note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask, classes = self.load_mask(self.root + "PATH_TO_JSON_ANNOTATIONS.json", int(self.imgs[idx].split('_')[1].split('.')[0]))
        obj_ids = []
        boxes_list = []
        labels = []
        area =[]
        for a in mask['annotations']:
            obj_ids.append(a)
            bb = mask['annotations'][a]['bounding_box']
            boxes_list.append(bb)
            labels.append(mask['annotations'][a]['category'])
            area.append(mask['annotations'][a]['area'])

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)


        boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = torch.as_tensor(area, dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        h, w = img_array.shape
        image_3d = np.zeros((3, h, w))
        image_3d[-1, :, :] = img_array
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def show(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    def __len__(self):
        return len(self.imgs)

    def load_mask(self, path, idx):
        out = {}
        out2 = {}
        out['annotations'] ={}
        flag_imgs = 1
        flag_annotations = 0
        flag_categories = 0
        place_holder_img = -100
        place_holder_annot = -100
        with open(path) as F:
            for rr, r in enumerate(F.readlines()):
                if flag_categories:
                    if 'id' in r:
                        idc = int(r.split(':')[1].split(',')[0])
                        out2[idc] = {}
                    elif 'name' in r:
                        out2[idc]['name'] = r.split(': "')[1].split('"')[0]
                    elif 'supercategory' in r:
                        out2[idc]['supercat'] = r.split(': "')[1].split('"')[0]
                if 'categories' in r:
                    flag_categories = 1
                    flag_annotations = 0
                if flag_annotations:
                    if rr == place_holder_annot + 2:
                        category = int(r.split(':')[1].split(',')[0])
                    elif rr == place_holder_annot + 3:
                        id_obj = int(r.split(':')[1].split(',')[0])
                        out['annotations'][id_obj] = {}
                        out['annotations'][id_obj]['category'] = category
                        out['annotations'][id_obj]['bounding_box'] = bb_coco
                    elif rr == place_holder_annot + 1:
                        bb = [float(x) for x in r.split('[')[1].split(']')[0].split(',')]
                        bb_coco = [bb[1], bb[0], bb[3], bb[2]]  #bb

                        if np.absolute(bb_coco[2]-bb_coco[0])<1:
                            bb_coco[2] += 1
                            #print('mod1')
                        if np.absolute(bb_coco[3] - bb_coco[1]) < 1:
                            bb_coco[3] += 1
                            #print('mod2')

                    elif rr == place_holder_annot + 4:
                        area = (bb_coco[2]-bb_coco[0])*(bb_coco[3]-bb_coco[1])
                        out['annotations'][id_obj]['area'] = area

                    if '"image_id": ' + str(idx)+',' in r:
                        place_holder_annot = rr
                if 'annotations' in r:
                    flag_annotations = 1
                if flag_imgs:
                    if rr == place_holder_img + 1:
                        out['height'] = int(r.split(':')[1].split(',')[0])
                    elif rr == place_holder_img + 2:
                        out['width'] = int(r.split(':')[1].split(',')[0])
                        flag_imgs = 0
                    if '"id": ' + str(idx) in r:
                        place_holder_img = rr
        return out, out2



def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.mask_predictor = None

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    # use our dataset and defined transformations
    dataset = CellsDataset('PATH_TO_IMAGES', get_transform(train=True))
    dataset_test = CellsDataset('PATH_TO_IMAGES', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)
    num_epochs = 50

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    torch.save(model, 'PATH_TO_PTH_FILE.pth')

if __name__ == "__main__":
    main()
