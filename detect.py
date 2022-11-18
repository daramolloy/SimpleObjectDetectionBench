import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models import detection
from torchvision.ops import nms
import numpy as np
import argparse
import pickle
import cv2
import yaml
import glob
import json

class Inference:
    def __init__(self,imgFiles,outDir,labelDict,confFilter,minIoU):
        self.imgFiles = imgFiles
        self.outDir = outDir
        self.labelFilter = labelDict.keys()
        self.labelDict = labelDict
        self.confFilter = confFilter
        self.minIoU = minIoU
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.colours = np.random.uniform(0, 255, size=(len(self.labelFilter), 3))

    def batch_run(self,saveJSON=True):
        # Populate model list
        models = []
        models.append(["yolov5n",torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)])
        models.append(["yolov5s",torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)])
        models.append(["yolov5m",torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)])
        models.append(["yolov5l",torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)])
        models.append(["yolov5x",torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)])
        # models.append(["fasterrcnn_mobilenet_v3_large_fpn",detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)])
        # models.append(["fasterrcnn_mobilenet_v3_large_320_fpn",detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)])
        # models.append(["fasterrcnn_resnet50_fpn",detection.fasterrcnn_resnet50_fpn(pretrained=True)])
        # models.append(["retinanet_resnet50_fpn",detection.retinanet_resnet50_fpn(pretrained=True)])
        # models.append(["fcos_resnet50_fpn",detection.fcos_resnet50_fpn(pretrained=True)])
        # models.append(["ssd300_vgg16",detection.ssd300_vgg16(pretrained=True)])
        # models.append(["ssdlite320_mobilenet_v3_large",detection.ssdlite320_mobilenet_v3_large(pretrained=True)])

        ## Iterate through each model
        for modelInfo in models:
            ## Initialise prediction dictionary
            predDict = []

            if "yolo" in modelInfo[0]:
                ## Load in model
                model = modelInfo[1]
                model.to(self.device)
                ## Set NMS Params
                model.conf = self.confFilter
                model.iou = self.minIoU/100
                model.classes = [0,1,2]
                for imgFile in self.imgFiles:
                    ## Initialise COCO Image ID index
                    imageID = 0
                    ## Reading in Image
                    image = self.loadImage(imgFile, False)
                    ## Inference & NMS
                    results = model(image)
                    ## Parse Results & Correct Datatypes
                    imagePred = results.pandas().xywh[0].values
                    imagePred[:,0] =  np.clip(imagePred[:,0].astype('int32') - ( imagePred[:,2].astype('int32')/2),0,image.shape[1]).astype('int32')
                    imagePred[:,1] =  np.clip(imagePred[:,1].astype('int32') - ( imagePred[:,3].astype('int32')/2),0,image.shape[0]).astype('int32')
                    imagePred[:,2] =  imagePred[:,2].astype('int32')
                    imagePred[:,3] =  imagePred[:,3].astype('int32')
                    imagePred[:,4] =  imagePred[:,4].astype('float64')
                    imagePred[:,5] =  imagePred[:,5].astype('int32') + 1
                    ## Visualise results
                    self.visualiseTorchResults(imagePred, imgFile)
                    for pred in imagePred:
                        predDict.append({
                            'image_id': int(imageID),
                            'category_id': int(pred[5]),
                            'bbox': [pred[0],pred[1],pred[2],pred[3]],
                            'score': float(pred[4])
                        })
                    imageID += 1
                if saveJSON:
                    jsonPath = f"{outDir}{modelInfo[0]}-pred.json"
                    with open(jsonPath, 'w') as f:
                        json.dump(predDict, f)


            else:
                ## Load in model
                model = modelInfo[1].to(self.device)
                model.eval()

                ## For each image
                for imgFile in self.imgFiles:
                    ## Initialise COCO Image ID index
                    imageID = 0
                    ## Reading in Image
                    image = self.loadImage(imgFile, True)
                    image = image.to(self.device)
                    ## Inference
                    results = model(image)[0]
                    ## Non-max suppressions
                    indexes = nms(results['boxes'], results['scores'], self.minIoU)
                    ## Tensor Res to X,Y,W,H,Label,Score
                    imagePred = self.parseResults(indexes,results)
                    ## Visualising Results
                    self.visualiseTorchResults(imagePred,imgFile)
                    for pred in imagePred:
                        predDict.append({
                            'image_id': int(imageID),
                            'category_id': int(pred[5]),
                            'bbox': [pred[0],pred[1],pred[2],pred[3]],
                            'score': float(pred[4])
                        })
                    imageID+=1
                if saveJSON:
                    jsonPath = f"{outDir}{modelInfo[0]}-pred.json"
                    with open(jsonPath, 'w') as f:
                        json.dump(predDict, f)

    def loadImage(self,imageFile, toTensor):
        image = cv2.imread(imageFile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if toTensor:
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, axis=0)
            image = image / 255.0
            image = torch.FloatTensor(image)
        return image

    def parseResults(self,indexes,results):
        imagePred = []
        for i in indexes:
            conf = results["scores"][i].item()
            label = results['labels'][i].item()
            if label in self.labelFilter:
                if conf > self.confFilter:
                    box = results["boxes"][i].detach().cpu().numpy()
                    (x1, y1, x2, y2) = box.astype("int")
                    width = x2 - x1
                    height = y2 - y1
                    imagePred.append([int(x1), int(y1), int(width), int(height),conf,label])
        return imagePred

    def visualiseTorchResults(self,imagePred,imgFile):
        image = cv2.imread(imgFile)
        for pred in imagePred:
            if float(pred[4]) > 0.1:
                cv2.rectangle(image, (pred[0], pred[1]), (pred[0]+pred[2], pred[1]+pred[3]), self.colours[pred[5]], 5)
                y = pred[1] - 15 if pred[1] - 15 > 15 else pred[1] + 15
                cv2.putText(image, f"{self.labelDict[pred[5]]}, {round(pred[4],2)}", (pred[0], y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, self.colours[pred[5]], 2)
        image = cv2.resize(image,(int(image.shape[1]/3),int(image.shape[0]/3)))
        cv2.imshow("VisualiseResults",image)
        cv2.waitKey(100)

# imgFiles = glob.glob("G:/OOF_Paper/Data/CookeTripletRGB_25mm/plus1p5waves/1000/*.png")
imgFiles = glob.glob("G:/OOF_Paper/Data/CookeTripletRGB_25mm/original/1000/*.png")
outDir = "G:/OOF_Paper/SDK/Inference/"
labelDict = {
    0: "Background",
    1: "Person",
    2: "Bicycle",
    3: "Car"
}
inference = Inference(imgFiles,outDir,labelDict,0.001,30)
inference.batch_run(saveJSON = True)

