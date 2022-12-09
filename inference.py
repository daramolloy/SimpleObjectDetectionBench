import os.path

import torch
from torchvision.models import detection
from torchvision.ops import nms
import numpy as np
import cv2
import json
from tqdm import tqdm
from csv import reader
from re import search, findall

class Inference:
    def __init__(self,imgFiles,outDir,labelDict,confFilter,minIoU,imageSize=None,annoPath=None, model_list="models.csv"):
        self.imgFiles = imgFiles
        self.outDir = outDir
        self.labelFilter = labelDict.keys()
        self.labelDict = labelDict
        self.confFilter = confFilter
        self.minIoU = minIoU
        self.model_list = model_list
        if not os.path.exists(model_list):
            self.set_default_models()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.colours = np.random.uniform(0, 255, size=(15, 3))
        self.orig_imageSize = ()
        self.infer_imageSize = imageSize
        self.anno = None
        if annoPath is not None:
            with open(annoPath,"r") as f:
                self.anno = json.load(f)

        # Populate model list
        self.models = []
        # open file
        with open(self.model_list, "r") as model_file:
            # pass the file object to reader()
            file_reader = reader(model_file)
            #iterate over model list from file and append to models 
            for model_name in file_reader:
                if search("yolo", model_name[0]):
                    try:
                        version = int(findall("\d", model_name[0])[0])
                        print(version)
                        # if version is not specified, use the latest version
                        if version in [3, 5]:
                            self.models.append([model_name[0],torch.hub.load('ultralytics/yolov' + str(version), model_name[0], pretrained=model_name[1])])
                        else:
                            self.models.append([model_name[0],torch.hub.load('ultralytics/yolov5', model_name[0], pretrained=model_name[1])])
                    except:
                            self.models.append([model_name[0],torch.hub.load('ultralytics/yolov5', model_name[0], pretrained=model_name[1])])

                else:
                    self.models.append([model_name[0],getattr(detection, model_name[0])(pretrained=model_name[1])])
    
    '''
        set_default_models() - sets the default models to be used in CSV if no model list is provided
    '''
    def set_default_models(self):
        f = open(self.model_list, 'w')
        f.write("yolov5n,True\n" +
                "yolov5s,True\n" +
                "yolov5m,True\n" +
                "yolov5l,True\n" +
                "yolov5x,True\n" +
                "fasterrcnn_mobilenet_v3_large_fpn,True\n" +
                "fasterrcnn_resnet50_fpn,True\n" +
                "retinanet_resnet50_fpn,True\n" +
                "fcos_resnet50_fpn,True\n" +
                "ssd300_vgg16,True\n")
        f.close()

    def batch_run(self, modelList=None, saveJSON=True,visualise=False):
        # Initialise batch prediction dictionary
        batch_predDict = {}
        ## If models are filtered
        if modelList is not None:
            tempModelList = []
            for modelName in modelList:
                for model in self.models:
                    if modelName.lower() in model[0].lower():
                        tempModelList.append(model)
            self.models = tempModelList
        ## Iterate through each model
        for modelInfo in tqdm(self.models):
            ## Initialise prediction dictionary
            predDict = []

            if "yolo" in modelInfo[0]:
                ## Load in model
                model = modelInfo[1]
                model.to(self.device)
                ## Set NMS Params
                model.conf = self.confFilter
                model.iou = self.minIoU/100
                model.classes = [0,1,2,9,11] ## Hard-coded classes, needs fixing
                ## Initialise COCO Image ID index
                imageID = 0
                for imgFile in self.imgFiles:
                    ## Reading in Image
                    image = self.loadImage(imgFile, False)
                    ## Inference & NMS
                    results = model(image)
                    ## Parse Results & Correct Datatypes
                    xRatio = self.infer_imageSize[0] / self.orig_imageSize[0]
                    yRatio = self.infer_imageSize[1] / self.orig_imageSize[1]
                    imagePred = results.pandas().xywh[0].values
                    imagePred[:,0] =  (np.clip(imagePred[:,0].astype('int32') - ( imagePred[:,2].astype('int32')/2),0,image.shape[1])/xRatio).astype('int32')
                    imagePred[:,1] =  (np.clip(imagePred[:,1].astype('int32') - ( imagePred[:,3].astype('int32')/2),0,image.shape[0])/yRatio).astype('int32')
                    imagePred[:,2] =  (imagePred[:,2]/xRatio).astype('int32')
                    imagePred[:,3] =  (imagePred[:,3]/yRatio).astype('int32')
                    imagePred[:,4] =  imagePred[:,4].astype('float64')
                    imagePred[:,5] =  imagePred[:,5].astype('int32') + 1
                    ## Visualise results
                    if visualise: self.visualiseTorchResults(imagePred, imgFile)
                    if self.anno is None:
                        currImageID = imageID
                    else:
                        currImageID = self.pullImageID(imgFile)
                    for pred in imagePred:
                        predDict.append({
                            'image_id': int(currImageID),
                            'category_id': int(pred[5]),
                            'bbox': [pred[0],pred[1],pred[2],pred[3]],
                            'score': float(pred[4])
                        })
                    imageID += 1

            else:
                ## Load in model
                model = modelInfo[1].to(self.device)
                model.eval()

                ## Initialise COCO Image ID index
                imageID = 0
                ## For each image
                for imgFile in tqdm(self.imgFiles):
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
                    if visualise: self.visualiseTorchResults(imagePred,imgFile)
                    if self.anno is None:
                        currImageID = imageID
                    else:
                        currImageID = self.pullImageID(imgFile)
                    ## Add pred information to predDict in the COCO style
                    for pred in imagePred:
                        predDict.append({
                            'image_id': int(currImageID),
                            'category_id': int(pred[5]),
                            'bbox': [pred[0],pred[1],pred[2],pred[3]],
                            'score': float(pred[4])
                        })
                    ## Iterate image ID for each image
                    imageID+=1
            ## Save predictions to COCO JSON file per model if desired
            if saveJSON:
                jsonPath = f"{self.outDir}{modelInfo[0]}-pred.json"
                with open(jsonPath, 'w') as f:
                    json.dump(predDict, f)
            ## Save predictions to batch dictionary
            batch_predDict[modelInfo[0]] = predDict
        return batch_predDict


    def loadImage(self,imageFile, toTensor):
        image = cv2.imread(imageFile)
        self.orig_imageSize = (image.shape[1],image.shape[0])
        if self.infer_imageSize is not None: image = cv2.resize(image,self.infer_imageSize,interpolation=cv2.INTER_AREA)
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
            ## Pull confidence and label from results
            conf = results["scores"][i].item()
            label = results['labels'][i].item()
            ## If the label is within our label filter
            if label in self.labelFilter:
                ## If confidence is above our threshold
                if conf > self.confFilter:
                    ## Pull box from results
                    box = results["boxes"][i].detach().cpu().numpy()
                    ## Split box into vertices
                    (x1, y1, x2, y2) = box
                    ## Scale box size back to original as the annotations are in the original size
                    xRatio = self.orig_imageSize[0]/self.infer_imageSize[0]
                    yRatio = self.orig_imageSize[1]/self.infer_imageSize[1]
                    x1 = x1*xRatio
                    x2 = x2*xRatio
                    y1 = y1*yRatio
                    y2 = y2*yRatio
                    ## Calculate width and height
                    width = x2 - x1
                    height = y2 - y1
                    ## Append prediction entry with correct format
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
        cv2.waitKey(10)

    def pullImageID(self,imagePath):
        currImageID = -1
        for image in self.anno['images']:
            if image['file_name'].split('.')[0] == os.path.basename(imagePath).split('.')[0]:
                currImageID = image['id']
        return currImageID


