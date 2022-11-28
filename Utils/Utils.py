import json
import glob
import os.path
import numpy as np

def filterPred():
    filteredPred = []

    return filteredPred


## Filter current annotation file to contain only images in the predictions
## (i.e. useful for when you want to benchmark a subset of the data, this allows you to not create a new anno file each time)
def filterAnno(imgGlob,fullAnnoPath):

    filteredAnno = {}
    with open(fullAnnoPath, "r") as f:
        allAnno = json.load(f)

    newImages = []
    newIDs = []
    for img in allAnno['images']:
        for globImg in imgGlob:
            if os.path.basename(globImg).split('.')[0] == img['file_name'] and img['id'] not in newIDs:
                newImages.append(img)
                newIDs.append(img['id'])

    newAnno = []
    for nImg in newImages:
        for ann in allAnno['annotations']:
            if nImg['id'] == ann['image_id']:
                newAnno.append(ann)

    filteredAnno['annotations'] = newAnno
    filteredAnno['images'] = newImages
    filteredAnno['categories'] = allAnno['categories']

    outPath = os.path.dirname(fullAnnoPath) + "/filtered_" + os.path.basename(fullAnnoPath)
    with open(outPath,"w") as f:
        json.dump(filteredAnno, f)

    return outPath