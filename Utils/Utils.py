import json
import glob


## Filter current annotation file to contain only images in the predictions
## (i.e. useful for when you want to benchmark a subset of the data, this allows you to not create a new anno file each time)
import numpy as np


def filterAnno(anno,pred,outAnno):

    with open(anno, "r") as f:
        fullAnno = json.load(f)
    with open(pred, "r") as f:
        pred = json.load(f)

    validImageIDs = np.unique([])