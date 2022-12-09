import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE
import tidecv.datasets as datasets
import pandas as pd
from tqdm import tqdm

class Benchmark:
    def __init__(self):
        header = ["Name","Model", "AP5095", "AP50", "AP75", "APsmall", "APmedium","APlarge","AR5095_1","AR5095_10","AR5095_100",
                  "AR5095_100_small","AR5095_100_medium","AR5095_100_large"]

        for className in ["background", "vehicles", "person", "bicycle", "traffic light", "traffic sign"]:
            header.append(f"AP5095_{className}")
            header.append(f"AP50_{className}")
            header.append(f"AP75_{className}")
            header.append(f"APsmall_{className}")
            header.append(f"APmedium_{className}")
            header.append(f"APlarge_{className}")
            header.append(f"AR5095_1_{className}")
            header.append(f"AR5095_10_{className}")
            header.append(f"AR5095_100_{className}")
            header.append(f"AR5095_100_small_{className}")
            header.append(f"AR5095_100_medium_{className}")
            header.append(f"AR5095_100_large_{className}")
        header.extend(["TIDE-AP5095", "AP50", "AP55", "AP60", "AP65", "AP70", "AP75", "AP80", "AP85", "AP90", "AP95",
                       "CLS", "LOC", "BOTH", "DUPE","BKG", "MISS", "FalsePos", "FalseNeg"])

        for className in ["background", "vehicles", "person", "bicycle", "traffic light", "traffic sign"]:
            header.append(f"CLS_{className}")
            header.append(f"LOC_{className}")
            header.append(f"BOTH_{className}")
            header.append(f"DUPE_{className}")
            header.append(f"BKG_{className}")
            header.append(f"MISS_{className}")

        self.header = header


    def batch_run(self,predictions=None,annoPath=None,saveCSV=None,outDir=None,batchName=None):
        data = []
        cocoAnno = COCO(annoPath)
        print(cocoAnno)
        tide_gt = datasets.COCO(annoPath)

        for pred in tqdm(predictions):
            currData = [batchName,os.path.basename(pred)[:-10]]

            ## Standard COCO Metrics
            coco_pred = cocoAnno.loadRes(pred)
            eval = COCOeval(cocoAnno,coco_pred,'bbox')
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            for metric in eval.stats:
                currData.append(metric)

            ## Class COCO Metrics
            for i in range(5):
                eval = COCOeval(cocoAnno, coco_pred, 'bbox')
                eval.params.catIds = [i + 1]
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                for metric in eval.stats:
                    currData.append(metric)

            ## TIDE Metrics
            tide_pred = datasets.COCOResult(pred)
            tide = TIDE()
            tide.evaluate_range(tide_gt, tide_pred, mode=TIDE.BOX)
            tide.summarize()

            ## Pull Out Errors
            errors = tide.get_all_errors()
            #class_errors = tide.get_main_per_class_errors()
            #class_errors = list(class_errors.values())[0]
            main_errors = list(list(errors.values())[0].values())[0]
            special_errors = list(list(errors.values())[1].values())[0]

            ## Pull Out APs
            tide_aps = list(tide.run_thresholds.values())[0]
            tide_threshs = []
            for i in range(10):
                tide_threshs.append(tide_aps[i].ap)
            tide_ap5095 = sum(tide_threshs) / len(tide_threshs)
            currData.append(tide_ap5095)
            for thresh in tide_threshs:
                currData.append(thresh)

            ## Add in Errors
            currData.append(main_errors['Cls'])
            currData.append(main_errors['Loc'])
            currData.append(main_errors['Both'])
            currData.append(main_errors['Dupe'])
            currData.append(main_errors['Bkg'])
            currData.append(main_errors['Miss'])
            currData.append(special_errors['FalsePos'])
            currData.append(special_errors['FalseNeg'])

            #for i in range(3):
            #    for err in class_errors.keys():
            #        currData.append(class_errors[err][i + 1])

            data.append(currData)
        dataframe = pd.DataFrame(data, columns=self.header)

        ## Save out Dataframe
        if saveCSV: dataframe.to_csv(outDir + "Results.csv")
        return dataframe

