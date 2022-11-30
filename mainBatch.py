from inference import Inference
from benchmark import Benchmark
from Utils.Utils import filterAnno
from Utils.Utils import sortTempOutputs
import glob
import plotly.express as px

## Path to annotation file that includes all globbed images
annoPath = "G:/OOF_Paper/Data/Annotations/fullAnno.json"
## Dir to save everything out to including predictions and final results
outDir = "G:/OOF_Paper/Results/ScriptOutput/"
## The label IDs and corresponding text labels that we want to predict, see COCO list if changes are needed
labelDict = {
    0: "Background",
    1: "Person",
    2: "Bicycle",
    3: "Car"
}

baseDir = "G:/OOF_Paper/Data/CookeTripletRGB_25mm/"
configs = ["nominal","original","minus0p5waves","minus1waves","minus1p5waves","plus0p5waves","plus1waves","plus1p5waves"]
areas = ["500","750","1000","1250","1500","2000","2500","3000","3500","4000","5000","10000","20000","40000","80000","160000"]

variations = {}
for config in configs:
    for area in areas:
        variations[f"{config}-{area}"] = f"{config}/{area}/"

## Init Classes
inference = Inference(outDir,labelDict,0.01,30,imageSize = (640,338),annoPath=annoPath)
benchmark = Benchmark()

## Loop through each config
for var in variations.keys():
    try:
        ## Inference

        ## Glob of all images
        imgFiles = glob.glob(baseDir + variations[var] + "/*.png")

        ## Infer based on above info
        print("Runnning Inference")
        predictions = inference.batch_run(imgFiles=imgFiles, modelList=None,saveJSON=True, visualise=True)

        ## Predictions

        ## Glob JSONs saved in benchmark
        predJSONs = glob.glob(outDir+"*.json")
        ## Path to annotation file containing only the images that are glob
        f_annoPath = filterAnno(imgFiles,annoPath)
        ## Run the benchmark for each prediction JSON against the newest annoPath annotation JSON and output a results file
        print("Runnning Benchmark")
        results = benchmark.batch_run(predictions=predJSONs,annoPath=f_annoPath,outDir = outDir,saveCSV=True,batchName=var)
        sortTempOutputs(var,outDir,f_annoPath,predJSONs)
    except Exception as e:
        print("Error")
        with open(f"{outDir}Errors.txt","a") as f:
            f.write(f"{var} : {e}")



