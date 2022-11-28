from inference import Inference
from benchmark import Benchmark
import glob
import plotly.express as px

# Sample Main File


## Inference

## Glob of all images
imgFiles = glob.glob("C:/path/to/images/*.png")
## Path to annotation file that includes all globbed images
annoPath = "C:/path/to/annotations/curr_anno.json"
## Dir to save everything out to including predictions and final results
outDir = "C:/InferenceOutput/"
## The label IDs and corresponding text labels that we want to predict, see COCO list if changes are needed
labelDict = {
    0: "Background",
    1: "Person",
    2: "Bicycle",
    3: "Car"
}
## Infer based on above info
print("Runnning Inference")
inference = Inference(imgFiles,outDir,labelDict,0.01,30,imageSize = (640,338),annoPath=annoPath)
predictions = inference.batch_run(modelList=None,saveJSON=True, visualise=False)


## Predictions

## Glob JSONs saved in benchmark
predJSONs = glob.glob(outDir+"*.json")
## Run the benchmark for each prediction JSON against the newest annoPath annotation JSON and output a results file
print("Runnning Benchmark")
benchmark = Benchmark()
results = benchmark.batch_run(predictions=predJSONs,annoPath=annoPath,outDir = outDir,saveCSV=True,batchName="Testing")


## Visualise Results
fig = px.bar(results, x='Model', y='AP5095', title="AP5095")
fig.show()