from inference import Inference
from benchmark import Benchmark
import glob
import plotly.express as px
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, required=True, help='directory of images to run inference on')
    parser.add_argument('--annotations', type=str, required=True, help='annotations for images')
    parser.add_argument('--inference-out', type=str, required=True, help='inference output directory')
    opt = parser.parse_args()
    # Sample Main File

    ## Inference

    ## Glob of all images
    imgFiles = glob.glob(opt.img_dir + "/*")
    ## Path to annotation file that includes all globbed images
    annoPath = opt.annotations
    ## Dir to save everything out to including predictions and final results
    outDir = opt.inference_out
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    ## The label IDs and corresponding text labels that we want to predict, see COCO list if changes are needed
    '''
    labelDict = {
        0: "Background",
        1: "Person",
        2: "Bicycle",
        3: "Car"
    }
    '''
    labelDict = {
        0: "Background",
        1: "Person",
        2: "Bicycle",
        3: "Car",
        10: "Traffic Light",
        12: "Traffic Sign"
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
    results = benchmark.batch_run(predictions=predJSONs,
                                    annoPath=annoPath,
                                    outDir = outDir,
                                    saveCSV=True,
                                    batchName="Testing")


    ## Visualise Results
    fig = px.bar(results, x='Model', y='AP5095', title="AP5095")
    fig.show()

if __name__ == "__main__":
    main()
