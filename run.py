import argparse
import os
from eval_otb import EvalOTB
from eval_uav import EvalUAV123
from eval_lasot import EvalLaSOT
from plot import plot

#############################################
# Here to implement the path to your datasets
#############################################
UAV123_path = "/data5/maoyy/dataset/UAV123"
OTB100_path = "/data5/maoyy/dataset/OTB100"
LaSOT_path = "/data5/maoyy/dataset/LaSOT"




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw success and precision plot on the chosen dataset.')
    parser.add_argument('--dataset', type=str, default='otb', help='Which dataset to be evaluated on.')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save result png file.')

    args = parser.parse_args()

    if args.dataset == "otb":
        evaluator = EvalOTB(OTB100_path)
        result_path = os.path.join( os.getcwd(), "Results", "OTB100" )
    elif args.dataset == "uav":
        evaluator = EvalUAV123(UAV123_path)
        result_path = os.path.join( os.getcwd(), "Results", "UAV123" )
    elif args.dataset == "lasot":
        evaluator = EvalLaSOT(LaSOT_path)
        result_path = os.path.join( os.getcwd(), "Results", "LaSOT" )
    else:
        raise ValueError("Un recognized dataset type")

    trackers = {}
    tracker_names = [ x for x in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, x)) ]
    for name in tracker_names:
        repeat = False if len([ x for x in os.listdir(os.path.join(result_path, name)) if x.endswith("txt") ]) else True
        trackers[name] = {"path": os.path.join(result_path, name), "repeat": repeat}
    
    plot(evaluator, trackers, args.output_path)