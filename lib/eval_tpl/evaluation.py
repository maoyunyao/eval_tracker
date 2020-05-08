import os
import sys
import numpy as np
import cv2


def IoU(rectA, rectB):
    '''
    Compute intersection over union between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rectA.ndim == 1:
        rectA = rectA[None,:]
    if rectB.ndim == 1:
        rectB = rectB[None,:]

    left = np.maximum(rectA[:,0], rectB[:,0])
    right = np.minimum(rectA[:,0]+rectA[:,2], rectB[:,0]+rectB[:,2])
    top = np.maximum(rectA[:,1], rectB[:,1])
    bottom = np.minimum(rectA[:,1]+rectA[:,3], rectB[:,1]+rectB[:,3])

    intersection = np.maximum(0, right-left) * np.maximum(0, bottom-top)
    union = rectA[:,2] * rectA[:,3] + rectB[:,2] * rectB[:,3] - intersection
    iou = np.clip(intersection / union, 0, 1)
    return iou

def convert_bbox_to_center(bboxes):
    '''
    Convert bbox(x,y,w,h) to center point
    - bboxes: 2d array of [xmin,ymin,w,h]
    '''
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

def convert_xywh_to_xyxy(bboxes):
    '''
    Convert bbox(x,y,w,h) to bbox(xmin,ymin,xmax,ymax)
    - bboxes: 2d array of [xmin,ymin,w,h]
    '''
    return np.array([(bboxes[:, 0]),
                     (bboxes[:, 1]),
                     (bboxes[:, 0] + (bboxes[:, 2] - 1)),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1))
                    ]).T    


class EvalTPL:
    def __init__(self, dataset_path, seq_len):
        '''
            dataset_path: Path to TempleColor128 dataset 
        '''
        self.dataset_path = dataset_path
        self.seq_info_list = self._get_sequence_info_list()
        self.seq_len = seq_len# Truncated sequence length

        self.gt = {}
        for seq_info in self.seq_info_list:
            seq_gt_path = os.path.join(self.dataset_path, seq_info["anno_path"])
            try:
                seq_gt = np.loadtxt(str(seq_gt_path), dtype=np.float64)
            except:
                seq_gt = np.loadtxt(str(seq_gt_path), delimiter=',', dtype=np.float64)
            
            init_omit = 0
            if "initOmit" in seq_info:
                init_omit = seq_info["initOmit"]
            
            self.gt[seq_info["name"]] = seq_gt[init_omit:, :]


    def evaluate(self, result_path):
        '''
        Params:
            result_path: Path to tracking result, result should be named as "SEQUENCE_NAME.txt"
        Return:
            avg_success: 
            avg_precision:
        '''
        result = {}
        for seq_name in self.gt:
            seq_result_path = os.path.join(result_path, seq_name + ".txt")
            try:
                seq_result = np.loadtxt(str(seq_result_path), dtype=np.float64)
            except:
                seq_result = np.loadtxt(str(seq_result_path), delimiter=',', dtype=np.float64)
            result[seq_name] = seq_result
        

        success_list = []
        precision_list = []
        for seq_name in self.gt:
            seq_success = self.compute_success(self.gt[seq_name], result[seq_name])
            seq_precision = self.compute_precision(self.gt[seq_name], result[seq_name])
            success_list.append(seq_success)
            precision_list.append(seq_precision)
        
        avg_success = np.mean(success_list, axis=0)
        avg_precision = np.mean(precision_list, axis=0)
        
        return avg_success, avg_precision 


    def visualize(self, result_path, seq_index = None):
        '''
            result_path: Path to the tracking result, sequence tracking result should be named as "SEQUENCE_NAME.txt"
            seq_index: integer between 0 and 99, indicate 100 sequences in OTB2015
        '''
        if seq_index:
            assert(seq_index >= 0 and seq_index <= len(self.seq_info_list))
            seq_to_show = [seq_index]
        else:
            seq_to_show = [i for i in range(len(self.seq_info_list))]
        
        for index in seq_to_show:
            seq_info = self.seq_info_list[index]
            
            name, path = seq_info["name"], seq_info["path"]
            start_frame, end_frame, ext =  seq_info["startFrame"], seq_info["endFrame"], seq_info["ext"]
            init_omit = 0
            if "initOmit" in seq_info:
                init_omit = seq_info["initOmit"]
            
            frame_names = [ x for x in os.listdir(os.path.join(self.dataset_path, path)) if x.endswith(tuple(ext)) ]
            frame_names.sort()
            frame_names = frame_names[start_frame+init_omit:end_frame+1]
            seq_gt = self.gt[name]
            
            seq_result_path = os.path.join(result_path, name + ".txt")
            try:
                seq_result = np.loadtxt(str(seq_result_path), dtype=np.float64)
            except:
                seq_result = np.loadtxt(str(seq_result_path), delimiter=',', dtype=np.float64)
            
            seq_gt = convert_xywh_to_xyxy(seq_gt)
            seq_result = convert_xywh_to_xyxy(seq_result)

            for index, frame in enumerate(frame_names):
                gt_xmin,gt_ymin,gt_xmax,gt_ymax = seq_gt[index].astype(np.int32)
                result_xmin,result_ymin,result_xmax,result_ymax = seq_result[index].astype(np.int32)
                frame = cv2.imread(os.path.join(self.dataset_path, path, frame))
                cv2.rectangle(frame, (gt_xmin,gt_ymin), (gt_xmax,gt_ymax), (0,0,255), 2)
                cv2.rectangle(frame, (result_xmin,result_ymin), (result_xmax,result_ymax), (0,255,0), 2)
                cv2.imshow(name, frame)
                cv2.waitKey(1)
            cv2.destroyAllWindows()

    
    def compute_success(self, gt, result):
        iou_thresholds = np.arange(0, 1.05, 0.05)
        score = np.zeros( len(iou_thresholds) )
        iou = IoU(gt, result)

        if self.seq_len != -1:
            seq_len = min(len(iou), self.seq_len)
            iou = iou[0:seq_len]
        
        frameNum = len(iou)
        
        for i in range(len(iou_thresholds)):
            score[i] = sum(iou > iou_thresholds[i]) / float(frameNum)
        return score

    def compute_precision(self, gt, result):
        gtCenter = convert_bbox_to_center(gt)
        resultCenter = convert_bbox_to_center(result)
        dist_thresholds = np.arange(0, 51, 1)
        score = np.zeros( len(dist_thresholds) )
        dist = np.sqrt( np.sum(np.power(gtCenter-resultCenter, 2), axis=1) )
        
        if self.seq_len != -1:
            seq_len = min(len(dist), self.seq_len)
            dist = dist[0:seq_len]

        frameNum = len(dist)

        for i in range( len(dist_thresholds) ):
            score[i] = sum(dist <= dist_thresholds[i]) / float(frameNum)
        return score

    def _get_sequence_info_list(self, exclude_otb=False):
        sequence_info_list = [
            {"name": "Skating2", "path": "Skating2/img", "startFrame": 1, "endFrame": 707, "nz": 4,
             "ext": "jpg", "anno_path": "Skating2/Skating2_gt.txt"},
            {"name": "Pool_ce3", "path": "Pool_ce3/img", "startFrame": 1, "endFrame": 124, "nz": 4,
             "ext": "jpg", "anno_path": "Pool_ce3/Pool_ce3_gt.txt"},
            {"name": "Microphone_ce1", "path": "Microphone_ce1/img", "startFrame": 1, "endFrame": 204, "nz": 4,
             "ext": "jpg", "anno_path": "Microphone_ce1/Microphone_ce1_gt.txt"},
            {"name": "Torus", "path": "Torus/img", "startFrame": 1, "endFrame": 264, "nz": 4, "ext": "jpg",
             "anno_path": "Torus/Torus_gt.txt"},
            {"name": "Lemming", "path": "Lemming/img", "startFrame": 1, "endFrame": 1336, "nz": 4, "ext": "jpg",
             "anno_path": "Lemming/Lemming_gt.txt"},
            {"name": "Eagle_ce", "path": "Eagle_ce/img", "startFrame": 1, "endFrame": 112, "nz": 4,
             "ext": "jpg", "anno_path": "Eagle_ce/Eagle_ce_gt.txt"},
            {"name": "Skating_ce2", "path": "Skating_ce2/img", "startFrame": 1, "endFrame": 497, "nz": 4,
             "ext": "jpg", "anno_path": "Skating_ce2/Skating_ce2_gt.txt"},
            {"name": "Yo-yos_ce3", "path": "Yo-yos_ce3/img", "startFrame": 1, "endFrame": 201, "nz": 4,
             "ext": "jpg", "anno_path": "Yo-yos_ce3/Yo-yos_ce3_gt.txt"},
            {"name": "Board", "path": "Board/img", "startFrame": 1, "endFrame": 598, "nz": 4, "ext": "jpg",
             "anno_path": "Board/Board_gt.txt"},
            {"name": "Tennis_ce3", "path": "Tennis_ce3/img", "startFrame": 1, "endFrame": 204, "nz": 4,
             "ext": "jpg", "anno_path": "Tennis_ce3/Tennis_ce3_gt.txt"},
            {"name": "SuperMario_ce", "path": "SuperMario_ce/img", "startFrame": 1, "endFrame": 146, "nz": 4,
             "ext": "jpg", "anno_path": "SuperMario_ce/SuperMario_ce_gt.txt"},
            {"name": "Yo-yos_ce1", "path": "Yo-yos_ce1/img", "startFrame": 1, "endFrame": 235, "nz": 4,
             "ext": "jpg", "anno_path": "Yo-yos_ce1/Yo-yos_ce1_gt.txt"},
            {"name": "Soccer", "path": "Soccer/img", "startFrame": 1, "endFrame": 392, "nz": 4, "ext": "jpg",
             "anno_path": "Soccer/Soccer_gt.txt"},
            {"name": "Fish_ce2", "path": "Fish_ce2/img", "startFrame": 1, "endFrame": 573, "nz": 4,
             "ext": "jpg", "anno_path": "Fish_ce2/Fish_ce2_gt.txt"},
            {"name": "Liquor", "path": "Liquor/img", "startFrame": 1, "endFrame": 1741, "nz": 4, "ext": "jpg",
             "anno_path": "Liquor/Liquor_gt.txt"},
            {"name": "Plane_ce2", "path": "Plane_ce2/img", "startFrame": 1, "endFrame": 653, "nz": 4,
             "ext": "jpg", "anno_path": "Plane_ce2/Plane_ce2_gt.txt"},
            {"name": "Couple", "path": "Couple/img", "startFrame": 1, "endFrame": 140, "nz": 4, "ext": "jpg",
             "anno_path": "Couple/Couple_gt.txt"},
            {"name": "Logo_ce", "path": "Logo_ce/img", "startFrame": 1, "endFrame": 610, "nz": 4, "ext": "jpg",
             "anno_path": "Logo_ce/Logo_ce_gt.txt"},
            {"name": "Hand_ce2", "path": "Hand_ce2/img", "startFrame": 1, "endFrame": 251, "nz": 4,
             "ext": "jpg", "anno_path": "Hand_ce2/Hand_ce2_gt.txt"},
            {"name": "Kite_ce2", "path": "Kite_ce2/img", "startFrame": 1, "endFrame": 658, "nz": 4,
             "ext": "jpg", "anno_path": "Kite_ce2/Kite_ce2_gt.txt"},
            {"name": "Walking", "path": "Walking/img", "startFrame": 1, "endFrame": 412, "nz": 4, "ext": "jpg",
             "anno_path": "Walking/Walking_gt.txt"},
            {"name": "David", "path": "David/img", "startFrame": 300, "endFrame": 770, "nz": 4, "ext": "jpg",
             "anno_path": "David/David_gt.txt"},
            {"name": "Boat_ce1", "path": "Boat_ce1/img", "startFrame": 1, "endFrame": 377, "nz": 4,
             "ext": "jpg", "anno_path": "Boat_ce1/Boat_ce1_gt.txt"},
            {"name": "Airport_ce", "path": "Airport_ce/img", "startFrame": 1, "endFrame": 148, "nz": 4,
             "ext": "jpg", "anno_path": "Airport_ce/Airport_ce_gt.txt"},
            {"name": "Tiger2", "path": "Tiger2/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg",
             "anno_path": "Tiger2/Tiger2_gt.txt"},
            {"name": "Suitcase_ce", "path": "Suitcase_ce/img", "startFrame": 1, "endFrame": 184, "nz": 4,
             "ext": "jpg", "anno_path": "Suitcase_ce/Suitcase_ce_gt.txt"},
            {"name": "TennisBall_ce", "path": "TennisBall_ce/img", "startFrame": 1, "endFrame": 288, "nz": 4,
             "ext": "jpg", "anno_path": "TennisBall_ce/TennisBall_ce_gt.txt"},
            {"name": "Singer_ce1", "path": "Singer_ce1/img", "startFrame": 1, "endFrame": 214, "nz": 4,
             "ext": "jpg", "anno_path": "Singer_ce1/Singer_ce1_gt.txt"},
            {"name": "Pool_ce2", "path": "Pool_ce2/img", "startFrame": 1, "endFrame": 133, "nz": 4,
             "ext": "jpg", "anno_path": "Pool_ce2/Pool_ce2_gt.txt"},
            {"name": "Surf_ce3", "path": "Surf_ce3/img", "startFrame": 1, "endFrame": 279, "nz": 4,
             "ext": "jpg", "anno_path": "Surf_ce3/Surf_ce3_gt.txt"},
            {"name": "Bird", "path": "Bird/img", "startFrame": 1, "endFrame": 99, "nz": 4, "ext": "jpg",
             "anno_path": "Bird/Bird_gt.txt"},
            {"name": "Crossing", "path": "Crossing/img", "startFrame": 1, "endFrame": 120, "nz": 4,
             "ext": "jpg", "anno_path": "Crossing/Crossing_gt.txt"},
            {"name": "Plate_ce1", "path": "Plate_ce1/img", "startFrame": 1, "endFrame": 142, "nz": 4,
             "ext": "jpg", "anno_path": "Plate_ce1/Plate_ce1_gt.txt"},
            {"name": "Cup", "path": "Cup/img", "startFrame": 1, "endFrame": 303, "nz": 4, "ext": "jpg",
             "anno_path": "Cup/Cup_gt.txt"},
            {"name": "Surf_ce2", "path": "Surf_ce2/img", "startFrame": 1, "endFrame": 391, "nz": 4,
             "ext": "jpg", "anno_path": "Surf_ce2/Surf_ce2_gt.txt"},
            {"name": "Busstation_ce2", "path": "Busstation_ce2/img", "startFrame": 6, "endFrame": 400, "nz": 4,
             "ext": "jpg", "anno_path": "Busstation_ce2/Busstation_ce2_gt.txt"},
            {"name": "Charger_ce", "path": "Charger_ce/img", "startFrame": 1, "endFrame": 298, "nz": 4,
             "ext": "jpg", "anno_path": "Charger_ce/Charger_ce_gt.txt"},
            {"name": "Pool_ce1", "path": "Pool_ce1/img", "startFrame": 1, "endFrame": 166, "nz": 4,
             "ext": "jpg", "anno_path": "Pool_ce1/Pool_ce1_gt.txt"},
            {"name": "MountainBike", "path": "MountainBike/img", "startFrame": 1, "endFrame": 228, "nz": 4,
             "ext": "jpg", "anno_path": "MountainBike/MountainBike_gt.txt"},
            {"name": "Guitar_ce1", "path": "Guitar_ce1/img", "startFrame": 1, "endFrame": 268, "nz": 4,
             "ext": "jpg", "anno_path": "Guitar_ce1/Guitar_ce1_gt.txt"},
            {"name": "Busstation_ce1", "path": "Busstation_ce1/img", "startFrame": 1, "endFrame": 363, "nz": 4,
             "ext": "jpg", "anno_path": "Busstation_ce1/Busstation_ce1_gt.txt"},
            {"name": "Diving", "path": "Diving/img", "startFrame": 1, "endFrame": 231, "nz": 4, "ext": "jpg",
             "anno_path": "Diving/Diving_gt.txt"},
            {"name": "Skating_ce1", "path": "Skating_ce1/img", "startFrame": 1, "endFrame": 409, "nz": 4,
             "ext": "jpg", "anno_path": "Skating_ce1/Skating_ce1_gt.txt"},
            {"name": "Hurdle_ce2", "path": "Hurdle_ce2/img", "startFrame": 27, "endFrame": 330, "nz": 4,
             "ext": "jpg", "anno_path": "Hurdle_ce2/Hurdle_ce2_gt.txt"},
            {"name": "Plate_ce2", "path": "Plate_ce2/img", "startFrame": 1, "endFrame": 181, "nz": 4,
             "ext": "jpg", "anno_path": "Plate_ce2/Plate_ce2_gt.txt"},
            {"name": "CarDark", "path": "CarDark/img", "startFrame": 1, "endFrame": 393, "nz": 4, "ext": "jpg",
             "anno_path": "CarDark/CarDark_gt.txt"},
            {"name": "Singer_ce2", "path": "Singer_ce2/img", "startFrame": 1, "endFrame": 999, "nz": 4,
             "ext": "jpg", "anno_path": "Singer_ce2/Singer_ce2_gt.txt"},
            {"name": "Shaking", "path": "Shaking/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg",
             "anno_path": "Shaking/Shaking_gt.txt"},
            {"name": "Iceskater", "path": "Iceskater/img", "startFrame": 1, "endFrame": 500, "nz": 4,
             "ext": "jpg", "anno_path": "Iceskater/Iceskater_gt.txt"},
            {"name": "Badminton_ce2", "path": "Badminton_ce2/img", "startFrame": 1, "endFrame": 705, "nz": 4,
             "ext": "jpg", "anno_path": "Badminton_ce2/Badminton_ce2_gt.txt"},
            {"name": "Spiderman_ce", "path": "Spiderman_ce/img", "startFrame": 1, "endFrame": 351, "nz": 4,
             "ext": "jpg", "anno_path": "Spiderman_ce/Spiderman_ce_gt.txt"},
            {"name": "Kite_ce1", "path": "Kite_ce1/img", "startFrame": 1, "endFrame": 484, "nz": 4,
             "ext": "jpg", "anno_path": "Kite_ce1/Kite_ce1_gt.txt"},
            {"name": "Skyjumping_ce", "path": "Skyjumping_ce/img", "startFrame": 1, "endFrame": 938, "nz": 4,
             "ext": "jpg", "anno_path": "Skyjumping_ce/Skyjumping_ce_gt.txt"},
            {"name": "Ball_ce1", "path": "Ball_ce1/img", "startFrame": 1, "endFrame": 391, "nz": 4,
             "ext": "jpg", "anno_path": "Ball_ce1/Ball_ce1_gt.txt"},
            {"name": "Yo-yos_ce2", "path": "Yo-yos_ce2/img", "startFrame": 1, "endFrame": 454, "nz": 4,
             "ext": "jpg", "anno_path": "Yo-yos_ce2/Yo-yos_ce2_gt.txt"},
            {"name": "Ironman", "path": "Ironman/img", "startFrame": 1, "endFrame": 166, "nz": 4, "ext": "jpg",
             "anno_path": "Ironman/Ironman_gt.txt"},
            {"name": "FaceOcc1", "path": "FaceOcc1/img", "startFrame": 1, "endFrame": 892, "nz": 4,
             "ext": "jpg", "anno_path": "FaceOcc1/FaceOcc1_gt.txt"},
            {"name": "Surf_ce1", "path": "Surf_ce1/img", "startFrame": 1, "endFrame": 404, "nz": 4,
             "ext": "jpg", "anno_path": "Surf_ce1/Surf_ce1_gt.txt"},
            {"name": "Ring_ce", "path": "Ring_ce/img", "startFrame": 1, "endFrame": 201, "nz": 4, "ext": "jpg",
             "anno_path": "Ring_ce/Ring_ce_gt.txt"},
            {"name": "Surf_ce4", "path": "Surf_ce4/img", "startFrame": 1, "endFrame": 135, "nz": 4,
             "ext": "jpg", "anno_path": "Surf_ce4/Surf_ce4_gt.txt"},
            {"name": "Ball_ce4", "path": "Ball_ce4/img", "startFrame": 1, "endFrame": 538, "nz": 4,
             "ext": "jpg", "anno_path": "Ball_ce4/Ball_ce4_gt.txt"},
            {"name": "Bikeshow_ce", "path": "Bikeshow_ce/img", "startFrame": 1, "endFrame": 361, "nz": 4,
             "ext": "jpg", "anno_path": "Bikeshow_ce/Bikeshow_ce_gt.txt"},
            {"name": "Kobe_ce", "path": "Kobe_ce/img", "startFrame": 1, "endFrame": 582, "nz": 4, "ext": "jpg",
             "anno_path": "Kobe_ce/Kobe_ce_gt.txt"},
            {"name": "Tiger1", "path": "Tiger1/img", "startFrame": 1, "endFrame": 354, "nz": 4, "ext": "jpg",
             "anno_path": "Tiger1/Tiger1_gt.txt"},
            {"name": "Skiing", "path": "Skiing/img", "startFrame": 1, "endFrame": 81, "nz": 4, "ext": "jpg",
             "anno_path": "Skiing/Skiing_gt.txt"},
            {"name": "Tennis_ce1", "path": "Tennis_ce1/img", "startFrame": 1, "endFrame": 454, "nz": 4,
             "ext": "jpg", "anno_path": "Tennis_ce1/Tennis_ce1_gt.txt"},
            {"name": "Carchasing_ce4", "path": "Carchasing_ce4/img", "startFrame": 1, "endFrame": 442, "nz": 4,
             "ext": "jpg", "anno_path": "Carchasing_ce4/Carchasing_ce4_gt.txt"},
            {"name": "Walking2", "path": "Walking2/img", "startFrame": 1, "endFrame": 500, "nz": 4,
             "ext": "jpg", "anno_path": "Walking2/Walking2_gt.txt"},
            {"name": "Sailor_ce", "path": "Sailor_ce/img", "startFrame": 1, "endFrame": 402, "nz": 4,
             "ext": "jpg", "anno_path": "Sailor_ce/Sailor_ce_gt.txt"},
            {"name": "Railwaystation_ce", "path": "Railwaystation_ce/img", "startFrame": 1, "endFrame": 413,
             "nz": 4, "ext": "jpg", "anno_path": "Railwaystation_ce/Railwaystation_ce_gt.txt"},
            {"name": "Bee_ce", "path": "Bee_ce/img", "startFrame": 1, "endFrame": 90, "nz": 4, "ext": "jpg",
             "anno_path": "Bee_ce/Bee_ce_gt.txt"},
            {"name": "Girl", "path": "Girl/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg",
             "anno_path": "Girl/Girl_gt.txt"},
            {"name": "Subway", "path": "Subway/img", "startFrame": 1, "endFrame": 175, "nz": 4, "ext": "jpg",
             "anno_path": "Subway/Subway_gt.txt"},
            {"name": "David3", "path": "David3/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg",
             "anno_path": "David3/David3_gt.txt"},
            {"name": "Electricalbike_ce", "path": "Electricalbike_ce/img", "startFrame": 1, "endFrame": 818,
             "nz": 4, "ext": "jpg", "anno_path": "Electricalbike_ce/Electricalbike_ce_gt.txt"},
            {"name": "Michaeljackson_ce", "path": "Michaeljackson_ce/img", "startFrame": 1, "endFrame": 393,
             "nz": 4, "ext": "jpg", "anno_path": "Michaeljackson_ce/Michaeljackson_ce_gt.txt"},
            {"name": "Woman", "path": "Woman/img", "startFrame": 1, "endFrame": 597, "nz": 4, "ext": "jpg",
             "anno_path": "Woman/Woman_gt.txt"},
            {"name": "TableTennis_ce", "path": "TableTennis_ce/img", "startFrame": 1, "endFrame": 198, "nz": 4,
             "ext": "jpg", "anno_path": "TableTennis_ce/TableTennis_ce_gt.txt"},
            {"name": "Motorbike_ce", "path": "Motorbike_ce/img", "startFrame": 1, "endFrame": 563, "nz": 4,
             "ext": "jpg", "anno_path": "Motorbike_ce/Motorbike_ce_gt.txt"},
            {"name": "Baby_ce", "path": "Baby_ce/img", "startFrame": 1, "endFrame": 296, "nz": 4, "ext": "jpg",
             "anno_path": "Baby_ce/Baby_ce_gt.txt"},
            {"name": "Gym", "path": "Gym/img", "startFrame": 1, "endFrame": 766, "nz": 4, "ext": "jpg",
             "anno_path": "Gym/Gym_gt.txt"},
            {"name": "Matrix", "path": "Matrix/img", "startFrame": 1, "endFrame": 100, "nz": 4, "ext": "jpg",
             "anno_path": "Matrix/Matrix_gt.txt"},
            {"name": "Kite_ce3", "path": "Kite_ce3/img", "startFrame": 1, "endFrame": 528, "nz": 4,
             "ext": "jpg", "anno_path": "Kite_ce3/Kite_ce3_gt.txt"},
            {"name": "Fish_ce1", "path": "Fish_ce1/img", "startFrame": 1, "endFrame": 401, "nz": 4,
             "ext": "jpg", "anno_path": "Fish_ce1/Fish_ce1_gt.txt"},
            {"name": "Hand_ce1", "path": "Hand_ce1/img", "startFrame": 1, "endFrame": 401, "nz": 4,
             "ext": "jpg", "anno_path": "Hand_ce1/Hand_ce1_gt.txt"},
            {"name": "Doll", "path": "Doll/img", "startFrame": 1, "endFrame": 3872, "nz": 4, "ext": "jpg",
             "anno_path": "Doll/Doll_gt.txt"},
            {"name": "Carchasing_ce3", "path": "Carchasing_ce3/img", "startFrame": 1, "endFrame": 572, "nz": 4,
             "ext": "jpg", "anno_path": "Carchasing_ce3/Carchasing_ce3_gt.txt"},
            {"name": "Thunder_ce", "path": "Thunder_ce/img", "startFrame": 1, "endFrame": 375, "nz": 4,
             "ext": "jpg", "anno_path": "Thunder_ce/Thunder_ce_gt.txt"},
            {"name": "Singer2", "path": "Singer2/img", "startFrame": 1, "endFrame": 366, "nz": 4, "ext": "jpg",
             "anno_path": "Singer2/Singer2_gt.txt"},
            {"name": "Basketball", "path": "Basketball/img", "startFrame": 1, "endFrame": 725, "nz": 4,
             "ext": "jpg", "anno_path": "Basketball/Basketball_gt.txt"},
            {"name": "Hand", "path": "Hand/img", "startFrame": 1, "endFrame": 244, "nz": 4, "ext": "jpg",
             "anno_path": "Hand/Hand_gt.txt"},
            {"name": "Cup_ce", "path": "Cup_ce/img", "startFrame": 1, "endFrame": 338, "nz": 4, "ext": "jpg",
             "anno_path": "Cup_ce/Cup_ce_gt.txt"},
            {"name": "MotorRolling", "path": "MotorRolling/img", "startFrame": 1, "endFrame": 164, "nz": 4,
             "ext": "jpg", "anno_path": "MotorRolling/MotorRolling_gt.txt"},
            {"name": "Boat_ce2", "path": "Boat_ce2/img", "startFrame": 1, "endFrame": 412, "nz": 4,
             "ext": "jpg", "anno_path": "Boat_ce2/Boat_ce2_gt.txt"},
            {"name": "CarScale", "path": "CarScale/img", "startFrame": 1, "endFrame": 252, "nz": 4,
             "ext": "jpg", "anno_path": "CarScale/CarScale_gt.txt"},
            {"name": "Sunshade", "path": "Sunshade/img", "startFrame": 1, "endFrame": 172, "nz": 4,
             "ext": "jpg", "anno_path": "Sunshade/Sunshade_gt.txt"},
            {"name": "Football1", "path": "Football1/img", "startFrame": 1, "endFrame": 74, "nz": 4,
             "ext": "jpg", "anno_path": "Football1/Football1_gt.txt"},
            {"name": "Singer1", "path": "Singer1/img", "startFrame": 1, "endFrame": 351, "nz": 4, "ext": "jpg",
             "anno_path": "Singer1/Singer1_gt.txt"},
            {"name": "Hurdle_ce1", "path": "Hurdle_ce1/img", "startFrame": 1, "endFrame": 300, "nz": 4,
             "ext": "jpg", "anno_path": "Hurdle_ce1/Hurdle_ce1_gt.txt"},
            {"name": "Basketball_ce3", "path": "Basketball_ce3/img", "startFrame": 1, "endFrame": 441, "nz": 4,
             "ext": "jpg", "anno_path": "Basketball_ce3/Basketball_ce3_gt.txt"},
            {"name": "Toyplane_ce", "path": "Toyplane_ce/img", "startFrame": 1, "endFrame": 405, "nz": 4,
             "ext": "jpg", "anno_path": "Toyplane_ce/Toyplane_ce_gt.txt"},
            {"name": "Skating1", "path": "Skating1/img", "startFrame": 1, "endFrame": 400, "nz": 4,
             "ext": "jpg", "anno_path": "Skating1/Skating1_gt.txt"},
            {"name": "Juice", "path": "Juice/img", "startFrame": 1, "endFrame": 404, "nz": 4, "ext": "jpg",
             "anno_path": "Juice/Juice_gt.txt"},
            {"name": "Biker", "path": "Biker/img", "startFrame": 1, "endFrame": 180, "nz": 4, "ext": "jpg",
             "anno_path": "Biker/Biker_gt.txt"},
            {"name": "Boy", "path": "Boy/img", "startFrame": 1, "endFrame": 602, "nz": 4, "ext": "jpg",
             "anno_path": "Boy/Boy_gt.txt"},
            {"name": "Jogging1", "path": "Jogging1/img", "startFrame": 1, "endFrame": 307, "nz": 4,
             "ext": "jpg", "anno_path": "Jogging1/Jogging1_gt.txt"},
            {"name": "Deer", "path": "Deer/img", "startFrame": 1, "endFrame": 71, "nz": 4, "ext": "jpg",
             "anno_path": "Deer/Deer_gt.txt"},
            {"name": "Panda", "path": "Panda/img", "startFrame": 1, "endFrame": 241, "nz": 4, "ext": "jpg",
             "anno_path": "Panda/Panda_gt.txt"},
            {"name": "Coke", "path": "Coke/img", "startFrame": 1, "endFrame": 291, "nz": 4, "ext": "jpg",
             "anno_path": "Coke/Coke_gt.txt"},
            {"name": "Carchasing_ce1", "path": "Carchasing_ce1/img", "startFrame": 1, "endFrame": 501, "nz": 4,
             "ext": "jpg", "anno_path": "Carchasing_ce1/Carchasing_ce1_gt.txt"},
            {"name": "Badminton_ce1", "path": "Badminton_ce1/img", "startFrame": 1, "endFrame": 579, "nz": 4,
             "ext": "jpg", "anno_path": "Badminton_ce1/Badminton_ce1_gt.txt"},
            {"name": "Trellis", "path": "Trellis/img", "startFrame": 1, "endFrame": 569, "nz": 4, "ext": "jpg",
             "anno_path": "Trellis/Trellis_gt.txt"},
            {"name": "Face_ce2", "path": "Face_ce2/img", "startFrame": 1, "endFrame": 148, "nz": 4,
             "ext": "jpg", "anno_path": "Face_ce2/Face_ce2_gt.txt"},
            {"name": "Ball_ce2", "path": "Ball_ce2/img", "startFrame": 1, "endFrame": 603, "nz": 4,
             "ext": "jpg", "anno_path": "Ball_ce2/Ball_ce2_gt.txt"},
            {"name": "Skiing_ce", "path": "Skiing_ce/img", "startFrame": 1, "endFrame": 511, "nz": 4,
             "ext": "jpg", "anno_path": "Skiing_ce/Skiing_ce_gt.txt"},
            {"name": "Jogging2", "path": "Jogging2/img", "startFrame": 1, "endFrame": 307, "nz": 4,
             "ext": "jpg", "anno_path": "Jogging2/Jogging2_gt.txt"},
            {"name": "Bike_ce1", "path": "Bike_ce1/img", "startFrame": 1, "endFrame": 801, "nz": 4,
             "ext": "jpg", "anno_path": "Bike_ce1/Bike_ce1_gt.txt"},
            {"name": "Bike_ce2", "path": "Bike_ce2/img", "startFrame": 1, "endFrame": 812, "nz": 4,
             "ext": "jpg", "anno_path": "Bike_ce2/Bike_ce2_gt.txt"},
            {"name": "Ball_ce3", "path": "Ball_ce3/img", "startFrame": 1, "endFrame": 273, "nz": 4,
             "ext": "jpg", "anno_path": "Ball_ce3/Ball_ce3_gt.txt"},
            {"name": "Girlmov", "path": "Girlmov/img", "startFrame": 1, "endFrame": 1500, "nz": 4, "ext": "jpg",
             "anno_path": "Girlmov/Girlmov_gt.txt"},
            {"name": "Bolt", "path": "Bolt/img", "startFrame": 1, "endFrame": 350, "nz": 4, "ext": "jpg",
             "anno_path": "Bolt/Bolt_gt.txt"},
            {"name": "Basketball_ce2", "path": "Basketball_ce2/img", "startFrame": 1, "endFrame": 455, "nz": 4,
             "ext": "jpg", "anno_path": "Basketball_ce2/Basketball_ce2_gt.txt"},
            {"name": "Bicycle", "path": "Bicycle/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg",
             "anno_path": "Bicycle/Bicycle_gt.txt"},
            {"name": "Face_ce", "path": "Face_ce/img", "startFrame": 1, "endFrame": 620, "nz": 4, "ext": "jpg",
             "anno_path": "Face_ce/Face_ce_gt.txt"},
            {"name": "Basketball_ce1", "path": "Basketball_ce1/img", "startFrame": 1, "endFrame": 496, "nz": 4,
             "ext": "jpg", "anno_path": "Basketball_ce1/Basketball_ce1_gt.txt"},
            {"name": "Messi_ce", "path": "Messi_ce/img", "startFrame": 1, "endFrame": 272, "nz": 4,
             "ext": "jpg", "anno_path": "Messi_ce/Messi_ce_gt.txt"},
            {"name": "Tennis_ce2", "path": "Tennis_ce2/img", "startFrame": 1, "endFrame": 305, "nz": 4,
             "ext": "jpg", "anno_path": "Tennis_ce2/Tennis_ce2_gt.txt"},
            {"name": "Microphone_ce2", "path": "Microphone_ce2/img", "startFrame": 1, "endFrame": 103, "nz": 4,
             "ext": "jpg", "anno_path": "Microphone_ce2/Microphone_ce2_gt.txt"},
            {"name": "Guitar_ce2", "path": "Guitar_ce2/img", "startFrame": 1, "endFrame": 313, "nz": 4,
             "ext": "jpg", "anno_path": "Guitar_ce2/Guitar_ce2_gt.txt"}

        ]

        otb_sequences = ['Skating2', 'Lemming', 'Board', 'Soccer', 'Liquor', 'Couple', 'Walking', 'David', 'Tiger2', 'Bird', 'Crossing', 'MountainBike',
                         'Diving', 'CarDark', 'Shaking', 'Ironman', 'FaceOcc1', 'Tiger1', 'Skiing', 'Walking2', 'Girl', 'Girlmov', 'Subway', 'David3', 'Woman',
                         'Gym', 'Matrix', 'Doll', 'Singer2', 'Basketball', 'MotorRolling', 'CarScale', 'Football1', 'Singer1', 'Skating1', 'Biker',
                         'Boy', 'Jogging1', 'Deer', 'Panda', 'Coke', 'Trellis', 'Jogging2', 'Bolt', ]
        if exclude_otb:
            sequence_info_list_nootb = []
            for seq in sequence_info_list:
                if seq['name'] not in otb_sequences:
                    sequence_info_list_nootb.append(seq)

            sequence_info_list = sequence_info_list_nootb

        return sequence_info_list