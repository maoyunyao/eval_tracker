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


class EvalOTB:
    def __init__(self, dataset_path, seq_len):
        '''
            dataset_path: Path to OTB dataset 
        '''
        self.dataset_path = dataset_path
        self.seq_info_list = self._get_sequence_info_list()
        self.seq_len = seq_len # Truncated sequence length

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

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "Basketball", "path": "Basketball/img", "startFrame": 1, "endFrame": 725, "nz": 4, "ext": "jpg", "anno_path": "Basketball/groundtruth_rect.txt"},
            {"name": "Biker", "path": "Biker/img", "startFrame": 1, "endFrame": 142, "nz": 4, "ext": "jpg", "anno_path": "Biker/groundtruth_rect.txt"},
            {"name": "Bird1", "path": "Bird1/img", "startFrame": 1, "endFrame": 408, "nz": 4, "ext": "jpg", "anno_path": "Bird1/groundtruth_rect.txt"},
            {"name": "Bird2", "path": "Bird2/img", "startFrame": 1, "endFrame": 99, "nz": 4, "ext": "jpg", "anno_path": "Bird2/groundtruth_rect.txt"},
            {"name": "BlurBody", "path": "BlurBody/img", "startFrame": 1, "endFrame": 334, "nz": 4, "ext": "jpg", "anno_path": "BlurBody/groundtruth_rect.txt"},
            {"name": "BlurCar1", "path": "BlurCar1/img", "startFrame": 247, "endFrame": 988, "nz": 4, "ext": "jpg", "anno_path": "BlurCar1/groundtruth_rect.txt"},
            {"name": "BlurCar2", "path": "BlurCar2/img", "startFrame": 1, "endFrame": 585, "nz": 4, "ext": "jpg", "anno_path": "BlurCar2/groundtruth_rect.txt"},
            {"name": "BlurCar3", "path": "BlurCar3/img", "startFrame": 3, "endFrame": 359, "nz": 4, "ext": "jpg", "anno_path": "BlurCar3/groundtruth_rect.txt"},
            {"name": "BlurCar4", "path": "BlurCar4/img", "startFrame": 18, "endFrame": 397, "nz": 4, "ext": "jpg", "anno_path": "BlurCar4/groundtruth_rect.txt"},
            {"name": "BlurFace", "path": "BlurFace/img", "startFrame": 1, "endFrame": 493, "nz": 4, "ext": "jpg", "anno_path": "BlurFace/groundtruth_rect.txt"},
            {"name": "BlurOwl", "path": "BlurOwl/img", "startFrame": 1, "endFrame": 631, "nz": 4, "ext": "jpg", "anno_path": "BlurOwl/groundtruth_rect.txt"},
            {"name": "Board", "path": "Board/img", "startFrame": 1, "endFrame": 698, "nz": 5, "ext": "jpg", "anno_path": "Board/groundtruth_rect.txt"},
            {"name": "Bolt", "path": "Bolt/img", "startFrame": 1, "endFrame": 350, "nz": 4, "ext": "jpg", "anno_path": "Bolt/groundtruth_rect.txt"},
            {"name": "Bolt2", "path": "Bolt2/img", "startFrame": 1, "endFrame": 293, "nz": 4, "ext": "jpg", "anno_path": "Bolt2/groundtruth_rect.txt"},
            {"name": "Box", "path": "Box/img", "startFrame": 1, "endFrame": 1161, "nz": 4, "ext": "jpg", "anno_path": "Box/groundtruth_rect.txt"},
            {"name": "Boy", "path": "Boy/img", "startFrame": 1, "endFrame": 602, "nz": 4, "ext": "jpg", "anno_path": "Boy/groundtruth_rect.txt"},
            {"name": "Car1", "path": "Car1/img", "startFrame": 1, "endFrame": 1020, "nz": 4, "ext": "jpg", "anno_path": "Car1/groundtruth_rect.txt"},
            {"name": "Car2", "path": "Car2/img", "startFrame": 1, "endFrame": 913, "nz": 4, "ext": "jpg", "anno_path": "Car2/groundtruth_rect.txt"},
            {"name": "Car24", "path": "Car24/img", "startFrame": 1, "endFrame": 3059, "nz": 4, "ext": "jpg", "anno_path": "Car24/groundtruth_rect.txt"},
            {"name": "Car4", "path": "Car4/img", "startFrame": 1, "endFrame": 659, "nz": 4, "ext": "jpg", "anno_path": "Car4/groundtruth_rect.txt"},
            {"name": "CarDark", "path": "CarDark/img", "startFrame": 1, "endFrame": 393, "nz": 4, "ext": "jpg", "anno_path": "CarDark/groundtruth_rect.txt"},
            {"name": "CarScale", "path": "CarScale/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg", "anno_path": "CarScale/groundtruth_rect.txt"},
            {"name": "ClifBar", "path": "ClifBar/img", "startFrame": 1, "endFrame": 472, "nz": 4, "ext": "jpg", "anno_path": "ClifBar/groundtruth_rect.txt"},
            {"name": "Coke", "path": "Coke/img", "startFrame": 1, "endFrame": 291, "nz": 4, "ext": "jpg", "anno_path": "Coke/groundtruth_rect.txt"},
            {"name": "Couple", "path": "Couple/img", "startFrame": 1, "endFrame": 140, "nz": 4, "ext": "jpg", "anno_path": "Couple/groundtruth_rect.txt"},
            {"name": "Coupon", "path": "Coupon/img", "startFrame": 1, "endFrame": 327, "nz": 4, "ext": "jpg", "anno_path": "Coupon/groundtruth_rect.txt"},
            {"name": "Crossing", "path": "Crossing/img", "startFrame": 1, "endFrame": 120, "nz": 4, "ext": "jpg", "anno_path": "Crossing/groundtruth_rect.txt"},
            {"name": "Crowds", "path": "Crowds/img", "startFrame": 1, "endFrame": 347, "nz": 4, "ext": "jpg", "anno_path": "Crowds/groundtruth_rect.txt"},
            {"name": "Dancer", "path": "Dancer/img", "startFrame": 1, "endFrame": 225, "nz": 4, "ext": "jpg", "anno_path": "Dancer/groundtruth_rect.txt"},
            {"name": "Dancer2", "path": "Dancer2/img", "startFrame": 1, "endFrame": 150, "nz": 4, "ext": "jpg", "anno_path": "Dancer2/groundtruth_rect.txt"},
            {"name": "David", "path": "David/img", "startFrame": 300, "endFrame": 770, "nz": 4, "ext": "jpg", "anno_path": "David/groundtruth_rect.txt"},
            {"name": "David2", "path": "David2/img", "startFrame": 1, "endFrame": 537, "nz": 4, "ext": "jpg", "anno_path": "David2/groundtruth_rect.txt"},
            {"name": "David3", "path": "David3/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg", "anno_path": "David3/groundtruth_rect.txt"},
            {"name": "Deer", "path": "Deer/img", "startFrame": 1, "endFrame": 71, "nz": 4, "ext": "jpg", "anno_path": "Deer/groundtruth_rect.txt"},
            {"name": "Diving", "path": "Diving/img", "startFrame": 1, "endFrame": 215, "nz": 4, "ext": "jpg", "anno_path": "Diving/groundtruth_rect.txt"},
            {"name": "Dog", "path": "Dog/img", "startFrame": 1, "endFrame": 127, "nz": 4, "ext": "jpg", "anno_path": "Dog/groundtruth_rect.txt"},
            {"name": "Dog1", "path": "Dog1/img", "startFrame": 1, "endFrame": 1350, "nz": 4, "ext": "jpg", "anno_path": "Dog1/groundtruth_rect.txt"},
            {"name": "Doll", "path": "Doll/img", "startFrame": 1, "endFrame": 3872, "nz": 4, "ext": "jpg", "anno_path": "Doll/groundtruth_rect.txt"},
            {"name": "DragonBaby", "path": "DragonBaby/img", "startFrame": 1, "endFrame": 113, "nz": 4, "ext": "jpg", "anno_path": "DragonBaby/groundtruth_rect.txt"},
            {"name": "Dudek", "path": "Dudek/img", "startFrame": 1, "endFrame": 1145, "nz": 4, "ext": "jpg", "anno_path": "Dudek/groundtruth_rect.txt"},
            {"name": "FaceOcc1", "path": "FaceOcc1/img", "startFrame": 1, "endFrame": 892, "nz": 4, "ext": "jpg", "anno_path": "FaceOcc1/groundtruth_rect.txt"},
            {"name": "FaceOcc2", "path": "FaceOcc2/img", "startFrame": 1, "endFrame": 812, "nz": 4, "ext": "jpg", "anno_path": "FaceOcc2/groundtruth_rect.txt"},
            {"name": "Fish", "path": "Fish/img", "startFrame": 1, "endFrame": 476, "nz": 4, "ext": "jpg", "anno_path": "Fish/groundtruth_rect.txt"},
            {"name": "FleetFace", "path": "FleetFace/img", "startFrame": 1, "endFrame": 707, "nz": 4, "ext": "jpg", "anno_path": "FleetFace/groundtruth_rect.txt"},
            {"name": "Football", "path": "Football/img", "startFrame": 1, "endFrame": 362, "nz": 4, "ext": "jpg", "anno_path": "Football/groundtruth_rect.txt"},
            {"name": "Football1", "path": "Football1/img", "startFrame": 1, "endFrame": 74, "nz": 4, "ext": "jpg", "anno_path": "Football1/groundtruth_rect.txt"},
            {"name": "Freeman1", "path": "Freeman1/img", "startFrame": 1, "endFrame": 326, "nz": 4, "ext": "jpg", "anno_path": "Freeman1/groundtruth_rect.txt"},
            {"name": "Freeman3", "path": "Freeman3/img", "startFrame": 1, "endFrame": 460, "nz": 4, "ext": "jpg", "anno_path": "Freeman3/groundtruth_rect.txt"},
            {"name": "Freeman4", "path": "Freeman4/img", "startFrame": 1, "endFrame": 283, "nz": 4, "ext": "jpg", "anno_path": "Freeman4/groundtruth_rect.txt"},
            {"name": "Girl", "path": "Girl/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg", "anno_path": "Girl/groundtruth_rect.txt"},
            {"name": "Girl2", "path": "Girl2/img", "startFrame": 1, "endFrame": 1500, "nz": 4, "ext": "jpg", "anno_path": "Girl2/groundtruth_rect.txt"},
            {"name": "Gym", "path": "Gym/img", "startFrame": 1, "endFrame": 767, "nz": 4, "ext": "jpg", "anno_path": "Gym/groundtruth_rect.txt"},
            {"name": "Human2", "path": "Human2/img", "startFrame": 1, "endFrame": 1128, "nz": 4, "ext": "jpg", "anno_path": "Human2/groundtruth_rect.txt"},
            {"name": "Human3", "path": "Human3/img", "startFrame": 1, "endFrame": 1698, "nz": 4, "ext": "jpg", "anno_path": "Human3/groundtruth_rect.txt"},
            {"name": "Human4_2", "path": "Human4/img", "startFrame": 1, "endFrame": 667, "nz": 4, "ext": "jpg", "anno_path": "Human4/groundtruth_rect.2.txt"},
            {"name": "Human5", "path": "Human5/img", "startFrame": 1, "endFrame": 713, "nz": 4, "ext": "jpg", "anno_path": "Human5/groundtruth_rect.txt"},
            {"name": "Human6", "path": "Human6/img", "startFrame": 1, "endFrame": 792, "nz": 4, "ext": "jpg", "anno_path": "Human6/groundtruth_rect.txt"},
            {"name": "Human7", "path": "Human7/img", "startFrame": 1, "endFrame": 250, "nz": 4, "ext": "jpg", "anno_path": "Human7/groundtruth_rect.txt"},
            {"name": "Human8", "path": "Human8/img", "startFrame": 1, "endFrame": 128, "nz": 4, "ext": "jpg", "anno_path": "Human8/groundtruth_rect.txt"},
            {"name": "Human9", "path": "Human9/img", "startFrame": 1, "endFrame": 305, "nz": 4, "ext": "jpg", "anno_path": "Human9/groundtruth_rect.txt"},
            {"name": "Ironman", "path": "Ironman/img", "startFrame": 1, "endFrame": 166, "nz": 4, "ext": "jpg", "anno_path": "Ironman/groundtruth_rect.txt"},
            {"name": "Jogging_1", "path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg", "anno_path": "Jogging/groundtruth_rect.1.txt"},
            {"name": "Jogging_2", "path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg", "anno_path": "Jogging/groundtruth_rect.2.txt"},
            {"name": "Jump", "path": "Jump/img", "startFrame": 1, "endFrame": 122, "nz": 4, "ext": "jpg", "anno_path": "Jump/groundtruth_rect.txt"},
            {"name": "Jumping", "path": "Jumping/img", "startFrame": 1, "endFrame": 313, "nz": 4, "ext": "jpg", "anno_path": "Jumping/groundtruth_rect.txt"},
            {"name": "KiteSurf", "path": "KiteSurf/img", "startFrame": 1, "endFrame": 84, "nz": 4, "ext": ["png", "jpg"], "anno_path": "KiteSurf/groundtruth_rect.txt"},
            {"name": "Lemming", "path": "Lemming/img", "startFrame": 1, "endFrame": 1336, "nz": 4, "ext": "jpg", "anno_path": "Lemming/groundtruth_rect.txt"},
            {"name": "Liquor", "path": "Liquor/img", "startFrame": 1, "endFrame": 1741, "nz": 4, "ext": "jpg", "anno_path": "Liquor/groundtruth_rect.txt"},
            {"name": "Man", "path": "Man/img", "startFrame": 1, "endFrame": 134, "nz": 4, "ext": "jpg", "anno_path": "Man/groundtruth_rect.txt"},
            {"name": "Matrix", "path": "Matrix/img", "startFrame": 1, "endFrame": 100, "nz": 4, "ext": "jpg", "anno_path": "Matrix/groundtruth_rect.txt"},
            {"name": "Mhyang", "path": "Mhyang/img", "startFrame": 1, "endFrame": 1490, "nz": 4, "ext": "jpg", "anno_path": "Mhyang/groundtruth_rect.txt"},
            {"name": "MotorRolling", "path": "MotorRolling/img", "startFrame": 1, "endFrame": 164, "nz": 4, "ext": "jpg", "anno_path": "MotorRolling/groundtruth_rect.txt"},
            {"name": "MountainBike", "path": "MountainBike/img", "startFrame": 1, "endFrame": 228, "nz": 4, "ext": "jpg", "anno_path": "MountainBike/groundtruth_rect.txt"},
            {"name": "Panda", "path": "Panda/img", "startFrame": 1, "endFrame": 1000, "nz": 4, "ext": "jpg", "anno_path": "Panda/groundtruth_rect.txt"},
            {"name": "RedTeam", "path": "RedTeam/img", "startFrame": 1, "endFrame": 1918, "nz": 4, "ext": "jpg", "anno_path": "RedTeam/groundtruth_rect.txt"},
            {"name": "Rubik", "path": "Rubik/img", "startFrame": 1, "endFrame": 1997, "nz": 4, "ext": "jpg", "anno_path": "Rubik/groundtruth_rect.txt"},
            {"name": "Shaking", "path": "Shaking/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg", "anno_path": "Shaking/groundtruth_rect.txt"},
            {"name": "Singer1", "path": "Singer1/img", "startFrame": 1, "endFrame": 351, "nz": 4, "ext": "jpg", "anno_path": "Singer1/groundtruth_rect.txt"},
            {"name": "Singer2", "path": "Singer2/img", "startFrame": 1, "endFrame": 366, "nz": 4, "ext": "jpg", "anno_path": "Singer2/groundtruth_rect.txt"},
            {"name": "Skater", "path": "Skater/img", "startFrame": 1, "endFrame": 160, "nz": 4, "ext": "jpg", "anno_path": "Skater/groundtruth_rect.txt"},
            {"name": "Skater2", "path": "Skater2/img", "startFrame": 1, "endFrame": 435, "nz": 4, "ext": "jpg", "anno_path": "Skater2/groundtruth_rect.txt"},
            {"name": "Skating1", "path": "Skating1/img", "startFrame": 1, "endFrame": 400, "nz": 4, "ext": "jpg", "anno_path": "Skating1/groundtruth_rect.txt"},
            {"name": "Skating2_1", "path": "Skating2/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg", "anno_path": "Skating2/groundtruth_rect.1.txt"},
            {"name": "Skating2_2", "path": "Skating2/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg", "anno_path": "Skating2/groundtruth_rect.2.txt"},
            {"name": "Skiing", "path": "Skiing/img", "startFrame": 1, "endFrame": 81, "nz": 4, "ext": "jpg", "anno_path": "Skiing/groundtruth_rect.txt"},
            {"name": "Soccer", "path": "Soccer/img", "startFrame": 1, "endFrame": 392, "nz": 4, "ext": "jpg", "anno_path": "Soccer/groundtruth_rect.txt"},
            {"name": "Subway", "path": "Subway/img", "startFrame": 1, "endFrame": 175, "nz": 4, "ext": "jpg", "anno_path": "Subway/groundtruth_rect.txt"},
            {"name": "Surfer", "path": "Surfer/img", "startFrame": 1, "endFrame": 376, "nz": 4, "ext": "jpg", "anno_path": "Surfer/groundtruth_rect.txt"},
            {"name": "Suv", "path": "Suv/img", "startFrame": 1, "endFrame": 945, "nz": 4, "ext": "jpg", "anno_path": "Suv/groundtruth_rect.txt"},
            {"name": "Sylvester", "path": "Sylvester/img", "startFrame": 1, "endFrame": 1345, "nz": 4, "ext": "jpg", "anno_path": "Sylvester/groundtruth_rect.txt"},
            {"name": "Tiger1", "path": "Tiger1/img", "startFrame": 1, "endFrame": 354, "nz": 4, "ext": "jpg", "anno_path": "Tiger1/groundtruth_rect.txt", "initOmit": 5},
            {"name": "Tiger2", "path": "Tiger2/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg", "anno_path": "Tiger2/groundtruth_rect.txt"},
            {"name": "Toy", "path": "Toy/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg", "anno_path": "Toy/groundtruth_rect.txt"},
            {"name": "Trans", "path": "Trans/img", "startFrame": 1, "endFrame": 124, "nz": 4, "ext": "jpg", "anno_path": "Trans/groundtruth_rect.txt"},
            {"name": "Trellis", "path": "Trellis/img", "startFrame": 1, "endFrame": 569, "nz": 4, "ext": "jpg", "anno_path": "Trellis/groundtruth_rect.txt"},
            {"name": "Twinnings", "path": "Twinnings/img", "startFrame": 1, "endFrame": 472, "nz": 4, "ext": "jpg", "anno_path": "Twinnings/groundtruth_rect.txt"},
            {"name": "Vase", "path": "Vase/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg", "anno_path": "Vase/groundtruth_rect.txt"},
            {"name": "Walking", "path": "Walking/img", "startFrame": 1, "endFrame": 412, "nz": 4, "ext": "jpg", "anno_path": "Walking/groundtruth_rect.txt"},
            {"name": "Walking2", "path": "Walking2/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg", "anno_path": "Walking2/groundtruth_rect.txt"},
            {"name": "Woman", "path": "Woman/img", "startFrame": 1, "endFrame": 597, "nz": 4, "ext": "jpg", "anno_path": "Woman/groundtruth_rect.txt"}
        ]

        return sequence_info_list
