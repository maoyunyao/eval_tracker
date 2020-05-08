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


class EvalUAV123:
    def __init__(self, dataset_path, seq_len):
        '''
            dataset_path: Path to UAV123 dataset 
        '''
        self.dataset_path = dataset_path
        self.seq_info_list = self._get_sequence_info_list()
        self.seq_len = seq_len

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
            seq_index: integer between 0 and 122, indicate 123 sequences in UAV123
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
        
        iou = iou[~np.isnan(iou)] # in uav123, nan means out of view
        
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
        
        dist = dist[~np.isnan(dist)] # in uav123, nan means out of view
        
        frameNum = len(dist)

        for i in range( len(dist_thresholds) ):
            score[i] = sum(dist <= dist_thresholds[i]) / float(frameNum)
        return score

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "uav_bike1", "path": "data_seq/UAV123/bike1", "startFrame": 1, "endFrame": 3085, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/bike1.txt"},
            {"name": "uav_bike2", "path": "data_seq/UAV123/bike2", "startFrame": 1, "endFrame": 553, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/bike2.txt"},
            {"name": "uav_bike3", "path": "data_seq/UAV123/bike3", "startFrame": 1, "endFrame": 433, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/bike3.txt"},
            {"name": "uav_bird1_1", "path": "data_seq/UAV123/bird1", "startFrame": 1, "endFrame": 253, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/bird1_1.txt"},
            {"name": "uav_bird1_2", "path": "data_seq/UAV123/bird1", "startFrame": 775, "endFrame": 1477, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/bird1_2.txt"},
            {"name": "uav_bird1_3", "path": "data_seq/UAV123/bird1", "startFrame": 1573, "endFrame": 2437, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/bird1_3.txt"},
            {"name": "uav_boat1", "path": "data_seq/UAV123/boat1", "startFrame": 1, "endFrame": 901, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat1.txt"},
            {"name": "uav_boat2", "path": "data_seq/UAV123/boat2", "startFrame": 1, "endFrame": 799, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat2.txt"},
            {"name": "uav_boat3", "path": "data_seq/UAV123/boat3", "startFrame": 1, "endFrame": 901, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat3.txt"},
            {"name": "uav_boat4", "path": "data_seq/UAV123/boat4", "startFrame": 1, "endFrame": 553, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat4.txt"},
            {"name": "uav_boat5", "path": "data_seq/UAV123/boat5", "startFrame": 1, "endFrame": 505, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat5.txt"},
            {"name": "uav_boat6", "path": "data_seq/UAV123/boat6", "startFrame": 1, "endFrame": 805, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat6.txt"},
            {"name": "uav_boat7", "path": "data_seq/UAV123/boat7", "startFrame": 1, "endFrame": 535, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat7.txt"},
            {"name": "uav_boat8", "path": "data_seq/UAV123/boat8", "startFrame": 1, "endFrame": 685, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat8.txt"},
            {"name": "uav_boat9", "path": "data_seq/UAV123/boat9", "startFrame": 1, "endFrame": 1399, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat9.txt"},
            {"name": "uav_building1", "path": "data_seq/UAV123/building1", "startFrame": 1, "endFrame": 469, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/building1.txt"},
            {"name": "uav_building2", "path": "data_seq/UAV123/building2", "startFrame": 1, "endFrame": 577, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/building2.txt"},
            {"name": "uav_building3", "path": "data_seq/UAV123/building3", "startFrame": 1, "endFrame": 829, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/building3.txt"},
            {"name": "uav_building4", "path": "data_seq/UAV123/building4", "startFrame": 1, "endFrame": 787, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/building4.txt"},
            {"name": "uav_building5", "path": "data_seq/UAV123/building5", "startFrame": 1, "endFrame": 481, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/building5.txt"},
            {"name": "uav_car1_1", "path": "data_seq/UAV123/car1", "startFrame": 1, "endFrame": 751, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car1_1.txt"},
            {"name": "uav_car1_2", "path": "data_seq/UAV123/car1", "startFrame": 751, "endFrame": 1627, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car1_2.txt"},
            {"name": "uav_car1_3", "path": "data_seq/UAV123/car1", "startFrame": 1627, "endFrame": 2629, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car1_3.txt"},
            {"name": "uav_car10", "path": "data_seq/UAV123/car10", "startFrame": 1, "endFrame": 1405, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car10.txt"},
            {"name": "uav_car11", "path": "data_seq/UAV123/car11", "startFrame": 1, "endFrame": 337, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car11.txt"},
            {"name": "uav_car12", "path": "data_seq/UAV123/car12", "startFrame": 1, "endFrame": 499, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car12.txt"},
            {"name": "uav_car13", "path": "data_seq/UAV123/car13", "startFrame": 1, "endFrame": 415, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car13.txt"},
            {"name": "uav_car14", "path": "data_seq/UAV123/car14", "startFrame": 1, "endFrame": 1327, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car14.txt"},
            {"name": "uav_car15", "path": "data_seq/UAV123/car15", "startFrame": 1, "endFrame": 469, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car15.txt"},
            {"name": "uav_car16_1", "path": "data_seq/UAV123/car16", "startFrame": 1, "endFrame": 415, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car16_1.txt"},
            {"name": "uav_car16_2", "path": "data_seq/UAV123/car16", "startFrame": 415, "endFrame": 1993, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car16_2.txt"},
            {"name": "uav_car17", "path": "data_seq/UAV123/car17", "startFrame": 1, "endFrame": 1057, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car17.txt"},
            {"name": "uav_car18", "path": "data_seq/UAV123/car18", "startFrame": 1, "endFrame": 1207, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car18.txt"},
            {"name": "uav_car1_s", "path": "data_seq/UAV123/car1_s", "startFrame": 1, "endFrame": 1475, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car1_s.txt"},
            {"name": "uav_car2", "path": "data_seq/UAV123/car2", "startFrame": 1, "endFrame": 1321, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car2.txt"},
            {"name": "uav_car2_s", "path": "data_seq/UAV123/car2_s", "startFrame": 1, "endFrame": 320, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car2_s.txt"},
            {"name": "uav_car3", "path": "data_seq/UAV123/car3", "startFrame": 1, "endFrame": 1717, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car3.txt"},
            {"name": "uav_car3_s", "path": "data_seq/UAV123/car3_s", "startFrame": 1, "endFrame": 1300, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car3_s.txt"},
            {"name": "uav_car4", "path": "data_seq/UAV123/car4", "startFrame": 1, "endFrame": 1345, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car4.txt"},
            {"name": "uav_car4_s", "path": "data_seq/UAV123/car4_s", "startFrame": 1, "endFrame": 830, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car4_s.txt"},
            {"name": "uav_car5", "path": "data_seq/UAV123/car5", "startFrame": 1, "endFrame": 745, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car5.txt"},
            {"name": "uav_car6_1", "path": "data_seq/UAV123/car6", "startFrame": 1, "endFrame": 487, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car6_1.txt"},
            {"name": "uav_car6_2", "path": "data_seq/UAV123/car6", "startFrame": 487, "endFrame": 1807, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car6_2.txt"},
            {"name": "uav_car6_3", "path": "data_seq/UAV123/car6", "startFrame": 1807, "endFrame": 2953, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car6_3.txt"},
            {"name": "uav_car6_4", "path": "data_seq/UAV123/car6", "startFrame": 2953, "endFrame": 3925, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car6_4.txt"},
            {"name": "uav_car6_5", "path": "data_seq/UAV123/car6", "startFrame": 3925, "endFrame": 4861, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car6_5.txt"},
            {"name": "uav_car7", "path": "data_seq/UAV123/car7", "startFrame": 1, "endFrame": 1033, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car7.txt"},
            {"name": "uav_car8_1", "path": "data_seq/UAV123/car8", "startFrame": 1, "endFrame": 1357, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car8_1.txt"},
            {"name": "uav_car8_2", "path": "data_seq/UAV123/car8", "startFrame": 1357, "endFrame": 2575, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car8_2.txt"},
            {"name": "uav_car9", "path": "data_seq/UAV123/car9", "startFrame": 1, "endFrame": 1879, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car9.txt"},
            {"name": "uav_group1_1", "path": "data_seq/UAV123/group1", "startFrame": 1, "endFrame": 1333, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group1_1.txt"},
            {"name": "uav_group1_2", "path": "data_seq/UAV123/group1", "startFrame": 1333, "endFrame": 2515, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group1_2.txt"},
            {"name": "uav_group1_3", "path": "data_seq/UAV123/group1", "startFrame": 2515, "endFrame": 3925, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group1_3.txt"},
            {"name": "uav_group1_4", "path": "data_seq/UAV123/group1", "startFrame": 3925, "endFrame": 4873, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group1_4.txt"},
            {"name": "uav_group2_1", "path": "data_seq/UAV123/group2", "startFrame": 1, "endFrame": 907, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group2_1.txt"},
            {"name": "uav_group2_2", "path": "data_seq/UAV123/group2", "startFrame": 907, "endFrame": 1771, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group2_2.txt"},
            {"name": "uav_group2_3", "path": "data_seq/UAV123/group2", "startFrame": 1771, "endFrame": 2683, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group2_3.txt"},
            {"name": "uav_group3_1", "path": "data_seq/UAV123/group3", "startFrame": 1, "endFrame": 1567, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group3_1.txt"},
            {"name": "uav_group3_2", "path": "data_seq/UAV123/group3", "startFrame": 1567, "endFrame": 2827, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group3_2.txt"},
            {"name": "uav_group3_3", "path": "data_seq/UAV123/group3", "startFrame": 2827, "endFrame": 4369, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group3_3.txt"},
            {"name": "uav_group3_4", "path": "data_seq/UAV123/group3", "startFrame": 4369, "endFrame": 5527, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group3_4.txt"},
            {"name": "uav_person1", "path": "data_seq/UAV123/person1", "startFrame": 1, "endFrame": 799, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person1.txt"},
            {"name": "uav_person10", "path": "data_seq/UAV123/person10", "startFrame": 1, "endFrame": 1021, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person10.txt"},
            {"name": "uav_person11", "path": "data_seq/UAV123/person11", "startFrame": 1, "endFrame": 721, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person11.txt"},
            {"name": "uav_person12_1", "path": "data_seq/UAV123/person12", "startFrame": 1, "endFrame": 601, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person12_1.txt"},
            {"name": "uav_person12_2", "path": "data_seq/UAV123/person12", "startFrame": 601, "endFrame": 1621, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person12_2.txt"},
            {"name": "uav_person13", "path": "data_seq/UAV123/person13", "startFrame": 1, "endFrame": 883, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person13.txt"},
            {"name": "uav_person14_1", "path": "data_seq/UAV123/person14", "startFrame": 1, "endFrame": 847, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person14_1.txt"},
            {"name": "uav_person14_2", "path": "data_seq/UAV123/person14", "startFrame": 847, "endFrame": 1813, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person14_2.txt"},
            {"name": "uav_person14_3", "path": "data_seq/UAV123/person14", "startFrame": 1813, "endFrame": 2923,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person14_3.txt"},
            {"name": "uav_person15", "path": "data_seq/UAV123/person15", "startFrame": 1, "endFrame": 1339, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person15.txt"},
            {"name": "uav_person16", "path": "data_seq/UAV123/person16", "startFrame": 1, "endFrame": 1147, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person16.txt"},
            {"name": "uav_person17_1", "path": "data_seq/UAV123/person17", "startFrame": 1, "endFrame": 1501, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person17_1.txt"},
            {"name": "uav_person17_2", "path": "data_seq/UAV123/person17", "startFrame": 1501, "endFrame": 2347,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person17_2.txt"},
            {"name": "uav_person18", "path": "data_seq/UAV123/person18", "startFrame": 1, "endFrame": 1393, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person18.txt"},
            {"name": "uav_person19_1", "path": "data_seq/UAV123/person19", "startFrame": 1, "endFrame": 1243, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person19_1.txt"},
            {"name": "uav_person19_2", "path": "data_seq/UAV123/person19", "startFrame": 1243, "endFrame": 2791,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person19_2.txt"},
            {"name": "uav_person19_3", "path": "data_seq/UAV123/person19", "startFrame": 2791, "endFrame": 4357,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person19_3.txt"},
            {"name": "uav_person1_s", "path": "data_seq/UAV123/person1_s", "startFrame": 1, "endFrame": 1600, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person1_s.txt"},
            {"name": "uav_person2_1", "path": "data_seq/UAV123/person2", "startFrame": 1, "endFrame": 1189, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person2_1.txt"},
            {"name": "uav_person2_2", "path": "data_seq/UAV123/person2", "startFrame": 1189, "endFrame": 2623, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person2_2.txt"},
            {"name": "uav_person20", "path": "data_seq/UAV123/person20", "startFrame": 1, "endFrame": 1783, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person20.txt"},
            {"name": "uav_person21", "path": "data_seq/UAV123/person21", "startFrame": 1, "endFrame": 487, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person21.txt"},
            {"name": "uav_person22", "path": "data_seq/UAV123/person22", "startFrame": 1, "endFrame": 199, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person22.txt"},
            {"name": "uav_person23", "path": "data_seq/UAV123/person23", "startFrame": 1, "endFrame": 397, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person23.txt"},
            {"name": "uav_person2_s", "path": "data_seq/UAV123/person2_s", "startFrame": 1, "endFrame": 250, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person2_s.txt"},
            {"name": "uav_person3", "path": "data_seq/UAV123/person3", "startFrame": 1, "endFrame": 643, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person3.txt"},
            {"name": "uav_person3_s", "path": "data_seq/UAV123/person3_s", "startFrame": 1, "endFrame": 505, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person3_s.txt"},
            {"name": "uav_person4_1", "path": "data_seq/UAV123/person4", "startFrame": 1, "endFrame": 1501, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person4_1.txt"},
            {"name": "uav_person4_2", "path": "data_seq/UAV123/person4", "startFrame": 1501, "endFrame": 2743, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person4_2.txt"},
            {"name": "uav_person5_1", "path": "data_seq/UAV123/person5", "startFrame": 1, "endFrame": 877, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person5_1.txt"},
            {"name": "uav_person5_2", "path": "data_seq/UAV123/person5", "startFrame": 877, "endFrame": 2101, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person5_2.txt"},
            {"name": "uav_person6", "path": "data_seq/UAV123/person6", "startFrame": 1, "endFrame": 901, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person6.txt"},
            {"name": "uav_person7_1", "path": "data_seq/UAV123/person7", "startFrame": 1, "endFrame": 1249, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person7_1.txt"},
            {"name": "uav_person7_2", "path": "data_seq/UAV123/person7", "startFrame": 1249, "endFrame": 2065, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person7_2.txt"},
            {"name": "uav_person8_1", "path": "data_seq/UAV123/person8", "startFrame": 1, "endFrame": 1075, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person8_1.txt"},
            {"name": "uav_person8_2", "path": "data_seq/UAV123/person8", "startFrame": 1075, "endFrame": 1525, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person8_2.txt"},
            {"name": "uav_person9", "path": "data_seq/UAV123/person9", "startFrame": 1, "endFrame": 661, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person9.txt"},
            {"name": "uav_truck1", "path": "data_seq/UAV123/truck1", "startFrame": 1, "endFrame": 463, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/truck1.txt"},
            {"name": "uav_truck2", "path": "data_seq/UAV123/truck2", "startFrame": 1, "endFrame": 385, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/truck2.txt"},
            {"name": "uav_truck3", "path": "data_seq/UAV123/truck3", "startFrame": 1, "endFrame": 535, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/truck3.txt"},
            {"name": "uav_truck4_1", "path": "data_seq/UAV123/truck4", "startFrame": 1, "endFrame": 577, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/truck4_1.txt"},
            {"name": "uav_truck4_2", "path": "data_seq/UAV123/truck4", "startFrame": 577, "endFrame": 1261, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/truck4_2.txt"},
            {"name": "uav_uav1_1", "path": "data_seq/UAV123/uav1", "startFrame": 1, "endFrame": 1555, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav1_1.txt"},
            {"name": "uav_uav1_2", "path": "data_seq/UAV123/uav1", "startFrame": 1555, "endFrame": 2377, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav1_2.txt"},
            {"name": "uav_uav1_3", "path": "data_seq/UAV123/uav1", "startFrame": 2473, "endFrame": 3469, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav1_3.txt"},
            {"name": "uav_uav2", "path": "data_seq/UAV123/uav2", "startFrame": 1, "endFrame": 133, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav2.txt"},
            {"name": "uav_uav3", "path": "data_seq/UAV123/uav3", "startFrame": 1, "endFrame": 265, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav3.txt"},
            {"name": "uav_uav4", "path": "data_seq/UAV123/uav4", "startFrame": 1, "endFrame": 157, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav4.txt"},
            {"name": "uav_uav5", "path": "data_seq/UAV123/uav5", "startFrame": 1, "endFrame": 139, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav5.txt"},
            {"name": "uav_uav6", "path": "data_seq/UAV123/uav6", "startFrame": 1, "endFrame": 109, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav6.txt"},
            {"name": "uav_uav7", "path": "data_seq/UAV123/uav7", "startFrame": 1, "endFrame": 373, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav7.txt"},
            {"name": "uav_uav8", "path": "data_seq/UAV123/uav8", "startFrame": 1, "endFrame": 301, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav8.txt"},
            {"name": "uav_wakeboard1", "path": "data_seq/UAV123/wakeboard1", "startFrame": 1, "endFrame": 421, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard1.txt"},
            {"name": "uav_wakeboard10", "path": "data_seq/UAV123/wakeboard10", "startFrame": 1, "endFrame": 469,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard10.txt"},
            {"name": "uav_wakeboard2", "path": "data_seq/UAV123/wakeboard2", "startFrame": 1, "endFrame": 733, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard2.txt"},
            {"name": "uav_wakeboard3", "path": "data_seq/UAV123/wakeboard3", "startFrame": 1, "endFrame": 823, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard3.txt"},
            {"name": "uav_wakeboard4", "path": "data_seq/UAV123/wakeboard4", "startFrame": 1, "endFrame": 697, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard4.txt"},
            {"name": "uav_wakeboard5", "path": "data_seq/UAV123/wakeboard5", "startFrame": 1, "endFrame": 1675, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard5.txt"},
            {"name": "uav_wakeboard6", "path": "data_seq/UAV123/wakeboard6", "startFrame": 1, "endFrame": 1165, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard6.txt"},
            {"name": "uav_wakeboard7", "path": "data_seq/UAV123/wakeboard7", "startFrame": 1, "endFrame": 199, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard7.txt"},
            {"name": "uav_wakeboard8", "path": "data_seq/UAV123/wakeboard8", "startFrame": 1, "endFrame": 1543, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard8.txt"},
            {"name": "uav_wakeboard9", "path": "data_seq/UAV123/wakeboard9", "startFrame": 1, "endFrame": 355, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard9.txt"}
        ]

        return sequence_info_list
    
