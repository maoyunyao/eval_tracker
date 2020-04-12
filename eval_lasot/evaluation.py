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


class EvalLaSOT:
    def __init__(self, dataset_path):
        '''
            dataset_path: Path to LaSOT dataset 
        '''
        self.dataset_path = dataset_path
        self.seq_list = self._get_sequence_list()

        self.gt = {}
        for seq_name in self.seq_list:
            class_name = seq_name.split('-')[0]
            seq_gt_path = '{}/{}/{}/groundtruth.txt'.format(self.dataset_path, class_name, seq_name)
            try:
                seq_gt = np.loadtxt(str(seq_gt_path), dtype=np.float64)
            except:
                seq_gt = np.loadtxt(str(seq_gt_path), delimiter=',', dtype=np.float64)
            
            self.gt[seq_name] = seq_gt


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
            seq_index: integer between 0 and 279, indicate 280 test sequences in LaSOT
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
        frameNum = len(gt)
        score = np.zeros( len(iou_thresholds) )
        iou = IoU(gt, result)
        for i in range(len(iou_thresholds)):
            score[i] = sum(iou > iou_thresholds[i]) / float(frameNum)
        return score

    def compute_precision(self, gt, result):
        gtCenter = convert_bbox_to_center(gt)
        resultCenter = convert_bbox_to_center(result)
        dist_thresholds = np.arange(0, 51, 1)
        frameNum = len(gt)
        score = np.zeros( len(dist_thresholds) )
        dist = np.sqrt( np.sum(np.power(gtCenter-resultCenter, 2), axis=1) )
        for i in range( len(dist_thresholds) ):
            score[i] = sum(dist <= dist_thresholds[i]) / float(frameNum)
        return score
    
    def _get_sequence_list(self):
        sequence_list = ['airplane-1',
                         'airplane-9',
                         'airplane-13',
                         'airplane-15',
                         'basketball-1',
                         'basketball-6',
                         'basketball-7',
                         'basketball-11',
                         'bear-2',
                         'bear-4',
                         'bear-6',
                         'bear-17',
                         'bicycle-2',
                         'bicycle-7',
                         'bicycle-9',
                         'bicycle-18',
                         'bird-2',
                         'bird-3',
                         'bird-15',
                         'bird-17',
                         'boat-3',
                         'boat-4',
                         'boat-12',
                         'boat-17',
                         'book-3',
                         'book-10',
                         'book-11',
                         'book-19',
                         'bottle-1',
                         'bottle-12',
                         'bottle-14',
                         'bottle-18',
                         'bus-2',
                         'bus-5',
                         'bus-17',
                         'bus-19',
                         'car-2',
                         'car-6',
                         'car-9',
                         'car-17',
                         'cat-1',
                         'cat-3',
                         'cat-18',
                         'cat-20',
                         'cattle-2',
                         'cattle-7',
                         'cattle-12',
                         'cattle-13',
                         'spider-14',
                         'spider-16',
                         'spider-18',
                         'spider-20',
                         'coin-3',
                         'coin-6',
                         'coin-7',
                         'coin-18',
                         'crab-3',
                         'crab-6',
                         'crab-12',
                         'crab-18',
                         'surfboard-12',
                         'surfboard-4',
                         'surfboard-5',
                         'surfboard-8',
                         'cup-1',
                         'cup-4',
                         'cup-7',
                         'cup-17',
                         'deer-4',
                         'deer-8',
                         'deer-10',
                         'deer-14',
                         'dog-1',
                         'dog-7',
                         'dog-15',
                         'dog-19',
                         'guitar-3',
                         'guitar-8',
                         'guitar-10',
                         'guitar-16',
                         'person-1',
                         'person-5',
                         'person-10',
                         'person-12',
                         'pig-2',
                         'pig-10',
                         'pig-13',
                         'pig-18',
                         'rubicCube-1',
                         'rubicCube-6',
                         'rubicCube-14',
                         'rubicCube-19',
                         'swing-10',
                         'swing-14',
                         'swing-17',
                         'swing-20',
                         'drone-13',
                         'drone-15',
                         'drone-2',
                         'drone-7',
                         'pool-12',
                         'pool-15',
                         'pool-3',
                         'pool-7',
                         'rabbit-10',
                         'rabbit-13',
                         'rabbit-17',
                         'rabbit-19',
                         'racing-10',
                         'racing-15',
                         'racing-16',
                         'racing-20',
                         'robot-1',
                         'robot-19',
                         'robot-5',
                         'robot-8',
                         'sepia-13',
                         'sepia-16',
                         'sepia-6',
                         'sepia-8',
                         'sheep-3',
                         'sheep-5',
                         'sheep-7',
                         'sheep-9',
                         'skateboard-16',
                         'skateboard-19',
                         'skateboard-3',
                         'skateboard-8',
                         'tank-14',
                         'tank-16',
                         'tank-6',
                         'tank-9',
                         'tiger-12',
                         'tiger-18',
                         'tiger-4',
                         'tiger-6',
                         'train-1',
                         'train-11',
                         'train-20',
                         'train-7',
                         'truck-16',
                         'truck-3',
                         'truck-6',
                         'truck-7',
                         'turtle-16',
                         'turtle-5',
                         'turtle-8',
                         'turtle-9',
                         'umbrella-17',
                         'umbrella-19',
                         'umbrella-2',
                         'umbrella-9',
                         'yoyo-15',
                         'yoyo-17',
                         'yoyo-19',
                         'yoyo-7',
                         'zebra-10',
                         'zebra-14',
                         'zebra-16',
                         'zebra-17',
                         'elephant-1',
                         'elephant-12',
                         'elephant-16',
                         'elephant-18',
                         'goldfish-3',
                         'goldfish-7',
                         'goldfish-8',
                         'goldfish-10',
                         'hat-1',
                         'hat-2',
                         'hat-5',
                         'hat-18',
                         'kite-4',
                         'kite-6',
                         'kite-10',
                         'kite-15',
                         'motorcycle-1',
                         'motorcycle-3',
                         'motorcycle-9',
                         'motorcycle-18',
                         'mouse-1',
                         'mouse-8',
                         'mouse-9',
                         'mouse-17',
                         'flag-3',
                         'flag-9',
                         'flag-5',
                         'flag-2',
                         'frog-3',
                         'frog-4',
                         'frog-20',
                         'frog-9',
                         'gametarget-1',
                         'gametarget-2',
                         'gametarget-7',
                         'gametarget-13',
                         'hand-2',
                         'hand-3',
                         'hand-9',
                         'hand-16',
                         'helmet-5',
                         'helmet-11',
                         'helmet-19',
                         'helmet-13',
                         'licenseplate-6',
                         'licenseplate-12',
                         'licenseplate-13',
                         'licenseplate-15',
                         'electricfan-1',
                         'electricfan-10',
                         'electricfan-18',
                         'electricfan-20',
                         'chameleon-3',
                         'chameleon-6',
                         'chameleon-11',
                         'chameleon-20',
                         'crocodile-3',
                         'crocodile-4',
                         'crocodile-10',
                         'crocodile-14',
                         'gecko-1',
                         'gecko-5',
                         'gecko-16',
                         'gecko-19',
                         'fox-2',
                         'fox-3',
                         'fox-5',
                         'fox-20',
                         'giraffe-2',
                         'giraffe-10',
                         'giraffe-13',
                         'giraffe-15',
                         'gorilla-4',
                         'gorilla-6',
                         'gorilla-9',
                         'gorilla-13',
                         'hippo-1',
                         'hippo-7',
                         'hippo-9',
                         'hippo-20',
                         'horse-1',
                         'horse-4',
                         'horse-12',
                         'horse-15',
                         'kangaroo-2',
                         'kangaroo-5',
                         'kangaroo-11',
                         'kangaroo-14',
                         'leopard-1',
                         'leopard-7',
                         'leopard-16',
                         'leopard-20',
                         'lion-1',
                         'lion-5',
                         'lion-12',
                         'lion-20',
                         'lizard-1',
                         'lizard-3',
                         'lizard-6',
                         'lizard-13',
                         'microphone-2',
                         'microphone-6',
                         'microphone-14',
                         'microphone-16',
                         'monkey-3',
                         'monkey-4',
                         'monkey-9',
                         'monkey-17',
                         'shark-2',
                         'shark-3',
                         'shark-5',
                         'shark-6',
                         'squirrel-8',
                         'squirrel-11',
                         'squirrel-13',
                         'squirrel-19',
                         'volleyball-1',
                         'volleyball-13',
                         'volleyball-18',
                         'volleyball-19']
        return sequence_list

if __name__ == "__main__":
    evalotb = EvalOTB("/media/myy/Data/OTB100")
    success, precision = evalotb.evaluate("/media/myy/Data/Repo/eval_otb/Results/ATOM")
    evalotb.visualize("/media/myy/Data/Repo/eval_otb/Results/ATOM",0)