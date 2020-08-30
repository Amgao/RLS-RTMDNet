import os
from os.path import join, isdir
from tracker import *
import numpy as np

import argparse

import pickle

import math
import warnings
warnings.filterwarnings('ignore')
torch.cuda.set_device(1)

def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)


    if set_type == 'OTB100':
        ############################################  have to refine #############################################

        img_list = sorted([seq_path + '/img/' + p for p in os.listdir(seq_path + '/img') if os.path.splitext(p)[1] == '.jpg'])

        if (seqname == 'Jogging_1') or (seqname == 'Skating2_1'):
            gt = np.loadtxt(seq_path + '/groundtruth_rect.1.txt')
        elif (seqname == 'Jogging_2') or (seqname == 'Skating2_2'):
            gt = np.loadtxt(seq_path + '/groundtruth_rect.2.txt')
        elif seqname =='Human4':
            gt = np.loadtxt(seq_path + '/groundtruth_rect.2.txt', delimiter=',')
        elif (seqname == 'BlurBody')  or (seqname == 'BlurCar1') or (seqname == 'BlurCar2') or (seqname == 'BlurCar3') \
                or (seqname == 'BlurCar4') or (seqname == 'BlurFace') or (seqname == 'BlurOwl') or (seqname == 'Board') \
                or (seqname == 'Box')   or (seqname == 'Car4')  or (seqname == 'CarScale') or (seqname == 'ClifBar') \
                or (seqname == 'Couple')  or (seqname == 'Crossing')  or (seqname == 'Dog') or (seqname == 'FaceOcc1') \
                or (seqname == 'Girl') or (seqname == 'Rubik') or (seqname == 'Singer1') or (seqname == 'Subway') \
                or (seqname == 'Surfer') or (seqname == 'Sylvester') or (seqname == 'Toy') or (seqname == 'Twinnings') \
                or (seqname == 'Vase') or (seqname == 'Walking') or (seqname == 'Walking2') or (seqname == 'Woman')   :
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt')
        else:
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt', delimiter=',')

        if seqname == 'David':
            img_list = img_list[299:]
          
        if seqname == 'Football1':
            img_list = img_list[0:74]
        if seqname == 'Freeman3':
            img_list = img_list[0:460]
        if seqname == 'Freeman4':
            img_list = img_list[0:283]
        if seqname == 'Diving':
            img_list = img_list[0:215]

    elif set_type == 'UAV123':
        img_list = sorted([seq_path + '/' + p for p in os.listdir(seq_path) if os.path.splitext(p)[1] == '.jpg'])

        gt = np.loadtxt(seq_path + '/anno/UAV123/' + seqname + '.txt', delimiter=',')
            

        if seqname == 'bird1_1':
            img_list = img_list[0:253]
        if seqname == 'bird1_2':
            img_list = img_list[774:1477]
        if seqname == 'bird1_3':
            img_list = img_list[1572:2437]

        if seqname == 'car1_1':
            img_list = img_list[0:751]
        if seqname == 'car1_2':
            img_list = img_list[750:1627]
        if seqname == 'car1_3':
            img_list = img_list[1626:2629]

        if seqname == 'car6_1':
            img_list = img_list[0:487]
        if seqname == 'car6_2':
            img_list = img_list[486:1807]
        if seqname == 'car6_3':
            img_list = img_list[1806:2953]
        if seqname == 'car6_4':
            img_list = img_list[2952:3925]
        if seqname == 'car6_5':
            img_list = img_list[3924:4861]
        
        if seqname == 'car8_1':
            img_list = img_list[0:1357]
        if seqname == 'car8_2':
            img_list = img_list[1356:2575]

        if seqname == 'car16_1':
            img_list = img_list[0:415]
        if seqname == 'car16_2':
            img_list = img_list[414:1993]


        if seqname == 'group1_1':
            img_list = img_list[0:1333]
        if seqname == 'group1_2':
            img_list = img_list[1332:2515]
        if seqname == 'group1_3':
            img_list = img_list[2514:3925]
        if seqname == 'group1_4':
            img_list = img_list[3924:4873]

        if seqname == 'group2_1':
            img_list = img_list[0:907]
        if seqname == 'group2_2':
            img_list = img_list[906:1771]
        if seqname == 'group2_3':
            img_list = img_list[1770:2683]

        if seqname == 'group3_1':
            img_list = img_list[0:1567]
        if seqname == 'group3_2':
            img_list = img_list[1566:2827]
        if seqname == 'group3_3':
            img_list = img_list[2826:4369]
        if seqname == 'group3_4':
            img_list = img_list[4368:5527]

        if seqname == 'person2_1':
            img_list = img_list[0:1189]
        if seqname == 'person2_2':
            img_list = img_list[1188:2623]

        if seqname == 'person4_1':
            img_list = img_list[0:1501]
        if seqname == 'person4_2':
            img_list = img_list[1500:2743]

        if seqname == 'person5_1':
            img_list = img_list[0:877]
        if seqname == 'person5_2':
            img_list = img_list[876:2101]

        if seqname == 'person7_1':
            img_list = img_list[0:1249]
        if seqname == 'person7_2':
            img_list = img_list[1248:2065]

        if seqname == 'person8_1':
            img_list = img_list[0:1075]
        if seqname == 'person8_2':
            img_list = img_list[1074:1525]

        if seqname == 'person12_1':
            img_list = img_list[0:601]
        if seqname == 'person12_2':
            img_list = img_list[600:1621]

        if seqname == 'person14_1':
            img_list = img_list[0:847]
        if seqname == 'person14_2':
            img_list = img_list[846:1813]
        if seqname == 'person14_3':
            img_list = img_list[1812:2923]

        if seqname == 'person17_1':
            img_list = img_list[0:1501]
        if seqname == 'person17_2':
            img_list = img_list[1500:2347]

        if seqname == 'person19_1':
            img_list = img_list[0:1243]
        if seqname == 'person19_2':
            img_list = img_list[1242:2791]
        if seqname == 'person19_3':
            img_list = img_list[2790:4357]

        if seqname == 'truck4_1':
            img_list = img_list[0:577]
        if seqname == 'truck4_2':
            img_list = img_list[576:1261]

        if seqname == 'uav1_1':
            img_list = img_list[0:1555]
        if seqname == 'uav1_2':
            img_list = img_list[1554:2377]
        if seqname == 'uav1_3':
            img_list = img_list[2472:3469]

        if seqname == 'truck2':
            img_list = img_list[0:385]

        ##polygon to rect
    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    return img_list, gt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default = 'OTB100' )
    parser.add_argument("-model_path", default = './models/rt-mdnet.pth')
    parser.add_argument("-result_path", default = './result.npy')
    parser.add_argument("-visual_log",default=False, action= 'store_true')
    parser.add_argument("-visualize",default=False, action='store_true')
    parser.add_argument("-adaptive_align",default=True, action='store_false')
    parser.add_argument("-padding",default=1.2, type = float)
    parser.add_argument("-jitter",default=True, action='store_false')

    args = parser.parse_args()

    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ## option setting
    opts['model_path']=args.model_path
    opts['result_path']=args.result_path
    opts['visual_log']=args.visual_log
    opts['set_type']=args.set_type
    opts['visualize'] = args.visualize
    opts['adaptive_align'] = args.adaptive_align
    opts['padding'] = args.padding
    opts['jitter'] = args.jitter
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    print (opts)


    ## path initialization
    dataset_path = '/home/jgao/Recent/'


    seq_home = dataset_path + opts['set_type']
    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
    mIoU_max = 0.0
    mIoU_min = 1.0
    mIoU_avg = 0.0
    res_list = []
    for iterloop in range(50):
        iou_list=[]
        fps_list=dict()
        bb_result = dict()
        result = dict()

        iou_list_nobb=[]
        bb_result_nobb = dict()
        for num,seq in enumerate(seq_list):
            if num<-1:
                continue
            seq_path = seq_home + '/' + seq
            img_list,gt=genConfig(seq_path,opts['set_type'])

            if os.path.exists(opts['result_path']+str(iterloop)+'replay.npy'):
                resultdic = np.load(opts['result_path']+str(iterloop)+'replay.npy', allow_pickle=True)
                resultdic = resultdic.tolist()
                result_bb = resultdic['bb_result'][seq]
                fps = resultdic['fps'][seq]
                result_nobb = resultdic['bb_result_nobb'][seq]
                iou_result = np.zeros((len(img_list), 1))
                for i in range(1, len(img_list)):
                    iou_result[i] = overlap_ratio(gt[i], result_bb[i])[0]
            else:
                iou_result, result_bb, fps, result_nobb = run_mdnet(img_list, gt[0], gt, seq = seq, display=opts['visualize'])

            enable_frameNum = 0.
            for iidx in range(len(iou_result)):
                if (math.isnan(iou_result[iidx])==False):
                    enable_frameNum += 1.
                else:
                    ## gt is not alowed
                    iou_result[iidx] = 0.

            iou_list.append(iou_result.sum()/enable_frameNum)
            bb_result[seq] = result_bb
            fps_list[seq]=fps

            bb_result_nobb[seq] = result_nobb
            print ('{} {} : {} , total mIoU:{}, fps:{}'.format(num,seq,iou_result.mean(), sum(iou_list)/len(iou_list),sum(fps_list.values())/len(fps_list)))

        res_list.append(sum(iou_list) / len(iou_list))
        mIoU_avg += sum(iou_list)/len(iou_list)
        if mIoU_max < sum(iou_list)/len(iou_list):
            mIoU_max = sum(iou_list)/len(iou_list)
        if mIoU_min > sum(iou_list)/len(iou_list):
            mIoU_min = sum(iou_list)/len(iou_list)
        result['bb_result']=bb_result
        result['fps']=fps_list
        result['bb_result_nobb']=bb_result_nobb
        np.save(opts['result_path'] + str(iterloop) + 'replay', result)
        print (mIoU_max)
        print (mIoU_min)
        print (res_list)
        #np.save(opts['result_path']+str(iterloop),result)
    mIoU_avg /= 50
    print (mIoU_max)
    print (mIoU_avg)
    print (mIoU_min)
    print (res_list)
