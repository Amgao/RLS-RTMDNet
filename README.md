## RLS-RTMDNet
Code and raw result files of our CVPR2020 oral paper "[Recursive Least-Squares Estimator-Aided Online Learning for Visual Tracking](https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_Recursive_Least-Squares_Estimator-Aided_Online_Learning_for_Visual_Tracking_CVPR_2020_paper.html)"

Created by [Jin Gao](http://www.nlpr.ia.ac.cn/users/gaojin/)

### Introduction
RLS-RTMDNet is dedicated to improving online tracking part of RT-MDNet ([project page](http://cvlab.postech.ac.kr/~chey0313/real_time_mdnet/) and [paper](https://arxiv.org/pdf/1808.08834.pdf)) based on our proposed recursive least-squares estimator-aided online learning method.

### Citation
If you're using this code in a publication, please cite our paper.

	@InProceedings{Gao_2020_CVPR,
   	author = {Gao, Jin and Hu, Weiming and Lu, Yan},
    	title = {Recursive Least-squares Estimator-aided Online Learning for Visual Tracking},
    	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    	month = {June},
    	year = {2020}
  	}
  

### System Requirements

This code is tested on 64 bit Linux (Ubuntu 16.04 LTS) with the following Anaconda environment:
>> * PyTorch (= 1.2.0)
>> * Python (= 3.7.4)
  
### Online Tracking

**Pretrained Model**
 The off-the-shelf pretrained model in RT-MDNet is used for our testing: [RT-MDNet-ImageNet-pretrained](https://www.dropbox.com/s/lr8uft05zlo21an/rt-mdnet.pth?dl=0).

**Demo**
>> * 'Run.py' for OTB and UAV123
>> * 'python_RLS_RTMDNet.py' for VOT16/17.
  
