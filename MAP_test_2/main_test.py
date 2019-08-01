#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:25:31 2019

@author: chengxueping
"""
from judgement import voc_ap,input_ground_truth,get_AP,class_division
import numpy as np
import os

input_data=[[{'classID':0,'x1':288.0,'x2':302.0,'y1':158.0,'y2':172.0,'score':0.931217},
{'classID':0,'x1':217.0,'x2':241.0,'y1':155.0,'y2':178.0,'score':0.87208753},
{'classID':0,'x1':247.5,'x2':258.5,'y1':160.0,'y2':170.0,'score':0.94877433},
{'classID':0,'x1':263.5,'x2':272.5,'y1':160.0,'y2':168.0,'score':0.80760478},
{'classID':0,'x1':304.5,'x2':341.5,'y1':155.5,'y2':187.0,'score':0.933003544},
{'classID':0,'x1':376.0,'x2':636.0,'y1':86.0,'y2':286.0,'score':0.94288516},
{'classID':0,'x1':1.0,'x2':53.0,'y1':154.0,'y2':226.0,'score':0.91569763},
{'classID':0,'x1':335.5,'x2':386.5,'y1':162.0,'y2':212.0,'score':0.9039805},
{'classID':0,'x1':52.0,'x2':222.0,'y1':133.5,'y2':272.5,'score':0.92768150}],
[{'classID':2,'x1':297.5,'x2':312.5,'y1':138.0,'y2':180.0,'score':0.87488305},
{'classID':2,'x1':357.5,'x2':376.5,'y1':136.5,'y2':187.5,'score':0.9209788},
{'classID':2,'x1':317.5,'x2':336.5,'y1':136.5,'y2':189.5,'score':0.8816238},
{'classID':2,'x1':455.5,'x2':482.5,'y1':134.5,'y2':197.5,'score':0.94544273},
{'classID':2,'x1':477.0,'x2':499.0,'y1':138.0,'y2':195.5,'score':0.94434338},
{'classID':0,'x1':9.0,'x2':259.0,'y1':104.5,'y2':285.5,'score':0.99504894},
{'classID':2,'x1':383.0,'x2':397.0,'y1':139.0,'y2':180.5,'score':0.7869959}]]
image=['0a0a0b1a-7c39d841.txt','0a0c3694-24b5193a.txt']


sum_class=[]
sum_class_get=[]
MAP=0

for i in range(10):
    sum_class_get.append([])
    sum_class_get.append([])
    sum_class_get.append(0)
    
#print(sum_class_get) 


for i,img in enumerate(input_data):        #image
    image_name=image[i]#获取图片名称
    x=[]
    L=[]   
#    z=[]
    img_division=class_division(img,image_name)#class 10list
    for class_namefile in img_division:
        TP,FP,npos=get_AP(0.5,class_namefile)#该图像上每个类别的TP,FP数
        x.append(TP)# TP class-img
        x.append(FP)
        x.append(npos)         
    sum_class.append(x)
#print(sum_class)
for i in sum_class:
    for j in range(30):
            sum_class_get[j]+=i[j]

for i in range(10):
    tps=np.array(sum_class_get[i])
    fps=np.array(sum_class_get[i+1])
    npos=sum_class_get[i+2]
    npos_f=np.zeros(len(tps))+npos
    rec = tps/npos_f
    prec = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
#TPs=sum_class_get[:,0]
#FPs=sum_class_get[:,1]
#npos=sum_class_get[:,2]
#for i in len(TPs):
#    prec=TPs[i]/(TPs+FPs)
#    rec=TPs[i]/npos
#    AP=voc_ap(rec,prec,use_07_metric=False)
#    MAP+=AP
#
#MAP=MAP/len(TPs)   