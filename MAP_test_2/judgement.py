import numpy as np
import os
#from collections import Counter

def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:  # 使用07年方法
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])  # 插值
            ap = ap + p / 11.
    else:  # 新方式，计算所有点
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # compute the precision 曲线值（也用了插值）
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def input_ground_truth(txt_file,classID):#对标签数据进行处理
    L1=[]
    L2=[]
    L3=[]
    Class = {'0':'car', '1':'bus', '2':'person', '3':'bike', '4':'truck', '5':'motor', '6':'train', '7':'rider', '8':'traffic_sign', '9':'traffic_light'}
    with open(txt_file, 'r') as file1:
        for line in file1:
            L1.append(line)
        for i in L1:
           if i.split(' ')[0]==Class[str(classID)]:
               L2.append(i.split(' ')[4])
               L2.append(i.split(' ')[5])
               L2.append(i.split(' ')[6])
               L2.append(i.split(' ')[7])
        for x in range(0,len(L2),4):
            L3.append(L2[x:x+4])
            l=len(L3)
        return L3,l
    



def get_AP(ovthresh,class_detected_file,use_07_metric=False):
   # {'classID': 0, 'x1': 285, 'x2': 253, 'y1': 52, 'y2': 89, 'scorce': 0.56}
    #confidence是该类别的检测结果的置信度,BB一行4个坐标，表示检测结果的框,image_ids图片的文件名,
    #class_recs应该是lables按照字典形式,npos所有的正样本数,ovthresh设定的IOU阈值,use_07_metric选择用什么方法
    new_class_file= sorted(class_detected_file,key = lambda e:e.__getitem__('score'),reverse=True)
    base_path=os.getcwd()
#    sorted_ind= np.argsort(-confidence) # 按照置信度降序排序
#    BB = BB[sorted_ind, :]   # 预测框坐标
#    image_ids = [image_ids[x] for x in sorted_ind] # 各个预测框的对应图片id  
    # 便利预测框，并统计TPs和FPs
    nd = len(class_detected_file)#统计该类别有多少个样本
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    over_lap=[]
    npos=0
    class_ids=[]
    for d in range(nd):
#        R = class_recs[[d]]#对应预测结果的真实属性值，字典形式，包括很多属性
        filename=new_class_file[d]['imageID']
        class_ids.append(filename)
        file_path=os.path.join(base_path,'DATA','detection_2d','labels',filename)
        BBGT,number=input_ground_truth(file_path,new_class_file[d]['classID'])
        bb=[new_class_file[d]['x1'],new_class_file[d]['x2'],new_class_file[d]['y1'],new_class_file[d]['y2']]
#        bb = BB[d, :].astype(float)#找到第d行框，即该预测结果的4个坐标
        ovmax = -np.inf
#        if (filename in class_ids) is False:
#            npos+=number
#        BBGT = R['bbox'].astype(float)  # ground truth，找到真实框，按字典属性寻找
        if np.size(BBGT) > 0:#IOU计算
            for i in range(number):
                ixmin = np.maximum(int(float(BBGT[i][0])),int( bb[0]))
                ixmax = np.maximum(int(float(BBGT[i][1])),int( bb[1]))
                iymin = np.minimum(int(float(BBGT[i][2])),int( bb[2]))
                iymax = np.minimum(int(float(BBGT[i][3])),int( bb[3]))
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (float(BBGT[i][1]) - float(BBGT[i][0]) + 1.) *
                       (float(BBGT[i][3]) - float(BBGT[i][2]) + 1.) - inters)
                overlaps = inters / uni
                over_lap.append(overlaps)  
            
            ovmax = np.max(over_lap)
#            jmax = np.argmax(overlaps)
        # 取最大的IoU
        if ovthresh<ovmax :  # 是否大于阈值
            tp[d] = 1.
        else:
            fp[d] = 1.
        if d==0:
            npos=number
    # 计算precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
#    rec = tp / float(npos)
#    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#    ap = voc_ap(rec, prec, use_07_metric)


#    rec = tp / float(npos)
#    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#    ap = voc_ap(rec, prec, use_07_metric)
    return  list(tp),list(fp),npos
#    return  tp,fp,ap


def class_division(rectangles,flie):#对每张图片里面检测出的结果进行分类
    L0=[]
    L1=[]
    L2=[]
    L3=[]
    L4=[]
    L5=[]
    L6=[]
    L7=[]
    L8=[]
    L9=[]
    L=[]
    for i,data in enumerate(rectangles):
        if data['classID']==0:
            data['imageID']=flie
#            data['ranking']=i
            L0.append(data)
        elif data['classID']==1:
            data['imageID']=flie
#            data['ranking']=i
            L1.append(data)
        elif data['classID']==2:
            data['imageID']=flie
#            data['ranking']=i
            L2.append(data)
        elif data['classID']==3:
            data['imageID']=flie
#            data['ranking']=i
            L3.append(data)
        elif data['classID']==4:
            data['imageID']=flie
#            data['ranking']=i
            L4.append(data)
        elif data['classID']==5:
            data['imageID']=flie
#            data['ranking']=i
            L5.append(data)
        elif data['classID']==6:
            data['imageID']=flie
#            data['ranking']=i
            L6.append(data)
        elif data['classID']==7:
            data['imageID']=flie
#            data['ranking']=i
            L7.append(data)
        elif data['classID']==8:
            data['imageID']=flie
#            data['ranking']=i
            L8.append(data)
        elif data['classID']==9:
            data['imageID']=flie
#            data['ranking']=i
            L9.append(data)  
        L.append(L0)
        L.append(L1)
        L.append(L2)
        L.append(L3)
        L.append(L4)
        L.append(L5)
        L.append(L6)
        L.append(L7)
        L.append(L8)
        L.append(L9)
    return L[:10]


#npos_class=[1021857,16505,129262,10229,42963,4296,179,6461,343777,265906]
#def get_map(detected_file,ground_truth,name):
#    for object in detected_file:

#def get_MAP(file):
#    L0=[]
#    L1=[]
#    L2=[]
#    L3=[]
#    L4=[]
#    L5=[]
#    L6=[]
#    L7=[]
#    L8=[]
#    L9=[]
#    for img in file:   
#        a,b,c,d,e,f,g,h,i,j=class_division(img,img_path)
#        for i in a:
#            L0.append(i)
#        for i in b:
#            L1.append(i)
#        for i in c:
#            L2.append(i)
#        for i in d:
#            L3.append(i)
#        for i in e:
#            L4.append(i)
#        for i in f:
#            L5.append(i)
#        for i in g:
#            L6.append(i)
#        for i in h:
#            L7.append(i)
#        for i in i:
#            L8.append(i)
#        for i in j:
#            L9.append(i)
#    class0_ap=get_AP(0.5,L0)    
#    class1_ap=get_AP(0.5,L1)    
#    class2_ap=get_AP(0.5,L2)    
#    class3_ap=get_AP(0.5,L3)    
#    class4_ap=get_AP(0.5,L4)    
#    class5_ap=get_AP(0.5,L5)    
#    class6_ap=get_AP(0.5,L6)    
#    class7_ap=get_AP(0.5,L7)    
#    class8_ap=get_AP(0.5,L8)    
#    class9_ap=get_AP(0.5,L9)
#    MAP=np.sum(class0_ap,class1_ap,class2_ap,class3_ap,class4_ap,class5_ap,class6_ap,class7_ap,class8_ap,class9_ap)/10
#    return MAP
#    
