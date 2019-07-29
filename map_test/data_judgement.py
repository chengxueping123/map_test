import os
import re

def input_detection_results(image_name,bbox,confidence,class_name):#生成input中的detection-results文件
    path=os.getcwd()
    with open(os.path.join(path,'input','detection-results',image_name.replace('jpg','txt')) ,'w') as f:
        f.write(class_name+' ')
        f.write(str(confidence)+' ')
        f.write(str(bbox[0])+' ')
        f.write(str(bbox[1])+' ')
        f.write(str(bbox[2])+' ')
        f.write(str(bbox[3]))

#input_detection_results('2007_000027.jpg',[12,56,65,23],0.45,'cat')
def input_ground_truth(txt_file,txt_name):#对标签数据进行处理
    path=os.getcwd()
    L1=[]
    L2=[]
    with open(txt_file, 'r') as file1:
        for line in file1:
            L1.append(line)
        for i in L1:
           L2.append(i.split(' ')[0])
           L2.append(i.split(' ')[4])
           L2.append(i.split(' ')[5])
           L2.append(i.split(' ')[6])
           L2.append(i.split(' ')[7])
    with open(os.path.join(path,'input','ground-truth',txt_name),'w') as file2:
        for i,j in enumerate(L2):
            file2.write(j+' ')
            if (i+1)%5==0:
                file2.write('\n')
           
#input_ground_truth('/home/chengxueping/Desktop/map_test/23ef1ac8-2c4b1a6d.txt','23ef1ac8-2c4b1a6d.txt')     


