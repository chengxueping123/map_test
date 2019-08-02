#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:20:00 2019

@author: chengxueping
"""

# -*- coding: utf-8 -*-
import os


#path2 = r'/home/cheng-xp/桌面/detection_2d/input/detection-results/'
path1='/home/chengxueping/Desktop/data/'

path2 = r'/home/chengxueping/Desktop/data_1/'
def file_name(file_dir1,file_dir2):
    jpg_list = []
    txt_list = []
    for root, dirs, files in os.walk(file_dir1):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                jpg_list.append(os.path.splitext(file)[0])
    for root, dirs, files in os.walk(file_dir2):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                txt_list.append(os.path.splitext(file)[0])
    
#            elif os.path.splitext(file)[1] == '.txt':
#                xml_list.append(os.path.splitext(file)[0])

    diff = set(txt_list).difference(set(jpg_list))  # 差集，在a中但不在b中的元素
    print(len(diff))
    for name in diff:
        print("no jpg", name + ".txt")
        os.remove(path2+name+'.txt')

    diff2 = set(jpg_list).difference(set(txt_list))  # 差集，在b中但不在a中的元素
    print(len(diff2))
    for name in diff2:
        print("no txt", name + ".jpg")
        os.remove(path1+name+'.jpg')
    return jpg_list,txt_list

    # 其中os.path.splitext()函数将路径拆分为文件名+扩展名

if __name__ == '__main__':

    a,b=file_name(path1,path2)


 


#inputimagePath = '/home/chengxueping/Desktop/data/'
#inputlabel = '/home/chengxueping/Desktop/data_1/'
#   
#
#delet(inputimagePath,inputlabel)