# Author by zhangyida
# 詹皇明年夺冠

import subprocess,sys
import argparse

print(''' \033[1;31m<<description=generate your subtomo.star
 example: python generate_subtomostar.py subtomo 64 64 
python generate_subtomostar.py subtomo CubeSize  CropSize>>\033[0m''')
 
folder_name=sys.argv[1]
CubeSize=sys.argv[2]
CropSize=sys.argv[3]

command = "ls -1 %s > star_list"%(folder_name)

# 使用subprocess.run()执行命令
result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 打印标准输出和标准错误
newlist=[]
f = open("star_list")               # 返回一个文件对象
line = f.readlines()               # 调用文件的 readline()方法

for i,k in enumerate(line):
    j=i+1
    l=k.strip()
    a='%s    subtomo/%s    %s    %s    4.44 \n'%(j,l,CubeSize,CropSize)
    newlist.append(a)
f = open("subtomo.star",'a')               # 返回一个文件对象
f.write('''
data_

loop_
_rlnSubtomoIndex #1
_rlnImageName #2
_rlnCubeSize #3
_rlnCropSize #4
_rlnPixelSize #5
''')
for j in newlist:
    f.write(j)
f.close()