# Author by zhangyida
# 詹皇明年夺冠
import fileinput
import os,re
# dirname = "tomoset"

path = "tomoset"
filelist = [i for i in os.listdir(path)]
# print(filelist)
# f = open('sep_tomo_stack/datetime.txt', 'r')
# datatime = f.readlines()
newlist=[]
for num,file in enumerate(filelist):
    if file.endswith(".mrc") :
        print (file)
        basename1=file.split('.')
        basename2=basename1[0]
        text= str(num) + '   tomoset/'  + str(file) +  '   ' + str(4.44)+ '   ./corrected_tomos/'+basename2+'_correct.mrc \n'
        print(text)
        newlist.append(text)

f = open("for_predict1.star", 'a')  # 返回一个文件对象
f.write('''
data_

loop_
_rlnIndex #1
_rlnMicrographName #2
_rlnPixelSize #3
_rlnCorrectedTomoName #4
''')
for j in newlist:
    f.write(j)
f.close()
        # file_path = path+file
        # file_path= os.path.join('sep_tomo_stack',file)
        # file_path=file_path.replace('\\', '/')
        # f2=open(file_path)
        # mdoc2=f2.readlines()
        # print(mdoc2)
        # newname=os.path.join(dirname,file)
        # newname=newname.replace('\\', '/')
        # newname =dirname + '/'+file
        # f3=open(newname, 'a')
        # i = 0
        # for line in mdoc2:
        #     if "FilterSlitAndLoss" in line:
        #         line=line.replace(line,line+"%s"%(datatime[i]))
        #         i=i+1
        #     f3.write(line)
