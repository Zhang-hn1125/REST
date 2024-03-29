# Author by zhangyida
# 詹皇明年夺冠
import logging
from REST.preprocessing.cubes import prepare_cubes
from REST.preprocessing.img_processing import normalize
from REST.preprocessing.prepare import prepare_first_iter,get_cubes_list,get_noise_level
from REST.util.dict2attr import save_args_json,load_args_from_json
import glob
import mrcfile
import numpy as np
import glob
import os
import sys
import shutil
from REST.util.metadata import MetaData, Item, Label
from REST.util.utils import mkfolder
from REST.util.dict2attr import Arg,check_parse,idx2list

import logging
import os, sys, traceback
from REST.util.dict2attr import Arg,check_parse,idx2list

from REST.util.metadata import MetaData,Label,Item
import os
import sys
import logging
import sys
import mrcfile
from REST.preprocessing.cubes import create_cube_seeds,crop_cubes,DataCubes
from REST.preprocessing.img_processing import normalize
from REST.preprocessing.simulate import apply_wedge1 as  apply_wedge
from multiprocessing import Pool
import numpy as np
from functools import partial
from REST.util.rotations import rotation_list
import argparse




parser2 = argparse.ArgumentParser(description='refine your model using simulated data')

parser2.add_argument('--initial_model', default=None,type=str, help=' Name of the pre-trained model. Leave empty to train a new one.')
parser2.add_argument('--new_model_name', default=None,type=str, help=' Name of the model you want to rename. Leave empty to use results/model_iter00.h5.')
parser2.add_argument('--epoch', default=2,type=int, help=' The epoch for training. Maybe you need find a suitable parameters to train your data')
parser2.add_argument('--step', default=1500,type=int, help=' The step of each epoch. Maybe you need find a suitable parameters to train your data')
parser2.add_argument('--num_gpu', default=1,type=int, help=' The number of gpu you want to use')



args_zhn = parser2.parse_args()

print(''' \033[1;34m<<Refine your model using simulated data start! Your new model will named %s>>\033[0m'''%(args_zhn.new_model_name))

dic={ 'gpuID': None,
 'iterations': None,
 'data_dir': None,
 'pretrained_model': None,
 'log_level': None,
 'result_dir': 'results',
 'preprocessing_ncpus': 16,
 'continue_from': None,
 'epochs': 10,
 'batch_size': None,
 'steps_per_epoch': None,
 'noise_level': None,
 'noise_start_iter': None,
 'noise_mode': None,
 'noise_dir': None,
 'learning_rate': None,
 'drop_out': 0.3,
 'convs_per_depth': 3,
 'kernel': (3, 3, 3),
 'pool': None,
 'unet_depth': 3,
 'filter_base': None,
 'batch_normalization': False,
 'normalize_percentile': True,
'subtomo_star':'subtomo.star'}
args=Arg(dic)
md = MetaData()
md.read(args.subtomo_star)
# args.crop_size = 18
# args.cube_size = 18
# args.predict_cropsize = 18
# num_noise_volume = 1000
# args.residual = False
args.crop_size = md._data[0].rlnCropSize
args.cube_size = md._data[0].rlnCubeSize
args.predict_cropsize = args.crop_size
num_noise_volume = 3000
args.residual = False
args.gpuID = 0
if args.data_dir is None:
    args.data_dir = args.result_dir + '/data'
if args.iterations is None:
    args.iterations = 30
args.ngpus = 1
if args.result_dir is None:
    args.result_dir = 'results'
if args.batch_size is None:
    args.batch_size = max(4, 2 * args.ngpus)
args.predict_batch_size = args.batch_size
if args.filter_base is None:
    if md._data[0].rlnPixelSize >15:
        args.filter_base = 32
    else:
        args.filter_base = 64
if args.steps_per_epoch is None:
    args.steps_per_epoch = min(int(len(md) * 6/args.batch_size) , 200)
if args.learning_rate is None:
    args.learning_rate = 0.0004
if args.noise_level is None:
    args.noise_level = (0.05,0.10,0.15,0.20)
if args.noise_start_iter is None:
    args.noise_start_iter = (11,16,21,26)
if args.noise_mode is None:
    args.noise_mode = 'noFilter'
if args.noise_dir is None:
    args.noise_dir = args.result_dir +'/training_noise'
if args.log_level is None:
    args.log_level = "info"

if len(md) <=0:
    logging.error("Subtomo list is empty!")
    sys.exit(0)
args.mrc_list = []
for i,it in enumerate(md):
    if "rlnImageName" in md.getLabels():
        args.mrc_list.append(it.rlnImageName)
import subprocess
sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out_str = sp.communicate()
out_str = out_str[0].decode('utf-8')
if 'CUDA Version' not in out_str:
    raise RuntimeError('No GPU detected, Please check your CUDA version and installation')
import subprocess
sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out_str = sp.communicate()
out_str = out_str[0].decode('utf-8')
if 'CUDA Version' not in out_str:
    raise RuntimeError('No GPU detected, Please check your CUDA version and installation')
from REST.training.predict import predict
from REST.training.train import prepare_first_model, train_data
settings=args
num_noise_volume = len(settings.mrc_list)
print('number of train set:',num_noise_volume)
from REST.preprocessing.cubes import create_cube_seeds,crop_cubes,DataCubes
from REST.preprocessing.img_processing import normalize
from REST.preprocessing.simulate import apply_wedge1 as  apply_wedge
from multiprocessing import Pool
import numpy as np
from functools import partial
num_iter=1
logging.info("Start Iteration{}!".format(num_iter))
# pretrained_model case
if num_iter ==1:
    args = prepare_first_model(args)
else:
    args.init_model = '{}/model_iter{:0>2d}.h5'.format(args.result_dir,args.iter_count-1)
noise_level_series = get_noise_level(args.noise_level,args.noise_start_iter,args.iterations)
args.noise_level_current =  noise_level_series[num_iter]
args.noise_level = (0.05,0.10,0.15,0.20)

args.noise_start_iter = (11,16,21,26)

args.noise_mode = 'noFilter'

args.noise_dir = args.result_dir +'/training_noise'

args.log_level = "info"
# all_path_x = os.listdir(settings.data_dir+'/train_x')
# num_test = int(len(all_path_x) * 0.1)
# num_test = num_test - num_test%settings.ngpus + settings.ngpus
# all_path_y = ['y_'+i.split('_')[1] for i in all_path_x ]
# ind = np.random.permutation(len(all_path_x))[0:num_test]
# for i in ind:
#     os.rename('{}/train_x/{}'.format(settings.data_dir, all_path_x[i]), '{}/test_x/{}'.format(settings.data_dir, all_path_x[i]) )
#     os.rename('{}/train_y/{}'.format(settings.data_dir, all_path_y[i]), '{}/test_y/{}'.format(settings.data_dir, all_path_y[i]) )
#     #os.rename('data/train_y/'+all_path_y[i], 'data/test_y/'+all_path_y[i])
from REST.training.train import prepare_first_model, train_data,train3D_continue
if args_zhn.initial_model is not None:
    shutil.copyfile(args_zhn.initial_model, 'results/model_init.h5')
epoch=args_zhn.epoch
num_gpu=args_zhn.num_gpu
step=args_zhn.step
history = train3D_continue('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,0),
                                        settings.init_model,
                                        data_dir = settings.data_dir,
                                        result_folder = settings.result_dir,
                                        epochs=epoch,
                                        steps_per_epoch=step,
                                        batch_size=batch_size1,
                                        lr = 0.0001,
                                        n_gpus=num_gpu)
if args_zhn.new_model_name is not None:
    shutil.copyfile( 'results/model_iter00.h5',args_zhn.new_model_name)

print(''' \033[1;34m<<Training completed>>\033[0m''')

