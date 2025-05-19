"""
 @Time    : 2021/7/6 09:46
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : config.py
 @Function: Configuration
 
"""
import os

backbone_path = './resnet50-19c8e357.pth'

datasets_root = '/home/aaa/LYW/dataset/DDANet'

cod_training_root = os.path.join(datasets_root, 'train')

chameleon_path = '/home/aaa/LYW/dataset/CODDataset/COD-TE/CHAMELEON-TE'
camo_path = os.path.join(datasets_root, 'test/CAMO-TE')
cod10k_path = os.path.join(datasets_root, 'test/COD10K-TE')
nc4k_path = '/home/aaa/LYW/dataset/CODDataset/COD-TE/NC4K'
#水下低质量图片
low_path = os.path.join(datasets_root, 'test/low')

cod10ktrain_path= os.path.join(datasets_root, 'train')