import os
import time
import datetime
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np

import py_sod_metrics
from config import *
from misc import *
from mynet import myNet

torch.manual_seed(2021)
device_ids = [0]
torch.cuda.set_device(device_ids[0])

exp_name = 'myNet_final3'
folder_name = 'myNet_final3'
model_name = '140_besteval_S 790.pth'

results_path = './results'
check_mkdir(results_path)

args = {
    'scale': 352,
    'save_results': False
}
model_to_load = f'ckpt/{exp_name}/{model_name}'

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



to_pil = transforms.ToPILImage()

to_test = OrderedDict([
                       #('CHAMELEON', chameleon_path),
                       #('CAMO', camo_path),
                       #('COD10K', cod10k_path)
                       #('NC4K', nc4k_path)
                       #('low', low_path)
                       ('COD10K-train', cod10ktrain_path)
                       ])

results = OrderedDict()

def main():
    net = myNet().cuda(device_ids[0])
    net.load_state_dict(torch.load(model_to_load))
    print('Load {} succeed!'.format(model_to_load))
    net.eval()

    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():
            time_list = []
            image_path = os.path.join(root, 'image')
            gt_path = os.path.join(root, 'mask')
            if args['save_results']:
                check_mkdir(os.path.join(results_path, folder_name, name))

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
            print(len(img_list))

            WFM = py_sod_metrics.WeightedFmeasure()
            SM = py_sod_metrics.Smeasure()
            EM = py_sod_metrics.Emeasure()
            MAE = py_sod_metrics.MAE()

            for idx, img_name in enumerate(img_list):
                img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')
                gt_img = Image.open(os.path.join(gt_path, img_name + '.png')).convert('L')
                w, h = gt_img.size
                gt = np.array(gt_img)
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])

                start_each = time.time()
                e_g_out = net(img_var)
                e_g_outp = torch.sigmoid(F.interpolate(e_g_out, scale_factor=4, mode='bilinear', align_corners=True))
                time_each = time.time() - start_each
                time_list.append(time_each)

                prediction = np.array(transforms.Resize((h, w))(to_pil(e_g_outp.data.squeeze(0).cpu())))
                assert prediction.shape == gt.shape, (prediction.shape, gt.shape)
                assert prediction.dtype == gt.dtype == np.uint8, (prediction.dtype, gt.dtype)

                SM.step(pred=prediction, gt=gt)
                WFM.step(pred=prediction, gt=gt)
                EM.step(pred=prediction, gt=gt)
                MAE.step(pred=prediction, gt=gt)

                if args['save_results']:
                    Image.fromarray(prediction).convert('L').save(os.path.join(results_path, folder_name, name, img_name + '.png'))

            print(f'{exp_name}')
            wfm = WFM.get_results()["wfm"]
            sm = SM.get_results()["sm"]
            em = EM.get_results()["em"]['adp']
            mae = MAE.get_results()["mae"]
            print(f"wfm:{wfm} sm:{sm} em:{em} mae:{mae}")
            print(f"{name}'s average Time Is : {np.mean(time_list):.3f} s")
            print(f"{name}'s average Time Is : {1/np.mean(time_list):.1f} fps")

        end = time.time()
        print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))

if __name__ == '__main__':
    main()
