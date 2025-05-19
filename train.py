import datetime
import time
import os

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

import joint_transforms
from config import cod_training_root
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from mynet import myNet as myNet_final4
import loss
from eval_intrain import eval_in_train

cudnn.benchmark = True
torch.manual_seed(2021)

device_ids = [0]  # 推荐默认0，用户自行修改

ckpt_path = './ckpt'
exp_name = 'myNet_final4'

args = {
    'epoch_num': 150,
    'train_batch_size': 8,
    'last_epoch': 0,
    'lr': 1e-4,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 352,
    'save_point': [80, 100, 120],
    'poly_train': True,
    'optimizer': 'SGD',
    'poly_80': True,
    'early': True,
    'early_min': 70,
    'final_lr': 1e-5,
    'SM_only': True
}

poly_max_num = 80
test_each_epoch = 1
overfit_count_max = 8

check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

min_test_loss = 10000
max_eval_score = 0
overfit_count = 0

# 数据预处理
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomVerticallyFlip(),
    joint_transforms.randomCrop(),
    joint_transforms.randomRotation(),
    joint_transforms.randomPeper(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# 数据加载
train_set = ImageFolder(cod_training_root, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)

total_epoch = args['epoch_num'] * len(train_loader)
total_epoch_max80 = poly_max_num * len(train_loader)

# 损失函数
structure_loss = loss.structure_loss().cuda(device_ids[0])
bce_loss = nn.BCEWithLogitsLoss().cuda(device_ids[0])
iou_loss = loss.IOU().cuda(device_ids[0])
dice_loss = loss.DiceLoss().cuda(device_ids[0])
ssim_loss = loss.SSIMLoss().cuda(device_ids[0])
object_loss = loss.objectLoss().cuda(device_ids[0])

def final_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    return bce_out + iou_out

def main(model_selected):
    print(args)
    print(exp_name)
    net = model_selected().cuda(device_ids[0]).train()
    optimizer = optim.SGD(
        net.parameters(), lr=args['lr'],
        momentum=args['momentum'], weight_decay=args['weight_decay']
    )
    if args['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            net.parameters(), lr=args['lr'], weight_decay=args['weight_decay']
        )
    if args['snapshot']:
        print(f"Training resumes from '{args['snapshot']}'")
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
    net = nn.DataParallel(net, device_ids=device_ids)
    print(f"Using {len(device_ids)} GPU(s) to Train.")
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()

def train(net, optimizer):
    curr_iter = 1 + args['last_epoch'] * len(train_loader)
    start_time = time.time()
    global max_eval_score, overfit_count

    for epoch in range(args['last_epoch'] + 1, args['epoch_num']):
        loss_record = AvgMeter()
        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            # Poly学习率调度
            if args['poly_train']:
                if args['poly_80']:
                    prograss = float(min(curr_iter, total_epoch_max80)) / float(total_epoch_max80)
                    base_lr = args['lr'] * (1 - prograss) ** args['lr_decay']
                    if prograss < 0.9999:
                        if epoch < 5:
                            optimizer.param_groups[0]['lr'] = (1.5 * base_lr + 0.00005) * ((epoch + 1) / 6)
                        else:
                            optimizer.param_groups[0]['lr'] = 1.5 * base_lr + 0.00005
                    else:
                        optimizer.param_groups[0]['lr'] = 1.5 * base_lr + args['final_lr']
                else:
                    base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                    optimizer.param_groups[0]['lr'] = 1.5 * base_lr + 0.000025
            else:
                optimizer.param_groups[0]['lr'] = 0.00005

            inputs, labels = data
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])
            optimizer.zero_grad()
            e_g_out = net(inputs)
            e_g_outp = F.interpolate(e_g_out, scale_factor=4, mode='bilinear', align_corners=True)
            loss = final_loss(e_g_outp, labels)
            loss.backward()
            optimizer.step()
            loss_record.update(loss.data, inputs.size(0))

            if curr_iter % 10 == 0:
                writer.add_scalar('loss', loss, curr_iter)

            log = f'[{epoch:3d}], [{curr_iter:6d}], [{optimizer.param_groups[0]["lr"]:.6f}], [{loss_record.avg:.5f}]'
            train_iterator.set_description(log)
            curr_iter += 1
        open(log_path, 'a').write(log + '\n')

        # 保存模型
        if epoch in args['save_point']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, f'{epoch}.pth'))
            net.cuda(device_ids[0])

        # 早停与验证
        if args['early'] and epoch >= args['early_min'] and epoch % test_each_epoch == 0:
            net.eval()
            with torch.no_grad():
                eval_score = eval_in_train(net)
                final_score = eval_score['sm'] if args['SM_only'] else eval_score['sm'] + eval_score['wfm']
                open(log_path, 'a').write(f"wfm:{eval_score['wfm']} sm:{eval_score['sm']}\n")
                if max_eval_score < final_score:
                    max_eval_score = final_score
                    overfit_count = max(0, overfit_count - 3)
                    net.cpu()
                    save_name = f"{epoch}_besteval_S{int(max_eval_score * 10000)}.pth" if args['SM_only'] else f"{epoch}_besteval_{int(max_eval_score * 1000)}.pth"
                    torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, save_name))
                    net.cuda(device_ids[0])
                else:
                    overfit_count += 1
                    if overfit_count >= overfit_count_max:
                        net.cpu()
                        torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, f'{epoch}.pth'))
                        print("Total Training Time:", str(datetime.timedelta(seconds=int(time.time() - start_time))))
                        print(exp_name)
                        print("Early stopping triggered!")
                        return
            net.train()

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, f'{epoch}.pth'))
            print("Total Training Time:", str(datetime.timedelta(seconds=int(time.time() - start_time))))
            print(exp_name)
            print("Optimization Done!")
            return

if __name__ == '__main__':
    main(myNet_final4)
