import torch
import argparse
import yaml
import math
import os
import time
import sys
module_dir = "xxx/code/U3M"
if module_dir not in sys.path:
    sys.path.append(module_dir)
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from math import ceil
import numpy as np
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

@torch.no_grad()
def sliding_predict(model, image, num_classes, flip=True):
    image_size = image[0].shape
    tile_size = (int(ceil(image_size[2]*1)), int(ceil(image_size[3]*1)))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = torch.zeros((num_classes, image_size[2], image_size[3]), device=torch.device('cuda'))
    count_predictions = torch.zeros((image_size[2], image_size[3]), device=torch.device('cuda'))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = [modal[:, :, y_min:y_max, x_min:x_max] for modal in image]
            padded_img = [pad_image(modal, tile_size) for modal in img]
            tile_counter += 1
            padded_prediction,_,_ = model(padded_img)
            if flip:
                fliped_img = [padded_modal.flip(-1) for padded_modal in padded_img]
                fliped_predictions,_,_ = model(fliped_img)
                padded_prediction += fliped_predictions.flip(-1)
            predictions = padded_prediction[:, :, :img[0].shape[2], :img[0].shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.squeeze(0)

    return total_predictions.unsqueeze(0)

def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette

@torch.no_grad()
def evaluate(model, dataloader, device, VIS_Saving, loss_fn=None):
    print('Evaluating...')
    model.eval()
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    sliding = False
    test_loss = 0.0
    iter = 0
    i = 0
    for images, labels in tqdm(dataloader):
        images = [x.to(device) for x in images]
        labels = labels.to(device)
        
        if sliding:
            # preds = sliding_predict(model, images, num_classes=n_classes).softmax(dim=1)
            preds,_,_ = sliding_predict(model, images, num_classes=n_classes)
        else:
            # preds = model(images).softmax(dim=1)
            preds,_,_ = model(images)
        
        # 为每个类别定义一个颜色
        palette = get_palette()
        image_path = f'{VIS_Saving}/semantic_segmentation_visualization_{i}.png'
        preds = preds.cpu().numpy().squeeze()
        img = np.zeros((preds.shape[1], preds.shape[2], 3), dtype=np.uint8)

        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[preds == cid] = palette[cid]

        img = Image.fromarray(np.uint8(img))
        img.save(pth+ image_name[i])
        plt.savefig(image_path)
        i += 1
    return 0


def main(cfg):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    VIS_Saving = eval_cfg['VIS_SAVE_DIR']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    # cases = ['cloud', 'fog', 'night', 'rain', 'sun']
    # cases = ['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    cases = [None] # all
    
    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): 
        raise FileNotFoundError
    print(f"Evaluating {model_path}...")

    exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    eval_path = os.path.join(os.path.dirname(eval_cfg['MODEL_PATH']), 'eval_{}.txt'.format(exp_time))

    for case in cases:
        dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, cfg['DATASET']['MODALS'], case)
        # --- test set
        # dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'test', transform, cfg['DATASET']['MODALS'], case)

        model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes, cfg['DATASET']['MODALS'])
        
        msg = model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        print(msg)
        model = model.to(device)
        sampler_val = None
        dataloader = DataLoader(dataset, batch_size=eval_cfg['BATCH_SIZE_VIS'], num_workers=eval_cfg['BATCH_SIZE'], pin_memory=False, sampler=sampler_val)
        
        ok = evaluate(model, dataloader, device,VIS_Saving)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='xxx/code/U3M/configs/fmb_rgbt.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    # gpu = setup_ddp()
    # main(cfg, gpu)
    main(cfg)