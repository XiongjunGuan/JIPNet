'''
Description: 
Author: Xiongjun Guan
Date: 2024-10-31 17:01:48
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-11-01 16:34:54

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

import os
import os.path as osp
import random
from glob import glob

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.JIPNet import JIPNet
from models.utils import AffinePatch


def norm_pred(align_pred, eps=1e-6):
    pred_b1 = align_pred[:, 0]
    pred_b2 = align_pred[:, 1]
    cosT = np.divide(pred_b1,
                     eps + np.sqrt(np.square(pred_b1) + np.square(pred_b2)))
    sinT = np.divide(pred_b2,
                     eps + np.sqrt(np.square(pred_b1) + np.square(pred_b2)))
    pred_norm = np.concatenate(
        [
            cosT[:, None], sinT[:, None], align_pred[:, 2][:, None],
            align_pred[:, 3][:, None]
        ],
        axis=1,
    )
    return pred_norm


def set_seed(seed=7):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    set_seed(7)
    cuda_ids = [0]
    batch_size = 1

    pth_dir = "./ckpts/best.pth"
    config_path = "./ckpts/config.yaml"
    with open(config_path, "r") as config:
        cfg = yaml.safe_load(config)

    patch_size = 160
    pad_width = 100  # for visualization

    data_dir = "./examples/data/"
    ftitle_lst = ["0", "1", "2", "3"]
    save_dir = "./examples/results/"
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda:{}".format(str(cuda_ids[0])) if torch.cuda.
                          is_available() else "cpu")

    model = JIPNet(input_size=patch_size,
                   img_channel=cfg["model_cfg"]["img_channel"],
                   num_classes=cfg["model_cfg"]["num_classes"],
                   width=cfg["model_cfg"]["width"],
                   enc_blk_nums=cfg["model_cfg"]["enc_blk_nums"],
                   dw_expand=cfg["model_cfg"]["dw_expand"],
                   ffn_expand=cfg["model_cfg"]["ffn_expand"],
                   mid_blk_nums=cfg["model_cfg"]["mid_blk_nums"],
                   mid_blk_strides=cfg["model_cfg"]["mid_blk_strides"],
                   mid_embed_dims=cfg["model_cfg"]["mid_embed_dims"],
                   dec_hidden_dim=cfg["model_cfg"]["dec_hidden_dim"],
                   dec_nhead=cfg["model_cfg"]["dec_nhead"],
                   dec_local_num=cfg["model_cfg"]["dec_local_num"])

    model.load_state_dict(
        torch.load(pth_dir, map_location=f'cuda:{cuda_ids[0]}'))

    model = torch.nn.DataParallel(
        model,
        device_ids=cuda_ids,
        output_device=cuda_ids[0],
    ).to(device)

    affine_func = AffinePatch()

    model.eval()
    with torch.no_grad():
        for ftitle in tqdm(ftitle_lst):
            img1 = cv2.imread(osp.join(data_dir, f"{ftitle}_1.png"), 0)
            img2 = cv2.imread(osp.join(data_dir, f"{ftitle}_2.png"), 0)

            input1 = ((255.0 - img1) / 255.0)[np.newaxis, np.newaxis, :, :]
            input2 = ((255.0 - img2) / 255.0)[np.newaxis, np.newaxis, :, :]

            input1 = torch.tensor(input1).float().to(device)
            input2 = torch.tensor(input2).float().to(device)

            cla_pred, align_pred = model([input1, input2])

            y = cla_pred.squeeze().cpu().numpy()
            align_pred = align_pred.squeeze().cpu().numpy()

            align_pred = norm_pred(align_pred[None, :])

            # visualization
            img2 = torch.tensor(np.float32(img2)[None, None, :, :] / 255.0)
            img2 = affine_func(img2,
                               torch.tensor(align_pred),
                               pad_width=pad_width).squeeze().numpy()
            img2 = np.clip(img2 * 255, 0, 255)
            img1 = np.pad(img1,
                          [[pad_width, pad_width], [pad_width, pad_width]],
                          mode='constant',
                          constant_values=0)
            img = (img1 * 1.0 + img2 * 1.0) * 0.5
            img = np.clip(img, 0, 255)
            cv2.imwrite(osp.join(save_dir, f"{ftitle}.png"), img)

            save_path = osp.join(save_dir, f"{ftitle}.txt")
            with open(save_path, 'w') as file:
                # 将文本写入文件
                file.write("1-2 matching probability:\n")

                file.write("{:.2f}\n\n".format(y))
                file.write("1-2 pose alignment (cos, sin, tx, ty):\n")
                for number in align_pred[0, :]:
                    file.write("{:.2f}, ".format(number))
                file.write("\n")
