"""
Description:
Author: Xiongjun Guan
Date: 2025-04-25 15:16:19
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-04-25 15:16:37

Copyright (C) 2025 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import datetime
import logging
import os
import os.path as osp
import random
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from data_loader import get_dataloader_train, get_dataloader_valid
from loss import CompareAlignLoss
from models.JIPNet import JIPNet


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(
    model,
    train_dataloader,
    valid_dataloader,
    device,
    cfg,
    save_dir=None,
    save_checkpoint=15,
):
    # -------------- init settings-------------- #
    lr = cfg["train_cfg"]["lr"]
    end_lr = cfg["train_cfg"]["end_lr"]
    optim = cfg["train_cfg"]["optimizer"]
    scheduler_type = cfg["train_cfg"]["scheduler_type"]
    num_epoch = cfg["train_cfg"]["epochs"]

    if valid_dataloader is None:
        valid = False
    else:
        valid = True

    # -------------- some global functions -------------- #
    criterion = CompareAlignLoss()

    # -------------- select optimizer -------------- #
    if optim == "sgd":
        optimizer = torch.optim.SGD(
            (param for param in model.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=0,
        )

    elif optim == "adam":
        optimizer = torch.optim.Adam(
            (param for param in model.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=1e-3,
        )

    elif optim == "adamW":
        optimizer = torch.optim.AdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=1e-2,
        )

    # -------------- select scheduler -------------- #
    if scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=np.round(num_epoch), eta_min=end_lr
        )

    elif scheduler_type == "StepLR":
        scheduler = StepLR(optimizer, 15, 0.1)

    # -------------- train & valid -------------- #
    for epoch in range(num_epoch):
        # train phase
        model.train()
        train_losses = {
            "total": 0,
            "focal": 0,
            "Lr": 0,
        }
        logging.info(
            "epoch: {}, lr:{:.8f}".format(
                epoch, optimizer.state_dict()["param_groups"][0]["lr"]
            )
        )
        pbar = tqdm(train_dataloader, desc=f"epoch:{epoch}, train")
        for input1, input2, align_target, target in pbar:
            input1 = input1.float().to(device)
            input2 = input2.float().to(device)
            align_target = align_target.float().to(device)
            target = target[:, 0:1].float().to(device)

            cla_pred, align_pred = model([input1, input2])

            loss, items = criterion(cla_pred, target, align_pred, align_target)
            klist = items.keys()
            for k in klist:
                train_losses[k] += items[k] / len(train_dataloader)
            train_losses["total"] += loss.item() / len(train_dataloader)
            pbar.set_postfix(**{"loss": loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del loss

        pbar.close()

        klist = train_losses.keys()
        logging_info = "\tTRAIN: ".format(epoch)
        for k in klist:
            logging_info = logging_info + "{}:{:.4f}, ".format(k, train_losses[k])
        logging.info(logging_info)

        scheduler.step()

        if save_dir is not None and epoch > save_checkpoint:
            if hasattr(model, "module"):
                torch.save(
                    model.module.state_dict(),
                    osp.join(save_dir, f"epoch_{epoch}.pth"),
                )
            else:
                torch.save(
                    model.state_dict(),
                    osp.join(save_dir, f"epoch_{epoch}.pth"),
                )

        # valid phase
        if valid is False:
            continue
        model.eval()
        with torch.no_grad():
            valid_losses = {
                "total": 0,
                "focal": 0,
                "Lr": 0,
            }
            pbar = tqdm(valid_dataloader, desc=f"epoch:{epoch}, val")

            for input1, input2, align_target, target in pbar:
                input1 = input1.float().to(device)
                input2 = input2.float().to(device)
                align_target = align_target.float().to(device)
                target = target[:, 0:1].float().to(device)

                cla_pred, align_pred = model([input1, input2])

                loss, items = criterion(cla_pred, target, align_pred, align_target)
                klist = items.keys()
                for k in klist:
                    valid_losses[k] += items[k] / len(valid_dataloader)
                valid_losses["total"] += loss.item() / len(valid_dataloader)
                pbar.set_postfix(**{"loss": loss.item()})

                del loss

            pbar.close()

            klist = valid_losses.keys()
            logging_info = "\tVALID: ".format(epoch)
            for k in klist:
                logging_info = logging_info + "{}:{:.4f}, ".format(k, valid_losses[k])
            logging.info(logging_info)

    return


if __name__ == "__main__":
    set_seed(seed=7)

    current_path = os.path.abspath(__file__)
    config_path = osp.join(osp.dirname(current_path), "configs", "JIPNet.yaml")
    with open(config_path, "r") as config:
        cfg = yaml.safe_load(config)

    # set save dir
    save_basedir = osp.join(cfg["save_cfg"]["save_basedir"], cfg["model_cfg"]["model"])
    if cfg["save_cfg"]["save_title"] == "time":
        now = datetime.datetime.now()
        save_dir = osp.join(save_basedir, now.strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        save_dir = osp.join(save_basedir, cfg["save_cfg"]["save_title"])

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    shutil.copy(config_path, osp.join(save_dir, "config.yaml"))

    # set database
    train_info_path = cfg["db_cfg"]["train_info_path"]
    valid_info_path = cfg["db_cfg"]["valid_info_path"]

    train_info = np.load(train_info_path, allow_pickle=True).item()
    valid_info = np.load(valid_info_path, allow_pickle=True).item()

    # logging losses
    logging_path = osp.join(save_dir, "info.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        filename=logging_path,
        filemode="w",
    )

    train_loader = get_dataloader_train(
        info_lst=train_info["info_lst"],
        patch_size=cfg["model_cfg"]["input_size"],
        batch_size=cfg["train_cfg"]["batch_size"],
        shuffle=True,
    )

    valid_loader = get_dataloader_valid(
        info_lst=valid_info["info_lst"],
        patch_size=cfg["model_cfg"]["input_size"],
        batch_size=cfg["train_cfg"]["batch_size"],
    )

    # set models
    model = JIPNet(
        input_size=cfg["model_cfg"]["input_size"],
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
        dec_local_num=cfg["model_cfg"]["dec_local_num"],
        encoder_pretrain_pth=cfg["pretrain_cfg"]["encoder_pth"],
    )

    logging.info("Model: {}".format(cfg["model_cfg"]["model"]))

    device = torch.device(
        "cuda:{}".format(str(cfg["train_cfg"]["cuda_ids"][0]))
        if torch.cuda.is_available()
        else "cpu"
    )

    model = torch.nn.DataParallel(
        model,
        device_ids=cfg["train_cfg"]["cuda_ids"],
        output_device=cfg["train_cfg"]["cuda_ids"][0],
    ).to(device)

    logging.info("******** begin training ********")
    train(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        device=device,
        cfg=cfg,
        save_dir=save_dir,
        save_checkpoint=cfg["train_cfg"]["epochs"] - 6,
    )
