import config


import data_loader
import doy
import paths
import torch
import utils
from doy import loop

import sys


import torch.nn.functional as F


from collections import deque
from functools import partial
from itertools import chain


import env_utils


import torch.nn as nn
import torch.nn.functional as F
from doy import PiecewiseLinearSchedule as PLS
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from utils import create_decoder
import sys
import datetime


import wandb

torch.backends.cudnn.benchmark = True   #

device_copy = config.DEVICE


cfg = config.get()
current_datetime = datetime.datetime.now()
formatted_datetime_1 = current_datetime.strftime("%Y%m%d")
formatted_datetime_2 = current_datetime.strftime("%H%M%S")
cfg.exp_name = formatted_datetime_1+'-'+formatted_datetime_2   + cfg.env_name

def offline_pretraining():
    doy.print("[bold green]BCV-LR: offline pretraining over videos:")
    config.print_cfg(cfg)

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    cfg.stage_exp_name = 'offline_pretraining' #+ formatted_datetime

    run, logger = config.wandb_init("BCV-LR-offline-continuous", config.get_wandb_cfg(cfg),wandb_enabled = True)

    idm, wm = utils.create_dynamics_models(cfg.model)
    wm.conv = idm.conv    #share the conv encoder

    train_data, test_data = data_loader.load(cfg.env_name)
    train_iter = train_data.get_iter(cfg.offline_pt.bs)
    test_iter = test_data.get_iter(128)


    opt, lr_sched = doy.LRScheduler.make(
    all=(
        doy.PiecewiseLinearSchedule(
            [0, cfg.offline_pt.steps + 1],
            [cfg.offline_pt.lr,  cfg.offline_pt.lr],
        ),
        [wm, idm],
    ),
    )





    ######## offline: self-supervised latent feature pre-training
    step_repre_max = 50000
    train_encoder_iter = train_data.get_iter(512)
    contrast_record = []
    
    for step_repre in loop(step_repre_max, desc="[green bold](offline) latent feature pre-training"):
        idm.train()
        batch = next(train_encoder_iter)
        contrastive_loss = idm.update_repre(batch['obs'])

        contrast_record.append(contrastive_loss.item())

    del(train_encoder_iter)

    ########




    def train_step():
        idm.train()
        wm.train()

        lr_sched.step(step)

        batch = next(train_iter)
        
        
        vq_loss, vq_perp = idm.label(batch)
        wm_loss = wm.label(batch)
        loss = wm_loss + vq_loss

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([*idm.parameters(), *wm.parameters()], 2)
        opt.step()

        logger(
            step,
            wm_loss=wm_loss,
            global_step=step,
            vq_perp=vq_perp,
            vq_loss=vq_loss,
            grad_norm=grad_norm,
            **lr_sched.get_state(),
        )

        if step<step_repre_max:
            logger(step, contrastive_loss = contrast_record[step])


    def test_step():
        idm.eval()  # disables idm.vq ema update
        wm.eval()

        # evaluate IDM + FDM generalization on (action-free) test data
        batch = next(test_iter)
        idm.label(batch)
        wm_loss = wm.label(batch)

        # train latent -> true action decoder and evaluate its predictiveness
        _, eval_metrics = utils.eval_latent_repr(test_data, idm, cfg.env_name)

        logger(step, wm_loss_test=wm_loss, global_step=step, **eval_metrics)


    


    if cfg.offline_pt.steps == 0:
        print('------- no la unsupervised pt -------')
        torch.save(
                dict(
                    **doy.get_state_dicts(wm=wm, idm=idm, opt=opt),
                    step=0,
                    cfg=cfg,
                    logger=logger,
                ),
                paths.get_models_path(cfg.exp_name),
            )
    else:
        for step in loop(cfg.offline_pt.steps + 1, desc="[green bold](offline) latent action pre-training"):
            train_step()

            if step % 500 == 0:
                test_step()

            if step > 0 and (step % 5_000 == 0 or step == cfg.offline_pt.steps):
                torch.save(
                    dict(
                        **doy.get_state_dicts(wm=wm, idm=idm, opt=opt),
                        step=step,
                        cfg=cfg,
                        logger=logger,
                    ),
                    paths.get_models_path(cfg.exp_name),
                )

    


def optional_pre_imitation():   
    # let the latent policy fully imitate the pre-trained latent actions before online interactions
    # 20k steps seems enough, 60k steps in paper

    state_dicts = torch.load(paths.get_models_path(cfg.exp_name))
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%H%M%S")
    cfg.stage_exp_name = 'optinal-pre-imitation-continuous' + formatted_datetime
    doy.print("[bold green] optional pre-imitation before online stage:")
    config.print_cfg(cfg)


    idm, _ = utils.create_dynamics_models(cfg.model, state_dicts=state_dicts)
    idm.eval()

    policy = utils.create_policy(cfg.model, cfg.model.la_dim)

    policy.conv.load_state_dict(idm.conv.state_dict())   #share encoder


    opt, lr_sched = doy.LRScheduler.make(
        policy=(
            doy.PiecewiseLinearSchedule(
                [0, 1000, cfg.latent_policy_imitaiton.steps + 1], [ cfg.latent_policy_imitaiton.lr, cfg.latent_policy_imitaiton.lr, cfg.latent_policy_imitaiton.lr]
            ),
            [policy],
        ),
    )

    train_data, test_data = data_loader.load(cfg.env_name)
    train_iter = train_data.get_iter(cfg.latent_policy_imitaiton.bs)
    test_iter = test_data.get_iter(128)

    _, eval_metrics = utils.eval_latent_repr(train_data, idm, cfg.env_name)
    doy.log(f"Decoder metrics sanity check: {eval_metrics}")

    run, logger = config.wandb_init("optional pre-imitation for BCV-LR continuous", config.get_wandb_cfg(cfg))


    for step in loop(
        cfg.latent_policy_imitaiton.steps + 1, desc="[green bold]optional pre-imitation before online stage"
    ):
        lr_sched.step(step)

        policy.train()
        batch = next(train_iter)
        idm.label(batch)

    
        preds = policy(batch["obs"][:, -2])  # the -2 selects last the pre-transition ob
        loss = F.mse_loss(preds, batch["la"])

        opt.zero_grad()
        loss.backward()
        opt.step()

        logger(
            step=step,
            loss=loss,
            **lr_sched.get_state(),
        )

        if step % 200 == 0:
            policy.eval()
            test_batch = next(test_iter)
            idm.label(test_batch)
            test_loss = F.mse_loss(policy(test_batch["obs"][:, -2]), test_batch["la"])
            logger(step=step, test_loss=test_loss)

    torch.save(
        dict(policy=doy.state_dict_orig(policy), cfg=cfg, logger=logger),
        paths.get_latent_policy_path(cfg.exp_name),
    )





print('BCV-LR offline stage starting')

offline_pretraining()

assert device_copy == config.DEVICE

wandb.finish()

assert device_copy == config.DEVICE

optional_pre_imitation()

wandb.finish()

