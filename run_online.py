from collections import deque
from functools import partial
from itertools import chain

import config
import doy
import env_utils
import paths
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from doy import PiecewiseLinearSchedule as PLS
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from utils import create_decoder
import numpy as np

import datetime

import dmc
from replay_buffer_tensordict import ReplayBuffer
from logger import Logger
import os
from video import VideoRecorder

import data_loader

print('device: ',config.DEVICE)
device_copy = config.DEVICE
print('seed: ',config.DMC_SEED)
seed_copy = config.DMC_SEED

state_dict = torch.load(paths.get_latent_policy_path(config.get().exp_name))
cfg = config.get()   

current_datetime = datetime.datetime.now()
formatted_datetime1 = current_datetime.strftime("%Y%m%d")
formatted_datetime2 = current_datetime.strftime("%H%M%S")
cfg.stage_exp_name =  formatted_datetime1+'_'+formatted_datetime2+'_seed'+str(seed_copy)

doy.print("[bold green]BCV-LR: online finetuning and policy imitation:")
config.print_cfg(cfg)


work_dir = paths.get_experiment_dir(cfg.exp_name)/formatted_datetime1/(formatted_datetime2+'_seed'+str(seed_copy))
os.makedirs(work_dir)
logger_dmc = Logger(work_dir,
                save_tb=False,
                log_frequency=10000,
                action_repeat=2,
                agent='BCV-LR')

eval_video_recorder = VideoRecorder(
            work_dir)

policy = utils.create_policy(
    cfg.model,
    action_dim=cfg.model.la_dim,
    state_dict=state_dict["policy"],
    strict_loading=True,
)

idm_decoder = create_decoder(
    in_dim=cfg.model.la_dim,
    out_dim=cfg.model.ta_dim,
    hidden_sizes=(192, 128, 64),
)


models_path = paths.get_models_path(cfg.exp_name)
idm, wm = utils.create_dynamics_models(cfg.model, state_dicts=torch.load(models_path))


train_data, test_data = data_loader.load(cfg.env_name)
expert_iter = train_data.get_iter(cfg.offline_pt.bs)
test_iter = test_data.get_iter(128)


run, logger = config.wandb_init("BCV-LR continuous finetuning stage", config.get_wandb_cfg(cfg),wandb_enabled = True)


if cfg.env_name == 'pointeasy250':
    env_name_fordmc = 'point_easy'
elif 'bottomleft' in cfg.env_name:
    env_name_fordmc = 'jaco_reach_bottom_left'
elif 'topright' in cfg.env_name:
    env_name_fordmc = 'jaco_reach_top_right'
elif cfg.env_name == 'reachereasy250':
    env_name_fordmc = 'reacher_easy'
elif cfg.env_name == 'finger250':
    env_name_fordmc = 'finger_spin'
elif 'cheetahrunback' in cfg.env_name:
    env_name_fordmc = 'cheetah_run_backward'
elif 'cupcatch' in cfg.env_name:
    env_name_fordmc = 'cup_catch'
elif 'reacherhard' in cfg.env_name:
    env_name_fordmc = 'reacher_hard'



frame_stack = 3
action_repeat = 2
seed = config.DMC_SEED
env = dmc.make(env_name_fordmc, 'pixels', frame_stack,
                                  action_repeat, seed)

eval_env = dmc.make(env_name_fordmc, 'pixels', frame_stack,
                                  action_repeat, seed)

obs_spec = env.observation_spec()
action_spec = env.action_spec()    # BoundedArray(shape=(2,), dtype=dtype('float32'), name='action', minimum=-1.0, maximum=1.0)

replay_buffer_capacity = cfg.replay_buffer_capacity # default 100000
replay_buffer = ReplayBuffer(max_size = replay_buffer_capacity, ta_dim = action_spec.shape[0])


if 'jaco' in env_name_fordmc or 'reacher' in env_name_fordmc:
    cfg.wm_loss_alpha = 1e-3
    cfg.wm_opt_lr = 1e-3


decoder_opt = torch.optim.Adam(utils.chain(idm.parameters(),idm_decoder.parameters()), lr = cfg.decoder_lr)
policy_opt = torch.optim.Adam(utils.chain(policy.parameters()), lr = cfg.policy_lr)


wm_opt_wm = torch.optim.Adam(wm.parameters(),lr=cfg.wm_opt_lr)





def action_selection_hook(obs):

    
    obs = torch.torch.from_numpy(obs).to(config.DEVICE)
    obs = obs.unsqueeze(0)
    obs = data_loader.normalize_obs(obs)

    with torch.no_grad():
        mu = idm_decoder(policy(obs))
    
    std = 0.5

    std = torch.ones_like(mu) * std

    dist = utils.TruncatedNormal(mu, std)

    action = dist.sample(clip=None)
    
    

    return action.cpu().numpy()[0]

def action_selection_eval(obs):

    
    obs = torch.torch.from_numpy(obs).to(config.DEVICE)
    obs = obs.unsqueeze(0)
    obs = data_loader.normalize_obs(obs)

    with torch.no_grad():
        mu = idm_decoder(policy(obs))
    
    std = 0.5

    std = torch.ones_like(mu) * std

    dist = utils.TruncatedNormal(mu, std)

    action = dist.mean

    return action.cpu().numpy()[0]



def reset_decoder(decoder):
    for layer in decoder.children():
        if isinstance(layer, torch.nn.Linear):
            layer.reset_parameters()
        else:
            assert isinstance(layer, torch.nn.ReLU)

def finetuning_policylearning_each_iteration(collect_iter,total_step):
    #print(buf_ta)

    if total_step == 1000:
        wm_loss, vq_perp = expert_wm_wandb()
             
            #if count % 1 ==0:   
        logger(
                step=0,
                wm_unsupervised_loss = wm_loss.item(),
                vq_perp = vq_perp,
            )


    assert device_copy == config.DEVICE
    assert seed_copy == config.DMC_SEED
    
    
    #wm_opt_idm = torch.optim.Adam(idm.parameters(),lr=1e-7)
    if True:
        # do decoder online SL training step
        idm.train()
        
        if 'jaco' in env_name_fordmc or 'reacher' in env_name_fordmc:
            times = 100
        else:
            times = (total_step/ 512)*2
        count = 0
        while count < times:
            batch = next(collect_iter)
            
            decoder_opt.zero_grad()
            wm_opt_wm.zero_grad()
 
            la = idm(batch["obs"])[0]["la"]
            predict_la = idm_decoder(la)
            ta = batch["ta"][:, -2]
            
            loss = F.mse_loss(predict_la, ta)
            wm_loss = wm_collect_loss(batch)

            loss_union = loss + cfg.wm_loss_alpha*wm_loss


            loss_union.backward()
            decoder_opt.step()
            wm_opt_wm.step()

            
            count+=1

        wm_loss, vq_perp = expert_wm_wandb()
             
            #if count % 1 ==0:   
        logger(
                step=total_step,
                idm_supervised_loss = loss.item(),
                wm_unsupervised_loss = wm_loss.item(),
                vq_perp = vq_perp,
                update_times = times
            )
                


        policy.train()
        times = 1000
        count = 0
        while count < times:
            batch = next(expert_iter)
            with torch.no_grad():
                idm.label(batch)

            preds = policy(batch["obs"][:, -2])  # the -2 selects last the pre-transition ob
            loss = F.mse_loss(preds, batch["la"])

            policy_opt.zero_grad()
            loss.backward()
            policy_opt.step()
            count +=1

            if count % 100 == 0: 
                logger(
                        step=total_step,
                        bc_loss = loss.item(),
                    
                    )
        



def wm_collect_loss(batch_collect):
    idm.train()
    wm.train()
    

    vq_loss_collect, vq_perp_collect = idm.label(batch_collect)
    wm_loss_collect = wm.label(batch_collect)


    loss = vq_loss_collect + wm_loss_collect


    return loss

def expert_wm_wandb():
    idm.train()
    wm.train()
    batch = next(expert_iter)

    vq_loss, vq_perp = idm.label(batch)
    wm_loss = wm.label(batch)

    return wm_loss, vq_perp



def test_step(total_step):
    idm.eval()  # disables idm.vq ema update
    wm.eval()

    # evaluate IDM + FDM generalization on (action-free) test data
    batch = next(test_iter)
    idm.label(batch)
    wm_loss = wm.label(batch)

    # train latent -> true action decoder and evaluate its predictiveness
    _, eval_metrics = utils.eval_latent_repr(test_data, idm, cfg.env_name)

    logger(step = total_step, **eval_metrics)


def evaluate(total_step):
    avg_episode_reward = 0
    num_episodes = 10
    reward_list = []
    for episode in range(num_episodes):
        time_step = eval_env.reset()
        eval_video_recorder.init(enabled=(episode == 0))
        episode_reward = 0
        episode_success = 0
        episode_step = 0
        while not time_step.last():
            with torch.no_grad():
                obs = time_step.observation
                action = action_selection_eval(obs)
            time_step = eval_env.step(action)
            eval_video_recorder.record(eval_env)
            episode_reward += time_step.reward
            episode_step += 1

        avg_episode_reward += episode_reward
        reward_list.append(episode_reward)
        eval_video_recorder.save(f'{total_step}.mp4')
    avg_episode_reward /= num_episodes
    logger_dmc.log('eval/episode_reward', avg_episode_reward, total_step)
    logger_dmc.dump(total_step, ty='eval')

    logger(
            step=total_step,
            eval_reward = avg_episode_reward,
                    )
    
    print(reward_list)


    
    
def train():

    spec = env.action_spec()
    done = False
    episode, episode_reward, episode_step = 0, 0, 0
    time_step = env.reset()
    obs = time_step.observation

    for total_step in doy.loop(1, cfg.online_ft.steps + 1, desc="Online finetuning and policy imitation"):
        if done:
            print(f"episode: {episode}, return: {episode_reward}")
            time_step = env.reset()
            obs = time_step.observation
            episode_reward = 0
            episode_step = 0
            episode += 1
            

        if total_step<1000:
            action = np.random.uniform(spec.minimum, spec.maximum,
                                           spec.shape)
        else:
            action = action_selection_hook(obs)

        time_step = env.step(action)
        next_obs = time_step.observation

        
        done = time_step.last()
        episode_reward += time_step.reward

        replay_buffer.add(obs, action,   done,time_step.reward,time_step.reward,time_step.reward) #rewards not used in BCV-LR

        obs = next_obs
        episode_step += 1

        if total_step%1000 ==0:
            collect_iter = replay_buffer.get_iter(batch_size=512,device = config.DEVICE)
            finetuning_policylearning_each_iteration(collect_iter,total_step)

        if total_step%1000 ==0:
            evaluate(total_step)
            


train()



out_path = paths.get_decoded_policy_path(cfg.exp_name)
torch.save(
    {
        "policy": policy.state_dict(),
        "cfg": cfg,
        "logger": logger,
    },
    out_path,
)