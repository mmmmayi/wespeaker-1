# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import fire
import kaldiio
import torch
import copy
import sys
import matplotlib.pyplot as plt
import librosa
import librosa.display
from torch.utils.data import DataLoader
from tqdm import tqdm
from wespeaker.dataset.dataset import Dataset
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.utils import parse_config_or_kwargs, validate_path, spk2id
from wespeaker.models.projections import get_projection
from wespeaker.utils.file_utils import read_table

def extract(config='conf/config.yaml', **kwargs):
    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)

    model_path = configs['model_path']
    embed_ark = configs['embed_ark']
    
    batch_size = configs.get('batch_size', 1)
    num_workers = configs.get('num_workers', 1)

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    model = get_speaker_model(configs['model'])(**configs['model_args'])
    projection = get_projection(configs['projection_args'])
    model.add_module("projection", projection)
    load_checkpoint(model, model_path)
    device = torch.device("cuda")
    model.to(device).eval()
    # test_configs
    test_conf = copy.deepcopy(configs['dataset_args'])
    test_conf['speed_perturb'] = False
    if 'fbank_args' in test_conf:
        test_conf['fbank_args']['dither'] = 0.0
    elif 'mfcc_args' in test_conf:
        test_conf['mfcc_args']['dither'] = 0.0
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False

    train_label = configs['train_label']
    train_utt_spk_list = read_table(train_label)
    spk2id_dict = spk2id(train_utt_spk_list)

    dataset = Dataset(configs['data_type'],
                      configs['data_list'],
                      test_conf,
                      spk2id_dict=spk2id_dict,
                      whole_utt=(batch_size == 1),
                      reverb_lmdb_file=None,
                      noise_lmdb_file=None)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            prefetch_factor=4)

    validate_path(embed_ark)
    embed_ark = os.path.abspath(embed_ark)
    embed_scp = embed_ark[:-3] + "scp"

    #with torch.no_grad():
    with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
                                 embed_scp) as writer:
        for _, batch in enumerate(dataloader):
            utts = batch['key'][0]
            features = batch['feat']
            targets = batch['label']
            features = features.float().to(device)  # (B,T,F)
            features = features.permute(0, 2, 1)
            
            targets = targets.long().to(device)
             
                # Forward through model
            embeds, feature = model(features)  # embed or (embed_a, embed_b)
            #embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            outputs = model.projection(embeds, targets)
            target_score = outputs[:,targets[0].item()]
           
            samap = torch.autograd.grad(target_score, feature, retain_graph=False)[0]
            samask = (samap-torch.min(samap))/(torch.max(samap)-torch.min(samap))+0.5
            samask = samask.squeeze().unsqueeze(0)
            #print(samask.shape)
            features = features+(samask+1e-6).log()
            #print(features.shape)
            embeds, feature = model(features) 
            
            
           
            model.zero_grad()  
            '''
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            librosa.display.specshow(feature.detach().cpu().squeeze().numpy(), x_axis=None, ax=ax[0])
            img = librosa.display.specshow(samap.detach().cpu().squeeze().numpy(), x_axis=None, ax=ax[1])
            fig.colorbar(img, ax=ax)
            if not os.path.exists(os.path.join(configs['exp_dir'],'pic',utts.split('/')[-3],utts.split('/')[-2])):
                os.makedirs(os.path.join(configs['exp_dir'],'pic',utts.split('/')[-3],utts.split('/')[-2]))
            plt.savefig(os.path.join(configs['exp_dir'],'pic',utts.replace('.wav','.png')))
            plt.close()
            continue
            '''
            embeds = embeds.cpu().detach().numpy()  # (B,F)
            torch.cuda.empty_cache()

            writer(utts, embeds)


if __name__ == '__main__':
    fire.Fire(extract)
