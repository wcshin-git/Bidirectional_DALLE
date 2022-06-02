import os

import argparse
from pathlib import Path

import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader

from dalle_pytorch import VQGanVAE, DALLE
from dalle_pytorch import distributed_utils
from dalle_pytorch.loader import MnistDataset
from tokenizers import Tokenizer
from tokenizers.processors import BertProcessing
from parser import MnistLabelParser, FashionLabelParser, ImgAcc
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm import tqdm
import random
import numpy as np
from img_classifier.pl_model import ModelModule
import datetime

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--dalle_path', type=str,
                   help='path to your partially trained DALL-E')

parser.add_argument('--vqgan_model_path', type=str, default = None,
                   help='path to your trained VQGAN weights. This should be a .ckpt file.')

parser.add_argument('--vqgan_config_path', type=str, default = None,
                   help='path to your trained VQGAN config. This should be a .yaml file.')

parser.add_argument('--img_classifier_path', type=str, default = None)

parser.add_argument('--parser', type=str, default = None, choices=['MNIST', 'FASHION'])

parser.add_argument('--save_dir', type=str, default='./checkpoints')

parser.add_argument('--valtest_name', type=str, required=True, choices=['mnist_test_seen', 'mnist_test_unseen'],
                    help='valtest folder name')

parser.add_argument('--val_use_ratio', type=float, default = 0.25, help='the ratio between used and whole valtest set to reduce validation time')

parser.add_argument('--image_text_folder', type=str, required=True,
                    help='path to your folder of images and text for learning the DALL-E')

parser.add_argument('--truncate_captions', dest='truncate_captions', action='store_true',
                    help='Captions passed in which exceed the max token length will be truncated if this is set.')

parser.add_argument('--bpe_path', type=str,
                    help='path to your huggingface BPE json file')

parser.add_argument('--wandb_name', default='dalle_train_transformer',
                    help='Name W&B will use when saving results.')

parser.add_argument('--saving_interval', default = 2, type = int, help = 'saving interval in terms of epochs')

parser = distributed_utils.wrap_arg_parser(parser)

train_group = parser.add_argument_group('Training settings')

train_group.add_argument('--seed', default = 42, type = int, help='Seed number')

train_group.add_argument('--task', default = 't2i', type = str, choices=['t2i', 'i2t', 'alternate'])

train_group.add_argument('--epochs', default = 20, type = int, help = 'Number of epochs')

train_group.add_argument('--batch_size', default = 4, type = int, help = 'Batch size')

train_group.add_argument('--learning_rate', default = 3e-4, type = float, help = 'Learning rate')

train_group.add_argument('--clip_grad_norm', default = 0.5, type = float, help = 'Clip gradient norm')

train_group.add_argument('--lr_decay', dest = 'lr_decay', action = 'store_true')

train_group.add_argument('--num_workers', default = 0, type = int, help = 'the number of dataloader workers')

model_group = parser.add_argument_group('Model settings')

model_group.add_argument('--dim', default = 512, type = int, help = 'Model dimension')

model_group.add_argument('--text_seq_len', default = 50, type = int, help = 'Text sequence length')

model_group.add_argument('--depth', default = 2, type = int, help = 'Model depth')

model_group.add_argument('--heads', default = 8, type = int, help = 'Model number of heads')

model_group.add_argument('--dim_head', default = 64, type = int, help = 'Model head dimension')

model_group.add_argument('--loss_weight', default = 7, type = int, help = 'loss weight')

model_group.add_argument('--attn_types', default = 'full', type = str, help = 'comma separated list of attention types. attention type can be: full or axial_row or axial_col or conv_like.')

model_group.add_argument('--pe_type', default = 'learnable', type = str, choices=['learnable', 'fixed'])

args = parser.parse_args()

# quit early if you used the wrong folder name

assert Path(args.image_text_folder).exists(), f'The path {args.image_text_folder} was not found.'

# make save folder

now = datetime.datetime.now()  # GMT
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
output_folder = os.path.join(args.save_dir, timestamp)
os.makedirs(output_folder, exist_ok=True)

# helpers

def exists(val):
    return val is not None

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(args.seed)

# constants

VQGAN_MODEL_PATH = args.vqgan_model_path
VQGAN_CONFIG_PATH = args.vqgan_config_path
DALLE_PATH = args.dalle_path
RESUME = exists(DALLE_PATH)

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

LEARNING_RATE = args.learning_rate
CLIP_GRAD_NORM = args.clip_grad_norm
LR_DECAY = args.lr_decay

TEXT_SEQ_LEN = args.text_seq_len
DIM = args.dim
DEPTH = args.depth
HEADS = args.heads
DIM_HEAD = args.dim_head
LOSS_WEIGHT = args.loss_weight

ATTN_TYPES = tuple(args.attn_types.split(','))
PE_TYPE = args.pe_type

IMG_CLASSIFIER_SPEC = None
for spec in ['Resnet18', 'Resnet34', 'Resnet50']:
    if spec in args.img_classifier_path:
        IMG_CLASSIFIER_SPEC = spec
assert IMG_CLASSIFIER_SPEC != None, 'cant find the img calssifier spec from the name of the ckpt file.'

# initialize distributed backend

distr_backend = distributed_utils.set_backend_from_args(args)
distr_backend.initialize()

# tokenizer

tokenizer = Tokenizer.from_file(args.bpe_path)

tokenizer.post_processor = BertProcessing(
    ("[SEP]", tokenizer.token_to_id("[SEP]")),  # EOS; at the very back
    ("[CLS]", tokenizer.token_to_id("[CLS]")),  # SOS; at the very front
)

tokenizer.enable_truncation(max_length=args.text_seq_len)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=args.text_seq_len)

SPECIAL_TOK_IDX = {
    '[PAD]': tokenizer.token_to_id("[PAD]"),
    '[CLS]': tokenizer.token_to_id("[CLS]"),
    '[SEP]': tokenizer.token_to_id("[SEP]"),
}

# reconstitute vae

if RESUME:
    dalle_path = Path(DALLE_PATH)
    assert dalle_path.exists(), 'DALL-E model file does not exist'
    loaded_obj = torch.load(str(dalle_path), map_location='cpu')

    dalle_params, weights = loaded_obj['hparams'], loaded_obj['weights']

    vae = VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)

    dalle_params = dict(
        **dalle_params
    )

else:
    if distr_backend.is_root_worker():
        print('using pretrained VAE for encoding images to tokens')

    vae = VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)

    dalle_params = dict(
        num_text_tokens=tokenizer.get_vocab_size(),
        text_seq_len=TEXT_SEQ_LEN,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        loss_weight=LOSS_WEIGHT,
        attn_types=ATTN_TYPES,
        pe_type=PE_TYPE,
        special_tok_idx = SPECIAL_TOK_IDX,
    )

IMAGE_SIZE = vae.image_size
IMAGE_SEQ_LEN = vae.fmap_size ** 2

# create dataset and dataloader

is_shuffle = not distributed_utils.using_backend(distributed_utils.HorovodBackend)  # is_shuffle = False  when using Horovod

train_text_folder = os.path.join(args.image_text_folder, 'mnist_train_text')
train_ing_folder = os.path.join(args.image_text_folder, 'mnist_train_img')
test_text_folder = os.path.join(args.image_text_folder, args.valtest_name+'_text')
test_img_foler = os.path.join(args.image_text_folder, args.valtest_name+'_img')
ds_train = MnistDataset(
            train_text_folder,
            train_ing_folder,
            image_size=IMAGE_SIZE,
            tokenizer=tokenizer,
            split='train',
            info_save_dir=output_folder,
            )
ds_val = MnistDataset(
            test_text_folder,
            test_img_foler,
            image_size=IMAGE_SIZE,
            tokenizer=tokenizer,
            split='val',
            info_save_dir=output_folder,
            use_ratio=args.val_use_ratio,
            )   

assert len(ds_train) > 0, 'dataset is empty'
if distr_backend.is_root_worker():
    print(f'{len(ds_train)} image-text pairs will be used for training')
    print(f'{len(ds_val)} image-text pairs will be used for validation')

if not is_shuffle:
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        ds_train,
        num_replicas=distr_backend.get_world_size(),
        rank=distr_backend.get_rank()
    )
else:
    data_sampler = None

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=is_shuffle, drop_last=True, sampler=data_sampler, num_workers=args.num_workers)
dl_val =  DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=args.num_workers) # shuffle=True for logging various samples

# initialize DALL-E

dalle = DALLE(vae=vae, **dalle_params)
dalle = dalle.cuda()

if RESUME:
    dalle.load_state_dict(weights)

# optimizer

opt = Adam(get_trainable_params(dalle), lr=LEARNING_RATE)

if LR_DECAY:
    print("len(train_loader):", len(dl_train))
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps = 1 * len(dl_train),
            num_training_steps = args.epochs * len(dl_train),
            num_cycles = 6,
        )

if distr_backend.is_root_worker():
    # experiment tracker

    model_config = dict(
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD
    )

    run = wandb.init(
        entity="t-i_gen",
        project=args.wandb_name,  # 'dalle_train_transformer' by default
        resume=RESUME,
        config=args,
    )

# distribute

distr_backend.check_batch_size(BATCH_SIZE)

(distr_dalle, distr_opt, distr_dl_train, distr_dl_val, distr_scheduler) = distr_backend.distribute(
    args=args,
    model=dalle,
    optimizer=opt,
    model_parameters=get_trainable_params(dalle),
    training_data=dl_train,
    val_data=dl_val,
    lr_scheduler=scheduler if LR_DECAY else None,
)

def save_model(path):
    save_obj = {
        'hparams': dalle_params,
    }

    if not distr_backend.is_root_worker():
        return

    save_obj = {
        **save_obj,
        'weights': dalle.state_dict()
    }

    torch.save(save_obj, path)

# training

task_change_switch = (args.task == 'alternate')

for epoch in range(EPOCHS):
    if data_sampler:
        data_sampler.set_epoch(epoch)
    for i, (text, images) in enumerate(distr_dl_train):
        if task_change_switch == False:
            task_list = [args.task]
            tensor_list = [(text, images)]
        else:
            task_list = ['i2t', 't2i']
            text_copy = text.clone().detach()
            images_copy = images.clone().detach()
            tensor_list = [(text, images), (text_copy, images_copy)]

        for args.task, (text, images) in zip(task_list, tensor_list):
            text, images = map(lambda t: t.cuda(), (text, images))

            text_mask = (text != tokenizer.token_to_id("[PAD]")).cuda()  # 'False' part will not be attended.

            loss = distr_dalle(text, images, return_loss=True, text_mask=text_mask, task = args.task)

            loss.backward()
            clip_grad_norm_(distr_dalle.parameters(), CLIP_GRAD_NORM)
            distr_opt.step()
            distr_opt.zero_grad()

            # Averaged loss
            avg_loss = distr_backend.average_all(loss)

            log = {}

            if (i % 10 == 0 and distr_backend.is_root_worker()):
                print(epoch, i, f'{args.task}_loss - {avg_loss.item()}')

                log = {
                    'epoch': epoch,
                    'iter': i,
                    f'{args.task}_train_loss': avg_loss.item(),
                    'lr': distr_scheduler.optimizer.param_groups[0]['lr']
                }

            if i % 100 == 0:
                if args.task == 't2i':
                    if distr_backend.is_root_worker():
                        sample_text = text[:1]  # [1,text_seq_len]
                        token_list = sample_text.squeeze().tolist()
                        decoded_text = tokenizer.decode(token_list, skip_special_tokens=True) # str
                        gen_image = distr_dalle.generate_images(sample_text, filter_thres=0.9) # [B, 3, H, W] 0. ~ 1.
                        log['gen_image_train'] = wandb.Image(gen_image, caption=decoded_text)
                elif args.task == 'i2t':
                    if distr_backend.is_root_worker():
                        sample_image = images[:1]  # [1, 3, H, W]  min: 0.  max: 1.
                        gen_txt = distr_dalle.generate_texts(sample_image, filter_thres=0.9) # gen_txt: tensor[1, <=text_seq_len]
                        token_list = gen_txt[0].tolist()
                        decoded_text = tokenizer.decode(token_list, skip_special_tokens=True)  # str
                        log['gen_caption_train'] = wandb.Image(sample_image, caption=decoded_text)

            if distr_backend.is_root_worker():
                wandb.log(log)
                
        distr_scheduler.step()


    # validation
    log = {}
    t2i_loss_list = []
    i2t_loss_list = []
    # parser
    if args.parser == 'MNIST':
        parser = MnistLabelParser()
    elif args.parser == 'FASHION':
        parser = FashionLabelParser()
    # ImgAcc
    imgacc = ImgAcc()
    if distr_backend.is_root_worker():
        img_classifier = ModelModule.load_from_checkpoint(args.img_classifier_path, model_type=IMG_CLASSIFIER_SPEC)
        img_classifier.eval()
        img_classifier = img_classifier.cuda()
        with torch.no_grad():
            for i, (text, images) in enumerate(tqdm(distr_dl_val)):  # text: [B, text_len]   image: [B, 3, H, W]
                text, images = map(lambda t: t.cuda(), (text, images))

                text_mask = (text != tokenizer.token_to_id("[PAD]")).cuda()

                t2i_loss = distr_dalle(text, images, return_loss=True, text_mask=text_mask, task='t2i')
                t2i_loss_list.append(t2i_loss.item())
                
                i2t_loss = distr_dalle(text, images, return_loss=True, text_mask=text_mask, task='i2t')
                i2t_loss_list.append(i2t_loss.item())

                # text generation 
                gen_texts = distr_dalle.generate_texts(images, filter_thres=0.9)
                decoded_gt_list = []
                for gt_text_tensor, gen_text_tensor in zip(text, gen_texts):
                    gt_text_decoded = tokenizer.decode(gt_text_tensor.tolist(), skip_special_tokens=True)  # str
                    decoded_gt_list.append(gt_text_decoded)
                    gen_text_decoded = tokenizer.decode(gen_text_tensor.tolist(), skip_special_tokens=True) 
                    parser.count_correct(gt_text_decoded, gen_text_decoded)

                # image generation
                gen_images = distr_dalle.generate_images(text, filter_thres=0.9) # [B, 3, H, W]  # 0. ~ 1.
                for gt_str, gen_image in zip(decoded_gt_list, gen_images): # str, [3, H, W]
                    gt_labels_list = parser.parse_label(gt_str)
                    gen_labels_list = []
                    for gt_label in gt_labels_list: # eg. gt_label: [0, 1, 3] ; loc_idx, color_idx, number_idx
                        if gt_label[0] == 0:
                            gen_label = [0]
                            cropped_image = gen_image[:, int(gen_image.size(1)/4+2):-int(gen_image.size(1)/4+2), int(gen_image.size(2)/4+2):-int(gen_image.size(2)/4+2)]
                        elif gt_label[0] == 1:
                            gen_label = [1]
                            cropped_image = gen_image[:, 2:int(gen_image.size(1)/2-2), 2:int(gen_image.size(2)/2-2)]
                        elif gt_label[0] == 2:
                            gen_label = [2]
                            cropped_image = gen_image[:, 2:int(gen_image.size(1)/2-2), int(gen_image.size(2)/2+2):-2]
                        elif gt_label[0] == 3:
                            gen_label = [3]
                            cropped_image = gen_image[:, int(gen_image.size(1)/2+2):-2, 2:int(gen_image.size(2)/2-2)]
                        elif gt_label[0] == 4:
                            gen_label = [4]
                            cropped_image = gen_image[:, int(gen_image.size(1)/2+2):-2, int(gen_image.size(2)/2+2):-2]
                        color_logit, number_logit = img_classifier(cropped_image.unsqueeze(0))
                        color_idx = color_logit.argmax(dim=1)
                        number_idx = number_logit.argmax(dim=1)
                        gen_label.append(color_idx.item())
                        gen_label.append(number_idx.item())
                        gen_labels_list.append(gen_label)
                    
                    imgacc.count_correct(gt_labels_list, gen_labels_list)

            # text acc  
            [text_center_acc, text_quad1_acc, text_quad2_acc, text_quad3_acc, text_quad4_acc], \
                [text_center_color_acc, text_center_number_acc, text_quad1_color_acc, text_quad1_number_acc, text_quad2_color_acc, text_quad2_number_acc, text_quad3_color_acc, text_quad3_number_acc, text_quad4_color_acc, text_quad4_number_acc] \
                        = parser.calc_accuracy()
            log['text_center_color_acc']= text_center_color_acc
            log['text_center_number_acc'] = text_center_number_acc
            log['text_quad1_color_acc'] = text_quad1_color_acc
            log['text_quad1_number_acc'] = text_quad1_number_acc
            log['text_quad2_color_acc'] = text_quad2_color_acc
            log['text_quad2_number_acc'] = text_quad2_number_acc
            log['text_quad3_color_acc'] = text_quad3_color_acc
            log['text_quad3_number_acc'] = text_quad3_number_acc
            log['text_quad4_color_acc'] = text_quad4_color_acc
            log['text_quad4_number_acc'] = text_quad4_number_acc
            print(f'text_center_number_acc: {text_center_number_acc:.3f} | tot_num_labels: {parser.center_total_count}')
            print(f'text_quad1_number_acc: {text_quad1_number_acc:.3f} | tot_num_labels: {parser.quad1_total_count}')
            print(f'text_quad2_number_acc: {text_quad2_number_acc:.3f} | tot_num_labels: {parser.quad2_total_count}')
            print(f'text_quad3_number_acc: {text_quad3_number_acc:.3f} | tot_num_labels: {parser.quad3_total_count}')
            print(f'text_quad4_number_acc: {text_quad4_number_acc:.3f} | tot_num_labels: {parser.quad4_total_count}')

            # img acc
            [img_center_acc, img_quad1_acc, img_quad2_acc, img_quad3_acc, img_quad4_acc], \
                [img_center_color_acc, img_center_number_acc, img_quad1_color_acc, img_quad1_number_acc, img_quad2_color_acc, img_quad2_number_acc, img_quad3_color_acc, img_quad3_number_acc, img_quad4_color_acc, img_quad4_number_acc] \
                        = imgacc.calc_accuracy()
            log['img_center_color_acc']= img_center_color_acc
            log['img_center_number_acc'] = img_center_number_acc
            log['img_quad1_color_acc'] = img_quad1_color_acc
            log['img_quad1_number_acc'] = img_quad1_number_acc
            log['img_quad2_color_acc'] = img_quad2_color_acc
            log['img_quad2_number_acc'] = img_quad2_number_acc
            log['img_quad3_color_acc'] = img_quad3_color_acc
            log['img_quad3_number_acc'] = img_quad3_number_acc
            log['img_quad4_color_acc'] = img_quad4_color_acc
            log['img_quad4_number_acc'] = img_quad4_number_acc
            print(f'img_center_number_acc: {img_center_number_acc:.3f} | tot_num_labels: {imgacc.center_total_count}')
            print(f'img_quad1_number_acc: {img_quad1_number_acc:.3f} | tot_num_labels: {imgacc.quad1_total_count}')
            print(f'img_quad2_number_acc: {img_quad2_number_acc:.3f} | tot_num_labels: {imgacc.quad2_total_count}')
            print(f'img_quad3_number_acc: {img_quad3_number_acc:.3f} | tot_num_labels: {imgacc.quad3_total_count}')
            print(f'img_quad4_number_acc: {img_quad4_number_acc:.3f} | tot_num_labels: {imgacc.quad4_total_count}')

            # averaged loss
            t2i_val_loss = np.array(t2i_loss_list).mean()
            i2t_val_loss = np.array(i2t_loss_list).mean()
            print(epoch, f't2i_val_loss - {t2i_val_loss}')
            print(epoch, f'i2t_val_loss - {i2t_val_loss}')
            log['epoch'] = epoch
            log['t2i_val_loss'] = t2i_val_loss
            log['i2t_val_loss'] = i2t_val_loss

            # log one sample
            sample_text = text[:1]  # [1,text_seq_len]
            token_list = sample_text.squeeze().tolist()
            decoded_text = tokenizer.decode(token_list, skip_special_tokens=True) # str
            gen_image = distr_dalle.generate_images(sample_text, filter_thres=0.9) # [B, 3, H, W] 0. ~ 1.
            log['gen_image_val'] = wandb.Image(gen_image, caption=decoded_text)
            
            sample_image = images[:1]  # [1, 3, H, W]  min: 0.  max: 1.
            gen_txt = distr_dalle.generate_texts(sample_image, filter_thres=0.9) # gen_txt: tensor[1, <=text_seq_len]
            token_list = gen_txt[0].tolist()
            decoded_text = tokenizer.decode(token_list, skip_special_tokens=True)  # str
            log['gen_caption_val'] = wandb.Image(sample_image, caption=decoded_text) # sample_image: Tensor, decoded_text: str

            wandb.log(log)


    # save model
    if epoch % args.saving_interval == 0:
        save_model(f'{output_folder}/dalle_ep{epoch}.pt')


# finish
save_model(f'{output_folder}/dalle-final.pt')

if distr_backend.is_root_worker():
    wandb.finish()
