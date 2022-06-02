import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from dalle_pytorch.vae import VQGanVAE
from dalle_pytorch.transformer import Transformer

import torch
from torch import nn
from operator import mul
from functools import reduce
import numpy as np


class FixedPositionalEmbedding2D(nn.Module):
    def __init__(self, channels, axial_shape):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(FixedPositionalEmbedding2D, self).__init__()
        self.ori_channels = channels
        channels = int(np.ceil(channels/4)*2)
        self.channels = channels
        self.shape = axial_shape
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

        x, y = self.shape
        pos_x = torch.arange(x).type(self.inv_freq.type())
        pos_y = torch.arange(y).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x,y,self.channels*2))
        emb[:,:,:self.channels] = emb_x
        emb[:,:,self.channels:2*self.channels] = emb_y
        emb = emb.reshape(-1, self.ori_channels)
        self.register_buffer('emb', emb)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x_times_y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x_times_y, ch)
        """
        b, t, e = tensor.shape

        return self.emb[None, :t, :].repeat(b, 1, 1).to(tensor)


class ParameterList(object):
    def __init__(self, kls, prefix, length):
        self.ind = 0
        self.kls = kls
        self.prefix = prefix
        self.length = length

    def _keyname(self, prefix, ind):
        return f'{prefix}_{ind}'

    def append(self, x):
        setattr(self.kls, self._keyname(self.prefix, self.ind), x)
        self.ind += 1

    def to_list(self):
        return [getattr(self.kls, self._keyname(self.prefix, i)) for i in range(self.length)]


class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, axial_shape, axial_dims = None):
        super().__init__()

        self.dim = dim
        self.shape = axial_shape
        self.max_seq_len = reduce(mul, axial_shape, 1)

        self.summed = axial_dims is None
        axial_dims = ((dim,) * len(axial_shape)) if self.summed else axial_dims

        assert len(self.shape) == len(axial_dims), 'number of axial dimensions must equal the number of dimensions in the shape'
        assert self.summed or not self.summed and sum(axial_dims) == dim, f'axial dimensions must sum up to the target dimension {dim}'

        self.weights = ParameterList(self, 'weights', len(axial_shape))

        for ind, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, axial_dim)
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0, 1))
            self.weights.append(ax_emb)
    
    def forward(self, x):
        b, t, e = x.shape
        assert (t <= self.max_seq_len), f'Sequence length ({t}) must be less than the maximum sequence length allowed ({self.max_seq_len})'
        embs = []

        for ax_emb in self.weights.to_list():
            axial_dim = ax_emb.shape[-1]
            expand_shape = (b, *self.shape, axial_dim)
            emb = ax_emb.expand(expand_shape).reshape(b, self.max_seq_len, axial_dim)
            embs.append(emb)

        pos_emb = sum(embs) if self.summed else torch.cat(embs, dim=-1)
        return pos_emb[:, :t].to(x)


class AxialPositionalEmbeddingImage(nn.Module): # Axial Positional Embedding for Images
    def __init__(self, dim, axial_shape, axial_dims = None):
        super().__init__()
        assert len(axial_shape) == 2, 'Axial shape must have 2 dimensions for images'
        self.pos_emb = AxialPositionalEmbedding(dim, axial_shape, axial_dims)

    def forward(self, img):
        b, c, h, w = img.shape
        img = img.permute(0, 2, 3, 1).reshape(b, h * w, c)
        pos_emb = self.pos_emb(img)
        return pos_emb.reshape(b, h, w, c).permute(0, 3, 1, 2)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):  # TODO: 아무데도 안 쓰임
    def inner(*args, **kwargs):
        return val
    return inner

def is_empty(t):
    return t.nelement() == 0

def masked_mean(t, mask, dim = 1):  # TODO: 아무데도 안 쓰임
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# main DALL-E class

class DALLE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae,
        num_text_tokens = 10000,
        text_seq_len = 256,
        depth,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0,
        attn_types = None,
        loss_weight = 7,
        pe_type = 'fixed',
        special_tok_idx = None,
    ):
        super().__init__()
        assert isinstance(vae, (VQGanVAE)), 'vae must be an instance of VQGanVAE'

        self.special_tok_idx = special_tok_idx

        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = vae.fmap_size
        image_seq_len = image_fmap_size ** 2
        
        num_text_tokens = num_text_tokens + text_seq_len  # reserve unique padding tokens for each position (text seq len)

        self.text_emb = nn.Embedding(num_text_tokens, dim)
        self.image_emb = nn.Embedding(num_image_tokens, dim)

        self.text_pos_emb = nn.Embedding(text_seq_len, dim)
        if pe_type == 'learnable':
            self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (image_fmap_size, image_fmap_size))
        elif pe_type == 'fixed':
            self.image_pos_emb = FixedPositionalEmbedding2D(dim, axial_shape = (image_fmap_size, image_fmap_size))
        self.num_text_tokens = num_text_tokens
        self.num_image_tokens = num_image_tokens

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        total_seq_len = text_seq_len + image_seq_len
        num_total_tokens = num_text_tokens + num_image_tokens
        self.num_total_tokens = num_total_tokens
        self.total_seq_len = total_seq_len

        self.vae = vae
        set_requires_grad(self.vae, False) # freeze VAE from being trained

        self.transformer = Transformer(
            dim = dim,
            causal = True,
            seq_len = total_seq_len,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            attn_types = attn_types,
            image_fmap_size = image_fmap_size,
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.num_total_tokens),
        )

        seq_range = torch.arange(total_seq_len)
        logits_range = torch.arange(self.num_total_tokens)

        seq_range = rearrange(seq_range, 'n -> () n ()')
        logits_range = rearrange(logits_range, 'd -> () () d')
        # logit space is composed of text and then image tokens
        logits_mask_t2i = ( # row-axis: seq_inx (text first),  col_axis: logit_inx
            ((seq_range < text_seq_len-1) & (logits_range < num_text_tokens)) | # Upper left block: True
            ((seq_range >= text_seq_len-1) & (logits_range >= num_text_tokens))   # Lower right block: True
        ) # 'False' part will be masked with -inf.

        self.register_buffer('logits_mask_t2i', logits_mask_t2i, persistent=False)
        
        logits_mask_i2t = ( # row-axis: seq_inx (image first),  col_axis: logit_inx
            ((seq_range >= image_seq_len-1) & (logits_range < num_text_tokens)) | # Lower left block: True
            ((seq_range < image_seq_len-1) & (logits_range >= num_text_tokens)) # Upper right block: True
        ) # 'False' part will be masked with -inf.

        self.register_buffer('logits_mask_i2t', logits_mask_i2t, persistent=False)

        self.loss_weight = loss_weight

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        text,   # tensor [B, text_seq_len]
        *,
        clip = None,
        filter_thres = 0.5,
        temperature = 1.,
        img = None,
        num_init_img_tokens = None,
        return_indices = False,
    ):
        vae, text_seq_len, image_seq_len, num_text_tokens = self.vae, self.text_seq_len, self.image_seq_len, self.num_text_tokens
        total_len = text_seq_len + image_seq_len

        text = text[:, :text_seq_len] # make sure text is within bounds
        out = text

        text_mask = (text != self.special_tok_idx['[PAD]']).cuda()

        if exists(img):
            image_size = vae.image_size
            assert img.shape[1] == 3 and img.shape[2] == image_size and img.shape[3] == image_size, f'input image must have the correct image size {image_size}'

            indices = vae.get_codebook_indices(img)
            num_img_tokens = default(num_init_img_tokens, int(0.4375 * image_seq_len))  # OpenAI used 14 * 32 initial tokens to prime
            assert num_img_tokens < image_seq_len, 'number of initial image tokens for priming must be less than the total image token sequence length'

            indices = indices[:, :num_img_tokens]
            out = torch.cat((out, indices), dim = -1)

        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len

            text, image = out[:, :text_seq_len], out[:, text_seq_len:]

            logits = self(text, image, text_mask = text_mask, return_loss=False, task='t2i')[:, -1, :]
            
            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)

            sample -= (num_text_tokens if is_image else 0) # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            out = torch.cat((out, sample), dim=-1)

        text_seq = out[:, :text_seq_len]

        img_seq = out[:, -image_seq_len:]
        if return_indices == True:
            return img_seq
        images = vae.decode(img_seq)  # eg. [B, 3, H, W]   # min: 0. max: 1.

        if exists(clip):
            scores = clip(text_seq, images, return_loss = False)
            return images, scores

        return images


    
    @torch.no_grad()
    @eval_decorator
    def generate_texts(
        self,
        image,   # tensor[B, 3, H, W]  min: 0.  max: 1.
        *,
        clip = None,
        filter_thres = 0.5,
        temperature = 1.,
    ):
        vae, text_seq_len, image_seq_len, num_text_tokens = self.vae, self.text_seq_len, self.image_seq_len, self.num_text_tokens
        total_len = text_seq_len + image_seq_len
        B = image.shape[0]

        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4

            if is_raw_image:
                image_size = self.vae.image_size
                assert tuple(image.shape[1:]) == (3, image_size, image_size), f'invalid image of dimensions {image.shape} passed in during training'

                image = self.vae.get_codebook_indices(image)  # -> [B, fmap_size**2]

        # Generated tokens are supposed to be added to 'out'
        out = torch.cat(
            ( image, torch.tensor([[self.special_tok_idx['[CLS]']]]*B).to(image) ), 
            dim=-1
            )

        for cur_len in range(out.shape[1], total_len):

            image, text = out[:, :image_seq_len], out[:, image_seq_len:]

            logits = self(text, image, return_loss=False, task='i2t')[:, -1, :]
            
            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)
            # logit space is composed of text and then image tokens
 
            out = torch.cat((out, sample), dim=-1)

        img_seq = out[:, :image_seq_len]
        text_seq = out[:, image_seq_len:]

        indices = [
            list(row).index(self.special_tok_idx['[SEP]']) if self.special_tok_idx['[SEP]'] in row 
            else -1 
            for row in text_seq
            ]
        for row, idx in enumerate(indices):
            if idx >= 0:
                text_seq[row, idx+1:] = self.special_tok_idx['[PAD]']

        if exists(clip):
            scores = clip(text_seq, image, return_loss = False)
            return text_seq, scores

        return text_seq


    def forward(
        self,
        text,
        image = None,
        text_mask = None,
        return_loss = False,
        task = 't2i',
    ):
        if task == 't2i':
            assert text.shape[-1] == self.text_seq_len, f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'
        
        device, total_seq_len = text.device, self.total_seq_len
        
        text_labels = text.detach().clone()

        # make sure padding in text tokens get unique padding token id
        text_range = torch.arange(text.shape[1], device = device) + (self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == self.special_tok_idx['[PAD]'], text_range, text)

        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        seq_len = tokens.shape[1]

        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4

            if is_raw_image:
                image_size = self.vae.image_size
                assert tuple(image.shape[1:]) == (3, image_size, image_size), f'invalid image of dimensions {image.shape} passed in during training'

                image = self.vae.get_codebook_indices(image)

            image_len = image.shape[1]
            image_emb = self.image_emb(image)

            image_emb += self.image_pos_emb(image_emb)

            if task == 't2i':
                tokens = torch.cat((tokens, image_emb), dim = 1)
            elif task == 'i2t':
                tokens = torch.cat((image_emb, tokens), dim = 1)

            seq_len += image_len

        B = image.shape[0]
        image_len = image.shape[1]

        img_mask = torch.ones((B, image_len), dtype=torch.bool).cuda()

        if text_mask is not None:
            if task == 't2i':
                mask = torch.cat((text_mask, img_mask), dim = 1) # -> [B, text_len+img_len]
            elif task == 'i2t':
                mask = torch.cat((img_mask, text_mask), dim = 1) # -> [B, img_len+text_len]
        else: 
            mask = torch.ones((B, seq_len), dtype=torch.bool).cuda()

        # tokens: [B, tot_len, dim] -> out: [B, tot_len, dim]
        out = self.transformer(tokens, mask=mask, task=task)
        logits = self.to_logits(out)

        if task == 't2i':
            logits_mask_t2i = self.logits_mask_t2i[:, :seq_len]
            max_neg_value = -torch.finfo(logits.dtype).max
            logits.masked_fill_(~logits_mask_t2i, max_neg_value)
        elif task == 'i2t':
            logits_mask_i2t = self.logits_mask_i2t[:, :seq_len]
            max_neg_value = -torch.finfo(logits.dtype).max
            logits.masked_fill_(~logits_mask_i2t, max_neg_value)

        if not return_loss:
            return logits

        assert exists(image), 'when training, image must be supplied'

        offsetted_image = image + self.num_text_tokens

        logits = rearrange(logits, 'b n c -> b c n')  # -> [B, num_total_tokens, tot_len]

        if task == 't2i':
            loss_text = F.cross_entropy(logits[:, :, :self.text_seq_len-1], text_labels[:, 1:], ignore_index=self.special_tok_idx['[PAD]'])
            loss_img = F.cross_entropy(logits[:, :, self.text_seq_len-1: -1], offsetted_image)
            loss = (loss_text + self.loss_weight * loss_img) / (self.loss_weight + 1)
        elif task == 'i2t':
            loss_img = F.cross_entropy(logits[:, :, :self.image_seq_len-1], offsetted_image[:, 1:])
            loss_text = F.cross_entropy(logits[:, :, self.image_seq_len-1:-1], text_labels, ignore_index=self.special_tok_idx['[PAD]'])
            loss = (loss_img + self.loss_weight * loss_text) / (self.loss_weight + 1)
        
        return loss