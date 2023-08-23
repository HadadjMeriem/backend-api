import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import timm
from timm.models.layers import to_2tuple,trunc_normal_
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor
import torchaudio
import librosa
import random
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim)  
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv[:, 0:2])
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        #x = self.proj(x).transpose(1, 3)

        return x
class CrossDeit(nn.Module):
   def __init__(self,classes=4,fstride=(6,10), tstride=(6,10), input_fdim=64, input_tdim=798,input_fdim2=64,input_tdim2=800,imagenet_pretrain=True,patch_size=(12, 16),in_chans=1,embed_dim=(192, 384), cross_attn_depth =1,cross_attn_heads = 3,heads=3,dropout = 0.1,multi_scale_enc_depth=1,scale_dim = 4):
       super().__init__()
       timm.models.vision_transformer.PatchEmbed = PatchEmbed
       self.num_classes=classes
       self.model_small= timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
       self.model_large= timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
       self.original_num_patches_small= self.model_small.patch_embed.num_patches
       self.original_num_patches_large = self.model_large.patch_embed.num_patches
       self.oringal_hw_small = int(self.original_num_patches_small ** 0.5) #the size of the image width and height
       self.oringal_hw_large= int(self.original_num_patches_large ** 0.5)
     
       self.original_embedding_dim_small = self.model_small.pos_embed.shape[2]
       self.original_embedding_dim_large = self.model_large.pos_embed.shape[2]
       self.norm_small=nn.LayerNorm(self.original_embedding_dim_small)
       self.norm_large=nn.LayerNorm(self.original_embedding_dim_large)
       self.mlp_head_small = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim_small),
            nn.Linear( self.original_embedding_dim_small,self.num_classes)
            #nn.Sigmoid()
        )
       self.mlp_head_large = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim_large),
            nn.Linear(self.original_embedding_dim_large,self.num_classes)
            #nn.Sigmoid()
        )
       self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
       self.bn1=nn.BatchNorm2d(num_features=16)
       self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
       self.bn1=nn.BatchNorm2d(num_features=16)
       self.bn2 = nn.BatchNorm2d(num_features=32)
    

     
       self.pos_drop = nn.Dropout(p=dropout)
       f_dim_small, t_dim_small = self.get_shape(fstride[0], tstride[0], self.original_embedding_dim_small, input_fdim2, input_tdim2,patch_size[0])
       f_dim_large,t_dim_large=self.get_shape(fstride[1], tstride[1],self.original_embedding_dim_large, input_fdim, input_tdim,16)
       num_patches_small= f_dim_small * t_dim_small
       num_patches_large=f_dim_large*t_dim_large
       patch_dim_large = in_chans * num_patches_large ** 2
       patch_dim_small = in_chans * num_patches_small ** 2
       self.model_small.patch_embed.num_patches = num_patches_small
       self.model_large.patch_embed.num_patches = num_patches_large
       new_proj_small= torch.nn.Conv2d(1,self.original_embedding_dim_small, kernel_size=(patch_size[0],patch_size[0]), stride=(fstride[0], tstride[0])) #in_chans=1
       new_proj_large= torch.nn.Conv2d(1, self.original_embedding_dim_large, kernel_size=(16,16), stride=(fstride[1], tstride[1])) #in_chans=1
       if imagenet_pretrain == True:
                #print(self.model_small.patch_embed)
                new_proj_small.weight = torch.nn.Parameter(torch.sum(self.model_small.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj_small.bias = self.model_small.patch_embed.proj.bias
                new_proj_large.weight = torch.nn.Parameter(torch.sum(self.model_large.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj_large.bias = self.model_large.patch_embed.proj.bias
       if patch_size[0]<new_proj_small.weight.shape[2]:  #nombre de patches n'est pas le meme
              new_proj_small.weight=torch.nn.Parameter(new_proj_small.weight.data[:,:,int(new_proj_small.weight.shape[2]/ 2) - int(patch_size[0] / 2): int(new_proj_small.weight.shape[2] / 2) - int(patch_size[0] / 2) +  int(patch_size[0]),int(new_proj_small.weight.shape[2] / 2) - int(patch_size[0] / 2): int(new_proj_small.weight.shape[2] / 2) - int(patch_size[0] / 2) + patch_size[0]])

       self.model_small.patch_embed.proj = new_proj_small
       self.model_large.patch_embed.proj = new_proj_large
       self.model_small.blocks=nn.ModuleList( self.model_small.blocks[:6])
         # the positional embedding
       if imagenet_pretrain == True:
                new_pos_embed_small = self.model_small.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches_small, self.original_embedding_dim_small).transpose(1, 2).reshape(1,  self.original_embedding_dim_small, self.oringal_hw_small, self.oringal_hw_small)
                new_pos_embed_large = self.model_large.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches_large,self.original_embedding_dim_large).transpose(1, 2).reshape(1,self.original_embedding_dim_large,self.oringal_hw_large, self.oringal_hw_large)
                if t_dim_small <= self.oringal_hw_small:
                    new_pos_embed_small = new_pos_embed_small[:, :, :, int(self.oringal_hw_small / 2) - int(t_dim_small / 2): int(self.oringal_hw_small / 2) - int(t_dim_small / 2) + t_dim_small]
                else:
                    new_pos_embed_small = torch.nn.functional.interpolate(new_pos_embed_small, size=(self.oringal_hw_small, t_dim_small), mode='bilinear')
                if t_dim_large<= self.oringal_hw_large:
                    new_pos_embed_large = new_pos_embed_large[:, :, :, int(self.oringal_hw_large / 2) - int(t_dim_large / 2): int(self.oringal_hw_large / 2) - int(t_dim_large / 2) + t_dim_large]
                else:
                    new_pos_embed_large= torch.nn.functional.interpolate(new_pos_embed_large, size=(self.oringal_hw_large, t_dim_large), mode='bilinear')
                if f_dim_small <= self.oringal_hw_small:
                    new_pos_embed_small = new_pos_embed_small[:, :, int(self.oringal_hw_small / 2) - int(f_dim_small / 2): int(self.oringal_hw_small / 2) - int(f_dim_small / 2) + f_dim_small, :]
                else:
                    new_pos_embed_small = torch.nn.functional.interpolate(new_pos_embed_small, size=(f_dim_small, t_dim_small), mode='bilinear')
                if f_dim_large <= self.oringal_hw_large:
                    new_pos_embed_large = new_pos_embed_large[:, :, int(self.oringal_hw_large / 2) - int(f_dim_large / 2): int(self.oringal_hw_large / 2) - int(f_dim_large / 2) + f_dim_large, :]
                else:
                    new_pos_embed_large = torch.nn.functional.interpolate(new_pos_embed_large, size=(f_dim_large, t_dim_large), mode='bilinear')

                # flatten the positional embedding
                new_pos_embed_small = new_pos_embed_small.reshape(1, self.original_embedding_dim_small, num_patches_small).transpose(1,2)
                new_pos_embed_large = new_pos_embed_large.reshape(1, self.original_embedding_dim_large, num_patches_large).transpose(1,2)
                self.model_small.pos_embed = nn.Parameter(torch.cat([self.model_small.pos_embed[:, :2, :].detach(), new_pos_embed_small], dim=1))
                #print("positional_small"+str(self.model_small.pos_embed.shape))
                self.model_large.pos_embed = nn.Parameter(torch.cat([self.model_large.pos_embed[:, :2, :].detach(), new_pos_embed_large], dim=1))

       else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed_small = nn.Parameter(torch.zeros(1, self.model_small.patch_embed.num_patches + 2, self.original_embedding_dim_small))
                new_pos_embed_large = nn.Parameter(torch.zeros(1, self.model_large.patch_embed.num_patches + 2, self.original_embedding_dim_large))


                self.model_small.pos_embed = new_pos_embed_small
                self.model_large.pos_embed = new_pos_embed_large
                trunc_normal_(self.model_small.pos_embed, std=.02)
                trunc_normal_(self.model_large.pos_embed, std=.02)
       #implement the multiscale transformer part
       
       self.cross_attn_layers = nn.ModuleList([])
       for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(self.original_embedding_dim_small, self.original_embedding_dim_large),
                nn.Linear(self.original_embedding_dim_large, self.original_embedding_dim_small),
                PreNorm(self.original_embedding_dim_large, CrossAttention(self.original_embedding_dim_large, heads = cross_attn_heads, dim_head = self.original_embedding_dim_large // heads, dropout = dropout)),
                nn.Linear(self.original_embedding_dim_large, self.original_embedding_dim_small),
                nn.Linear(self.original_embedding_dim_small, self.original_embedding_dim_large),
                PreNorm(self.original_embedding_dim_small, CrossAttention(self.original_embedding_dim_small, heads = cross_attn_heads, dim_head = self.original_embedding_dim_small // heads, dropout = dropout)),
            ]))
       
   def get_shape(self, fstride, tstride,original_embedding, input_fdim=128, input_tdim=1024,patch_size=8):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, original_embedding , kernel_size=(patch_size, patch_size), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim



   def forward(self,x1,x2):
        #print(x1.shape)
        B = x1.shape[0]
        x1=x1.transpose(2,3)
        xl= self.model_large.patch_embed(x1)
        cls_tokens_large = self.model_large.cls_token.expand(B, -1, -1)
        dist_token_large = self.model_large.dist_token.expand(B, -1, -1)
        xl = torch.cat((cls_tokens_large, dist_token_large, xl), dim=1)
        xl = xl + self.model_large.pos_embed
        xs= self.model_small.patch_embed(x2)
        cls_tokens_small = self.model_small.cls_token.expand(B, -1, -1)
        dist_token_small = self.model_small.dist_token.expand(B, -1, -1)
        xs = torch.cat((cls_tokens_small, dist_token_small, xs), dim=1)
        xs = xs + self.model_small.pos_embed
        xl = self.model_large.pos_drop(xl)
        xs = self.model_small.pos_drop(xs)
        for blk in self.model_large.blocks:
            xl = blk(xl)
        xl=self.model_large.norm(xl)
        for blk in self.model_small.blocks:
            xs = blk(xs)
        xs=self.model_small.norm(xs)
        
        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
            small_class = (xs[:, 0]+xs[:,1])/2
            x_small = xs[:, 2:]
            large_class = (xl[:, 0]+xl[:,1])/2
            x_large = xl[:, 2:]

            # Cross Attn for Large Patch

            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)
            xl=self.pos_drop(xl)
            xs=self.pos_drop(xs)    
        xs = (xs[:, 0] + xs[:, 1]) / 2
        xl = (xl[:, 0] + xl[:, 1]) / 2
        xs = self.mlp_head_small(xs)
        xl = self.mlp_head_large(xl)
        return xs+xl
#from RespireNet
class Utils():
 def __init__(self):
       super().__init__()
 def split_and_pad(self,original, desiredLength, sample_rate, types='repeat'):
     output_buffer_length = int(desiredLength*sample_rate)
     soundclip=original[0].copy()
     n_samples=len(soundclip)
     output=[]
     # if n_samples>output_buffer_length ==> slice samples with the desired length, and the last sample is padded until the desired length
     if n_samples > output_buffer_length:
         #décomposer le cycle en frames
         frames = librosa.util.frame(soundclip, frame_length=output_buffer_length, hop_length=output_buffer_length//2, axis=0)
         for i in range(frames.shape[0]):
             output.append((frames[i],original[1]))
         last_id=frames.shape[0]*(output_buffer_length//2)
         last_audio=soundclip[last_id:]
         padded_sample=self.duplicate_padding(soundclip, last_audio, output_buffer_length, sample_rate, types)
         output.append((padded_sample,original[1]))
     else:
         padded_sample=self.duplicate_padding(soundclip, soundclip, output_buffer_length, sample_rate, types)
         output.append((padded_sample,original[1]))
     return output
 def duplicate_padding(self,sample, source, output_length, sample_rate, types='repeat'):
    # pad_type == 1 or 2
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)
    left = output_length - src_length # amount to be padded

    if types == 'repeat':
        aug = sample
    while len(aug) < left:
        aug = np.concatenate([aug, aug])

    prob = random.random()
    if prob < 0.5:
        # pad the back part of original sample
        copy[left:] = source
        copy[:left] = aug[len(aug)-left:]
    else:
        # pad the front part of original sample
        copy[:src_length] = source[:]
        copy[src_length:] = aug[:left]

    return copy
 def preprocess_audio(self,filepath):
     audio, sr = librosa.load(filepath, sr=16000)
     #split or pad the audio to the desired length 
     audio=self.split_and_pad([audio, 0,0,0,0],8,16000, types='repeat')[0][0]
     train_transform_stft= [transforms.ToTensor()]
     transform_stft=transforms.Compose(train_transform_stft,transforms.Resize(size=(int(8), int(100))))
     transform_fbank=transforms.Compose(train_transform_stft,transforms.Resize(size=(int(8), int(100))))
     transform=transforms.ToTensor()
     fbank = torchaudio.compliance.kaldi.fbank(torch.tensor(audio).unsqueeze(0), htk_compat=True, sample_frequency=16000, use_energy=False, window_type='hanning', num_mel_bins=64, dither=0.0, frame_shift=10)
     fbank = fbank.unsqueeze(-1).numpy()
     fbank = (fbank - fbank.min()) / (fbank.max()-fbank.min()) # mean / std
     #stft=librosa.stft(audio,n_fft=1024, hop_length=512, win_length=40, window='hann', center=True, pad_mode='constant')
     stft=librosa.stft(audio,n_fft=512, hop_length=40, win_length=512, window='hann', center=True, pad_mode='constant')  
     stft=(stft-stft.min())/(stft.max() - stft.min())
     stft=transform_stft(stft)
     fbank=transform_fbank(fbank)
     return fbank,stft
 def load_model(self,model_path,filepath,patch_size,split,type='son',DEVICE='cpu'):
     lung_sound_model = torch.load(model_path,map_location=torch.device('cpu'))
     if type=='son':
        net=CrossDeit(cross_attn_depth=3,classes=4,patch_size=patch_size)
        print('ok')
     elif type=='pathologie':
        if split=='officiel':
            net=CrossDeit(cross_attn_depth=7,classes=6,patch_size=patch_size)
        else: 
            net=CrossDeit(cross_attn_depth=3,classes=6,patch_size=patch_size)
     #fbank,stft=self.preprocess_audio(filepath)
     #print(fbank)
     #fbank=fbank.unsqueeze(0).to('cpu')
     #stft=torch.abs(stft)
     #print(stft.shape)
     #stft=stft.unsqueeze(0).to('cpu')
     #net.load_state_dict(lung_sound_model)
     #print(net)
     #net=net.to(DEVICE)
     #net.eval()
     #output = net(fbank,stft)
     #print(output)
     #_, predicted_class = torch.max(output, dim=1)
     pridected_class=0
     #print("pridected_class"+str(predicted_class.item()))
     return(predicted_class)







