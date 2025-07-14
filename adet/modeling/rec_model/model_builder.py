import torch.nn as nn

from .decoder import *
from .attn_decoder import AttentionRecognitionHead

from timm.models.layers import to_2tuple

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if not isinstance(img_size, tuple):
            img_size = to_2tuple(img_size)
        if not isinstance(patch_size, tuple):
            patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class CTCRecModel(nn.Module):
  def __init__(self, args):
    super(CTCRecModel, self).__init__()

    self.encoder = create_encoder(args)
    d_embedding = 512
    self.ctc_classifier = nn.Sequential(nn.Linear(self.encoder.num_features, d_embedding),
                                        nn.LayerNorm(d_embedding, eps=1e-6),
                                        nn.GELU(),
                                        nn.Linear(d_embedding, args.nb_classes + 1))

    # some function and variable should be inherited.
    self.patch_embed = self.encoder.patch_embed
    self.pos_embed = self.encoder.pos_embed

  def no_weight_decay(self):
    skip_weight_decay_list = self.encoder.no_weight_decay()
    return {'encoder.' + item for item in skip_weight_decay_list}

  def get_num_layers(self):
    return self.encoder.get_num_layers()

  def forward(self, x):
    x, tgt, tgt_lens = x
    enc_x = self.encoder(x)

    B, N, C = enc_x.shape
    reshaped_enc_x = enc_x.view(B, *self.encoder.patch_embed.patch_shape, C).mean(1)
    ctc_logit = self.ctc_classifier(reshaped_enc_x)

    return ctc_logit

class AttnRecModel(nn.Module):
  def __init__(self, args):
    super(AttnRecModel, self).__init__()

    self.patch_embed = PatchEmbed(
      img_size=(32, 128), patch_size=4, in_chans=256, embed_dim=384
    )
    
    self.decoder = AttentionRecognitionHead(
                      num_classes=args.MODEL.ATTENTION.NB_CLASSES,
                      in_planes=384,
                      sDim=512,
                      attDim=512,
                      max_len_labels=args.MODEL.ATTENTION.MAX_LEN) 

  def no_weight_decay(self):
    skip_weight_decay_list = self.encoder.no_weight_decay()
    return {'encoder.' + item for item in skip_weight_decay_list}

  def get_num_layers(self):
    return self.encoder.get_num_layers()

  def forward(self, x):
    enc_x, tgt, tgt_lens = x
    enc_x = self.patch_embed(enc_x)

    dec_output, _ = self.decoder((enc_x, tgt, tgt_lens))
    return dec_output, None, None, None

class RecModel(nn.Module):
  def __init__(self, args):
    super(RecModel, self).__init__()

    self.patch_embed = PatchEmbed(
      img_size=(32, 128), patch_size=4, in_chans=256, embed_dim=384
    )
    self.decoder = create_decoder(args)

    d_embedding = self.decoder.d_embedding
    self.linear_norm = nn.Sequential(
      nn.Linear(384, d_embedding),     # 384 -> encoder.num_features
      nn.LayerNorm(d_embedding),
    )

    self.trg_word_emb = None
    self.insert_sem = False

    # 1d or 2d features
    self.use_1d_attdec = False
    self.beam_width = getattr(args, 'beam_width', 0)

  def no_weight_decay(self):
    skip_weight_decay_list = self.encoder.no_weight_decay()
    return {'encoder.' + item for item in skip_weight_decay_list}

  def get_num_layers(self):
    return self.encoder.get_num_layers()

  def forward(self, x):
    enc_x, tgt, tgt_lens = x   # enc_x : [len of pred_boxes, 256, 32, 128]
    enc_x = self.patch_embed(enc_x)  ### enc_x : [len of pred_boxes, 256, 384]

    cls_logit = None
    cls_logit_attn_maps = None

    dec_in = self.linear_norm(enc_x)
    dec_output, dec_attn_maps = self.decoder(dec_in,
                                             dec_in,
                                             targets=tgt,
                                             tgt_lens=tgt_lens,
                                             train_mode=self.training,
                                             cls_query_attn_maps=cls_logit_attn_maps,
                                             trg_word_emb=self.trg_word_emb,
                                             beam_width=self.beam_width,)
    
    # return dec_output, None, None, None
    return dec_output, None, None, dec_attn_maps

  def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True, use_conv=False):
      mlp = []
      for l in range(num_layers):
          dim1 = input_dim if l == 0 else mlp_dim
          dim2 = output_dim if l == num_layers - 1 else mlp_dim

          if use_conv:
            mlp.append(nn.Conv1d(dim1, dim2, 1, bias=False))
          else:
            mlp.append(nn.Linear(dim1, dim2, bias=False))

          if l < num_layers - 1:
              mlp.append(nn.BatchNorm1d(dim2))
              mlp.append(nn.ReLU(inplace=True))
          elif last_bn:
              # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
              # for simplicity, we further removed gamma in BN
              mlp.append(nn.BatchNorm1d(dim2, affine=False))

      return nn.Sequential(*mlp)
