import numpy as np
import pyaudio
from GANLatentDiscovery.loading import load_generator
from audio.features import magnitude

from GANLatentDiscovery.latent_deformator import LatentDeformator
from GANLatentDiscovery.constants import DEFORMATOR_TYPE_DICT, SHIFT_DISTRIDUTION_DICT, WEIGHTS


G = Editor(seed, 128, 512, 8, channel_multiplier=2)

G.load_state_dict(torch.load("./GANLatentDiscovery/models/StyleGAN2/017000.pt", map_location='cpu')['g_ema'])
G.cuda().eval()


state_dict = torch.load("./GANLatentDiscovery/models/checkpoint.pt")

deformator = LatentDeformator(shift_dim=G.dim_shift,
                                input_dim=512,
                                out_dim=512,
                                type=DEFORMATOR_TYPE_DICT["ortho"],
                                random_init=True).cuda()
deformator.load_state_dict(state_dict['deformator'])