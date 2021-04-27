from loading import load_generator
from latent_deformator import LatentDeformator
from constants import DEFORMATOR_TYPE_DICT, SHIFT_DISTRIDUTION_DICT, WEIGHTS
import torch
G = load_generator({"gan_type":"StyleGAN2", "gan_resolution":512, "w_shift":True}, "./models/StyleGAN2/017000.pt")

state_dict = torch.load("./models/checkpoint.pt")

deformator = LatentDeformator(shift_dim=G.dim_shift,
                 input_dim=512,
                 out_dim=512,
                 type=DEFORMATOR_TYPE_DICT["ortho"],
                 random_init=True).cuda()
deformator.load_state_dict(state_dict['deformator'])

target_indices = torch.randint(0, 8, [1], device='cuda')

shifts = 2.0 * torch.rand(target_indices.shape, device='cuda') - 1.0

z_shift = torch.zeros([1] + [512], device='cuda')
for i, (index, val) in enumerate(zip(target_indices, shifts)):
    z_shift[i][index] += val

shift = deformator(z_shift)
img = G.gen_shifted(torch.randn(1,512, device="cuda"), shift)
