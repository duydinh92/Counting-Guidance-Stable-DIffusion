import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
import yaml
import errno

from helper import OptimizerDetails, get_seg_text
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from torchvision import transforms, utils
import torch
import torch.nn as nn
from torchvision import transforms, utils
import torchvision.transforms.functional as TF

from src.LearningToCountAnything.utils.data_utils import resize_instance
from src.LearningToCountAnything.models.CountingAnything import CountingAnything



# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()

    return model



def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass



def mse_loss(y_count, gt_count):
    return (torch.abs(y_count - gt_count) ** 2).mean()

def mae_loss(y_count, gt_count):
    return torch.abs(y_count - gt_count).mean()

def mape_loss(y_count, gt_count):
    return (torch.abs(y_count - gt_count) / gt_count).mean()

def square_mape_loss(y_count, gt_count):
    return ((torch.abs(y_count - gt_count) / gt_count) ** 2).mean()

class ObjectCounting(nn.Module):
    def __init__(self, model, trans):
        super(ObjectCounting, self).__init__()
        self.model = model.cuda()
        self.trans = trans

    def forward(self, image):
        image = (image + 1) * 0.5
        image = TF.resize(image, (224, 224), interpolation=TF.InterpolationMode.BILINEAR)
        input = self.trans(image)
        input = input.cuda()
        feats = self.model.backbone(input)
        y_count, _ = self.model.counting_head(feats)
    
        return y_count

def get_optimation_details(args):
    config = yaml.safe_load(open("./src/LearningToCountAnything/configs/_DEFAULT.yml"))
    test_config = yaml.safe_load(open(f"./src/LearningToCountAnything/configs/test.yml"))
    config.update(test_config)

    trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = CountingAnything(config)
    model = model.eval()

    for param in model.parameters():
        param.requires_grad = False

    operation_func = ObjectCounting(model, trans)
    operation_func = torch.nn.DataParallel(operation_func).cuda()
    operation_func.eval()

    for param in operation_func.parameters():
        param.requires_grad = False

    operation = OptimizerDetails()

    operation.num_steps = args.optim_num_steps
    operation.operation_func = operation_func
    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr

    if args.count_loss == "mse":
        operation.loss_func = mse_loss
    elif args.count_loss == "mae":
        operation.loss_func = mae_loss
    elif args.count_loss == "mape":
        operation.loss_func = mape_loss
    elif args.count_loss == "square_mape":
        operation.loss_func = square_mape_loss

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff

    operation.guidance_3 = args.optim_forward_guidance
    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt
    operation.guidance_2 = args.optim_backward_guidance
    operation.original_guidance = args.optim_original_conditioning

    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 5
    operation.folder = args.optim_folder

    return operation



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6868,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--count_loss",
        type=str,
        default="mse",
        help="Loss type for using gradient from counting network: mse, mae, mape, square_mape",
    )

    parser.add_argument("--optim_lr", default=1e-2, type=float)
    parser.add_argument('--optim_max_iters', type=int, default=1)
    parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
    parser.add_argument('--optim_forward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_backward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_original_conditioning', action='store_true', default=False)
    parser.add_argument("--optim_forward_guidance_wt", default=5.0, type=float)
    parser.add_argument("--optim_tv_loss", default=None, type=float)
    parser.add_argument('--optim_warm_start', action='store_true', default=False)
    parser.add_argument('--optim_print', action='store_true', default=False)
    parser.add_argument('--optim_folder', default='./temp/')
    parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
    parser.add_argument("--text", default=None)
    parser.add_argument('--text_type', type=int, default=1)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument('--seg_folder', default='./data/segmentation_data')
    parser.add_argument('--special_prompt', default='Walker hound, Walker foxhound')
    parser.add_argument("--trials", default=10, type=int)


    opt = parser.parse_args()

    results_folder = opt.optim_folder
    create_folder(results_folder)

    seed_everything(opt.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()

    sampler = DDIMSamplerWithGrad(model)

    operation = get_optimation_details(opt)
    batch_size = opt.batch_size

    torch.set_grad_enabled(False)

    if opt.text != None:
        prompt = opt.text
    else:
        prompt = get_seg_text(opt.text_type)

    print(prompt)

    if opt.text != None:
        final_prompt = prompt
    else:
        final_prompt = opt.special_prompt + prompt

    print(final_prompt)

    import re
    temp = re.findall(r'\d+', final_prompt)
    gt_count = list(map(int, temp))[0]

    uc = None

    if opt.scale != 1.0:
        uc = model.module.get_learned_conditioning(batch_size * [""])

    c = model.module.get_learned_conditioning([final_prompt])

    for multiple_tries in range(opt.trials):
        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

        samples_ddim, start_zt = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=batch_size,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 operated_image=gt_count,
                                                 operation=operation)

        x_samples_ddim = model.module.decode_first_stage(samples_ddim)
        x_samples_ddim_unnorm = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        utils.save_image(x_samples_ddim_unnorm, f'{results_folder}/new_img_{multiple_tries}.png')
        torch.save(start_zt, f'{results_folder}/start_zt_{multiple_tries}.pt')



if __name__ == "__main__":
    main()
