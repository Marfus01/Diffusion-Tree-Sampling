import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.core.config import cache_path

# Optional dependencies: only the SD rewards need them, and torch_tensorrt is only
# importable on machines with an NVIDIA driver.
try:
    import torch_tensorrt  # registers the "torch_tensorrt" / "tensorrt" torch.compile backends
except ImportError:
    torch_tensorrt = None
try:
    import clip
except ImportError:
    clip = None
try:
    import hpsv2
except ImportError:
    hpsv2 = None


# --- Classifier Reward Functions ---

def classifier_log_likelihood_reward(images, target_class_index, classifier, device):
    """Calculates log p(y|x) for a batch of images using a pre-trained classifier."""
    classifier = classifier.to(device)
    classifier.eval()
    input_images = images.to(device)
    logits = classifier(input_images)
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_likelihood = log_probs[:, target_class_index]
    if len(target_log_likelihood.shape) > 1:
        target_log_likelihood = torch.max(target_log_likelihood, dim=-1)[0]
    return target_log_likelihood


# --- SD Reward Functions ---

# Stores the reward models
REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
    "LLMGrader": None,
    "AestheticScore": None,
    "JPEG-Compressibility": None,
}

# Returns the reward function based on the guidance_reward_fn name
def get_reward_function(reward_name, images, prompts, metric_to_chase="overall_score"):
    if reward_name == "ImageReward":
        return do_image_reward(images=images, prompts=prompts)
    
    elif reward_name == "Clip-Score":
        return do_clip_score(images=images, prompts=prompts)
    
    elif reward_name == "HumanPreference":
        return do_human_preference_score(images=images, prompts=prompts)

    elif reward_name == "AestheticScore":
        return do_aesthetic_score(images=images)
    
    elif reward_name == "JPEG-Compressibility":
        return do_jpeg_compressibility(images=images)
    
    # elif reward_name == "LLMGrader":
    #     return do_llm_grading(images=images, prompts=prompts, metric_to_chase=metric_to_chase)
    
    else:
        raise ValueError(f"Unknown metric: {reward_name}")
    
# Compute human preference score
def do_human_preference_score(*, images, prompts, use_paths=False):
    if use_paths:
        scores = hpsv2.score(images, prompts, hps_version="v2.1")
        scores = [float(score) for score in scores]
    else:
        scores = []
        for i, image in enumerate(images):
            score = hpsv2.score(image, prompts[i], hps_version="v2.1")
            # print(f"Human preference score for image {i}: {score}")
            score = float(score[0])
            scores.append(score)

    # print(f"Human preference scores: {scores}")
    return scores

# Compute CLIP-Score and diversity
def do_clip_score_diversity(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=cache_path("clip"), device="cuda")
    with torch.no_grad():
        arr_clip_result = []
        arr_img_features = []
        for i, prompt in enumerate(prompts):
            clip_result, feature_vect = REWARDS_DICT["Clip-Score"].score(
                prompt, images[i], return_feature=True
            )

            arr_clip_result.append(clip_result.item())
            arr_img_features.append(feature_vect['image'])

    # calculate diversity by computing pairwise similarity between image features
    diversity = torch.zeros(len(images), len(images))
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            diversity[i, j] = (arr_img_features[i] - arr_img_features[j]).pow(2).sum()
            diversity[j, i] = diversity[i, j]
    n_samples = len(images)
    diversity = diversity.sum() / (n_samples * (n_samples - 1))

    return arr_clip_result, diversity.item()

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes)

    return _fn

def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew = jpeg_fn(images, prompts, metadata)
        return -rew

    return _fn

def do_jpeg_compressibility(*, images, prompts=None):
    global REWARDS_DICT
    if REWARDS_DICT["JPEG-Compressibility"] is None:
        REWARDS_DICT["JPEG-Compressibility"] = jpeg_compressibility()

    with torch.no_grad():
        jpeg_compressibility_result = REWARDS_DICT["JPEG-Compressibility"](images, prompts, metadata={})
    
    return jpeg_compressibility_result

# Compute ImageReward
def do_image_reward(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["ImageReward"] is None:
        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")

    with torch.no_grad():
        image_reward_result = REWARDS_DICT["ImageReward"].score_batched(prompts, images)
        # image_reward_result = [REWARDS_DICT["ImageReward"].score(prompt, images[i]) for i, prompt in enumerate(prompts)]

    return image_reward_result

# Compute CLIP-Score
def do_clip_score(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=cache_path("clip"), device="cuda")
    with torch.no_grad():
        clip_result = [
            REWARDS_DICT["Clip-Score"].score(prompt, images[i])
            for i, prompt in enumerate(prompts)
        ]
    return clip_result


def do_aesthetic_score(*, images, prompts=None):
    global REWARDS_DICT
    if REWARDS_DICT["AestheticScore"] is None:
        REWARDS_DICT["AestheticScore"] = get_aesthetic_model(clip_model="vit_l_14").to("cuda")
    
    with torch.no_grad():
        # Get CLIP image features first
        if REWARDS_DICT["Clip-Score"] is None:
            REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=cache_path("clip"), device="cuda")
        
        scores = []
        for image in images:
            # Get CLIP image features
            _, features = REWARDS_DICT["Clip-Score"].score("dummy", image, return_feature=True)
            image_features = features['image']
            
            # Predict aesthetic score
            score = REWARDS_DICT["AestheticScore"](image_features)
            scores.append(score.item())
            
    return scores

# # Compute LLM-grading
# def do_llm_grading(*, images, prompts, metric_to_chase="overall_score"):
#     global REWARDS_DICT
    
#     if REWARDS_DICT["LLMGrader"] is None:
#         REWARDS_DICT["LLMGrader"]  = LLMGrader()
#     llm_grading_result = [
#         REWARDS_DICT["LLMGrader"].score(images=images[i], prompts=prompt, metric_to_chase=metric_to_chase)
#         for i, prompt in enumerate(prompts)
#     ]
#     return llm_grading_result


'''
@File       :   CLIPScore.py
@Time       :   2023/02/12 13:14:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   CLIPScore.
* Based on CLIP code base
* https://github.com/openai/CLIP
'''


class CLIPScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device, jit=False, download_root=download_root
        )

        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model
            )  # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, prompt, pil_image, return_feature=False):
        # if (type(image_path).__name__=='list'):
        #     _, rewards = self.inference_rank(prompt, image_path)
        #     return rewards

        # text encode
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))

        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image))

        # score
        rewards = torch.sum(
            torch.mul(txt_features, image_features), dim=1, keepdim=True
        )

        if return_feature:
            return rewards, {'image': image_features, 'txt': txt_features}

        return rewards.detach().cpu().numpy().item()
    
from typing import Union
import os
import torch

from PIL import Image
try:
    import ImageReward as RM
except ImportError:
    RM = None


'''
@File       :   ImageReward.py
@Time       :   2023/01/28 19:53:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   ImageReward Reward model.
* Based on CLIP code base and improved-aesthetic-predictor code base
* https://github.com/openai/CLIP
* https://github.com/christophschuhmann/improved-aesthetic-predictor
'''

import os
import torch
import torch.nn as nn
from PIL import Image
try:
    from ImageReward.models.BLIP.blip_pretrain import BLIP_Pretrain
except ImportError:
    BLIP_Pretrain = None
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from torchvision.transforms.functional import pil_to_tensor

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

    def forward(self, input):
        return self.layers(input)


class IRSMC(nn.Module):
    def __init__(self, med_config, device='cpu'):
        super().__init__()
        self.device = device

        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)
        self.preprocess = _transform(224)
        self.mlp = MLP(768)

        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

    def score_batched_old(self, prompts, images):
        # batch
        results = []
        for i, prompt in enumerate(prompts):
            results.append(self.score(prompt, images[i]))

        return results

    def score_gard(self, prompt_ids, prompt_attention_mask, image):
        image_embeds = self.blip.visual_encoder(image)
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_output = self.blip.text_encoder(
            prompt_ids,
            attention_mask=prompt_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :]  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards

    def score(self, prompt, image):
        if type(image).__name__ == 'list':
            _, rewards = self.inference_rank(prompt, image)
            return rewards

        # text encode
        text_input = self.blip.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str) and os.path.isfile(image):
            pil_image = Image.open(image)
        else:
            raise TypeError(
                r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.'
            )

        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_embeds = self.blip.visual_encoder(image)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards.detach().cpu().numpy().item()

    def score_batched(self, prompts, images):
        assert isinstance(prompts, list)
        # assert isinstance(images, list)

        # text encode
        text_input = self.blip.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        # image encode
        images = [
            self.preprocess(image).unsqueeze(0).to(self.device) for image in images
        ]
        images = torch.cat(images, 0).to(self.device)

        image_embeds = self.blip.visual_encoder(images)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards.view(txt_features.shape[0]).detach().cpu().numpy()

    def inference_rank(self, prompt, generations_list):
        text_input = self.blip.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)
        txt_set = []
        for generation in generations_list:
            # image encode
            if isinstance(generation, Image.Image):
                pil_image = generation
            elif isinstance(generation, str):
                if os.path.isfile(generation):
                    pil_image = Image.open(generation)
            else:
                raise TypeError(
                    r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.'
                )
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_embeds = self.blip.visual_encoder(image)

            # text encode cross attention with image
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )
            text_output = self.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            txt_set.append(text_output.last_hidden_state[:, 0, :])

        txt_features = torch.cat(txt_set, 0).float()  # [image_num, feature_dim]
        rewards = self.mlp(txt_features)  # [image_num, 1]
        rewards = (rewards - self.mean) / self.std
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1

        return (
            indices.detach().cpu().numpy().tolist(),
            rewards.detach().cpu().numpy().tolist(),
        )


def rm_load(
    name: str = "ImageReward-v1.0",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    download_root: str = None,
    med_config: str = None,
):
    """Load a ImageReward model

    Parameters
    ----------
    name : str
        A model name listed by `ImageReward.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    download_root: str
        path to download the model files; by default, it uses "~/.cache/ImageReward"

    Returns
    -------
    model : torch.nn.Module
        The ImageReward model
    """
    if name in RM.utils._MODELS:
        model_path = RM.ImageReward_download(
            RM.utils._MODELS[name],
            download_root or cache_path("ImageReward"),
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found;")

    print('load checkpoint from %s' % model_path)
    state_dict = torch.load(model_path, map_location='cpu')
    # state_dict = torch.load(model_path, map_location=device)

    # med_config
    if med_config is None:
        med_config = RM.ImageReward_download(
            "https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json",
            download_root or cache_path("ImageReward"),
        )

    model = IRSMC(device=device, med_config=med_config).to(device)
    msg = model.load_state_dict(state_dict, strict=False)

    if torch_tensorrt is not None and os.environ.get("DTS_NO_COMPILE", "0") != "1":
        model = torch.compile(
            model,
            backend="torch_tensorrt",
            options={
                "truncate_long_and_double": True,
                "enabled_precisions": {torch.float32, torch.float16},
            },
            dynamic=False,
        )
    
    print("ImageReward checkpoint loaded")
    model.eval()
    # import pdb; pdb.set_trace()
    return model

'''
Aesthetic Score Predictor by LAION
'''

import os
import torch
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    cache_folder = cache_path("emb_reader")
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1).half()  # Convert to float16
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1).half()  # Convert to float16
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m
