import fnmatch
import inspect

import torch

from .multi_head import MultiHeadClassifier

_AVAILABLE_MODELS = (
    "dino_resnet50",
    "dino_vits16",
    "dino_vitb16",
    "timm_resnet50",
    "timm_vit_small_patch16_224",
    "timm_vit_base_patch16_224",
    "timm_vit_large_patch16_224",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "msn_vit_small",
    "msn_vit_base",
    "msn_vit_large",
    "mocov3_vit_small",
    "mocov3_vit_base",
    "clip_ViT-B/16",
    "clip_ViT-L/14",
    "clip_RN50",
    "mae_vit_base",
    "mae_vit_large",
    "mae_vit_huge",
)


def available_models(pattern=None):
    if pattern is None:
        return _AVAILABLE_MODELS
    return tuple(fnmatch.filter(_AVAILABLE_MODELS, pattern))

def load_model(config, head=True):
    """
    config/args file
    head=False returns just the backbone for baseline evaluation
    split_preprocess=True returns resizing etc. and normalization/ToTensor as separate transforms
    """
    from main_args import set_default_args
    config = set_default_args(config)
    model = None

    if head:
        if getattr(config, "embed_dim", None) is None:
            raise ValueError("Specify embed_dim")
        # Just get everything via reflection
        mmc_params = inspect.signature(MultiHeadClassifier).parameters
        mmc_args = {k: v for k, v in config.__dict__.items() if k in mmc_params}
        model = MultiHeadClassifier("PaSST", **mmc_args)

        if config.embed_norm:
            model.set_mean_std(*load_embed_stats(config, test=False))
        print("Head loaded.")

    return model, None

def load_embeds(config=None,
                arch=None,
                dataset=None,
                test=False,
                norm=False,
                datapath='data',
                with_label=False):
    p, test_str = _embedding_path(arch, config, datapath, dataset, test)
    emb = torch.load(p / f'embeddings{test_str}.pt', map_location='cpu')
    if norm:
        emb /= emb.norm(dim=-1, keepdim=True)
    if not with_label:
        return emb
    label = torch.load(p / f'label{test_str}.pt', map_location='cpu')
    return emb, label


def _embedding_path(arch, config, datapath, dataset, test):
    assert bool(config) ^ bool(arch and dataset)
    if config:
        arch = config.arch
        dataset = config.dataset
    import gen_embeds
    test_str = '-test' if test else ''
    p = gen_embeds.get_outpath(arch, dataset, datapath)
    return p, test_str


def load_embed_stats(
        config=None,
        arch=None,
        dset=None,
        test=False,
        datapath='data'):
    p, test_str = _embedding_path(arch, config, datapath, dset, test)
    mean = torch.load(p / f'mean{test_str}.pt', map_location='cpu')
    std = torch.load(p / f'std{test_str}.pt', map_location='cpu')
    return mean, std

