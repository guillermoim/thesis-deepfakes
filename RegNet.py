import os
import pycls.core.checkpoint as checkpoint
from pycls.core.io import cache_url
from pycls.models.anynet import AnyNet
from pycls.models.effnet import EffNet
import torch

# URL prefix for pretrained models
_URL_PREFIX = "https://dl.fbaipublicfiles.com/pycls/dds_baselines"
# Model weights download cache directory
_DOWNLOAD_CACHE = "/tmp/pycls-download-cache"

# RegNetY -> URL
_REGNETY_URLS = {
    "200MF": "176245422/RegNetY-200MF_dds_8gpu.pyth",
    "400MF": "160906449/RegNetY-400MF_dds_8gpu.pyth",
    "600MF": "160981443/RegNetY-600MF_dds_8gpu.pyth",
    "800MF": "160906567/RegNetY-800MF_dds_8gpu.pyth",
    "1.6GF": "160906681/RegNetY-1.6GF_dds_8gpu.pyth",
    "3.2GF": "160906834/RegNetY-3.2GF_dds_8gpu.pyth",
    "4.0GF": "160906838/RegNetY-4.0GF_dds_8gpu.pyth",
    "6.4GF": "160907112/RegNetY-6.4GF_dds_8gpu.pyth",
    "8.0GF": "161160905/RegNetY-8.0GF_dds_8gpu.pyth",
    "12GF": "160907100/RegNetY-12GF_dds_8gpu.pyth",
    "16GF": "161303400/RegNetY-16GF_dds_8gpu.pyth",
    "32GF": "161277763/RegNetY-32GF_dds_8gpu.pyth",
}

# RegNetY -> cfg
_REGNETY_CFGS = {
    "200MF": {"ds": [1, 1, 4, 7], "ws": [24, 56, 152, 368], "g": 8},
    "400MF": {"ds": [1, 3, 6, 6], "ws": [48, 104, 208, 440], "g": 8},
    "600MF": {"ds": [1, 3, 7, 4], "ws": [48, 112, 256, 608], "g": 16},
    "800MF": {"ds": [1, 3, 8, 2], "ws": [64, 128, 320, 768], "g": 16},
    "1.6GF": {"ds": [2, 6, 17, 2], "ws": [48, 120, 336, 888], "g": 24},
    "3.2GF": {"ds": [2, 5, 13, 1], "ws": [72, 216, 576, 1512], "g": 24},
    "4.0GF": {"ds": [2, 6, 12, 2], "ws": [128, 192, 512, 1088], "g": 64},
    "6.4GF": {"ds": [2, 7, 14, 2], "ws": [144, 288, 576, 1296], "g": 72},
    "8.0GF": {"ds": [2, 4, 10, 1], "ws": [168, 448, 896, 2016], "g": 56},
    "12GF": {"ds": [2, 5, 11, 1], "ws": [224, 448, 896, 2240], "g": 112},
    "16GF": {"ds": [2, 4, 11, 1], "ws": [224, 448, 1232, 3024], "g": 112},
    "32GF": {"ds": [2, 5, 12, 1], "ws": [232, 696, 1392, 3712], "g": 116},
}

def regnety(name, pretrained=True, nc=1000):
    """Constructs a RegNetY model."""
    is_valid = name in _REGNETY_URLS.keys() and name in _REGNETY_CFGS.keys()
    assert is_valid, "RegNetY-{} not found in the model zoo.".format(name)
    # Construct the model
    cfg = _REGNETY_CFGS[name]
    model = AnyNet(**{
        "stem_type": "simple_stem_in", "stem_w": 32, "block_type": "res_bottleneck_block",
        "ss": [2, 2, 2, 2], "bms": [1.0, 1.0, 1.0, 1.0], "se_r": 0.25, "nc": nc,
        "ds": cfg["ds"], "ws": cfg["ws"], "gws": [cfg["g"] for _ in range(4)]
    })
    # Download and load the weights
    if pretrained:
        url = os.path.join(_URL_PREFIX, _REGNETY_URLS[name])
        ws_path = cache_url(url, _DOWNLOAD_CACHE)
        checkpoint.load_checkpoint(ws_path, model)
    return model

