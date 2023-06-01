from typing import List, Optional, Tuple, Union, cast
import torch
import torch.autograd
import torch.nn.functional as F
import numpy as np
import torchvision.models as torchvision_models
from torch.autograd import Variable
from math import exp
from torch import nn
from typing_extensions import Literal

from .models import AlexNetFeatureModel, FeatureModel
from static_vars import StaticVars


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift',
                             torch.tensor([-.030, -.088, -.188], device=StaticVars.DEVICE)[None, :, None, None])
        self.register_buffer('scale', torch.tensor([.458, .448, .450], device=StaticVars.DEVICE)[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''

    def __init__(self, chn_in, chn_out=1, use_dropout=True):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers).to(StaticVars.DEVICE)

    def forward(self, x):
        return self.model(x)


class ImageNetNormalizer(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        mean = torch.tensor(self.mean, device=x.device)
        std = torch.tensor(self.std, device=x.device)

        return (
                (x - mean[None, :, None, None]) /
                std[None, :, None, None]
        )


class OriginalLPIPSDistance(nn.Module):
    model: torchvision_models.AlexNet

    def __init__(self, path):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.normalizer = ImageNetNormalizer()
        self.model = torchvision_models.alexnet(pretrained=True).to(StaticVars.DEVICE).eval()

        assert len(self.model.features) == 13
        self.layer1 = nn.Sequential(self.model.features[:2])
        self.layer2 = nn.Sequential(self.model.features[2:5])
        self.layer3 = nn.Sequential(self.model.features[5:8])
        self.layer4 = nn.Sequential(self.model.features[8:10])
        self.layer5 = nn.Sequential(self.model.features[10:12])

        self.chns = [64, 192, 384, 256, 256]
        self.lin0 = NetLinLayer(self.chns[0])
        self.lin1 = NetLinLayer(self.chns[1])
        self.lin2 = NetLinLayer(self.chns[2])
        self.lin3 = NetLinLayer(self.chns[3])
        self.lin4 = NetLinLayer(self.chns[4])

        self.only_conv_layers = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        [layer.eval() for layer in self.only_conv_layers]
        self.load_state_dict(torch.load(path, map_location=StaticVars.DEVICE), strict=False)

    def normalize_tensor(self, in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
        return in_feat / (norm_factor + eps).detach()

    def spatial_average(self, arr):
        return arr.mean([2, 3], keepdim=True)

    def features(self, x):
        # x = self.normalizer(x)
        x = self.scaling_layer(x)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        x_layer5 = self.layer5(x_layer4)

        return (self.lin0(self.normalize_tensor(x_layer1)), self.lin1(self.normalize_tensor(x_layer2)),
                self.lin2(self.normalize_tensor(x_layer3)), self.lin3(self.normalize_tensor(x_layer4)),
                self.lin4(self.normalize_tensor(x_layer5)))

    def forward(self, img1, img2):
        for img_parts in img1:
            img_parts.detach()
        for img_parts in img2:
            img_parts.detach()
        img_1_and_img_2_diff, res = {}, {}
        for idx in range(5):
            img_1_and_img_2_diff[idx] = (img1[idx] - img2[idx]) ** 2
            img_1_and_img_2_diff[idx].detach()
            res[idx] = self.spatial_average(img_1_and_img_2_diff[idx]).reshape(-1).detach()

        res_sum = 0
        for i in range(len(res)):
            res_sum += res[i]
        return torch.round(res_sum, decimals=4).detach()


class LPIPSDistance(nn.Module):
    """
    Calculates the square root of the Learned Perceptual Image Patch Similarity
    (LPIPS) between two images, using a given neural network.
    """

    model: FeatureModel

    def __init__(
            self,
            model: Optional[Union[FeatureModel, nn.DataParallel]] = None,
            activation_distance: Literal['l2'] = 'l2',
            include_image_as_activation: bool = False,
    ):
        """
        Constructs an LPIPS distance metric. The given network should return a
        tuple of (activations, logits). If a network is not specified, AlexNet
        will be used. activation_distance can be 'l2' or 'cw_ssim'.
        """

        super().__init__()

        if model is None:
            alexnet_model = torchvision_models.alexnet(pretrained=True)
            self.model = AlexNetFeatureModel(alexnet_model)
        elif isinstance(model, nn.DataParallel):
            self.model = cast(FeatureModel, model.module)
        else:
            self.model = model

        self.activation_distance = activation_distance
        self.include_image_as_activation = include_image_as_activation

        self.eval()

    def features(self, image: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        features = self.model.features(image)
        if self.include_image_as_activation:
            features = (image,) + features
        return features

    def forward(self, image1, image2):
        features1 = self.features(image1)
        features2 = self.features(image2)

        if self.activation_distance == 'l2':
            return (
                    normalize_flatten_features(features1) -
                    normalize_flatten_features(features2)
            ).norm(dim=1)
        else:
            raise ValueError(
                f'Invalid activation_distance "{self.activation_distance}"')


class LinearizedLPIPSDistance(LPIPSDistance):
    """
    An approximation of the LPIPS distance using the Jacobian of the feature
    network, i.e. d(x1, x2) = || D phi(x1) (x2 - x1) ||_2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.activation_distance != 'l2':
            raise ValueError(
                f'Invalid activation_distance "{self.activation_distance}"'
            )

    def forward(self, image1, image2):
        # Use the double-autograd trick for forward derivatives from
        # https://j-towns.github.io/2017/06/12/A-new-trick.html
        # and https://github.com/pytorch/pytorch/issues/10223#issuecomment-547104071

        image1 = image1.detach().requires_grad_()
        diff = image2 - image1
        features1 = normalize_flatten_features(self.features(image1))
        v = torch.ones_like(features1, requires_grad=True)
        vjp, = torch.autograd.grad(
            features1,
            image1,
            grad_outputs=v,
            create_graph=True,
        )
        output, = torch.autograd.grad(vjp, v, grad_outputs=diff)
        return output.norm(dim=1)


def normalize_flatten_features(
        features: Tuple[torch.Tensor, ...],
        eps=1e-10,
) -> torch.Tensor:
    """
    Given a tuple of features (layer1, layer2, layer3, ...) from a network,
    flattens those features into a single vector per batch input. The
    features are also scaled such that the L2 distance between features
    for two different inputs is the LPIPS distance between those inputs.
    """

    normalized_features: List[torch.Tensor] = []
    for feature_layer in features:
        norm_factor = torch.sqrt(
            torch.sum(feature_layer ** 2, dim=1, keepdim=True)) + eps
        normalized_features.append(
            (feature_layer / (norm_factor *
                              np.sqrt(feature_layer.size()[2] *
                                      feature_layer.size()[3])))
            .view(feature_layer.size()[0], -1)
        )
    return torch.cat(normalized_features, dim=1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    """
    Copied from https://github.com/Po-Hsun-Su/pytorch-ssim
    """

    def __init__(self, window_size=11, size_average=True, dissimilarity=False):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.dissimilarity = dissimilarity

    def forward(self, imgs1, imgs2):
        (_, channel, _, _) = imgs1.size()

        if channel == self.channel and self.window.data.type() == imgs1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if imgs1.is_cuda:
                window = window.cuda(imgs1.get_device())
            window = window.type_as(imgs1)

            self.window = window
            self.channel = channel

        sim = torch.tensor([
            _ssim(img1[None], img2[None], window, self.window_size, channel, self.size_average)
            for img1, img2 in zip(imgs1, imgs2)
        ])
        return 1 - sim if self.dissimilarity else sim


class L2Distance(nn.Module):
    def forward(self, img1, img2):
        return (img1 - img2).reshape(img1.shape[0], -1).norm(dim=1)


class LinfDistance(nn.Module):
    def forward(self, img1, img2):
        return (img1 - img2).reshape(img1.shape[0], -1).abs().max(dim=1)[0]
