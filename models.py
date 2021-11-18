from utils.google_utils import *
from utils.layers import *
from utils.parse_config import *
import torch.nn as nn
from torchvision.transforms import GaussianBlur
import math
import numpy as np
from functools import cmp_to_key

ONNX_EXPORT = False


def create_modules(module_defs, img_size, cfg):
    # Constructs module list of layer blocks from module configuration in module_defs

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # expand if necessary
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef['pad'] else 0,
                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                       bias=not bn))
            else:  # multiple-size conv
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                          out_ch=filters,
                                                          k=k,
                                                          stride=stride,
                                                          bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())

        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:  # normalize RGB image
                # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef['type'] == 'maxpool':
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            stride = [32, 16, 8]  # P5, P4, P3 strides
            if any(x in cfg for x in ['panet', 'yolov4', 'cd53']):  # stride order reversed
                stride = list(reversed(stride))
            layers = mdef['from'] if 'from' in mdef else []
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
                                nc=mdef['classes'],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=layers,  # output layers
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = layers[yolo_index] if 'from' in mdef else -1
                # If previous layer is a dropout layer, get the one before
                if module_list[j].__class__.__name__ == 'Dropout':
                    j -= 1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                bias[:, 4] += -4.5  # obj
                bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        elif mdef['type'] == 'dropout':
            perc = float(mdef['probability'])
            modules = nn.Dropout(p=perc)

        elif mdef['type'] == 'sift':
            modules = SIFT()

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # index in layers, number of layers
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

            # outputs and weights
            # w = F.softmax(p[:, -n:], 1)  # normalized weights
            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # weighted ASFF sum
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * \
                         F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)
        self.yolo_layers = get_yolo_layers(self)
        # torch_utils.initialize_weights(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, augment=False, verbose=False):

        if not augment:
            return self.forward_once(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
            #     area = yi[..., 2:4].prod(2)[:, :, None]
            #     if i == 1:
            #         yi *= (area < 96. ** 2).float()
            #     elif i == 2:
            #         yi *= (area > 32. ** 2).float()
            #     y[i] = yi

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ''

        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:  # train
            return yolo_out
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            return x, p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


class SIFT(nn.Module):
    def __init__(self):
        super(SIFT, self).__init__()
        self.sigma = 1.6
        self.assumed_blur = 0.5
        self.num_intervals = 3
        self.extrema_thresh = math.floor(0.5 * 0.04 / self.num_intervals * 255)
        self.image_border_width = 5
        self.eigen_value_ratio = 10
        self.convergence_attempts = 5
        self.contrast_thresh = 0.04
        self.radius_factor = 3
        self.num_bins = 36
        self.peak_ratio = 0.8
        self.scale_factor = 1.5
        self.float_tolerance = 1e-7
        self.window_width = 4
        self.descriptor_bins = 8
        self.scale_multiplier = 3
        self.descriptor_max_value = 0.2

    def forward(self, input):
        # Input should be batched images
        # TODO Step 1: Scale Space
        # Upsample input images by 2 and gaussian blur
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        input_up = upsample(input)

        sigma = math.sqrt(max((self.sigma ** 2) - ((2 ** self.assumed_blur) ** 2), 0.01))
        blur = GaussianBlur(sigma=sigma, kernel_size=1)
        input_blur = blur(input_up)

        # TODO if this needs to be converted to int
        # Calc number of octaves in image pyramid
        num_octaves = torch.round(torch.log(torch.min(input_blur.size())) / math.log(2) - 1)

        # Calc list of gaussian kernels to blur input image
        num_images_per_octave = self.num_intervals + 3
        k = 2 ** (1.0 / self.num_intervals)

        gaus_kernels = torch.zeros(num_images_per_octave)  # To create image pyramid from base image
        gaus_kernels[0] = self.sigma

        for img_idx in range(1, num_images_per_octave):
            sigma_prev = (k ** (img_idx - 1)) * self.sigma
            sigma_total = k ** sigma_prev
            gaus_kernels[img_idx] = math.sqrt(sigma_total ** 2 - sigma_prev ** 2)

        # Generate Gaussian images of image pyramid
        gaussian_images = []
        image = input_blur
        for octave_idx in range(num_octaves):
            gauss_images_in_octave = []
            gauss_images_in_octave.append(image)
            for gaus_kernel in gaus_kernels[1:]:
                blur = GaussianBlur(sigma=gaus_kernel, kernel_size=1)
                image = blur(image)
                gauss_images_in_octave.append(image)
            gaussian_images.append(gauss_images_in_octave)
            octave_base = gauss_images_in_octave[-3]
            image = torch.reshape(octave_base, (octave_base.size()[1] / 2, octave_base.size()[0] / 2))  # TODO Confirm

        # Generate DOG Images
        dog_images = []

        for gauss_img_in_octave in gaussian_images:
            dog_images_in_octave = []
            for first_img, second_img in zip(gauss_img_in_octave, gauss_img_in_octave[1:]):
                sub_imgs = torch.subtract(second_img, first_img)
                dog_images_in_octave.append(sub_imgs)
            dog_images.append(dog_images_in_octave)

        # TODO Step 2: Keypoint Localization and Step 3: Orientation Assignment
        # FInd pixel positions of all scale-space extrema in pyramid
        keypoints = []

        for octave_idx, dog_images_in_octave in enumerate(dog_images):
            for img_idx, (first_img, second_img, third_img) in enumerate(zip(dog_images_in_octave,
                                                                             dog_images_in_octave[1:],
                                                                             dog_images_in_octave[2:])):
                # TODO Rework below for 3D colour images?
                for i in range(self.image_border_width, first_img.size()[0] - self.image_border_width):
                    for j in range(self.image_border_width, first_img.size()[1] - self.image_border_width):
                        if self.check_pixel_extrema(first_img[i-1:i+2, j-1:j+2],
                                                    second_img[i-1:i+2, j-1:j+2],
                                                    third_img[i-1:i+2, j-1:j+2]):
                            localization_result = self.localize_extrema(i, j, img_idx + 1, octave_idx,
                                                                        dog_images_in_octave)
                            if localization_result is not None:
                                keypoint, localized_img_idx = localization_result
                                orientated_keypoints = self.keypoints_with_orientation(keypoint,
                                                                                       octave_idx,
                                                                                       gaussian_images[octave_idx][localized_img_idx])
                                for keypoint_with_orientation in orientated_keypoints:
                                    keypoints.append(keypoint_with_orientation)

        # Remove duplicate keypoints and sort
        if len(keypoints) < 2:
            return keypoints

        keypoints.sort(key=cmp_to_key(self.compare_keypoints))
        unique_keypoints = [keypoints[0]]

        for next_keypoint in keypoints[1:]:
            last_unique_keypoint = unique_keypoints[-1]
            if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
                    last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
                    last_unique_keypoint.size != next_keypoint.size or \
                    last_unique_keypoint.angle != next_keypoint.angle:
                unique_keypoints.append(next_keypoint)

        # Convert keypoints to input image size
        converted_keypoints = []
        for keypoint in unique_keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)

        # TODO Step 4: Keypoint Descriptor
        # Create descriptors for each key point
        descriptors = []
        for keypoint in converted_keypoints:
            octave, layer, scale = self.unpack_octave(keypoint)
            gaussian_image = gaussian_images[octave + 1, layer]
            num_rows, num_cols = gaussian_image.size()
            point = round(scale * np.array(keypoint.pt)).astype('int')
            bins_per_degree = self.num_bins / 360.
            angle = 360.0 - keypoint.angle
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * self.window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            histogram_tensor = torch.zeros((self.window_width + 2, self.window_width + 2, self.descriptor_bins))

            hist_width = self.scale_multiplier * 0.5 * scale * keypoint.size
            half_width = int(round(hist_width * math.sqrt(2) * (self.window_width + 1) * 0.5))
            half_width = int(min(half_width, math.sqrt(num_rows ** 2 + num_cols ** 2)))

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * self.window_width - 0.5
                    col_bin  = (col_rot / hist_width) + 0.5 * self.window_width - 0.5
                    if row_bin > -1 and row_bin < self.window_width and col_bin > -1 and col_bin < self.window_width:
                        window_row = int(round(point[1] + row))
                        window_col = int(round(point[0] + col))
                        if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                            dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row -1, window_col] - gaussian_image[window_row + 1, window_col]
                            gradient_magnitude = math.sqrt(dx * dx + dy * dy)
                            gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                            weight = math.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                # Use trilinear interpolation for smoothing
                row_bin_floor, col_bin_floor, orientation_bin_floor = math.floor([row_bin, col_bin, orientation_bin]).astype(int)
                row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, \
                                                                   col_bin - col_bin_floor, \
                                                                   orientation_bin - orientation_bin_floor
                if orientation_bin_floor < 0:
                    orientation_bin_floor += self.descriptor_bins
                if orientation_bin_floor >= self.descriptor_bins:
                    orientation_bin_floor -= self.descriptor_bins

                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)

                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % self.descriptor_bins] += c001
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % self.descriptor_bins] += c011
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % self.descriptor_bins] += c101
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % self.descriptor_bins] += c111

            descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
            # Threshold and normalize descriptor_vector
            threshold = np.linalg.norm(descriptor_vector) * self.descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(np.linalg.norm(descriptor_vector), self.float_tolerance)
            # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
            descriptor_vector = round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)

        return np.array(descriptors, dtype='float32')

    def check_pixel_extrema(self, first_subimg, second_subimg, third_subimg):
        # Checks if center element of 3x3x3 array is greater than or less than all neighbours

        center_pixel_value = second_subimg[1, 1]
        if torch.abs(center_pixel_value) > self.extrema_thresh:
            if center_pixel_value > 0:
                # TODO Rework for colour images?
                result = all(center_pixel_value >= first_subimg) and all(center_pixel_value >= third_subimg) and \
                         all(center_pixel_value >= second_subimg[0, :]) and \
                         all(center_pixel_value >= second_subimg[2, :]) \
                         and center_pixel_value >= second_subimg[1, 0] and center_pixel_value >= second_subimg[1, 2]
                return result

            elif center_pixel_value < 0:
                result = all(center_pixel_value <= first_subimg) and all(center_pixel_value <= third_subimg) and \
                         all(center_pixel_value <= second_subimg[0, :]) and \
                         all(center_pixel_value <= second_subimg[2, :]) \
                         and center_pixel_value <= second_subimg[1, 0] and center_pixel_value <= second_subimg[1, 2]
                return result

        return False

    def localize_extrema(self, i, j, image_idx, octave_idx, dog_images_in_octave):
        # Use quadratic fit to refine pixel position
        extremum_outside_image = False
        image_shape = dog_images_in_octave[0].size()
        for attempt in range(self.convergence_attempts):
            first_img, second_img, third_img = dog_images_in_octave[image_idx-1:image_idx+2]
            # TODO Maybe torch stack?
            pixel_cube = np.stack([first_img[i-1:i+2, j-1:j+2],
                                   second_img[i-1:i+2, j-1:j+2],
                                   third_img[i-1:i+2, j-1:j+2]])
            gradient = self.compute_center_pix_gradient(pixel_cube)
            hessian = self.compute_center_pix_hessian(pixel_cube)

            extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
                break

            j += int(round(extremum_update[0]))
            i += int(round(extremum_update[1]))
            image_idx += int(round(extremum_update[2]))

            if i < self.image_border_width or \
                    i >= image_shape[0] - self.image_border_width or \
                    j < self.image_border_width or \
                    j >= image_shape[1] - self.image_border_width or \
                    image_idx < 1 or \
                    image_idx > self.num_intervals:
                extremum_outside_image = True
                break

            if extremum_outside_image:
                return None
            if attempt >= self.convergence_attempts - 1:
                return None

            value_at_updated_extrenum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
            if abs(value_at_updated_extrenum) * self.num_intervals >= self.contrast_thresh:
                xy_hessian = hessian[:2, :2]
                xy_hessian_trace = np.trace(xy_hessian)
                xy_hessian_det = np.linalg.det(xy_hessian)
                # Contrast Check
                if xy_hessian_det > 0 and \
                        self.eigen_value_ratio * (xy_hessian_trace ** 2) < ((self.eigen_value_ratio + 1) ** 2) * xy_hessian_det:
                    keypoint = KeyPoint()
                    keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_idx),
                                   (i + extremum_update[1]) * (2 ** octave_idx))
                    keypoint.octave = octave_idx + image_idx * (2 ** 8) + \
                                      int(round(extremum_update[2] + 0.5 * 255)) * (2 ** 16)
                    keypoint.size = self.sigma * (2 ** ((image_idx + extremum_update[2]) / np.float32(self.num_intervals))) * (2 ** (octave_idx + 1))
                    keypoint.response = abs(value_at_updated_extrenum)

                    return keypoint, image_idx
        return None

    def compute_center_pix_gradient(self, cube):
        # Gradient at center pixel [1,1,1] of 3x3x3 array using central difference formula
        dx = 0.5 * (cube[1, 1, 2] - cube[1, 1, 0])
        dy = 0.5 * (cube[1, 2, 1] - cube[1, 0, 1])
        ds = 0.5 * (cube[2, 1, 1] - cube[0, 1, 1])
        gradient = np.array([dx, dy, ds])

        return gradient

    def compute_center_pix_hessian(self, cube):
        # Hessian at center pixel [1,1,1] of 3x3x3 array using central difference formula
        center_pix_value = cube[1, 1, 1]
        dxx = cube[1, 1, 2] - 2 * center_pix_value + cube[1, 1, 0]
        dyy = cube[1, 2, 1] - 2 * center_pix_value + cube[1, 0, 1]
        dss = cube[2, 1, 1] - 2 * center_pix_value + cube[0, 1, 1]
        dxy = 0.25 * (cube[1, 2, 2] - cube[1, 2, 0] - cube[1, 0, 2] + cube[1, 0, 0])
        dxs = 0.25 * (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 0])
        dys = 0.25 * (cube[2, 2, 1] - cube[2, 0, 1] - cube[0, 2, 1] + cube[0, 0, 1])
        hessian = np.array([[dxx, dxy, dxs],
                            [dxy, dyy, dys],
                            [dxs, dys, dss]])

        return hessian

    def keypoints_with_orientation(self, keypoint, octave_idx, gaussian_img):
        # Compute Orientation for each keypoint
        keypoints_with_orientations = []
        image_shape = gaussian_img.size()

        scale = self.scale_factor * keypoint.size / np.float32(2 ** (octave_idx + 1))
        radius = int(round(self.radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        raw_histo = torch.zeros(self.num_bins)
        smooth_histo = torch.zeros(self.num_bins)

        for i in range(-radius, radius + 1):
            region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_idx))) + i
            if region_y > 0 and region_y < image_shape[0] - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_idx))) + j
                    if region_x > 0 and region_x < image_shape[1] - 1:
                        dx = gaussian_img[region_y, region_x + 1] - gaussian_img[region_y, region_x - 1]
                        dy = gaussian_img[region_y - 1, region_x] - gaussian_img[region_y + 1, region_x]
                        gradient_magnitude = math.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                        weight = math.exp(weight_factor * (i ** 2 + j ** 2))
                        histogram_idx = int(round(gradient_orientation * self.num_bins / 360.0))
                        raw_histo[histogram_idx % self.num_bins] += weight * gradient_magnitude

        for n in range(self.num_bins):
            smooth_histo[n] = (6 * raw_histo[n] + 4 * (raw_histo[n - 1] + raw_histo[(n + 1) % self.num_bins]) +
                               raw_histo[n - 2] + raw_histo[(n + 2) % self.num_bins])
        orientation_max = max(smooth_histo)
        orientation_peaks = np.where(np.logical_and(smooth_histo > np.roll(smooth_histo, 1),
                                                    smooth_histo > np.roll(smooth_histo, -1)))[0]
        for peak_idx in orientation_peaks:
            peak_value = smooth_histo[peak_idx]
            if peak_value >= self.peak_ratio * orientation_max:
                # Quadratic peak interpolation
                left_value = smooth_histo[(peak_idx - 1) % self.num_bins]
                right_value = smooth_histo[(peak_idx + 1) % self.num_bins]
                interpolated_peak_idx = (peak_idx + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % self.num_bins
                orientation = 360.0 - interpolated_peak_idx * 360.0 / self.num_bins
                if abs(orientation - 360.0) < self.float_tolerance:
                    orientation = 0
                new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                keypoints_with_orientations.append(new_keypoint)

        return keypoints_with_orientations

    def compare_keypoints(self, key1, key2):
        # Checks if keypoint 1 is less than keypoint 2
        if key1.pt[0] != key2.pt[0]:
            return key1.pt[0] - key2.pt[0]
        if key1.pt[1] != key2.pt[1]:
            return key1.pt[1] - key2.pt[1]
        if key1.size != key2.size:
            return key2.size - key1.size
        if key1.angle != key2.angle:
            return key1.angle - key2.angle
        if key1.response != key2.response:
            return key2.response - key1.response
        if key1.octave != key2.octave:
            return key2.octave - key1.octave

        return key2.class_id - key1.class_id

    def unpack_octave(self, key):
        octave = key.octave & 255
        layer = (key.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)

        return octave, layer, scale


class KeyPoint:
    def __init__(self, pt=None, size=None, angle=None, response=None, octave=None, class_id=None):
        self.pt = pt
        self.size = size
        self.angle = angle
        self.response = response
        self.octave = octave
        self.class_id = class_id


class HOG(nn.Module):
    def __init__(self):
        super(HOG, self).__init__()
        self.patch_size = 1
        self.num_ang_bins = 1
        self.num_space_bins = 1
        self.clipval = 1

    def forward(self, input):

        return


def get_yolo_layers(model):
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        target = weights.rsplit('.', 1)[0] + '.weights'
        save_weights(model, path=target, cutoff=-1)
        print("Success: converted '%s' to '%s'" % (weights, target))

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        target = weights.rsplit('.', 1)[0] + '.pt'
        torch.save(chkpt, target)
        print("Success: converted '%s' to '%s'" % (weights, target))

    else:
        print('Error: extension not supported.')


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip().replace("'", '')
    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if len(weights) > 0 and not os.path.isfile(weights):
        d = {'yolov3-spp.weights': '16lYS4bcIdM2HdmyJBVDOvt3Trx6N3W2R',
             'yolov3.weights': '1uTlyDWlnaqXcsKOktP5aH_zRDbfcDp-y',
             'yolov3-tiny.weights': '1CCF-iNIIkYesIDzaPvdwlcf7H9zSsKZQ',
             'yolov3-spp.pt': '1f6Ovy3BSq2wYq4UfvFUpxJFNDFfrIDcR',
             'yolov3.pt': '1SHNFyoe5Ni8DajDNEqgB2oVKBb_NoEad',
             'yolov3-tiny.pt': '10m_3MlpQwRtZetQxtksm9jqHrPTHZ6vo',
             'darknet53.conv.74': '1WUVBid-XuoUBmvzBVUCBl_ELrzqwA8dJ',
             'yolov3-tiny.conv.15': '1Bw0kCpplxUqyRYAJr9RY9SGnOJbo9nEj',
             'yolov3-spp-ultralytics.pt': '1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4'}

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:  # download from pjreddie.com
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        # Error check
        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)
