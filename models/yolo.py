# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

### nn.Module用法详解：https://blog.csdn.net/qq_27825451/article/details/90550890

class Detect(nn.Module):
    """Detect模块是用来构建Detect层的，将输入feature map 通过一个卷积操作和公式计算到我们想要的shape, 为后面的计算损失或者NMS作准备"""
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    ### __init__函数在parse_model函数中调用：
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        """
        detection layer 相当于yolov3中的YOLOLayer层
        :params nc: 类别数量
        :params anchors: 传入3个feature map上的所有anchor的大小（P3、P4、P5）
        :params ch: [128, 256, 512],3个输出feature map的channel
        """
        ### 对feature map的channel的理解：
        ###一种卷积核得到一个通道，所以特征图个数=输出通道数=卷积核个数。
        ###这样看来，图片中的通道就是某种意义上的特征图。
        ###一个通道是对某个特征的检测，通道中某一处数值的强弱就是对当前特征强弱的反应 。
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers  #Detect的个数 3
        self.na = len(anchors[0]) // 2  # number of anchors  #每个feature map的anchor个数 3
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid  #{list: 3}  tensor([0.]) X 3
        # self.anchor_grid的维度为[3, 3, 2]  anchors以[w, h]对的形式存储 ，3个feature map，每个feature map上有三个anchor（w,h）
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # 对每个输出的feature map都要调用一次conv1x1
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # 一般都是True 默认不使用AWS Inferentia加速
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    ### forward函数在BaseModel类的forward_once中调用：
    def forward(self, x):
        """
        :return     train: 一个tensor list,存放三个元素,每个元素均为[bs, anchor_num, grid_w, grid_h, xywh+confidence+classes]
        
                    inference: 0 [1, 3*80*80+3*40*40+3*20*20 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+classes]
                               1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+confidece+classes]
        """
        z = []  # inference output
        # 对三个feature map分别进行处理
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv  xi[bs, 128/256/512, 80, 80] to [bs, 75, 80, 80]
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # [bs, 255, 80, 80] to [1, 3, 85, 80, 80] to [1, 3, 80, 80, 85]
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                # 构造网格
                # 因为推理返回的不是归一化后的网格偏移量 需要再加上网格的位置 得到最终的推理坐标 再送入nms
                # 所以这里构建网格就是为了纪录每个grid的网格坐标 方面后面使用
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                #如果是分割模型
                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                #如果是检测或者分类模型
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                 # z是一个tensor list，三个元素，分别是[1, 19200, 25]、[1, 4800, 25]、[1, 1200, 25]
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        """
        构造网格
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    # 定义一次前向传播
    # 前向传播示意图：https://img-blog.csdnimg.cn/20210312191719915.png
    def _forward_once(self, x, profile=False, visualize=False):
        """
        :params x: 输入图像
        :params profile: True 可以做一些性能评估
        :params visualize: True 可以做一些特征可视化
        :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+classes_num]
                       维度分别是 [bs, 3, 32, 32, 85] [bs, 3, 16, 16, 85] [bs, 3, 8, 8, 85]
                inference: 0 [bs, anchor_num*grid_w*grid_h, xywh+c+classes_num]
                           1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+classes_num]
                             
        """
        # y: 存放着self.save=True的每一层的输出，因为后面的层结构concat等操作要用到
        # dt: 在profile中做性能评估时使用
        y, dt = [], []  # outputs
        for m in self.model:
            #print(m.i)
            # 前向推理每一层结构  m.i=index   m.f=from   m.type=类名   m.np=number of params
            if m.f != -1:  # if not from previous layer
                # 这里需要做4个concat操作和1个Detect操作
                # concat操作：如m.f=[-1, 6] x就有两个元素,一个是上一层的输出,另一个是index=6的层的输出 再送到x=m(x)做concat操作
                # Detect操作：如m.f=[17, 20, 23] x有三个元素,分别存放第17层第20层第23层的输出 再送到x=m(x)做Detect的forward
                # 如果该层的数据来源m.f是-1，就保留x为上一层的输出，否则，就从y中取出对应层的输出结果。
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # 打印日志信息  FLOPs time等
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run #正向推理 
            # # 存放着self.save的每一层的输出，因为后面需要用来作concat等操作要用到  不在self.save层的输出就为None
            y.append(x if m.i in self.save else None)  # save output
            # 特征可视化 可以自己改动想要哪层的特征进行可视化
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  
        """用在detect.py、val.py
        fuse model Conv2d() + BatchNorm2d() layers
        调用torch_utils.py中的fuse_conv_and_bn函数和common.py中Conv模块的fuseforward函数
        """
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            # 如果当前层是卷积层Conv且有bn结构, 那么就调用fuse_conv_and_bn函数讲conv和bn进行融合, 加速推理
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv #融合
                delattr(m, 'bn')  # remove batchnorm #移除BN
                # 更新前向传播（反向传播不用管，因为这种推理只用在推理阶段）
                m.forward = m.forward_fuse  # update forward
        self.info() # 打印conv+bn融合后的模型信息
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """
        :params cfg:模型配置文件
        :params ch: input img channels 一般是3(RGB文件)
        :params nc: number of classes 数据集的类别个数
        :anchors: 一般是None
        """
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml(一般执行这里)
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            # 如果配置文件中有中文，打开时要加ascii参数
            with open(cfg, encoding='ascii', errors='ignore') as f:
                #取到配置文件中每条的信息（没有注释内容）
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        # ch=3
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        # 设置类别数 一般不执行, 因为nc=self.yaml['nc']恒成立
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors: # 重写anchor，一般不执行, 因为传进来的anchors一般都是None
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        
        # 创建网络模型
        # self.model: 初始化的整个网络模型(包括Detect层结构)
        # self.save: 所有层结构中from不等于-1的序号，并排好序  [4, 6, 10, 14, 17, 20, 23]
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default class names ['0', '1', '2',..., '19']
        self.inplace = self.yaml.get('inplace', True)  # self.inplace默认True,不使用加速推理

        # 获取Detect模块的stride(相对输入图像的下采样率)和anchors在当前Detect输出的feature map的尺度
        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)           
            # 计算三个feature map下采样的倍率  [8, 16, 32]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)  # 检查anchor顺序与stride顺序是否一致
            # 求出相对当前feature map的anchor大小 如[10, 13]/8 -> [1.25, 1.625]
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once 初始化偏置
        ## 在某些时候，我们可能希望模型中的某些参数参数不更新，但又希望参数保存下来，这时我们就会用到register_buffer()
        ## register_buffer函数详解：https://blog.csdn.net/weixin_46197934/article/details/119518497
        self.register_buffer('pre_features0', torch.zeros(16, 128)) 
        self.register_buffer('pre_features1', torch.zeros(16, 256)) 
        self.register_buffer('pre_features2', torch.zeros(16, 512)) 
        self.register_buffer('pre_weight0', torch.ones(16, 1))
        self.register_buffer('pre_weight1', torch.ones(16, 1))
        self.register_buffer('pre_weight2', torch.ones(16, 1))
        # torch.zeros()函数返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor
        # 下面两句中的pre_features和pre_weight1分别是之前训练得到的，在当前训练保持不变的全局特征和全局样本权重
        # Init weights, biases
        initialize_weights(self) # 调用torch_utils.py下initialize_weights初始化模型权重
        self.info()  # 打印模型信息
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            x=self._forward_augment(x)  # augmented inference, None
        else:
            x= self._forward_once(x, profile, visualize) # single-scale inference, train
        return x

    def _forward_augment(self, x):
        """
        TTA(Test Time Augmentation)，是否在测试的时候也用数据增强
        """
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            # scale_img缩放图片尺寸
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi= self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            # _descale_pred将推理结果恢复到相对原图图片尺寸
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1),None  # augmented inference, train

       #### 新增加的函数，提取三个特征图
    def get_feature(self,x):
        ##### 新增加的池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #####
        y = [] # outputs
        for m in self.model:
            #print(m.i)
            # 前向推理每一层结构  m.i=index   m.f=from   m.type=类名   m.np=number of params
            if m.f != -1:  # if not from previous layer
                # 这里需要做4个concat操作和1个Detect操作
                # concat操作：如m.f=[-1, 6] x就有两个元素,一个是上一层的输出,另一个是index=6的层的输出 再送到x=m(x)做concat操作
                # Detect操作：如m.f=[17, 20, 23] x有三个元素,分别存放第17层第20层第23层的输出 再送到x=m(x)做Detect的forward
                # 如果该层的数据来源m.f是-1，就保留x为上一层的输出，否则，就从y中取出对应层的输出结果。
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if m.i == 17:
                flatten_features0=self.avgpool(m(x))
                #print(flatten_features0.size()) #torch.Size([16, 128, 1, 1])
                flatten_features0=torch.flatten(flatten_features0,1) # torch.Size([16,819200])
                #print(flatten_features0.size())
                x = m(x)
            elif m.i == 20:
                flatten_features1=self.avgpool(m(x))
                flatten_features1=torch.flatten(flatten_features1,1) # torch.Size([bs, 409600])
                x = m(x)
            elif m.i == 23:
                flatten_features2=self.avgpool(m(x))
                flatten_features2=torch.flatten(flatten_features2,1) # torch.Size([bs, 204800])
                x = m(x)
            else:
                x = m(x)
            ####
            #x = m(x)  # run #正向推理 
            # # 存放着self.save的每一层的输出，因为后面需要用来作concat等操作要用到  不在self.save层的输出就为None
            y.append(x if m.i in self.save else None)  # save output
            # 特征可视化 可以自己改动想要哪层的特征进行可视化
        return flatten_features0,flatten_features1,flatten_features2

    def _descale_pred(self, p, flips, scale, img_size):
        """用在上面的__init__函数上
        将推理结果恢复到原图图片尺寸  Test Time Augmentation(TTA)中用到
        de-scale predictions following augmented inference (inverse operation)
        :params p: 推理结果
        :params flips:
        :params scale:
        :params img_size:
        """
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace: # 默认执行 不使用AWS Inferentia
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None): 
        """用在上面的__init__函数上
        initialize biases into Detect(), cf is class frequency
        https://arxiv.org/abs/1708.02002 section 3.3
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        # 下面这句将yaml文件中定义的每一层的数据来源保存到该层的对象中
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=5, help='total batch size for all GPUs')
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    #print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device) 
    print(im.size()) # torch.Size([5, 3, 640, 640])
    model = Model(opt.cfg).to(device)
    pred=model(im)
    print(pred[2].size()) # torch.Size([5, 3, 20, 20, 85])
    #print(cfeatures0.size())
    #print(cfeatures1.size())
    #print(cfeatures2.size())
    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
