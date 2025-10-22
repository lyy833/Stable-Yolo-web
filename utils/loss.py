# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

##### 新增加的函数功能，用于计算一个batch中单独每张image的分类损失
def compute_img_loss(b,loss,batch_size,device):
    num=b.size(0)
    img_loss=torch.zeros(batch_size,device=device)
    for i in range(batch_size):
        for j in range(num):
            if b[j] !=i:
                j=j+1
            else:
                img_loss[i]=img_loss[i]+loss[j]
    return img_loss

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False # 是否根据iou排序

    # Compute losses 计算损失
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device  获得模型的设备
        h = model.hyp  # hyperparameters 获取超参数

        '''
		定义分类损失和置信度损失为带sigmoid的二值交叉熵损失，
		即会先将输入进行sigmoid再计算BinaryCrossEntropyLoss(BCELoss)。
		pos_weight参数是正样本损失的权重参数。
		'''
        ### Define criteria 定义损失的criteria
        # 用于分类的损失criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device),reduction='none')
        # 用于置信度的损失criteria
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        '''
		对标签做平滑,eps=0就代表不做标签平滑,那么默认cp=1,cn=0
        后续对正类别赋值cp，负类别赋值cn
		'''
        # 类别标签平滑
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        '''
		超参设置g>0则计算FocalLoss
		'''
        # Focal loss 
        # g默认为0，因此默认这块儿不执行
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        
        # 获取detect层
        m = de_parallel(model).model[-1]  # Detect() module
        '''
        每一层预测值所占的权重比，分别代表浅层到深层，小特征到大特征，4.0对应着P3，1.0对应P4,0.4对应P5。
        如果是自己设置的输出不是3层，则返回[4.0, 1.0, 0.25, 0.06, .02]，可对应1-5个输出层P3-P7的情况。
        '''
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        '''
        autobalance 默认为 False，yolov5中目前也没有使用，ssi = 0即可
        '''
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        '''
        赋值各种参数,gr是用来设置IoU的值在objectness loss中做标签的系数, 
        使用代码如下：
		tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
        train.py源码中model.gr=1，也就是说完全使用标签框与预测框的CIoU值来作为该预测框的objectness标签。
        '''
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        #####新增加的compute_loss类的属性

    def __call__(self, p, targets,weight0,weight1,weight2,bs):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        
        '''
        从build_targets函数中构建目标标签，获取标签中的tcls, tbox, indices, anchors
        tbox = [[[gx1,gy1,gw1,gh1],[gx2,gy2,gw2,gh2],...],
        
        indices = [[image indices1,anchor indices1,gridj1,gridi1],
        		   [image indices2,anchor indices2,gridj2,gridi2],
        		   ...]]
        anchors = [[aw1,ah1],[aw2,ah2],...]		  
        '''
        # tcls是一个list,分别保存3个预测层对应的类别标签，每个层对应的类别标签是一个一维张量，三个层的类别标签数目不同
        #tcls = [[cls1,cls2,...clsN1],[cls1,cls2,...,clsN2],[cls1,cls2,...,clsN3]]
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses

        '''
		p.shape = [nl,bs,na,nx,ny,no]
		nl 为 预测层数，一般为3
		na 为 每层预测层的anchor数，一般为3
		nx,ny 为 grid的w和h
		no 为 输出数，为5 + nc (5:x,y,w,h,obj,nc:分类数)
		'''
        # i和pi分别代表层的index和预测
        for i, pi in enumerate(p):  # layer index, layer predictions
            '''
            a:所有anchor的索引
            b:标签所属image的索引
            gridy:标签所在grid的y，在0到ny-1之间
            gridy:标签所在grid的x，在0到nx-1之间
            '''
            # 这里得到的b是一个n维张量，n为正样本数
            # 不同的i对应三个不同的预测层，不同的预测层中正样本数不同
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            '''
            pi.shape = [bs,na,nx,ny,no]
            tobj.shape = [bs,na,nx,ny]
            '''
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets  正样本数
            if n:
                '''
            	pi[b, a, gj, gi]为batch中第b个图像第a个anchor的第gj行第gi列的output
            	pi[b, a, gj, gi].shape = [N,5+nc],N = a[0].shape，即符合anchor大小的所有标签数
            	'''
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                
                # pcls.shape=[n,nc],
                # 其中n为当前预测层的预测的类别数，
                # 这里令这个数和标签中的正样本数相同，这样才能进行后面的分类损失计算
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
                '''
				xy的预测范围为-0.5~1.5
                wh的预测范围是0~4倍anchor的w和h，
				'''
                # Regression   anchor回归
                # 因为在build_targets中，作者相当于扩充了标签框所在网格的上下左右４个网格，
                # 左上网格的中心点偏移了－0.5，右下网格偏移了0.5，所以在中心坐标回归到的时候限制了回归的范围到偏移的网格范围内。
                # yolov4中sigmoid(tx)+cx的范围是cx-1,cx+1,当tx=0时，cx+0.5，认为中心点偏移的范围在相邻两个网格内
                # 当前方法范围是1.5+cx, cx-0.5, 当tx=0时，cx=0.5+cx，认为中心点坐标的偏移的最大范围就是扩充正样本时的范围，
                # 左上角网格-0.5,右下角网格+0.5,所以总的范围就是-0.5 ~ 1.5
                pxy = pxy.sigmoid() * 2 - 0.5
                # w,h 回归没有采用exp操作，而是直接乘上anchors[i]。
                # yolov4中求回归框的长和高的时候，直接对tw做指数操作保证缩放的系数大于0,但是宽度和高度的最大值完全不受限制，
                # 这种指数的运算很危险，因为它可能导致失控的梯度、不稳定、NaN损失并最终完全失去训练。
                # 所以yolov5对w,h 回归其没有采用exp操作，而是用sigmoid来限制了缩放系数的最大值。因为作者在YOLO5有一个超参数为anchor_t，就是4。
                # 该超参数的使用方法是，在训练时，如果真实框与锚框尺寸的比值大于4，限于预测框回归公式上界，该锚框是没有办法与该真实框重合的，
                # 所以训练时会把比值大于4的锚框删除掉。
                # 作者认为回归框和anchor的最大比值是４，所以将缩放系数的最大值设为４=2**2
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                # 找到和targets中相同图片相同anchor,相同网格的预测框，回归后的集合pbox
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                '''
                只有当CIOU=True时，才计算CIOU，否则默认为GIOU
                '''
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness

                # 获取target所对应的obj,网格中存在gt目标的会被标记为iou与gt的交并比  gr定义在train.py  model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)

                # torch.clamp(input, min, max, out=None) 限幅。将input的值限制在[min, max]之间，并返回结果
                # torch.tensor.type该方法的功能是:当不指定dtype时,返回类型.当指定dtype时,返回类型转换后的数据,如果类型已经符合要求,那么不做额外的复制,返回原对象.
                # iou.detach().clamp(0)将iou中所有的小于0的置0，认为小于0的CIOU是负样本

                # 给正样本的tobj赋初值，初值里用到了iou取代1，代表该点对应置信度，负样本（包括背景）的置信度为0
                # 引入了大量正样本anchor，但是不同anchor和gt bbox匹配度是不一样，预测框和gt bbox的匹配度也不一样，
                # 如果权重设置一样肯定不是最优的，故作者将预测框和bbox的iou作为权重乘到conf分支，用于表征预测质量。

                # 一般检测网络的分类头，在计算loss阶段，标签往往是非0即1的状态，即是否为当前类别。
                # yolov5则是将anchor与目标匹配时的iou作为该位置样本的标签值。iou值在0-1之间，label值的缩小导致了最后预测结果值偏小。
                # 通过model.gr可以修改iou值所占权重，默认是1.0，即用iou值完全作为标签值,而不是非0即1。iou的最大值为1
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                bs0 = tobj.shape[0]  # batch size
                if self.nc > 1:  # cls loss (only if multiple classes)
                    '''
               		ps[:, 5:].shape = [N,nc],用 self.cn 来填充型为[N,nc]得Tensor。
               		self.cn通过smooth_BCE平滑标签得到的，使得负样本不再是0，而是0.5 * eps
                	'''
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    '''
                    self.cp 是通过smooth_BCE平滑标签得到的，使得正样本不再是1，而是1.0 - 0.5 * eps
                    '''
                    t[range(n), tcls[i]] = self.cp

                    #################### 增加的代码
                    if i == 0:
                        weight=weight0
                    elif i== 1:
                        weight=weight1
                    else:
                        weight=weight2
                    b= b.long()
                    b= b-1
                    loss_cls=self.BCEcls(pcls, t) # 计算用sigmoid+BCE分类损失
                    loss_cls=loss_cls.sum(dim=1, keepdim=True)
                    # loss_cls=loss_cls.div_(self.nc)
                    loss_cls=loss_cls.view(-1)# 将损失转换成一维张量，并且将b中的索引还原回来
                    #print(loss_cls)
                    #print(b)
                    img_loss=compute_img_loss(b,loss_cls,bs,device=self.device)
                    img_loss = img_loss.div_(self.nc)
                    #print(img_loss)   # 是一个1x16的张量
                    #print(weight)       #是一个16x1的张量
                    img_weight_loss = torch.matmul(img_loss,weight)
                    # img_weight_loss = img_weight_loss / bs
                    img_weight_loss = img_weight_loss / bs0
                    # print(img_weight_loss)
                    lcls +=  img_weight_loss
                    # lcls += self.BCEcls(pcls, t)  # BCE
                    #print(lcls.size())
                    b = b+1
                    b = b.float()

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            '''
            pi[..., 4]所存储的是预测的obj
			self.balance[i]为第i层输出层所占的权重，在init函数中已介绍
			将每层的损失乘上权重计算得到obj损失
			'''
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        '''
        hyp.yaml中设置了每种损失所占比重，分别对应相乘
        '''
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        # bs = tobj.shape[0]  # batch size
        # 最后计算得到三个层的一个总损失，用于更新梯度
        #return (lbox + lobj+lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
        return (lbox + lobj+lcls) * bs0, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):  # p表示预测值
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        '''
        na = 3,表示每个预测层anchors的个数
        targets 为一个batch中所有的标签，包括标签所属的image，以及class,x,y,w,h
        targets = [[image1,class1,x1,y1,w1,h1],
        		   [image2,class2,x2,y2,w2,h2],
        		   ...
        		   [imageN,classN,xN,yN,wN,hN]]
        nt为一个batch中所有标签的数量
        '''
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        '''
        gain是为了最终将坐标所属grid坐标限制在坐标系内，不要超出范围,
        其中7是为了对应: image class x y w h ai,
        但后续代码只对x y w h赋值，x,y,w,h = nx,ny,nx,ny,
        nx和ny为当前输出层的grid大小。
        '''        
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        '''
        ai.shape = [na,nt]
        ai = [[0,0,0,.....],
        	  [1,1,1,...],
        	  [2,2,2,...]]
        这么做的目的是为了给targets增加一个属性，即当前标签所属的anchor索引
        '''
        # ai表示当前bbox和当前层哪个anchor匹配
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        '''
        targets.repeat(na, 1, 1).shape = [na,nt,6]
        ai[:, :, None].shape = [na,nt,1](None在list中的作用就是在插入维度1)
        ai[:, :, None] = [[[0],[0],[0],.....],
        	  			  [[1],[1],[1],...],
        	  	  		  [[2],[2],[2],...]]
        cat之后：
        targets.shape = [na,nt,7]
        重点：targets = [[[image1,class1,x1,y1,w1,h1,0],
        			[image2,class2,x2,y2,w2,h2,0],
        			...
        			[imageN,classN,xN,yN,wN,hN,0]],
        			[[image1,class1,x1,y1,w1,h1,1],
        			 [image2,class2,x2,y2,w2,h2,1],
        			...],
        			[[image1,class1,x1,y1,w1,h1,2],
        			 [image2,class2,x2,y2,w2,h2,2],
        			...]]
        这么做是为了纪录每个label对应的anchor。
        '''
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        
        '''
		定义每个grid(网格)偏移量，会根据标签在grid中的相对位置来进行偏移
		'''
        g = 0.5  # bias
        
        '''
        [0, 0]代表中间,
		[1, 0] * g = [0.5, 0]代表往左偏移半个grid， [0, 1]*0.5 = [0, 0.5]代表往上偏移半个grid，与后面代码的j,k对应
		[-1, 0] * g = [-0.5, 0]代代表往右偏移半个grid， [0, -1]*0.5 = [0, -0.5]代表往下偏移半个grid，与后面代码的l,m对应
		具体原理在代码后讲述
        '''
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl): # self.nl是模型的layers数目
            #####
            '''
        	原本yaml中加载的anchors.shape = [3,6],但在yolo.py的Detect中已经通过代码
        	a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        	self.register_buffer('anchors', a) 
        	将anchors进行了reshape。

        	self.anchors.shape = [3,3,2]
        	anchors.shape = [3,2]

            p是一个list,p.shape = [nl,bs,na,nx,ny,no]
            p[i].shape = [bs,na,nx,ny,no]
        	'''
            anchors, shape = self.anchors[i], p[i].shape
            ######
            '''
            gain = [1,1,nx,ny,nx,ny,1]
            '''
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            ##### Match targets to anchors
            #因为targets进行了归一化，默认在w = 1, h =1 的坐标系中，
            #需要将其映射到当前输出层w = nx, h = ny的坐标系中。
            t = targets * gain  # shape(3,n,7)

            #######
            if nt: # 如果分层检测的特征层上有目标
                ##### Matches
                '''
                t[:, :, 4:6].shape = [na,nt,2] = [3,nt,2],存放的是标签的w和h
                anchor[:,None] = [3,1,2]
                r.shape = [3,nt,2],存放的是标签和当前层anchor的长宽比
                '''
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                #######
                '''
                torch.max(r, 1. / r)求出最大的宽比和最大的长比，shape = [3,nt,2],再max(2)求出同一标签中宽比和长比较大的一个，shape = [2，3,nt],之所以第一个维度变成2，
                因为torch.max如果不是比较两个tensor的大小，而是比较1个tensor某一维度的大小，则会返回values和indices：
                	torch.return_types.max(
						values=tensor([...]),
						indices=tensor([...]))
                所以还需要加上索引0获取values，torch.max(r, 1. / r).max(2)[0].shape = [3,nt],将其和hyp.yaml中的anchor_t超参比较，小于该值则认为标签属于当前输出层的anchor
                
                j是一个list,有三个分量，每一个分量对应当前层的一个layer
                j = [[bool,bool,....],[bool,bool,...],[bool,bool,...]]
                j.shape = [3,nt]
                '''
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                
                #######
                '''
                t.shape = [na,nt,7]=[3,nt,7] 
                j.shape = [3,nt]
                假设j中有NTrue个True值，则
                t[j].shape = [NTrue,7]
                返回的是所有属于当前层anchor的标签。
                
                三个不同的layer中NTure数目不相同。
                '''
                t = t[j]  # filter

                # Offsets
                #######
                '''
                t.shape = [NTrue,7] ,其中7:image,class,x,y,h,w,ai
                gxy.shape = [NTrue,2] 存放的是x,y,相当于坐标到坐标系左边框和上边框的记录
                gxi.shape = [NTrue,2] 存放的是w-x,h-y,相当于测量坐标到坐标系右边框和下边框的距离
                '''
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse

                #######
                '''
                因为grid单位为1，共nx*ny个gird,gxy % 1相当于求得标签在第gxy.long()个grid中以grid左上角为原点的相对坐标，
                gxi % 1相当于求得标签在第gxy.long()个grid中以grid右下角为原点的相对坐标，下面这两行代码作用在于:
                1、筛选中心坐标 左、上方偏移量小于0.5,并且中心点大于1的标签
                2、筛选中心坐标 右、下方偏移量小于0.5,并且中心点大于1的标签          
                
                j.shape = [NTrue], j = [bool,bool,...]
                k.shape = [NTrue], k = [bool,bool,...]
                l.shape = [NTrue], l = [bool,bool,...]
                m.shape = [NTrue], m = [bool,bool,...]
                
                j,k,l,m都是一维张量且的维度是相同的，这个维度数与t[j]的第一个维度数相同。
                '''
                j, k = ((gxy % 1 < g) & (gxy > 1)).T # 筛选中心坐标 左、上方偏移量小于0.5,并且中心点大于1的标签
                l, m = ((gxi % 1 < g) & (gxi > 1)).T #筛选中心坐标 右、下方偏移量小于0.5,并且中心点大于1的标签 
                
                #######
                '''
                这段代码使用了PyTorch的torch.stack()函数来将5个张量沿着一个新的维度(默认情况下是第 0 维)进行堆叠，并返回一个新的张量。
                在该代码中，j是一个PyTorch张量，而k, l,m分别是三个其他张量。
                代码中的 torch.ones_like(j)将返回与j张量形状相同的张量，其中所有元素均为 1,
                因此，代码将创建包含这5个张量(5个张量shape相同)的5D张量
                
                j.shape = [5,NTrue]
                t.repeat之后shape为[5,NTrue,7], 通过索引j后t.shape = [NOff,7],NOff表示NTrue + (j,k,l,m中True的总数量)。
                ''' 
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]

                #######
                '''
                torch.zeros_like(gxy)[None].shape = [1,NTrue,2]
                off[:, None].shape = [5,1,2]
                相加之和shape = [5,NTrue,2]
                通过索引j后offsets.shape = [NOff,2]
                这段代码的表示当标签在grid左侧半部分时，会将标签往左偏移0.5个grid，上下右同理
                '''
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            '''
            t.shape = [NOff,7],(image,class,x,y,w,h,ai)
            
            这段代码使用了PyTorch的chunk()函数将t分割成4个张量。具体来说,它将t沿着第1个维度分成4个形状相等的块,即:
                bc:包含 t 中的前两列，形状为 [NOff, 2]
                gxy:包含 t 中的第 3、4 列，形状为 [NOff, 2]
                gwh:包含 t 中的第 5、6 列，形状为 [NOff, 2]
                a:包含 t 中的最后一列，形状为 [NOff, 1]
            接下来，代码将对新创建的张量进行进一步操作：
                a.long().view(-1):将a张量转换为整数类型(long)并展平为一维张量。结果是一个形状为[NOff]的一维张量。
                bc.long().T:将bc转换为整数类型,并将其转置。结果是一个形状为[2, NOff]的张量。
                (b, c):使用元组解构,将bc.long().T中的即第1行和第2行分别赋值给变量b和c。
            '''
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            
            #######
            '''
            offsets.shape = [NOff,2]
            gxy - offsets为gxy偏移后的坐标，
            gxi通过long()得到偏移后坐标所在的grid坐标
            '''
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            ####### Append
            '''
            a:所有anchor的索引 a.shape = [NOff]
            b:标签所属image的索引 b.shape = [NOff]
            gj.clamp_(0, shape[3] - 1)将标签所在grid的y限定在0到ny-1之间
            gi.clamp_(0, shape[2] - 1)将标签所在grid的x限定在0到nx-1之间
            '''
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            # tbox存放的是标签在所在grid内的相对坐标，∈[0,1] 最终shape = [nl,NOff]
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            # anch存放的是anchors 最终shape = [nl,NOff,2]
            anch.append(anchors[a])  # anchors
            # tcls存放的是标签的分类,是一个list,最终shape = [nl,NOff]
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

class ComputeLoss_old:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        # indices = [image, anchor, gridy, gridx] 最终shape = [nl,4,NOff]
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        ### tcls这个list的形状
        # print('tcls的形状')
        #tcls0=torch.tensor(tcls[0])
        # print(tcls0.size())
        #result00 = tcls0.cpu().numpy()
        #print(result00)
        '''
        np.savetxt("tcls0.txt", result00)
        tcls1=torch.tensor(tcls[1])
        print(tcls1.size())
        result01 = tcls1.cpu().numpy()
        np.savetxt("tcls1.txt", result01)
        tcls2=torch.tensor(tcls[2])
        print(tcls2.size())
        result02 = tcls2.cpu().numpy()
        np.savetxt("tcls2.txt", result02)
        '''
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            '''
            print(i)
            if(i==0):
               print('第0层b的维度')
               print(b.size())
               np.savetxt("b0.txt", b.cpu().numpy()) 
            if(i==1):
               print('第1层b的维度')
               np.savetxt("b1.txt", b.cpu().numpy()) 
               print(b.size()) 
            '''
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            '''
            print('n:')
            print(n)
            '''
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
                
                '''
                ##########
                print('pcls.size()')
                print(pcls.size())
                result2 = pcls.cpu().detach().numpy()
                if(i==0):
                    np.savetxt("pcls0.txt", result2)
                if(i==1):
                    np.savetxt("pcls1.txt", result2)
                ###########
                '''
                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # print('t的维度')
                    # print(t.size())
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        # print('nt:')
        # print(nt)
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            # print('现在在buld_targets()函数的第i层遍历中,i如下:')
            # print(i)
            anchors, shape = self.anchors[i], p[i].shape
            # print(shape)
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter
                # print('t[j]的维度：')
                # print(t.size())

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                # print('j,k的维度')
                # print(j.size())
                # print(k.size())
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                # print('l,m的维度')
                # print(l.size())
                # print(m.size())
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
