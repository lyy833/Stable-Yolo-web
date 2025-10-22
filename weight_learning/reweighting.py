# coding=utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

def lr_setter(optimizer, epoch, opt, bl=False):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = opt.lr
    if bl:
        lr = opt.lrbl * (0.1 ** (epoch // (opt.epochb * 0.5)))
    else:
        if opt.cos:
            lr *= ((0.01 + math.cos(0.5 * (math.pi * epoch / opt.epochs))) / 1.01)
        else:
            if epoch >= opt.epochs_decay[0]:
                lr *= 0.1
            if epoch >= opt.epochs_decay[1]:
                lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        cov = torch.matmul((w * x).t(), x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())

    return res

# 提取随机傅里叶特征
# x为特征提取器的输出
def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    if num_f is None:
        num_f = 1
    n = x.size(0)  # x在传参时传入的是all_feature,因此这里的n的值为batch-size(对应cfeaturec.size[0])+n_feature(对应pre_features.size(0))
    r = x.size(1)  # 这里为num_ftr
    x = x.view(n, r, 1)  #将x由n*r的张量reshape为n*r*1的张量,这一步的处理是为了后面的x和w的矩阵乘法
    '''
    2*8*1的张量举例:
    tensor([[[-0.1031],[ 0.1033],[-0.2219],[-0.1180],[-0.3010],[-0.4108],[ 0.8669],[-0.1419]],
            [[ 0.6703],[ 0.3508],[ 0.9204],[-0.9787],[ 0.1451],[-0.2817],[-0.6551],[ 0.9206]]])
    '''
    c = x.size(2)  #这里c的值为1
    if sigma is None or sigma == 0:
        sigma = 1
    
    '''
    HRFF =h: x →sqart(2)*cos(ωx + φ) |ω ~ N(0, 1), φ ~ Uniform(0, 2π)
    '''
    if w is None:
        # w的维度为[num_f,1]，randn生成的随机数服从N(0, 1)，num_f为傅里叶空间的数目，默认值为1
        w = 1 / sigma * (torch.randn(size=(num_f, c)))  
        '''
        w为[8,1]的张量举例：
        tensor([[ 1.0039],[-0.2019],[-1.3042],[-0.2585],[ 0.8485],[ 0.0246],[-1.2062],[ 0.2352]])
        '''
        # 这一步操作后b为随机生成的维度为[r,num_f],b服从Uniform(0, 2π)
        b = 2 * np.pi * torch.rand(size=(r, num_f)) 
        b = b.repeat((n, 1, 1)) # 这一步之后b的维度变为[n,r,num_f]

    Z = torch.sqrt(torch.tensor(2.0 / num_f).cuda())  # 在stablenet中tensor(1.4142, device='cuda:0')
    # mid由x[n,r,1]和w的转置[1,num_f]两个张量矩阵相乘得到,下一步操作后mid的维度为[n,r,num_f],这就和偏置b一致了
    mid = torch.matmul(x.cuda(), w.t().cuda()) # mid=ωx

    # 下面的计算操作不再改变mid的维度
    mid = mid + b.cuda() # mid=ωx + φ,其中φ=b
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0].cuda()
    mid *= np.pi / 2.0

    if sum:
        Z = Z * (torch.cos(mid).cuda() + torch.sin(mid).cuda())
    else:
        Z = Z * torch.cat((torch.cos(mid).cuda(), torch.sin(mid).cuda()), dim=-1)

    return Z

def lossb_expect(cfeaturec, weight, num_f, sum=True):
    # cfeaturecs为在cfeaturec基础上提取的傅里叶特征
    # 提取得到的随机傅里叶特征的维度为[n_cfeaturec,num_ftr,num_ftr]，其中n_cfeaturec=cfeaturec.size[0]
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, sum=sum).cuda()
    # 构建神经网络的计算图时，需用torch.autograd.Variable将Tensor包装起来，
    # 形成计算图中的节点，backward()自动计算出所有需要的梯度。
    # 来针对某个变量执行grad获得想要的梯度值。
    # https://blog.csdn.net/CSDN_of_ding/article/details/110691635#%E7%AE%80%E4%BB%8B
    loss = Variable(torch.FloatTensor([0]).cuda())
    weight = weight.cuda()
    for i in range(cfeaturecs.size()[-1]):
        cfeaturec = cfeaturecs[:, :, i]

        cov1 = cov(cfeaturec, weight)
        cov_matrix = cov1 * cov1
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)

    return loss

# train函数对应一个batch的训练，在train()函数中一次前向传播后使用到weight_learner,将epoch参数传递给这里的global_epoch
# 因此，global_epoch从0到epochs递增，但注意的的是global_epoch一个值不仅仅出现一次，而是出现一个epoch中的iterator次
def weight_learner(cfeatures, pre_features, pre_weight, opt, global_epoch=0, iter=0):
    ### 初始化一些变量
    softmax = nn.Softmax(0)   # nn.Softmax()函数详解：https://blog.csdn.net/qq_43665602/article/details/126576622?ops_request_misc=&request_id=&biz_id=102&utm_term=nn.Softmax(0)&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~sobaiduweb~default-2-126576622.blog_rank_default&spm=1018.2226.3001.4450
    # 下面这步用1初始化weight张量，这个张量的维度为[batch-size,1]
    weight = Variable(torch.ones(cfeatures.size()[0], 1).cuda())
    weight.requires_grad = True # weight反向传播可以更新
    # 下面这句创建一个数据类型为float32的张量，这个张量有着和cfeatures一样的维度
    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).cuda())
    # cfeaturec仅仅相当于把cfeatures张量中的数据类型转换为float的
    cfeaturec.data.copy_(cfeatures.data)

    ### 重新加载全局特征
    # detach 意为分离
    # 对某个张量调用函数 detach()的作用是返回一个 Tensor，它和原张量的数据相同，但 requires_grad=False
    #记detach()得到的张量为de，后续基于de继续进行计算，
    # 那么反向传播过程中,遇到调用了detach()方法的张量就会终止 ,不会继续向后计算梯度。
    # 下面这句将cfeaturec与pre_features进行横向拼接得到all_feature，
    # 但注意的是all_feature中来自pre_features的部分在反向传播中不能更新。
    all_feature = torch.cat([cfeaturec, pre_features.detach()], dim=0) 
    # 定义权重优化器
    optimizerbl = torch.optim.SGD([weight], lr=opt.lrbl, momentum=0.9)

    # for epoch balancing ← 1 to BALANCING EPOCH NUMBER do
    for epoch in range(opt.epochb):
        ### Optimize sample weights
        lr_setter(optimizerbl, epoch, opt, bl=True)
        #下面这句通过横向concat得到all_weight，做法与得到全局特征类似 
        # all_weight的维度为[batch-size*2,1]，
        # 也要注意的是all_weight中来自pre_weight1的部分在反向传播中不能更新
        all_weight = torch.cat((weight, pre_weight.detach()), dim=0) 

        # 根据pytorch中的backward()函数的计算，
        # 当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
        # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，
        # 因此这里就需要每个batch设置一遍zero_grad 了。
        optimizerbl.zero_grad()

        # lossb计算的是WGi部分的损失
        lossb = lossb_expect(all_feature, softmax(all_weight), opt.num_f, opt.sum)
        # 计算weight的softmax值，并对其进行一次幂操作（即求指数），然后对所有元素求和，这个结果被赋给了变量lossp。
        # lossp计算的是WL部分的损失
        lossp = softmax(weight).pow(opt.decay_pow).sum()
        # 根据输入参数args中的一些设定，计算一个标量lambdap。其中包括一个初始的lambda值args.lambdap，以及一些关于训练进程的信息。
        lambdap = opt.lambdap * max((opt.lambda_decay_rate ** (global_epoch // opt.lambda_decay_epoch)),
                                     opt.min_lambda_times)
        # lossg是一次权重平衡过程中的所有权重（包括全局权重和局部权重）的损失
        lossg = lossb / lambdap + lossp

        # 如果是第一次迭代
        if global_epoch == 0:
            # first_step_cons参数的作用：constrain the weight at the first step
            lossg = lossg * opt.first_step_cons
        # 反向传播优化权重
        lossg.backward(retain_graph=True)
        optimizerbl.step()
    
    ## 更新pre_features和pre_weight1 
    if global_epoch == 0 and iter < 10:
        pre_features = (pre_features * iter + cfeatures) / (iter + 1)
        pre_weight = (pre_weight * iter + weight) / (iter + 1)

    elif cfeatures.size()[0] < pre_features.size()[0]:
        pre_features[:cfeatures.size()[0]] = pre_features[:cfeatures.size()[0]] * opt.presave_ratio + cfeatures * (
                    1 - opt.presave_ratio)
        pre_weight[:cfeatures.size()[0]] = pre_weight[:cfeatures.size()[0]] * opt.presave_ratio + weight * (
                    1 - opt.presave_ratio)

    else:
        # Z'Gi = αi*ZGi + (1 ? αi)*ZL
        pre_features = pre_features * opt.presave_ratio + cfeatures * (1 - opt.presave_ratio)
        # w'Gi = αi*wGi + (1 ? αi)*wL.
        pre_weight = pre_weight * opt.presave_ratio + weight * (1 - opt.presave_ratio)

    softmax_weight = softmax(weight)

    # 返回的softmax_weight用于给weight1赋值
    return softmax_weight, pre_features, pre_weight
