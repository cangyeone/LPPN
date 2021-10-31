import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
class ConvBNReLU(nn.Sequential):
    """
    三个层在计算过程中应当进行融合
    使用ReLU作为激活函数可以限制
    数值范围，从而有利于量化处理。
    """
    def __init__(self, n_in, n_out, 
                 kernel_size=3, stride=1, 
                 groups=1, norm_layer=nn.BatchNorm2d):
        # padding为same时两边添加(K-1)/2个0
        padding = (kernel_size - 1) // 2
        # 本层构建三个层，即0：卷积，1：批标准化，2：ReLU
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(n_in, n_out, [1, kernel_size], 
                      stride, [0, padding], groups=groups, 
                      bias=False),
            nn.BatchNorm2d(n_out),
            nn.ReLU(inplace=True)
        )
class ConvTBNReLU(nn.Sequential):
    """
    三个层在计算过程中应当进行融合
    使用ReLU作为激活函数可以限制
    数值范围，从而有利于量化处理。
    """
    def __init__(self, n_in, n_out, 
                 kernel_size=3, stride=1, padding=1, output_padding=1, bias=True, dilation=1,  
                 groups=1, norm_layer=nn.BatchNorm2d):
        # padding为same时两边添加(K-1)/2个0
        # 本层构建三个层，即0：卷积，1：批标准化，2：ReLU
        super(ConvTBNReLU, self).__init__(
            nn.UpsamplingBilinear2d(scale_factor=tuple(stride)), 
            nn.Conv2d(n_in, n_out, kernel_size, stride=1, padding=padding), 
            nn.BatchNorm2d(n_out),
            nn.ReLU(inplace=True)
        )
class InvertedResidual(nn.Module):
    """
    本个模块为MobileNetV2中的可分离卷积层
    中间带有扩张部分，如图10-2所示
    """
    def __init__(self, n_in, n_out, 
                 stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = stride
        # 隐藏层需要进行特征拓张，以防止信息损失
        hidden_dim = int(round(n_in * expand_ratio))
        # 当输出和输出维度相同时，使用残差结构
        self.use_res = self.stride == 1 and n_in == n_out
        # 构建多层
        layers = []
        if expand_ratio != 1:
            # 逐点卷积，增加通道数
            layers.append(
                ConvBNReLU(n_in, hidden_dim, kernel_size=1, 
                            norm_layer=norm_layer))
        layers.extend([
            # 逐层卷积，提取特征。当groups=输入通道数时为逐层卷积
            ConvBNReLU(
                hidden_dim, hidden_dim, 
                stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # 逐点卷积，本层不加激活函数
            nn.Conv2d(hidden_dim, n_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(n_out),
        ])
        # 定义多个层
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        else:
            return self.conv(x)
class QInvertedResidual(InvertedResidual):
    """量化模型修改"""
    def __init__(self, *args, **kwargs):
        super(QInvertedResidual, self).__init__(*args, **kwargs)
        # 量化模型应当使用量化计算方法
        self.skip_add = nn.quantized.FloatFunctional()
    def forward(self, x):
        if self.use_res:
            # 量化加法
            #return self.skip_add.add(x, self.conv(x))
            return x + self.conv(x)
        else:
            return self.conv(x)
    def fuse_model(self):
        # 模型融合
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                # 将本个模块最后的卷积层和BN层融合
                fuse_modules(
                    self.conv, 
                    [str(idx), str(idx + 1)], inplace=True)
class Model(nn.Module):
    def __init__(self, n_stride=8, n_channel=4):
        super().__init__()
        self.n_stride = n_stride # 总步长 
        base = n_channel 
        if n_stride == 8:
            S = [1, 1, 2, 1, 2, 1, 2]
        elif n_stride == 16:
            S = [2, 1, 2, 1, 2, 1, 2]
        elif n_stride == 32:
            S = [2, 2, 2, 1, 2, 1, 2]
        elif n_stride == 64:
            S = [2, 2, 2, 2, 2, 1, 2]
        elif n_stride == 128:
            S = [2, 2, 2, 2, 2, 2, 2]
        else:
            raise ValueError("S must in 8, 16, 32, 64 or 128")
        self.layers = nn.Sequential(
            QInvertedResidual(     3, base*1, S[0], 2), 
            QInvertedResidual(base*1, base*2, S[1], 2), 
            QInvertedResidual(base*2, base*2, S[2], 2), 
            QInvertedResidual(base*2, base*3, S[3], 2), 
            QInvertedResidual(base*3, base*3, S[4], 2),
            QInvertedResidual(base*3, base*4, S[5], 2), 
            QInvertedResidual(base*4, base*5, S[6], 2)             
        )
        self.class_encoder = nn.Sequential(
            QInvertedResidual(base*5, base*5, 2, 2), 
            QInvertedResidual(base*5, base*5, 2, 2), 
            QInvertedResidual(base*5, base*5, 2, 2), 
            ConvTBNReLU(base*5, base*5, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1), 
            ConvTBNReLU(base*5, base*5, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1),  
            ConvTBNReLU(base*5, base*5, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1), 
        )
        self.cl = nn.Conv2d(base * 5 * 2, 3, 1) 
        self.tm = nn.Conv2d(base * 5 * 2, 1, 1)
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == ConvTBNReLU:
                fuse_modules(m, ['1', '2', '3'], inplace=True)
            if type(m) == QInvertedResidual:
                m.fuse_model()    
    def forward(self, x, device):
        B, C, T = x.shape 
        t = torch.arange(T) * 2 * 3.141592658 / 4 
        p = torch.stack([torch.sin(t), torch.sin(2*t), torch.sin(4*t)], dim=0).to(device)
        p = torch.unsqueeze(p, 0) 
        x = x + p 
        x = x.unsqueeze(2)
        x1 = self.layers(x)
        x2 = self.class_encoder(x1) 
        x = torch.cat([x1, x2], dim=1)
        out_class = self.cl(x).squeeze()
        out_time = self.tm(x).squeeze()
        out_time = out_time * self.n_stride 
        out_time = out_time.squeeze()
        B, C, T = out_class.shape 
        outputs = []
        prob = F.softmax(out_class, 1)
        return prob, out_time
        
import utils 
import time 
import numpy as np 
import scipy.signal as signal  
#import tensorflow as tf 


def find_phase(prob, regr, delta=1.0, height=0.80, dist=1):
    shape = np.shape(prob) 
    all_phase = []
    phase_name = {0:"N", 1:"P", 2:"S"}
    for itr in range(shape[0]):
        phase = []
        for itr_c in [0, 1]:
            p = prob[itr, itr_c+1, :] 
            #p = signal.convolve(p, np.ones([10])/10., mode="same")
            h = height 
            peaks, _ = signal.find_peaks(p, height=h, distance=dist) 
            for itr_p in peaks:
                phase.append(
                    [
                        itr_c+1, #phase_name[itr_c], 
                        itr_p*delta+regr[itr, itr_p], 
                        prob[itr, itr_c, itr_p], 
                        itr_p*delta
                    ]
                    )
        all_phase.append(phase)
    return all_phase 
def main(args):
    data_tool = utils.DataTest(batch_size=100, n_length=3072)
    models = []
    outfiles = []
    stride = args.stride 
    nchannel = args.feature 
    model_name = f"ckpt/{stride}-{nchannel}.wave"
    device = torch.device(args.device)
    model = Model(n_stride=stride, n_channel=nchannel)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    model.to(device)
    model.fuse_model()
    acc_time = 0
    file_ = open(f"{args.output}/{nchannel}-{stride}.stat.txt", "w")
    models.append(model) 
    outfiles.append(file_)
    datalen = 3072 
    for step in range(400):
        a1, a2, a3, a4 = data_tool.batch_data()
        time1 = time.perf_counter()
        a1 = torch.tensor(a1, dtype=torch.float32, device=device)
        a1 = a1.permute(0, 2, 1)
        for model, outfile in zip(models, outfiles):
            with torch.no_grad():
                oc, ot = model(a1, device)
                oc = oc.cpu().numpy() 
                ot = ot.cpu().numpy()
                phase = find_phase(oc, ot, stride, height=0.3, dist=500)
            
            for idx in range(len(a2)):
                is_noise = a2[idx] 
                pt, st = a4[idx] 
                snr = np.mean(a3[idx]) 
                if pt<0 or st<0:
                    continue 
                if is_noise:
                    outfile.write("#none\n")
                else:
                    if st > datalen:
                        outfile.write(f"#phase,{pt},{-100},{snr}\n") 
                    else:
                        outfile.write(f"#phase,{pt},{st},{snr}\n") 
                for p in phase[idx]:
                    outfile.write(f"{p[0]},{p[1]},{p[2]}\n") 
                outfile.flush()
            time2 = time.perf_counter()
            print(step, f"Finished! {time2-time1}")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")          
    parser.add_argument('-f', '--feature', default=8, type=int, help="base number of feature")       
    parser.add_argument('-s', '--stride', default=8, type=int, help="stride of model")       
    parser.add_argument('-i', '--input', default="data", help="dataset dir") 
    parser.add_argument('-o', '--output', default="outdata", help="output dir") 
    args = parser.parse_args()
    main(args)