"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from modules.OCResNet import OCResNet_FeatureExtractor


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}
        self.baseline = opt.baseline

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'OCResNet':
            self.FeatureExtraction = OCResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        
        self.AdaptiveAvgPool = nn.AdaptiveMaxPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            # self.SequenceModeling = nn.Sequential(
            #     BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
            #     BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.BiLSTM1 = BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size)
            self.BiLSTM2 = BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size)
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        # diff: 2D Attention 先试用论文垂直降采样的方法
        feature_map = self.FeatureExtraction(input)
        if self.baseline == True: # only baseline model train with 1d vector
            visual_feature = self.AdaptiveAvgPool(feature_map.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        
        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            # diff: 2D Attention 先试用两层 Bi-LSTM 进行编码
            contextual_feature, hidden = self.BiLSTM1(visual_feature)
            contextual_feature, hidden = self.BiLSTM2(contextual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM
       
        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            # diff: 2D Attention 除了LSTM进行编码的特征之外，还要引入 feature map
            prediction = self.Prediction(feature_map.contiguous(), contextual_feature.contiguous(), hidden, text, is_train, batch_max_length=self.opt.batch_max_length)
            
        return prediction


if __name__ == "__main__":
    import argparse
    import torch
    opt = argparse.ArgumentParser()
    opt.Transformation = "None"
    opt.FeatureExtraction = "OCResNet"
    opt.SequenceModeling = "BiLSTM"
    opt.Prediction = "Attn"
    opt.input_channel = 3
    opt.output_channel = 256
    opt.hidden_size = 256
    opt.num_class = 37
    opt.batch_max_length = 25
    opt.baseline = True
    net = Model(opt)

    x = torch.randn(10, 3, 100, 100)
    text = torch.zeros(10, 38).long()
    y = net(x, text)
    print(y.shape)
