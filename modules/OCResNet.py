import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        """
            input channel: in_ch + mid_ch 
            output channel: out_ch
        """
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class OCResNet(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers):
        super(OCResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]
        # 128 256 512 512
        self.inplanes = int(output_channel / 8) # 64
        self.conv0_1 = nn.Conv2d(input_channel, self.inplanes, 
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(self.inplanes)
        self.conv0_2 = nn.Conv2d(self.inplanes, self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
                               0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
                               1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))#, padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
                               2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))#, padding=(0, 1))
        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                 3], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.output_channel_block[3])

        """ U network"""
        self.upconv1 = double_conv(self.output_channel_block[3]//2, self.output_channel_block[3]//2, 256)
        self.upconv2 = double_conv(self.output_channel_block[2], 256, 256)
        self.upconv3 = double_conv(self.output_channel_block[1], 256, 256)

        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.head.modules())


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        c1 = self.relu(x) # B * 64 * H * W

        x = self.maxpool1(c1)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        c2 = self.relu(x) # B * 128 * (H/2) * (W/2)

        x = self.maxpool2(c2)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        c3 = self.relu(x) # B * 256 * (H/4) * (W/4)

        x = self.maxpool3(c3)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        c4 = self.relu(x) # B * 512 * (H/8) * (W/4)

        x = self.maxpool4(c4)
        x = self.layer4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        c5 = self.relu(x) # B * 512 * (H/16) * (W/4)

        p5 = self.upconv1(c5) # B * 256 * (H/16) * (W/4)
        x = F.interpolate(p5, size=c4.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c4], dim=1)

        p4 = self.upconv2(x)
        x = F.interpolate(p4, size=c3.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c3], dim=1)

        p3 = self.upconv3(x)
        x = F.interpolate(p3, size=c2.size()[2:], mode='bilinear', align_corners=False)
        x = self.head(x)

        return x

class OCResNet_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=512):
        super(OCResNet_FeatureExtractor, self).__init__()
        self.ConvNet = OCResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        return self.ConvNet(input)


if __name__ == "__main__":
    import torch
    x = torch.zeros(10, 3, 32, 100)
    net = OCResNet(3, 512, BasicBlock, [1, 2, 5, 3])
    print(net)
    y = net(x)
    print(y.shape)
    