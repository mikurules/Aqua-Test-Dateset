import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet
from config import backbone_path

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.map = nn.Conv2d(self.chanel_in, 128, 7, 1, 3)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        map = self.map(out)
        return map

class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.map = nn.Conv2d(self.chanel_in, 128, 7, 1, 3)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        map = self.map(out)
        return map

class SEA(nn.Module):
    def __init__(self):
        super(SEA, self).__init__()
        self.conv1h = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2h = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3h = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4h = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv1v = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2v = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3v = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4v = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = BasicConv2d(32+32, 32, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(32, 1, 1)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.conv1h(left), inplace=True)
        out2h = F.relu(self.conv2h(out1h), inplace=True)
        out1v = F.relu(self.conv1v(down), inplace=True)
        out2v = F.relu(self.conv2v(out1v), inplace=True)
        fuse = out2h * out2v
        out3h = F.relu(self.conv3h(fuse), inplace=True) + out1h
        out4h = F.relu(self.conv4h(out3h), inplace=True)
        out3v = F.relu(self.conv3v(fuse), inplace=True) + out1v
        out4v = F.relu(self.conv4v(out3v), inplace=True)
        edge_feature = self.conv5(torch.cat((out4h, out4v), dim=1))
        edge_out = self.conv_out(edge_feature)
        return edge_out

class INTERFACE(nn.Module):
    def __init__(self, position):
        super(INTERFACE, self).__init__()
        if position == 3.5:
            all_in_channel = 128 + 128
        if position == 2.5:
            all_in_channel = 32 + 32
        intern_channel = int(all_in_channel // 2)
        self.ra_conv1 = BasicConv2d(all_in_channel, intern_channel, kernel_size=3, padding=1)
        self.ra_conv2S = BasicConv2d(intern_channel, intern_channel, kernel_size=3, padding=1)
        self.ra_conv2C = BasicConv2d(intern_channel, intern_channel, kernel_size=3, padding=1)

    def forward(self, SA_features, CA_features):
        x1 = torch.cat((SA_features, CA_features), dim=1)
        x = self.ra_conv1(x1)
        x_S = F.relu(self.ra_conv2S(x))
        x_C = F.relu(self.ra_conv2C(x))
        return x_S + x, x_C + x

class SA_G_DECODER(nn.Module):
    def __init__(self, feature_channel, intern_channel, position):
        super(SA_G_DECODER, self).__init__()
        if position == 4:
            all_in_channel = feature_channel + 2048 + 128
        if position == 3:
            all_in_channel = feature_channel + 2048 + 128 + 128
        if position == 2:
            all_in_channel = feature_channel + 2048 + 32 + 32
        self.ra_conv1 = BasicConv2d(all_in_channel, intern_channel, kernel_size=3, padding=1)
        self.ra_conv2 = BasicConv2d(intern_channel, intern_channel, kernel_size=3, padding=1)
        self.ra_conv3 = BasicConv2d(intern_channel, intern_channel, kernel_size=3, padding=1)
        self.ra_out = nn.Conv2d(intern_channel, intern_channel, kernel_size=3, padding=1)

    def forward(self, features, neighbor_features, cross_guidance, global_guidance):
        if cross_guidance is not None:
            x1 = torch.cat((neighbor_features, cross_guidance), dim=1)
        else:
            x1 = neighbor_features
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        crop_sal = F.interpolate(global_guidance, size=features.size()[2:], mode='bilinear', align_corners=True)
        x3 = torch.cat((torch.cat((x1, crop_sal), dim=1), features), dim=1)
        x = self.ra_conv1(x3)
        x = F.relu(self.ra_conv2(x))
        x = F.relu(self.ra_conv3(x))
        x = self.ra_out(x)
        return x

class CA_E_DECODER(nn.Module):
    def __init__(self, feature_channel, intern_channel, position):
        super(CA_E_DECODER, self).__init__()
        if position == 4:
            all_in_channel = feature_channel + 64 + 128
        if position == 3:
            all_in_channel = feature_channel + 64 + 128 + 128
        if position == 2:
            all_in_channel = feature_channel + 64 + 32 + 32
        self.ra_conv1 = BasicConv2d(all_in_channel, intern_channel, kernel_size=3, padding=1)
        self.ra_conv2 = BasicConv2d(intern_channel, intern_channel, kernel_size=3, padding=1)
        self.ra_conv3 = BasicConv2d(intern_channel, intern_channel, kernel_size=3, padding=1)
        self.ra_out = nn.Conv2d(intern_channel, intern_channel, kernel_size=3, padding=1)

    def forward(self, features, neighbor_features, cross_guidance, edge_guidance):
        if cross_guidance is not None:
            x1 = torch.cat((neighbor_features, cross_guidance), dim=1)
        else:
            x1 = neighbor_features
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        crop_edge = F.interpolate(edge_guidance, size=features.size()[2:], mode='bilinear', align_corners=True)
        x3 = torch.cat((torch.cat((x1, crop_edge), dim=1), features), dim=1)
        x = self.ra_conv1(x3)
        x = F.relu(self.ra_conv2(x))
        x = F.relu(self.ra_conv3(x))
        x = self.ra_out(x)
        return x

class myNet(nn.Module):
    def __init__(self, channel=32):
        super(myNet, self).__init__()
        self.resnet = resnet.resnet50(backbone_path, pretrained=True)
        self.s_deco_4 = SA_G_DECODER(feature_channel=1024, intern_channel=128, position=4)
        self.s_deco_3 = SA_G_DECODER(feature_channel=512, intern_channel=32, position=3)
        self.s_deco_2 = SA_G_DECODER(feature_channel=256, intern_channel=32, position=2)
        self.c_deco_4 = CA_E_DECODER(feature_channel=1024, intern_channel=128, position=4)
        self.c_deco_3 = CA_E_DECODER(feature_channel=512, intern_channel=32, position=3)
        self.c_deco_2 = CA_E_DECODER(feature_channel=256, intern_channel=32, position=2)
        self.sab = SA_Block(2048)
        self.cab = CA_Block(2048)
        self.sea = SEA()
        self.inter2_5 = INTERFACE(2.5)
        self.inter3_5 = INTERFACE(3.5)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x0 = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x0)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        ca = self.cab(x4)
        sa = self.sab(x4)
        sc_4 = self.c_deco_4(x3, ca, None, x0)
        ss_4 = self.s_deco_4(x3, sa, None, x4)
        inter_S_35, inter_C_35 = self.inter3_5(ss_4, sc_4)
        sc_3 = self.c_deco_3(x2, sc_4, inter_S_35, x0)
        ss_3 = self.s_deco_3(x2, ss_4, inter_C_35, x4)
        inter_S_25, inter_C_25 = self.inter2_5(ss_3, sc_3)
        sc_2 = self.c_deco_2(x1, sc_3, inter_S_25, x0)
        ss_2 = self.s_deco_2(x1, ss_3, inter_C_25, x4)
        e_g_out = self.sea(sc_2, ss_2)
        return e_g_out

if __name__ == "__main__":
    model = myNet()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(sum(p.numel() for p in model.parameters()))
