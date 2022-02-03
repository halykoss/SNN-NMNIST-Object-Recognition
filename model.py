import torch
from typing import NamedTuple
import torch.nn.functional as F

from norse.torch import LIFParameters, LIFState
from norse.torch.module.lif import LIFCell, LIFRecurrentCell
# Notice the difference between "LIF" (leaky integrate-and-fire) and "LI" (leaky integrator)
from norse.torch import LICell, LIState


class Model(torch.nn.Module):
    def __init__(self, snn):
        super(Model, self).__init__()
        self.snn = snn

    def forward(self, x):
        x = self.snn(x)
        return x


class SNNState(NamedTuple):
    lif0: LIFState
    readout: LIState


class ConvNet(torch.nn.Module):
    def __init__(self, input_features, hidden_features, num_channels=2, dt=0.001,
                 method="super", alpha=100):
        super(ConvNet, self).__init__()

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = 4
        self.output_features_cl = 10

        # Common stream
        self.conv1 = torch.nn.Conv2d(num_channels, 30, 7, 1)
        self.conv2 = torch.nn.Conv2d(30, 70, 5, 1)
        self.flatten = torch.nn.Flatten()

        self.lif0 = LIFCell(dt=dt, p=LIFParameters(method=method, alpha=alpha))
        self.lif1 = LIFCell(dt=dt, p=LIFParameters(method=method, alpha=alpha))

        # Object detection stream
        self.l1 = LIFRecurrentCell(
            input_features,
            hidden_features,
            p=LIFParameters(alpha=10, v_th=torch.tensor(0.4)),
            dt=dt
        )

        self.fc_out = torch.nn.Linear(
            hidden_features, self.output_features, bias=False)

        self.out = LICell(dt=dt)

        # Classification stream

        self.cl_l1 = LIFRecurrentCell(
            input_features,
            hidden_features * 2,
            p=LIFParameters(alpha=10, v_th=torch.tensor(0.4)),
            dt=dt
        )

        self.cl_fc_out = torch.nn.Linear(
            hidden_features * 2, self.output_features_cl, bias=False)

        self.cl_out = LICell(dt=dt)

    def object_recognition_stream(self, z, s1, so):
        z1, s1 = self.l1(z, s1)
        z1 = self.fc_out(z1)
        vo, so = self.out(z1, so)
        return vo, s1, so

    def classification_stream(self, z, s1, so):
        z1, s1 = self.cl_l1(z, s1)
        z1 = self.cl_fc_out(z1)
        vo, so = self.cl_out(z1, so)
        return vo, s1, so

    def forward(self, x):
        _, seq_length, _, _, _ = x.shape
        s0 = s1 = s2 = so = s1_c = so_c = None
        voltages = []
        voltages_class = []

        for ts in range(seq_length):
            z = x[:, ts, :, :]
            z = self.conv1(z)
            z = torch.nn.functional.max_pool2d(z, 3, 3)
            z, s0 = self.lif0(z, s0)
            z = self.conv2(z)
            z = torch.nn.functional.max_pool2d(z, 3, 3)
            z, s2 = self.lif1(z, s2)
            z = self.flatten(z)
            vo, s1, so = self.object_recognition_stream(z, s1, so)
            voltages += [vo]
            vo_c, s1_c, so_c = self.classification_stream(z, s1_c, so_c)
            voltages_class += [vo_c]
        # Object detecti
        x = torch.stack(voltages)
        y_hat, _ = torch.max(x, 0)
        # Classification
        x_c = torch.stack(voltages_class)
        y_hat_c, _ = torch.max(x_c, 0)
        log_probs = F.log_softmax(y_hat_c, dim=1)
        return y_hat, log_probs
