import torch
from typing import NamedTuple

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
    def __init__(self, input_features, hidden_features, output_features, num_channels=2, record=False, dt=0.001,
                 method="super", alpha=100):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channels, 30, 7, 1)
        self.conv2 = torch.nn.Conv2d(30, 70, 5, 1)
        self.flatten = torch.nn.Flatten()
        self.l1 = LIFRecurrentCell(
            input_features,
            hidden_features,
            p=LIFParameters(alpha=10, v_th=torch.tensor(0.4)),
            dt=dt
        )

        self.lif0 = LIFCell(dt=dt, p=LIFParameters(method=method, alpha=alpha))
        self.lif1 = LIFCell(dt=dt, p=LIFParameters(method=method, alpha=alpha))

        self.input_features = input_features
        self.fc_out = torch.nn.Linear(
            hidden_features, output_features, bias=False)
        self.out = LICell(dt=dt)

        self.hidden_features = hidden_features
        self.output_features = output_features
        self.record = record

    def forward(self, x):
        batch_size, seq_length, _, _, _ = x.shape
        s0 = s1 = s2 = so = s3 = None
        voltages = []

        if self.record:
            self.recording = SNNState(
                LIFState(
                    z=torch.zeros(seq_length, batch_size,
                                  self.hidden_features),
                    v=torch.zeros(seq_length, batch_size,
                                  self.hidden_features),
                    i=torch.zeros(seq_length, batch_size, self.hidden_features)
                ),
                LIState(
                    v=torch.zeros(seq_length, batch_size,
                                  self.output_features),
                    i=torch.zeros(seq_length, batch_size, self.output_features)
                )
            )

        for ts in range(seq_length):
            z = x[:, ts, :, :]
            z = self.conv1(z)
            z = torch.nn.functional.max_pool2d(z, 3, 3)
            z, s0 = self.lif0(z, s0)
            z = self.conv2(z)
            z = torch.nn.functional.max_pool2d(z, 3, 3)
            z, s2 = self.lif1(z, s2)
            z = self.flatten(z)
            z, s1 = self.l1(z, s1)
            z = self.fc_out(z)
            vo, so = self.out(z, so)
            if self.record:
                self.recording.lif0.z[ts, :] = s1.z
                self.recording.lif0.v[ts, :] = s1.v
                self.recording.lif0.i[ts, :] = s1.i
                self.recording.readout.v[ts, :] = so.v
                self.recording.readout.i[ts, :] = so.i
            voltages += [vo]

        x = torch.stack(voltages)
        y_hat, _ = torch.max(x, 0)
        return y_hat
