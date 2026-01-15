import torch


class BidirectionalLSTM(torch.nn.Module):

    def __init__(self, input_size, hidden_size, out_features):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = torch.nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.embedding = torch.nn.Linear(hidden_size * 2, out_features)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        output = self.embedding(recurrent)  # [b, T, nOut]
        return output.permute(1, 0, 2)


class ConvNormAct(torch.nn.Module):
    def __init__(self,
                 *args,
                 activation=torch.nn.LeakyReLU,
                 normalization=torch.nn.BatchNorm2d,
                 **kwargs):
        super().__init__()

        self.module = torch.nn.Sequential(
            torch.nn.Conv2d(*args, **kwargs),
            normalization(args[1]),
            activation()
        )

    def forward(self, input):
        return self.module(input)


class CRNN(torch.nn.Module):
    def __init__(self,
                 out_features,
                 activation=torch.nn.LeakyReLU,
                 normalization=torch.nn.BatchNorm2d):
        super().__init__()
        cnn_acts = {"activation": activation, "normalization": normalization}
        self.conv = torch.nn.Sequential(
            ConvNormAct(1, 32, 3, **cnn_acts),
            torch.nn.MaxPool2d(2),
            ConvNormAct(32, 64, 3, **cnn_acts),
            torch.nn.MaxPool2d(2),
            ConvNormAct(64, 128, 3, **cnn_acts),
            torch.nn.MaxPool2d(2),
            ConvNormAct(128, 256, 3, **cnn_acts),
            torch.nn.MaxPool2d(2))
        self.lstm = BidirectionalLSTM(input_size=256, hidden_size=256, out_features=out_features)
        self.softmax = torch.nn.LogSoftmax(dim=2)
        self.register_full_backward_hook(self.backward_hook)

    def forward(self, input):
        res = self.conv(input)
        res = res.mean(dim=2)
        res = res.permute(0, 2, 1)
        res = self.lstm(res)
        res = self.softmax(res)
        return res

    def backward_hook(self, module, grad_input, grad_output):
        cleaned_grad_input = []
        for g in grad_input:
            if g is None:
                cleaned_grad_input.append(None)
                continue
            cleaned_grad_input.append(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))
        return tuple(cleaned_grad_input)