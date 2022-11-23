import torch


class BidirectionalLSTM(torch.nn.Module):

    def __init__(self, input_size, hidden_size, out_features):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = torch.nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.embedding = torch.nn.Linear(hidden_size * 2, out_features)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        b, T, h = recurrent.size()
        t_rec = recurrent.reshape(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


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

    def forward(self, input):
        res = self.conv(input)
        # print(f"LSTM weights: {self.lstm.all_weights[0][0][:20]}")
        # print(f"DENSE weights: {self.dense.weight[0][:20]}")
        res = res.reshape([res.shape[0], res.shape[3], 256])
        res = self.lstm(res)
        res = self.softmax(res)
        return res

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero