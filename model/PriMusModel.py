import numpy as np
import pytorch_lightning as pl
import torch
from scipy.spatial.distance import cosine
from Levenshtein import distance as lev


class PriMusModel(pl.LightningModule):
    def __init__(self,
                 model,
                 weight_decay=.1,
                 blank_symbol=0):
        super().__init__()
        self.model = model
        self.train_dir = ('loss/train')
        self.valid_dir = ('loss/valid')
        self.blank_symb = blank_symbol
        self.loss_function = torch.nn.CTCLoss(blank=blank_symbol, zero_infinity=True)
        self.weight_decay = weight_decay

    def forward(self, input):
        res = self.model(input)
        return res

    @staticmethod
    def get_output_seq(preds):
        out_seq = torch.zeros(preds.shape[1:])
        for seq_ind, seq in enumerate(preds.argmax(dim=0)):
            seq_pred = torch.zeros(out_seq.shape[1])
            prev_symb = -1
            seq_pred_ind = 0
            for cur_symb in seq:
                if cur_symb != prev_symb and cur_symb != 0:
                    seq_pred[seq_pred_ind] = int(cur_symb)
                    seq_pred_ind += 1
                prev_symb = cur_symb
            out_seq[seq_ind] = seq_pred
        return out_seq

    @staticmethod
    def dists(output_seq_preds, y, dists_func):
        dists = np.empty(shape=(y.shape[0], len(dists_func)))
        for ind, (pred_, target_) in enumerate(zip(output_seq_preds, y)):
            dists[ind] = [func(pred_, target_) for func in dists_func]
        return dists

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        batch_size = x.shape[0]
        # print("TRAIN")
        # print(f"x.mean:{x.mean()}")
        # print(f"x>0:{x[x > 0]}")
        # print(f"len(x>0):{len(x[x > 0])}")
        # print(f"image path: {batch['image_path']}")
        preds = self.forward(x)
        # preds = torch.permute(preds, (1, 0, 2))
        loss_train = self.loss_function(preds, y, torch.LongTensor([preds.size(0)] * batch_size),
                                        batch['target_lengths'])
        output_seq = self.get_output_seq(preds)
        lev_mean = self.dists(output_seq, y, [lev]).mean()/x.shape[-1]
        print(f'LEV MEAN:{lev_mean}')
        # print(f"loss_train:{loss_train}")
        # print(f"optim state: {self.optim.state_dict()}")
        # self.log('loss/train', torch.tensor(loss_train), on_step=True, on_epoch=True)
        # self.log('lev/train', torch.tensor(lev_mean), on_step=True, on_epoch=False)
        self.logger.experiment.log_metric(run_id=self.logger.run_id, key='loss/train', value=loss_train)
        self.logger.experiment.log_metric(run_id=self.logger.run_id, key='lev/train',  value=lev_mean)
        return loss_train

    # def training_step_end(self, outputs):
    #     # print("GRAD NORM")
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, norm_type=2.0)

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        batch_size = x.shape[0]
        # print("\n")
        # print("VALID")
        # print(f"x.mean:{x.mean()}")
        # print(f"x>0:{x[x > 0]}")
        # print(f"len(x>0):{len(x[x > 0])}")

        preds = self.forward(x)
        # preds = torch.permute(preds, (1, 0, 2))
        loss_test = self.loss_function(preds, y, torch.LongTensor([preds.size(0)] * batch_size),
                                       batch['target_lengths'])
        output_seq = self.get_output_seq(preds)
        lev_mean = self.dists(output_seq, y, [lev]).mean()/x.shape[-1]
        # self.logger.experiment.log_metric(run_id=self.logger.run_id, key='loss/valid', value=loss_test)
        # self.logger.experiment.log_metric(run_id=self.logger.run_id, key='lev/valid', value=lev_mean)
        self.log('lev/valid', lev_mean)
        return {'loss': loss_test}

    def configure_optimizers(self):
        self.optim = torch.optim.Adam(self.model.parameters(), lr=3e-5, weight_decay=self.weight_decay)
        self.shed = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.3, verbose=True)
        return self.optim


    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        epoch = args[0]
        batch_idx = args[1]

        if epoch != 0 and batch_idx == 0:
            val_accuracy = self.trainer.logged_metrics['lev/valid']
            self.shed.step(val_accuracy)
