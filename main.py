from model import CRNN, PriMusModel
from dataset import PriMusDatasetLoader, collate_wrapper
import yaml
import json
import os
import torch
import mlflow
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar

if __name__ == '__main__':
    experiment_name = "Training"
    run_name = "baseline_xlmr"
    exp = mlflow.set_experiment(experiment_name)

    mlflow.pytorch.autolog()
    mlflow.start_run(run_name="baseline_xlmr")

    torch.random.manual_seed(1)
    pl.seed_everything(1)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, "config.yaml")
    with open(config_path, "r") as config_stream_r:
        config = yaml.safe_load(config_stream_r)

    primus_loader = PriMusDatasetLoader(root=config['root'], agnostic=config['agnostic'])
    alphabet_path = os.path.join(dir_path, "alphabet", "agnostic.json") if config['agnostic'] else os.path.join(
        dir_path, "agnostic.json")
    if not os.path.exists(alphabet_path):
        primus_loader.set_alphabet_dict()
        with open(alphabet_path, "w") as alphabet_stream_wb:
            json.dump(primus_loader.alphabet_dict, alphabet_stream_wb)
    else:
        with open(alphabet_path, "r") as stream:
            alphabet = json.load(stream)
        primus_loader.alphabet_dict = alphabet

    train, valid = primus_loader.get_train_test(random_state=0, test_size=.3)

    train_loader = DataLoader(
        train,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_wrapper,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid,
        batch_size=32,
        num_workers=8,
        collate_fn=collate_wrapper
    )
    logger = pl.loggers.MLFlowLogger(experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
                                    tracking_uri=mlflow.get_tracking_uri(),
                                    run_id=mlflow.active_run().info.run_id)

    early_stopping = EarlyStopping(
        monitor='lev/valid', min_delta=0.01, patience=3, verbose=False, mode="min"
    )

    trainer = pl.Trainer(
        detect_anomaly=True,
        # track_grad_norm=2,
        # gradient_clip_val=1,
        # gradient_clip_algorithm="value",
        max_epochs=5,
        deterministic=True,
        accelerator="cpu",
        logger=logger,
        callbacks=[early_stopping])

    crnn_model = CRNN(out_features=len(primus_loader.alphabet_dict))
    model = PriMusModel(model=crnn_model)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
