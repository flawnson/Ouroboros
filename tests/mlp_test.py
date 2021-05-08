import json
import time
import torch
import logzero
import logging
import argparse
import torch.nn.functional as F

from typing import *
from logzero import logger
from optim.losses import loss as loss_fun
from data.linear_preprocessing import get_data
from optim.algos import OptimizerObj, LRScheduler
from utils.scores import scores
from utils.holdout import MNISTSplit
from utils.logging import PTTBLogger


class Model(torch.nn.Module):
    def __init__(self, config, device):
        super(Model, self).__init__()
        self.config = config
        self.device = device
        self.model = self.get_model()

    def get_model(self):
        model_layers = []

        ## Flatten for Colab
        model_layers.append(torch.nn.Flatten())
        ##
        
        model_layers.append(torch.nn.Linear(784, 80, bias=True))
        model_layers.append(torch.nn.ReLU())
        model_layers.append(torch.nn.Dropout())
        model_layers.append(torch.nn.Linear(80, 40, bias=True))
        model_layers.append(torch.nn.ReLU())
        model_layers.append(torch.nn.Dropout())
        model_layers.append(torch.nn.Linear(40, 40, bias=True))
        model_layers.append(torch.nn.ReLU())
        model_layers.append(torch.nn.Dropout())
        model_layers.append(torch.nn.Linear(40, 20, bias=True))
        model_layers.append(torch.nn.ReLU())
        model_layers.append(torch.nn.Dropout())
        model_layers.append(torch.nn.Linear(20, 10, bias=True))
        return torch.nn.Sequential(*model_layers)

        ###COLAB VERSION###
        # model_layers = []
        # model_layers.append(torch.nn.Flatten())
        # model_layers.append(torch.nn.Linear(784, 250, bias=True))
        # model_layers.append(torch.nn.ReLU())
        # #model_layers.append(torch.nn.Dropout())
        # model_layers.append(torch.nn.Linear(250, 100, bias=True))
        # model_layers.append(torch.nn.ReLU())
        # #model_layers.append(torch.nn.Dropout())
        # model_layers.append(torch.nn.Linear(100, 10, bias=True))
        # return torch.nn.Sequential(*model_layers)

    def forward(self, x):
        return self.model(x)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def run():
    ### Configuring ###
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="json config file", type=str)
    parser.add_argument("-s", "--scheme", help="json schema file", type=str)
    args = parser.parse_args()

    config: Dict = json.load(open(args.config))
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    logzero.loglevel(eval(config["log_level"]))
    tb_logger = PTTBLogger(config)
    logger.info(f"Successfully retrieved config json. Running {config['run_name']} on {device}.")
    logger.info(f"Using PyTorch version: {torch.__version__}")

    model = Model(config, device)

    # datasets = get_data(config)
    # dataloaders = MNISTSplit(config, datasets, None, device).partition()  # MNIST split appears to work fine with CIFAR

    ### MODIFIED ###
    BATCH_SIZE = 64
    train_data, test_data = get_data(config)
    train_iterator = torch.utils.data.DataLoader(train_data,
                                     shuffle=True,
                                     batch_size=BATCH_SIZE)

    test_iterator = torch.utils.data.DataLoader(test_data,
                                    batch_size=BATCH_SIZE)
    criterion = torch.nn.CrossEntropyLoss()
    ####
    optimizer = OptimizerObj(config, model).optim_obj
    scheduler = LRScheduler(config, optimizer).schedule_obj

    epoch_data = {"loss": [0, 0], "correct": [0, 0]}
    for epoch in range(config["run_config"]["num_epochs"]):
        ### MODIFIED ####
        start_time = time.monotonic()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, test_iterator, criterion, device)
        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Test Loss: {valid_loss:.3f} |  Test Acc: {valid_acc * 100:.2f}%')
        ###

        # batch_data = {"loss": [0, 0]}
        # for batch_idx, data in enumerate(dataloaders[list(dataloaders)[0]]):
        #     logger.info(f"Running train batch: #{batch_idx}")
        #     model.train()
        #     optimizer.zero_grad()
        #
        #     logits = model(data[0].reshape(-1))
        #     predictions = logits.argmax(keepdim=True)
        #     loss = loss_fun(config, model, logits, data[1])
        #
        #     if ((batch_idx + 1) % config["data_config"]["batch_size"]) == 0:
        #         loss["loss"].backward()
        #         optimizer.step()
        #         epoch_data["loss"][0] += batch_data["loss"][0]
        #         batch_data["loss"][0] = 0.0
        #         optimizer.zero_grad()
        #
        #     epoch_data["correct"][0] += predictions.eq(data[1].view_as(predictions)).sum().item()
        #     batch_data["loss"][0] += loss["loss"].item()
        #
        # for batch_idx, data in enumerate(dataloaders[list(dataloaders)[1]]):
        #     with torch.no_grad():
        #         logger.info(f"Running test batch: #{batch_idx}")
        #         model.eval()
        #
        #         logits = model(data[0].reshape(-1))
        #         predictions = logits.argmax(keepdim=True)
        #         loss = loss_fun(config, model, logits, data[1])
        #
        #         if ((batch_idx + 1) % config["data_config"]["batch_size"]) == 0:
        #             epoch_data["loss"][1] += batch_data["loss"][0]
        #             batch_data["loss"][1] = 0.0
        #
        #         epoch_data["correct"][1] += predictions.eq(data[1].view_as(predictions)).sum().item()
        #         batch_data["loss"][1] += loss["loss"].item()

        # epoch_scores = scores(config, dataloaders, epoch_data["correct"], device)
        #
        # # Log values for training
        # train_epoch_length = len(dataloaders[list(dataloaders)[0]])
        # actual_train_loss = epoch_data["loss"][0] / (train_epoch_length // config["data_config"]["batch_size"])
        # tb_logger.scalar_summary('loss (train)', actual_train_loss, epoch)
        # tb_logger.scalar_summary('scores (train)', epoch_scores["acc"][0], epoch)
        #
        # # Log values for testing
        # test_epoch_length = len(dataloaders[list(dataloaders)[0]])
        # actual_test_loss = epoch_data["loss"][1] / (test_epoch_length // config["data_config"]["batch_size"])
        # tb_logger.scalar_summary('loss (test)', actual_test_loss, epoch)
        # tb_logger.scalar_summary('scores (train)', epoch_scores["acc"][1], epoch)
        #
        # logger.info("Successfully wrote logs to tensorboard")
        #
        # epoch_data["loss"][0] = 0
        # epoch_data["correct"][0] = 0
        # epoch_data["loss"][1] = 0
        # epoch_data["correct"][1] = 0


if __name__ == "__main__":
    run()



