import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
import argparse
import pprint
from tqdm import tqdm
from utils import get_dataloaders, print_out, plot_train_eval_summary, resnet_get_grid_search, mlp_get_grid_search, cnn_get_grid_search
from models.CNN import MyCNN
from train import train_one_epoch
from eval import eval_model
from models.resnet import ResNet
from models.MLP import MLPModel

def analyze_model(model, log):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(list(model.state_dict().keys()))

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for name, param in model.named_parameters():
        print_out(f"{name}: {param.shape}", log)
    print_out(f"\nTotal parameters: {total_num:,};\tTrainable: {trainable_num:,}", log)


def main(config):
    """
    model_type: MLP or CNN or ResNet
    """
    ## Hyperparameters
    n_epochs = config["n_epochs"]
    batch_size = config["batch_size"]
    h, w = config["h"], config["w"]
    lr = config["lr"]
    momentum = config["momentum"]
    device = config["device"]
    kernel_size = config.get("kernel_size",None)
    model = config["model"]
    kernel_list = config.get("kernel_list", None)
    optimizer = config["optimizer"]
    debug = config.get("debug", False)
    num_blocks_list = config.get("num_blocks_list",None)
    is_batchnorm = config.get("is_batchnorm",None)
    weight_decay = config.get("weight_decay",None)
    h_dim_list = config.get("h_dim_list",None)
    use_pooling = config.get("use_pooling",None)
    num_kernel_conv_list = config.get("num_kernel_conv_list",None)

    ## check if cuda is available
    if not torch.cuda.is_available():
        device = "cpu"

    img_size = (h, w)

    ## create a new folder to store log files named logs
    os.makedirs("logs", exist_ok=True)
    ## create a new folder to store figures named figures
    os.makedirs("figures", exist_ok=True)

    datetime_now = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/{model}_epoch{n_epochs}_lr{lr}_{datetime_now}.log"
    log = open(log_filename, "a")

    print_out(f"config: {config}", log)

    ## prepare datasets
    trainloader, validloader, testloader = get_dataloaders(batch_size, img_size)
    ## TODO: test on testloader at the end of training

    if model == "MLP":
        net = MLPModel(h, w, h_dim_list)
    elif model == "CNN":
        net = MyCNN(num_kernel_conv_list=num_kernel_conv_list, 
                    kernel_size=kernel_size,
                    use_pooling=use_pooling
        )  # type: ignore
    elif model == "resnet":
        net = ResNet(input_dim=3, num_classes=2, num_blocks_list=num_blocks_list, is_batchnorm=is_batchnorm)
    else:
        raise NotImplementedError()
    
    analyze_model(net, log)

    if config.get("n_gpus", 0) > 1:
        net = torch.nn.DataParallel(net)
    net = net.to(device)

    # Optimizer
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {config['optimizer']} not implemented")
    
    # Loss function
    loss = torch.nn.CrossEntropyLoss()  ### Compute Loss (CELoss, MSE)

    train_acc_summary = []
    train_loss_summary = []
    valid_acc_summary = []
    valid_loss_summary = []

    best_valid_acc = 0.0
    final_test_acc = 0.0

    for e in range(1, n_epochs+1):
        start_time = time.time()  ## time at the beginning of epoch, for logging purpose
        train_acc, train_loss = train_one_epoch(
            model=net,
            device=device,
            trainloader=trainloader,
            optimizer=optimizer,
            criterion=loss,
            debug=debug
        )

        fig_name = f"{model}_epoch{e}_lr{lr}_{datetime_now}.png"
        roc_name_valid = f"figures/roc_valid_e{e}_{fig_name}"
        cm_name_valid = f"figures/cm_valid_e{e}_{fig_name}"
        roc_name_test = f"figures/roc_test_e{e}_{fig_name}"
        cm_name_test = f"figures/cm_test_e{e}_{fig_name}"

        valid_acc, valid_loss = eval_model(
            model=net,
            device=device,
            evalloader=validloader,
            criterion=loss,
            cm_name = cm_name_valid,
            roc_name = roc_name_valid
        )
        test_acc, test_loss = eval_model(
            model=net,
            device=device,
            evalloader=testloader,
            criterion=loss,
            cm_name = cm_name_test,
            roc_name = roc_name_test
        )

        if valid_acc["avg_acc"] > best_valid_acc:
            best_valid_acc = valid_acc["avg_acc"]
            final_test_acc = test_acc
            os.makedirs("best_figures", exist_ok=True)
            for magnif in ["40X", "100X", "200X", "400X"]:
                best_cm_name_test = f"best_figures/cm_test_{fig_name}"
                command_cm = f"cp {cm_name_test.replace('.png', f'_{magnif}.png')} {best_cm_name_test.replace('.png', f'_{magnif}.png')}"
                os.system(command_cm)

            best_roc_name_test = f"best_figures/roc_test_{fig_name}"
            command_roc = f"cp {roc_name_test} {best_roc_name_test}"
            os.system(command_roc)


        train_acc_summary.append(train_acc)
        train_loss_summary.append(train_loss)
        valid_acc_summary.append(valid_acc)
        valid_loss_summary.append(valid_loss)

        end_time = time.time()  ## time at the end of epoch
        runtime = end_time - start_time  ## runtime of 1 epoch, in seconds
        runtime_mins = round(runtime / 60, 1)  ## runtime of 1 epoch, in minutes

        print_out(
            f"Epoch: {e} | Train loss: {train_loss} | Train acc: {train_acc} | Valid loss: {valid_loss} | Valid acc: {valid_acc}| Runtime: {runtime_mins} mins",
            log,
        )

    # Create table result
    df_epoch = pd.DataFrame({"epoch": range(1, n_epochs + 1)})
    df_train = pd.DataFrame(train_acc_summary)
    df_eval = pd.DataFrame(valid_acc_summary)
    df_train_summary = pd.concat([df_epoch, df_train], axis=1)
    df_eval_summary = pd.concat([df_epoch, df_eval], axis=1)

    print_out(f"Train summary: {df_train_summary}", log)
    print_out(f"Eval summary: {df_eval_summary}", log)

    print_out(f"Final test accuracy: {final_test_acc}", log)


    # Visulaize the results
    plot_train_eval_summary(df_train_summary, df_eval_summary)
    # Create new folder name figures to store accuracy plot
  
    plt.savefig(f"figures/Accuracy_{model}_epoch{n_epochs}_lr{lr}_{datetime_now}.png")

if __name__ == "__main__":
    # all_configs = mlp_get_grid_search(config_path="configs/mlp_cuda0.yaml",
    #                               optimizers=["sgd"],
    #                               learning_rates=[0.00001],
    #                               weight_decay=[0.00005]
    #                             )

    # # for i, config in tqdm(enumerate(all_configs, 1)):
    # #     print(f"Running config {i}/{len(all_configs)}")
    # #     main(config)

    all_configs = cnn_get_grid_search(config_path="configs/cnnv1.yaml",
                                    optimizers=["adam"],
                                    learning_rates=[0.0001],
                                    num_kernel_conv_list=[[32,64,64]],
                                    use_pooling=[True])
    
    for i, config in tqdm(enumerate(all_configs, 1)):
        print(f"Running config {i}/{len(all_configs)}")
        main(config)

   