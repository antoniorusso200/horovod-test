import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.booster.plugin.dp_plugin_base import DPPluginBase
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam



# ==============================
# Prepare Hyperparameters
# ==============================
LEARNING_RATE = 1e-3


def build_dataloader(batch_size: int, coordinator: DistCoordinator, plugin: DPPluginBase):
    # transform
    transform_train = transforms.Compose(
        [transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()]
    )
    transform_test = transforms.ToTensor()

    # CIFAR-10 dataset
    with coordinator.priority_execution():
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',  # Cambia la destinazione in una directory accessibile
            train=True,
            download=True,
            transform=transform_train
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',  # Cambia la destinazione in una directory accessibile
            train=False,
            download=True,
            transform=transform_test
        )


    # Data loader
    train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = plugin.prepare_dataloader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dataloader, test_dataloader


@torch.no_grad()
def evaluate(model: nn.Module, test_dataloader: DataLoader, coordinator: DistCoordinator) -> float:
    model.eval()
    
    # For total accuracy
    correct = torch.zeros(1, dtype=torch.int64, device=get_accelerator().get_current_device())
    total = torch.zeros(1, dtype=torch.int64, device=get_accelerator().get_current_device())

    # For per-class accuracy
    class_correct = torch.zeros(10, dtype=torch.int64, device=get_accelerator().get_current_device())
    class_total = torch.zeros(10, dtype=torch.int64, device=get_accelerator().get_current_device())

    # For calculating Precision, Recall, and F1 score
    true_positives = torch.zeros(10, dtype=torch.int64, device=get_accelerator().get_current_device())
    false_positives = torch.zeros(10, dtype=torch.int64, device=get_accelerator().get_current_device())
    false_negatives = torch.zeros(10, dtype=torch.int64, device=get_accelerator().get_current_device())

    for images, labels in test_dataloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Count per-class accuracy and update confusion matrix
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1
            
            if predicted[i] == label:
                true_positives[label] += 1
            else:
                false_negatives[label] += 1
                false_positives[predicted[i]] += 1

    # Perform distributed all-reduce
    dist.all_reduce(correct)
    dist.all_reduce(total)
    dist.all_reduce(class_correct)
    dist.all_reduce(class_total)
    dist.all_reduce(true_positives)
    dist.all_reduce(false_positives)
    dist.all_reduce(false_negatives)

    accuracy = correct.item() / total.item()

    # Calculate Precision, Recall, and F1 Score
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Print the results only for the master node
    if coordinator.is_master():
        print(f"Accuracy of the model on the test images: {accuracy * 100:.2f} %")
        
        # Print accuracy per class
        classes = (
            "plane", "car", "bird", "cat", "deer", "dog", "frog", 
            "horse", "ship", "truck"
        )
        for i in range(10):
            if class_total[i] > 0:
                class_accuracy = (class_correct[i] / class_total[i]).item()
                print(f"Accuracy of {classes[i]:>5s}: {class_accuracy * 100:.2f} %")
                print(f"Precision of {classes[i]:>5s}: {precision[i].item() * 100:.2f} %")
                print(f"Recall of {classes[i]:>5s}: {recall[i].item() * 100:.2f} %")
                print(f"F1 Score of {classes[i]:>5s}: {f1_score[i].item() * 100:.2f} %")
        
        # Print overall F1 Score (average over all classes)
        avg_f1_score = f1_score.mean().item()
        print(f"Average F1 Score: {avg_f1_score * 100:.2f} %")
#organizzare in un dizionario per gestirlo in un json
#nel dizionario metto una chiave (la classe), che punta un altro dizionario che ha una struttura fissa (accuracy ecc), queste chiavi interne mostrano il valore
#un altro dizionario metto le metriche medie (avg_f1_score ecc)
#memorizzo anche deviazione standard 
#chiave mean e std, seconda chiave accuracy_f1_score ecc
#per memorizzare due dizionari all'interno di json bisogna passarla come lista di dizionari
#vedere il file bash ()    
    return accuracy


def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
    num_epoch: int
):
    model.train()
    with tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{num_epoch}]", disable=not coordinator.is_master()) as pbar:
        for images, labels in pbar:
            images = images.cuda()
            labels = labels.cuda()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()

            # Print log info
            pbar.set_postfix({"loss": loss.item()})


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    # FIXME(ver217): gemini is not supported resnet now
    parser.add_argument(
        "-p",
        "--plugin",
        type=str,
        default="torch_ddp",
        choices=["torch_ddp", "torch_ddp_fp16", "low_level_zero", "gemini"],
        help="plugin to use",
    )
    parser.add_argument("-e", "--epoch", type=int, default=20)
    parser.add_argument("-r", "--resume", type=int, default=-1, help="resume from the epoch's checkpoint")
    parser.add_argument("-c", "--checkpoint", type=str, default="./checkpoint", help="checkpoint directory")
    parser.add_argument("-i", "--interval", type=int, default=5, help="interval of saving checkpoint")
    parser.add_argument(
        "--target_acc", type=float, default=None, help="target accuracy. Raise exception if not reached"
    )
    args = parser.parse_args()

    # ==============================
    # Prepare Checkpoint Directory
    # ==============================
    if args.interval > 0:
        Path(args.checkpoint).mkdir(parents=True, exist_ok=True)

    # ==============================
    # Launch Distributed Environment
    # ==============================
    colossalai.launch_from_torch()
    coordinator = DistCoordinator()

    # update the learning rate with linear scaling
    # old_gpu_num / old_lr = new_gpu_num / new_lr
    global LEARNING_RATE
    LEARNING_RATE *= coordinator.world_size

    # ==============================
    # Instantiate Plugin and Booster
    # ==============================
    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        booster_kwargs["mixed_precision"] = "fp16"
    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)

    booster = Booster(plugin=plugin, **booster_kwargs)

    # ==============================
    # Prepare Dataloader
    # ==============================
    train_dataloader, test_dataloader = build_dataloader(100, coordinator, plugin)

    # ====================================
    # Prepare model, optimizer, criterion
    # ====================================
    # resent50
    model = torchvision.models.resnet18(num_classes=10)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = HybridAdam(model.parameters(), lr=LEARNING_RATE)

    # lr scheduler
    lr_scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=1 / 3)

    # ==============================
    # Boost with ColossalAI
    # ==============================
    model, optimizer, criterion, _, lr_scheduler = booster.boost(
        model, optimizer, criterion=criterion, lr_scheduler=lr_scheduler
    )

    # ==============================
    # Resume from checkpoint
    # ==============================
    if args.resume >= 0:
        booster.load_model(model, f"{args.checkpoint}/model_{args.resume}.pth")
        booster.load_optimizer(optimizer, f"{args.checkpoint}/optimizer_{args.resume}.pth")
        booster.load_lr_scheduler(lr_scheduler, f"{args.checkpoint}/lr_scheduler_{args.resume}.pth")

    # ==============================
    # Train model
    # ==============================
    start_epoch = args.resume if args.resume >= 0 else 0
    for epoch in range(start_epoch, args.epoch):
        train_epoch(epoch, model, optimizer, criterion, train_dataloader, booster, coordinator,args.epoch)
        lr_scheduler.step()

        # save checkpoint
        if args.interval > 0 and (epoch + 1) % args.interval == 0:
            booster.save_model(model, f"{args.checkpoint}/model_{epoch + 1}.pth")
            booster.save_optimizer(optimizer, f"{args.checkpoint}/optimizer_{epoch + 1}.pth")
            booster.save_lr_scheduler(lr_scheduler, f"{args.checkpoint}/lr_scheduler_{epoch + 1}.pth")

  

if __name__ == "__main__":
    main()
