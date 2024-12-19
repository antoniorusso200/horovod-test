import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
import colossalai
from colossalai.cluster import DistCoordinator
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
import json
import numpy as np
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.booster.plugin.dp_plugin_base import DPPluginBase

# ==============================
# Prepare Hyperparameters
# ==============================
NUM_CLASSES = 10

def build_dataloader(batch_size: int, coordinator: DistCoordinator, plugin: DPPluginBase):
    # transform
    transform_test = torchvision.transforms.ToTensor()

    # CIFAR-10 dataset
    with coordinator.priority_execution():
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',  
            train=False,
            download=True,
            transform=transform_test
        )

    # Data loader
    test_dataloader = plugin.prepare_dataloader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return test_dataloader


@torch.no_grad()
def evaluate(model: nn.Module, test_dataloader: DataLoader, coordinator: DistCoordinator) -> float:
    model.eval()
    
    # Initialize tensors for metrics
    correct = torch.zeros(1, dtype=torch.int64, device=get_accelerator().get_current_device())
    total = torch.zeros(1, dtype=torch.int64, device=get_accelerator().get_current_device())
    
    class_correct = torch.zeros(10, dtype=torch.int64, device=get_accelerator().get_current_device())
    class_total = torch.zeros(10, dtype=torch.int64, device=get_accelerator().get_current_device())
    
    true_positives = torch.zeros(10, dtype=torch.int64, device=get_accelerator().get_current_device())
    false_positives = torch.zeros(10, dtype=torch.int64, device=get_accelerator().get_current_device())
    false_negatives = torch.zeros(10, dtype=torch.int64, device=get_accelerator().get_current_device())

    accuracies = []

    for images, labels in test_dataloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update per-class metrics
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1
            
            if predicted[i] == label:
                true_positives[label] += 1
            else:
                false_negatives[label] += 1
                false_positives[predicted[i]] += 1

        # Collect accuracy for each batch to calculate std later
        batch_accuracy = (predicted == labels).float().mean().item()
        accuracies.append(batch_accuracy)

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

    # Collect metrics for each class
    metrics = {}
    classes = (
        "plane", "car", "bird", "cat", "deer", "dog", "frog", 
        "horse", "ship", "truck"
    )
    for i in range(10):
        if class_total[i] > 0:
            class_accuracy = (class_correct[i] / class_total[i]).item()
            metrics[classes[i]] = {
                "accuracy": class_accuracy * 100,
                "precision": precision[i].item() * 100,
                "recall": recall[i].item() * 100,
                "f1_score": f1_score[i].item() * 100
            }
    
    # Calculate mean and std for F1 score, precision, recall, and accuracy across all classes
    avg_f1_score = f1_score.mean().item()
    avg_precision = precision.mean().item()
    avg_recall = recall.mean().item()
    avg_accuracy = accuracy

    f1_std = torch.std(f1_score).item()
    precision_std = torch.std(precision).item()
    recall_std = torch.std(recall).item()

    # Calculate std for accuracy across all batches
    std_accuracy = np.std(accuracies)

    avg_metrics = {
        "mean": {
            "accuracy": avg_accuracy * 100,
            "f1_score": avg_f1_score * 100,
            "precision": avg_precision * 100,
            "recall": avg_recall * 100
        },
        "std": {
            "accuracy": std_accuracy,
            "f1_score": f1_std,
            "precision": precision_std,
            "recall": recall_std
        }
    }

    # Prepare the final result structure
    result = {
        "per_class_metrics": metrics,
        "average_metrics": avg_metrics
    }

    # Save the results to a JSON file
    if coordinator.is_master():
        with open("evaluation_metrics.json", "w") as f:
            json.dump(result, f, indent=4)

    return accuracy

def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=10)
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="checkpoint directory")
    parser.add_argument("-b", "--batch_size", type=int, default=100, help="batch size for evaluation")
    parser.add_argument(
        "--plugin", type=str, default="torch_ddp", choices=["torch_ddp", "low_level_zero", "gemini"],
        help="plugin to use for distributed training"
    )
    args = parser.parse_args()

    # ==============================
    # Launch Distributed Environment
    # ==============================
    colossalai.launch_from_torch()
    coordinator = DistCoordinator()

    # ==============================
    # Instantiate Plugin and Booster
    # ==============================
    if args.plugin == "torch_ddp":
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)

    booster = Booster(plugin=plugin)

    # ==============================
    # Load model, optimizer, criterion
    # ==============================
    model = torchvision.models.resnet18(num_classes=NUM_CLASSES)

    # Boost with ColossalAI
    model, _, _, _, _ = booster.boost(model)

    # ==============================
    # Prepare Dataloader
    # ==============================
    test_dataloader = build_dataloader(args.batch_size, coordinator, plugin)

    # Load model checkpoint
    model_path = f"{args.checkpoint}/model_{args.epoch}.pth"
    if coordinator.is_master():
        print(f"Loading model from {model_path}")
    booster.load_model(model, model_path)

    # ==============================
    # Evaluate the model
    # ==============================
    evaluate(model, test_dataloader, coordinator)


if __name__ == "__main__":
    main()
