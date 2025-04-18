import os
from collections import Counter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

def get_data_loaders(data_dir, exclude_classes=None, batch_size=64, val_split=0.2, input_size=256):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    all_classes = os.listdir(data_dir)
    included_classes = [cls for cls in all_classes if cls not in (exclude_classes or [])]

    filtered_samples = []
    for cls in included_classes:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            filepath = os.path.join(cls_dir, fname)
            filtered_samples.append((filepath, included_classes.index(cls)))

    dataset = ImageFolder(data_dir, transform=transform)
    dataset.samples = filtered_samples
    dataset.targets = [label for _, label in filtered_samples]
    dataset.class_to_idx = {cls: idx for idx, cls in enumerate(included_classes)}

    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    target_list = [dataset.targets[i] for i in train_dataset.indices]
    class_counts = Counter(target_list)
    class_weights = {c: 1.0 / class_counts[c] for c in class_counts}
    sample_weights = [class_weights[label] for label in target_list]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, included_classes
