import os
from torch.utils.data import Dataset
from PIL import Image
import json
from torch.utils.data import DataLoader
from torchvision import transforms


class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "Labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                target = self.syn_to_class[entry]
                sample_path = os.path.join(samples_dir, entry)
                for sample in os.listdir(sample_path):
                    self.samples.append(os.path.join(samples_dir, entry, sample))
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


def get_dataset(batch_size, num_workers, data_path, split='val', shuffle=False, drop_last=False, pin_memory=True):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset = ImageNetKaggle(data_path, split, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # may need to reduce this depending on your GPU
        num_workers=num_workers,  # may need to reduce this depending on your num of CPUs and RAM
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory
    )

    return dataset, dataloader
