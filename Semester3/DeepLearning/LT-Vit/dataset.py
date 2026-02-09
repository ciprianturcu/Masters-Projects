import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]


class NIHChestXrayDataset(Dataset):

    def __init__(self, data_dir, split="train", image_size=224,
                 val_ratio=0.1, seed=42, preprocessed=False, memmap_dir=None):
        super().__init__()
        assert split in ("train", "val", "test")
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.preprocessed = preprocessed
        self.use_memmap = memmap_dir is not None

        csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
        df = pd.read_csv(csv_path)

        for label in LABELS:
            df[label] = df["Finding Labels"].map(lambda s, l=label: 1.0 if l in s else 0.0)

        train_val_list = set(
            open(os.path.join(data_dir, "train_val_list.txt")).read().strip().split("\n")
        )
        test_list = set(
            open(os.path.join(data_dir, "test_list.txt")).read().strip().split("\n")
        )

        if split == "test":
            df = df[df["Image Index"].isin(test_list)].reset_index(drop=True)
        else:
            df = df[df["Image Index"].isin(train_val_list)].reset_index(drop=True)

            patient_ids = df["Patient ID"].unique()
            rng = np.random.RandomState(seed)
            rng.shuffle(patient_ids)
            n_val = int(len(patient_ids) * val_ratio)
            val_patients = set(patient_ids[:n_val])
            train_patients = set(patient_ids[n_val:])

            if split == "val":
                df = df[df["Patient ID"].isin(val_patients)].reset_index(drop=True)
            else:
                df = df[df["Patient ID"].isin(train_patients)].reset_index(drop=True)

        self.filenames = df["Image Index"].tolist()
        self.labels = df[LABELS].values.astype(np.float32)

        if self.use_memmap:
            with open(os.path.join(memmap_dir, "index.json")) as f:
                meta = json.load(f)
            self._memmap_path = os.path.join(memmap_dir, "images.dat")
            self._memmap_shape = tuple(meta["shape"])
            self._memmap_dtype = meta["dtype"]
            self._memmap = None
            fname_to_idx = meta["fname_to_idx"]
            self.memmap_indices = np.array(
                [fname_to_idx[fn] for fn in self.filenames], dtype=np.int32
            )
        else:
            self.image_paths = {}
            for folder in sorted(os.listdir(data_dir)):
                folder_path = os.path.join(data_dir, folder, "images")
                if os.path.isdir(folder_path):
                    for fname in os.listdir(folder_path):
                        self.image_paths[fname] = os.path.join(folder_path, fname)
                        if fname.lower().endswith(".jpg"):
                            png_name = os.path.splitext(fname)[0] + ".png"
                            self.image_paths[png_name] = os.path.join(folder_path, fname)

        resize_ops = [] if (preprocessed or self.use_memmap) else [
            transforms.Resize((image_size, image_size))
        ]

        if split == "train":
            self.transform = transforms.Compose(resize_ops + [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose(resize_ops + [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.use_memmap:
            if self._memmap is None:
                self._memmap = np.memmap(
                    self._memmap_path, dtype=self._memmap_dtype,
                    mode="r", shape=self._memmap_shape,
                )
            arr = self._memmap[self.memmap_indices[idx]]
            img = Image.fromarray(arr, mode="RGB")
        else:
            fname = self.filenames[idx]
            img_path = self.image_paths[fname]
            img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label
