from torchvision.transforms import ToTensor, ToPILImage, Resize, Compose, Normalize
from PIL import Image
from torch.nn.functional import pad
from torch.utils.data import default_collate
import matplotlib.pyplot as plt
import torch

def collate_wrapper(batch):
    batch_size = len(batch)
    max_img_width_batch = 0
    max_tg_len_batch = 0
    for img in batch:
        max_img_width_batch = max(img['image'].shape[2], max_img_width_batch)
        max_tg_len_batch = max(img['label'].shape[0], max_tg_len_batch)

    sequence_batch = batch
    for batch_idx in range(batch_size):
        item = sequence_batch[batch_idx]
        seq_len = item['image'].shape[2]
        label_len = item['label'].shape[0]
        image_path = item["image_path"].split('/')
        item['image_path'] = image_path[len(image_path)-1]
        item['input_lengths'] = seq_len
        item['target_lengths'] = label_len
        item['image'] = pad(input=item['image'], pad=(0, max_img_width_batch-seq_len), mode='constant', value=0)
        item['label'] = pad(input=item['label'], pad=(0, max_tg_len_batch-label_len), mode='constant', value=0)
    return default_collate(sequence_batch)


class PriMusDataset:
    def __init__(self, data: list, target_dict: dict, height: int = 61):
        self.img_label_paths = data
        self.target_dict = target_dict
        self.height = height

    def __getitem__(self, index):
        item = self.img_label_paths[index]
        img = Image.open(item[0])
        tensor_transform = ToTensor()
        img = tensor_transform(img)
        resize = Resize(size=self.height)
        img = resize(img)
        mean, std = img.mean([1, 2]), img.std([1, 2])
        normalize_transorfm = Normalize(mean, std)
        img = normalize_transorfm(img)

        with open(item[1], "r") as f:
            seq = f.read().split("\t")
            seq.pop()

        target = [self.target_dict[symb] for symb in seq]
        return {"image": img, "label": torch.tensor(target).long(), "image_path": item[0].split('/')[-1]}

    def show(self, idx: int):
        trans = ToPILImage()
        plt.figure(figsize=(10, 15))
        plt.imshow(trans(self[idx]['image']))
        label = open(self.img_label_paths[idx][1], "r").read().split("\t")
        label.pop()
        return label

    def __len__(self):
        return len(self.img_label_paths)