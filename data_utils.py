import math    
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image, ImageEnhance, ImageOps
import random
import numpy as np
import dill as pickle
import multiprocess as multiprocessing
from datasets import load_dataset, DatasetDict, load_from_disk
from pathlib import Path
import os
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, RobertaTokenizer, RobertaForSequenceClassification, DefaultDataCollator, DataCollatorWithPadding

def generate_quadratic(dim:int, size:int):
    # Linear regression problem
    data_X = torch.randn(size, dim)/math.sqrt(dim)
    hessian = torch.diag(torch.sqrt(torch.linspace(dim*2,dim,dim)))/dim
    # hessianR = torch.diag(dim,dim)/dim
    data_X= torch.matmul(data_X, hessian)
    param = torch.randn(dim)
    data_y = torch.matmul(data_X, param) #+ torch.randn(size)/size
    hessian = torch.matmul(data_X.T, data_X)
    # Lambda, MatR = torch.linalg.eigh(hessian)
    # print(torch.linalg.norm(hessian - torch.matmul(torch.matmul(MatR.T, torch.diag(Lambda)),MatR)))
    # print(Lambda)
    return data_X, data_y, hessian, param

class synthetic(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y.unsqueeze(1)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.y)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

class glue_data_collator():
    def __init__(self, tokenizer, max_seq_length) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.warning_state = self.tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    
    def __call__(self, features):
        batch = [{k: feature[k] for k in ["input_ids", "attention_mask","label"]} for feature in features]
        try:
            batch = self.tokenizer.pad(batch, max_length=self.max_seq_length)
        finally:
            # Restore the state of the warning.
            self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = self.warning_state
        batch['labels'] = batch.pop('label')
        # first = features[0]
        # batch = {}
        # if "label" in first and first["label"] is not None:
        #     label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        #     dtype = torch.long if isinstance(label, int) else torch.float
        #     batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        # elif "label_ids" in first and first["label_ids"] is not None:
        #     if isinstance(first["label_ids"], torch.Tensor):
        #         batch["labels"] = torch.stack([f["label_ids"] for f in features])
        #     else:
        #         dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
        #         batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # try:
        #     padded = self.tokenizer.pad()
        # finally:
        #     # Restore the state of the warning.
        #     self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = self.warning_state

        # for k, v in first.items():
        #     if k in ("input_ids", "attention_mask") and v is not None and not isinstance(v, str):
        #         if isinstance(v, torch.Tensor):
        #             batch[k] = torch.stack([f[k] for f in features])
        #         elif isinstance(v, np.ndarray):
        #             batch[k] = torch.tensor(np.stack([f[k] for f in features]))
        #         else:
        #             batch[k] = torch.tensor([f[k] for f in features])

        return batch

def generate_glue(dataset, data_path='./data', batch_size = 32, max_sequence_length = 256):
    if not os.path.isdir(Path(data_path+'/tokenizer')):
        tokenizer = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base', truncation=True, do_lower_case=True)
        os.makedirs(Path(data_path+'/tokenizer'))
        tokenizer.save_pretrained(Path(data_path+'/tokenizer'))
    else:
        tokenizer = RobertaTokenizer.from_pretrained(Path(data_path+'/tokenizer'))
    if not os.path.isdir(Path(data_path+'/mapped/'+dataset)):
        raw_data = load_dataset(
                "nyu-mll/glue",
                dataset,
                cache_dir=data_path
            )
        label_list = raw_data["train"].features["label"].names
        # num_labels = len(label_list)
        label_to_id = {v: i for i, v in enumerate(label_list)}
        # print(label_list, label_to_id)
        sentence1_key, sentence2_key = task_to_keys[dataset]
        max_seq_length = min(tokenizer.model_max_length, max_sequence_length)

    # tokenizer.pad_token = tokenizer.eos_token
        def tokenize(examples):
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=False, max_length=max_seq_length, truncation=True)
            # print(examples["label"])
            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = examples["label"]
            result = {k: result[k] for k in ["input_ids", "attention_mask","label"]}
            return result
        tokenized_datasets = raw_data.map(
            tokenize, batched=False, #cache_file_names=cache_file_names
        )
        tokenized_datasets.set_format("torch")
        os.makedirs(Path(data_path+'/mapped/'+dataset))
        tokenized_datasets.save_to_disk(Path(data_path+'/mapped/'+dataset))
    else:
        tokenized_datasets = load_from_disk(Path(data_path+'/mapped/'+dataset))
    label_list = tokenized_datasets["train"].features["label"].names
    label_to_id = {v: i for i, v in enumerate(label_list)}
    num_labels = len(label_list)
    data_collator = glue_data_collator(tokenizer, min(tokenizer.model_max_length, max_sequence_length))#DataCollatorWithPadding(tokenizer, max_length=tokenizer.model_max_length)
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["validation_matched" if dataset == "mnli" else "validation"], batch_size=batch_size, collate_fn=data_collator)
    return train_dataloader, eval_dataloader, tokenizer, label_to_id, num_labels

def generate_imgnet1k(batchsize, datapath):
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method("spawn")
    trans_imgnet =  transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    trans_imgnet_train = transforms.Compose([transforms.Resize(256),ImageNetPolicy(), transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset_train = datasets.ImageNet(datapath+'/imgnet1k', split='train', transform = trans_imgnet_train)
    dataset_test = datasets.ImageNet(datapath+'/imgnet1k', split='val', transform=trans_imgnet)
    def train_loader():
        return DataLoader(dataset_train,batch_size=batchsize,shuffle=True,drop_last=True, pin_memory = True,num_workers=4)
    def test_loader():
        return DataLoader(dataset_test,batch_size=batchsize,shuffle=False,drop_last=False, pin_memory = True)
    return train_loader, test_loader

def generate_Mnist(batchsize, datapath):
    trans_mnist = transforms.ToTensor()#transforms.Resize(224)
    dataset_train = datasets.MNIST(datapath+'/mnist', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST(datapath+'/mnist', train=False, download=True, transform=trans_mnist)
    train_loader = DataLoader(dataset_train,batch_size=batchsize,shuffle=True,drop_last=False, pin_memory = True,num_workers=4)
    test_loader = DataLoader(dataset_test,batch_size=batchsize*4,shuffle=False,drop_last=False, pin_memory = True)
    return train_loader, test_loader


def generate_Cifar(batchsize, dataset, model, data_path):
    if model == 'cnn5' or model.startswith('wrn'):
        size = 32
    else:
        size = 224
    trans_cifar = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])#transforms.Resize(224)
    trans_cifar_train = transforms.Compose([transforms.Resize(size),CIFAR10Policy(), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    if dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(data_path+'/cifar100', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR100(data_path+'/cifar100', train=False, download=True, transform=trans_cifar)
    else:
        dataset_train = datasets.CIFAR10(data_path+'/cifar10', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10(data_path+'/cifar10', train=False, download=True, transform=trans_cifar)
    train_loader = DataLoader(dataset_train,batch_size=batchsize,shuffle=True,drop_last=False, pin_memory = True,num_workers=4)
    test_loader = DataLoader(dataset_test,batch_size=batchsize*4,shuffle=False,drop_last=False, pin_memory = True)
    return train_loader, test_loader

# def generate_Imagenet(batchsize, dataset):


from PIL import Image, ImageEnhance, ImageOps
import random


class ShearX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)


class ShearY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)


class TranslateX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=self.fillcolor)


class TranslateY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
            fillcolor=self.fillcolor)


class Rotate(object):
    # from https://stackoverflow.com/questions/
    # 5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
    def __call__(self, x, magnitude):
        rot = x.convert("RGBA").rotate(magnitude * random.choice([-1, 1]))
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(x.mode)


class Color(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Color(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Posterize(object):
    def __call__(self, x, magnitude):
        return ImageOps.posterize(x, magnitude)


class Solarize(object):
    def __call__(self, x, magnitude):
        return ImageOps.solarize(x, magnitude)


class Contrast(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Contrast(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Sharpness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Sharpness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Brightness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Brightness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class AutoContrast(object):
    def __call__(self, x, magnitude):
        return ImageOps.autocontrast(x)


class Equalize(object):
    def __call__(self, x, magnitude):
        return ImageOps.equalize(x)


class Invert(object):
    def __call__(self, x, magnitude):
        return ImageOps.invert(x)


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform = transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        func = {
            "shearX": ShearX(fillcolor=fillcolor),
            "shearY": ShearY(fillcolor=fillcolor),
            "translateX": TranslateX(fillcolor=fillcolor),
            "translateY": TranslateY(fillcolor=fillcolor),
            "rotate": Rotate(),
            "color": Color(),
            "posterize": Posterize(),
            "solarize": Solarize(),
            "contrast": Contrast(),
            "sharpness": Sharpness(),
            "brightness": Brightness(),
            "autocontrast": AutoContrast(),
            "equalize": Equalize(),
            "invert": Invert()
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img