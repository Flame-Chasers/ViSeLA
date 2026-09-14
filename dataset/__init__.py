import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.tokenizers import build_tokenizer
from dataset.pretrain_dataset import ImageTextJsonDataset, RegionTextJsonDataset, TextJsonDataset, FrameTextDataset
from dataset.randaugment import RandomAugment


def create_dataset(dataset, config, evaluate=False):

    if dataset == 'pretrain_text':
        text_dataset = TextJsonDataset(config, config['train_file_text'], rank=int(os.environ.get('RANK') or 0),
                                       world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True)
        return text_dataset

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    box_transform = transforms.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'pretrain':
        if len(config.get('train_file', [])):
            image_dataset = ImageTextJsonDataset(config, config['train_file'], rank=int(os.environ.get('RANK') or 0),
                                                   world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                                   transform=pretrain_transform)
        else:
            image_dataset = None

        if len(config.get('train_file_regions', [])):
            region_dataset = RegionTextJsonDataset(config, config['train_file_regions'], rank=int(os.environ.get('RANK') or 0),
                                                    world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                                    transform=pretrain_transform, box_transform=box_transform)
        else:
            region_dataset = None

        if len(config.get('train_file_aux', [])):
            image_dataset_aux = ImageTextJsonDataset(config, config['train_file_aux'], rank=int(os.environ.get('RANK') or 0),
                                                   world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                   repeat=True, transform=pretrain_transform, is_aux=True)
        else:
            image_dataset_aux = None

        if len(config.get('train_file_videos', [])):
            video_dataset = FrameTextDataset(config, config['train_file_videos'], rank=int(os.environ.get('RANK') or 0),
                                                 world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                 repeat=True, transform=pretrain_transform, training=True)
        else:
            video_dataset = None

        if len(config.get('train_file_videos_aux', [])):
            video_dataset_aux = FrameTextDataset(config, config['train_file_videos_aux'], rank=int(os.environ.get('RANK') or 0),
                                                 world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                 repeat=True, transform=pretrain_transform, training=True)
        else:
            video_dataset_aux = None

        if len(config.get('train_file_text', [])):
            text_dataset = TextJsonDataset(config, config['train_file_text'], rank=int(os.environ.get('RANK') or 0),
                                           world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True)
        else:
            text_dataset = None

        # mtext_dataset not used for now
        mtext_dataset = None

        return image_dataset, region_dataset, text_dataset, image_dataset_aux, video_dataset, video_dataset_aux, mtext_dataset

    else:
        raise NotImplementedError(f"dataset == {dataset}")


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders
