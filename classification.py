# 1. read images
# 2. create loader object (iterable of train and test sets)
# 3. finetune classifier 

from pathlib import Path
import yaml 
import logging
import argparse

from reader import ImageReader
from loader import ImageLoader, get_sampling_weights
from cnn_model import VGG_finetuned, seed_worker
from vgg_trainer import ModelTrainer

from collections import Counter
import numpy as np
import torch
from torch.utils.data import random_split,  WeightedRandomSampler, DataLoader
from torchvision import transforms as T

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

print('creating parser')
parser = argparse.ArgumentParser(description="DESCCRIPTION HERE")
parser.add_argument("--root", type=str, help="root directory where data and eval folders exists.")
parser.add_argument("--data", type=str, help="directory where images exists.")
parser.add_argument("--config_file", type=str, help="File containing run configurations.")
args = parser.parse_args()

ROOT = Path(args.root)
DATA = ROOT.joinpath(args.data)

CONFIG_FILE = ROOT.joinpath(args.config_file)
with open(CONFIG_FILE, "r") as f:
    CONFIG = yaml.safe_load(f)
print(CONFIG)

SAVEPATH = ROOT.joinpath(f"cnn_models")
SAVEPATH.mkdir(parents=True, exist_ok=True)

fileids = [i for i in DATA.glob('*.png')]

if __name__ == '__main__':

    print('\nreading images')
    # =======================
    # to apply different transforms to train and validation, create different folders
    # to add images from other datasets to have varied data for training
    reader = ImageReader(ROOT, fileids)
    images = list(reader._read_images(fileids))
    labels = list(reader.labels(fileids))
    print(f"No. of images {len(images)} and No. of labels {len(labels)}")
    item_count = Counter(labels)
    print(f"Item frequency: {item_count}")

    print('\ngetting data loaders for training')
    # ==========================================
    data_config = CONFIG['DATA']
    train_config = CONFIG['TRAINING_CNN']
    transform = T.Compose([T.ToPILImage(),
                           T.Resize((224, 224)),
                           T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                           ])

    # try augmentation just on the training set
    # if with_augmentation:
    #     train_transform = create_transform(224, is_training=True)
    # else:
    #     train_transform = create_transform(224)
    # # Create datasets
    # train_dataset = PetsDataset(imbalanced_train_df, label_to_id, data_path, train_transform)

    loader = ImageLoader(images, labels, transform) 
    print(f"data shape: {loader.__getitem__(1)[0].shape}")
    
    # data split
    N = len(images)
    n_train = int(N*train_config['TRAIN_SIZE'])
    n_valid = N - n_train
    
    g_seed = torch.Generator()
    g_seed.manual_seed(data_config['SEED'])
    train, valid = random_split(loader, [n_train, n_valid])
    print(f"training and validation size: {len(train)}, {len(valid)}")
    
    # Create sampler and dataloaders 
    train_labels = [i[1] for i in train]
    class_count = Counter(train_labels)
    print(f"Item frequency in training: {class_count}")
    if data_config['SAMPLE']==1:
        # weighted sampler to have balanced class 
        sample_weights = get_sampling_weights(len(train_labels), train_labels)
        sample_weights = torch.DoubleTensor(sample_weights)
        # print(sample_weights)
        # print(train_labels)
        
        # try weighted sampler to oversample minority class and test performance
        # with and without augmentation on the training set
        # target_ratio = {0:8.0256, 1:1.1423}
        # sample_weights = [(target_ratio[i])*1/class_count[i] for i in train_labels]
        # sample_weights = torch.DoubleTensor(sample_weights)
        # print(sample_weights)
        
        # # for increasing the probablity of seeing all images from majority class
        # # i.e. images are sampled with varying frequency and by the end of each epoch
        # #      model may not have seen all images
        # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=int(2*len(sample_weights)))  
        
        # for upsampling the minority class and create more balanced dataset
        # all images may not be seen in an epoch 
        # takes about 10 epoch for the model to see all images
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)                                                                                        
        train_dataloader = DataLoader(train, batch_size=train_config['BATCH_SIZE'], sampler=sampler,
                                      num_workers=2, worker_init_fn=seed_worker, generator=g_seed)  
    else:
        train_dataloader = DataLoader(train, batch_size=train_config['BATCH_SIZE'], shuffle=True,
                                      num_workers=2, worker_init_fn=seed_worker, generator=g_seed)
    # do not sample trainingto replicate real data
    valid_dataloader = DataLoader(valid, batch_size=train_config['BATCH_SIZE'], shuffle=True,
                                  num_workers=2, worker_init_fn=seed_worker, generator=g_seed)

    # train
    print('\nfinetune vgg on custom data')
    # ====================================
    # set seed for training 
    model = VGG_finetuned(num_classes=len(item_count))
    # trainer = ModelTrainer(model, train_config, train_dataloader, valid_dataloader)
    model_name = f"vgg16_finetuned{train_config['MODEL_NAME']}.pth"
    trainer = ModelTrainer(model, train_config, train_dataloader, valid_dataloader, savepath=SAVEPATH.joinpath(model_name))
    fine_tuned = trainer.train_eval()

    # for scores in score_models(binary_models, loader):
    #     print(scores)
    #     result_filename = 'TRACE_results'+str(i)+'.json'
    #     with open(Path.joinpath(RESULTS,result_filename), 'a') as f:
    #         f.write(json.dumps(scores) + '\n')