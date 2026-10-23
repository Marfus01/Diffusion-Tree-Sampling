from pathlib import Path
import copy
import logging

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from tqdm import tqdm


logger = logging.getLogger(__name__)

from src.utils.datasets import get_img_dataset


class CNN(nn.Module):
    def __init__(self, input_size, channels, num_classes):
        super(CNN, self).__init__()
        self.input_shape = (channels, input_size, input_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.embed = nn.Sequential(self.conv1, self.conv2)

        x = self.embed(torch.zeros((1,) + tuple(self.input_shape)))
        input_dim = x.view(x.size(0), -1).shape[-1]
        self.out = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        output = self.out(x)
        return output


class ResNet(nn.Module):

    name = "ResNet"

    def __init__(self, input_size, channels, num_classes, depth=18, finetune=False):
        super(ResNet, self).__init__()
        self.input_shape = (channels, input_size, input_size)
        self.depth = depth
        self.num_classes = num_classes

        # create the embedding (sub) model
        self.resnet = self._init_resnet(finetune)

        if finetune:
            for name, param in self.resnet.named_parameters():
                param.requires_grad = False

        # we dynamically create readout layer dimensions
        x = self.resnet(torch.zeros((2,) + tuple(self.input_shape)))

        # make sure to set the input dimension, so the paent class knows how to build the models
        input_dim = x.view(x.size(0), -1).shape[-1]

        # creates and initializes all models and associated optimizers within ensemble
        self.out_layer = self._create_readout_layer(input_dim, num_classes)

    def _init_resnet(self, finetune=False):

        # Dynamically select the ResNet architecture
        resnet_class = getattr(models, f'resnet{self.depth}')

        # Initialize a pre-trained model
        resnet_model = resnet_class(weights=models.ResNet18_Weights.DEFAULT if finetune else None)
        # resnet_model = resnet_class(weights=models.ResNet18_Weights.DEFAULT)

        # Modify the first convolutional layer (for different input channels)
        if self.input_shape[0] != 3:
            resnet_model.conv1 = torch.nn.Conv2d(self.input_shape[0], 64,
                                                 kernel_size=(7, 7),
                                                 stride=(2, 2),
                                                 padding=(0, 3),
                                                 bias=False)

        modules = list(resnet_model.children())[:-1]  # Exclude the last fc layer
        resnet_model = nn.Sequential(*modules)

        return resnet_model

    def _create_readout_layer(self, input_dimension, output_dimension):
        return nn.Sequential(nn.Flatten(),
                             nn.Linear(input_dimension, output_dimension))

    def forward(self, x):
        x = self.resnet(x)
        return self.out_layer(x)


class CIFARResNet18(nn.Module):
    """Pretrained CIFAR-10 ResNet-18 of edadaltocg/resnet18_cifar10 (torchvision resnet18 with a
    3x3 stride-1 stem and no max-pool). Takes images in [-1, 1]; the network expects [0, 1]
    inputs normalized by the CIFAR-10 mean and std below, so both the reward and its gradient go
    through that conversion."""

    name = "resnet18_cifar10_edadaltocg"

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = models.resnet18(num_classes=num_classes)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        self.register_buffer('mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net(((x + 1) / 2 - self.mean) / self.std)


def train_classifier(model, config, patience=20, epochs=100, device='cpu'):
    """Train a classifier model."""

    model = nn.DataParallel(model).to(device)

    # Load dataset
    data = get_img_dataset(
        config=config,
    )

    train_loader = data.get('train_loader', None)
    valid_loader = data.get('valid_loader', data.get('test_loader', None))
    validate = True

    if train_loader is None:
        raise ValueError("Training loader not found in dataset dictionary.")
    if valid_loader is None:
        logger.info("Warning: Validation/Test loader not found. Skipping validation.")
        validate = False

    patience_lv = 0
    best_params = None
    best_epoch = 0
    best_acc = None

    opt_classifier = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Train")):
            x = batch[config.dataset.x_tensor].to(device)
            y = batch[config.dataset.y_tensor].to(device).flatten()

            logits = model(x)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(1) == y).float().mean()
            train_loss += loss.item()
            train_acc += acc.item()

            opt_classifier.zero_grad()
            loss.backward()
            opt_classifier.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        valid_loss = 0.
        valid_acc = 0.
        if validate:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(valid_loader, desc=f"Epoch {epoch} Valid")):
                    x = batch[config.dataset.x_tensor].to(device)
                    y = batch[config.dataset.y_tensor].to(device).flatten()

                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    acc = (logits.argmax(1) == y).float().mean()
                    valid_loss += loss.item()
                    valid_acc += acc.item()

            valid_loss /= len(valid_loader)
            valid_acc /= len(valid_loader)

            if best_acc is None or best_acc < valid_acc:
                best_params = copy.deepcopy(model.module.state_dict())   # state_dict tensors alias the live weights
                best_epoch = epoch
                best_acc = valid_acc
                patience_lv = 0
            else:
                patience_lv += 1

        logger.info(f"epoch: {epoch}, patience: {patience_lv}/{patience}, "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
              f"valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}")

        if patience_lv >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_params is None:
        best_params = model.module.state_dict()
        best_epoch = epoch

    return best_params, best_epoch


def get_train_classifier(config):
    """Retrieve a classifier if checkpoint exists, else train one."""
    logger.info("--- Getting Classifier --- ")

    if config.model.classifier_model == CIFARResNet18.name:
        if not config.paths.classifier_load_path or not Path(config.paths.classifier_load_path).exists():
            raise FileNotFoundError(f"{CIFARResNet18.name} needs --classifier_path to its pytorch_model.bin")
        classifier = CIFARResNet18(num_classes=config.dataset.num_classes)
        classifier.net.load_state_dict(torch.load(config.paths.classifier_load_path, map_location='cpu'))
        logger.info(f"Loaded pretrained {CIFARResNet18.name} from {config.paths.classifier_load_path}")
        classifier = classifier.to(config.common.device).eval()
        for p in classifier.parameters():
            p.requires_grad = False
        return classifier

    # Instantiate the model based on config
    if 'cnn' in config.model.classifier_model.lower():
        classifier = CNN(
            input_size=config.dataset.image_size,
            channels=config.dataset.channels,
            num_classes=config.dataset.num_classes
        )
    elif 'resnet' in config.model.classifier_model.lower():
        classifier = ResNet(
            input_size=config.dataset.image_size,
            channels=config.dataset.channels,
            num_classes=config.dataset.num_classes,
            depth=config.model.classifier_depth,
            finetune=config.model.classifier_pretrained
        )
    else:
        raise ValueError(f"Unsupported classifier model type: {config.model.classifier_model}")

    classifier_load_path = config.paths.classifier_load_path
    do_training = False

    if not classifier_load_path:
        classifier_ckpt_path = Path(config.paths.checkpoint_dir) / f"{config.dataset.name}_{config.model.classifier_model}_best.pt"
        logger.info(f"No valid classifier load path provided. Saving to: {classifier_ckpt_path}")
        do_training = True
    else:
        classifier_ckpt_path = Path(classifier_load_path)
        logger.info(f"Loading existing classifier from: {classifier_ckpt_path}")
        try:
            classifier.load_state_dict(
                torch.load(classifier_ckpt_path, map_location='cpu')
            )
            logger.info(f"Successfully loaded classifier from {classifier_ckpt_path}")
        except Exception as e:
            logger.info(f"Failed to load classifier checkpoint from {classifier_ckpt_path}. Training new model. Error: {e}")
            do_training = True

    if do_training:
        classifier_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        trained_params, best_epoch = train_classifier(
            model=classifier,
            config=config,
            device=config.common.device,
            epochs=config.model.classifier_epochs,
            patience=config.model.classifier_patience
        )

        torch.save(trained_params, classifier_ckpt_path)
        logger.info(f"Training complete. Best classifier at epoch {best_epoch} saved to {classifier_ckpt_path}")
        classifier.load_state_dict(trained_params)

    logger.info(f'Classifier - Total params: {(sum(p.numel() for p in classifier.parameters()) / 1e6):.2f}M ')

    classifier = classifier.to(config.common.device)
    classifier.eval()

    for p in classifier.parameters():
        p.requires_grad = False

    return classifier