import os
import numpy as np
from tqdm import tqdm

from generator import generator
from discriminator import discriminator
from cifar10_featmap import CIFAR10_featmap as CIFAR10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

# Set Seeds
# np.random.seed(seed=123)
# torch.manual_seed(123)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Prefer GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Save paths
INIT_D_MODEL_CHECKPOINT_PATH = './saved_models/init_d/init_d_feats_seed123.sav'
CLASS_FEATS_VECTOR_PATH = './saved_models/class_feats_vectors/class_feats_vectors.sav'

# Hyperparameters
BATCH_SIZE = 100

INITIAL_D_LR = 0.001  # Train D on initial X_0 classes
INITIAL_D_EPOCHS = 32

ADV_LR = 0.0002
ADV_EPOCHS = 10

# Labels for real and fake
REAL_LABEL = 1.0
FAKE_LABEL = 0.0


def get_disc_cls_acc(model, loader):
    '''
    Param #loader must be set to testloader or trainloader
    '''
    model.eval()  # Turn off training mode.
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            featmaps, labels = data[1].to(device), data[2].to(
                device)  # data[1] = featmaps | data[2] = label
            _, logits_cls, _ = model(featmaps)
            _, predicted = torch.max(logits_cls.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return ('%.2f%%' % (100.0 * (correct / total)))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adversarial_loss(outputs, is_real, criterion=nn.BCELoss(), target_real_label=REAL_LABEL, target_fake_label=FAKE_LABEL):
    real_label = torch.tensor(target_real_label)
    fake_label = torch.tensor(target_fake_label)

    labels = (real_label if is_real else fake_label).expand_as(
        outputs).to(device)
    loss = criterion(outputs, labels)
    return loss


# Dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# trainset = CIFAR10(root='../data', train=True,
#                    download=True, transform=transform)
# testset = CIFAR10(root='../data', train=False,
#                   download=True, transform=transform)

# train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
#                                            shuffle=True, num_workers=0)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
#                                           shuffle=False, num_workers=0)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Models
disc = discriminator().to(device)
gen = generator().to(device)

# Optimizers & Schedulers
initial_d_optimizer = optim.Adam(
    disc.parameters(), lr=INITIAL_D_LR, betas=(0.5, 0.999))
intial_d_scheduler = optim.lr_scheduler.StepLR(
    initial_d_optimizer, step_size=8, gamma=0.5)

optimizer_g = optim.Adam(gen.parameters(), lr=ADV_LR, betas=(0.5, 0.999))
optimizer_d = optim.Adam(disc.parameters(), lr=ADV_LR, betas=(0.5, 0.999))

# Set models to training mode
disc.train()
gen.train()

# Loss functions
cls_criterion = nn.CrossEntropyLoss()
adv_criterion = nn.BCELoss()

# Replay buffer
class_feats_vectors = np.zeros((10, 5000, 512))
class_feats_counts = [0] * 10


def save_class_feats_vectors(save=True):
    # Save features for inputs to generator
    print("Saving feature vectors for inputs to generator")
    global class_feats_vectors
    global class_feats_counts

    class_feats_vectors = np.zeros((10, 5000, 512))
    class_feats_counts = [0] * 10

    for batch_idx, (inputs, featmaps, targets) in enumerate(tqdm(train_loader)):
        featmaps, targets = featmaps.to(device), targets.to(device)
        feats, _, _ = disc(featmaps)

        for i, target in enumerate(targets.detach().cpu().numpy().astype(int)):
            class_feats_vectors[target, class_feats_counts[target]
                                ] = feats[i].detach().cpu().numpy()
            class_feats_counts[target] += 1
    class_feats_vectors = np.array(class_feats_vectors)

    if save:
        torch.save({
            'class_feats_vectors': class_feats_vectors,
            'class_feats_counts': class_feats_counts
        }, CLASS_FEATS_VECTOR_PATH)
        print("Features vectors saved.")


def sample_vecs(batch_size):
    vecs = []
    vec_targets = []
    for _ in range(batch_size):
        label = np.random.randint(10)
        vecs.append(torch.Tensor(class_feats_vectors[label][np.random.choice(
            class_feats_vectors[label].shape[0], 1)][0]))
        # vecs.append(torch.Tensor(class_feats_vectors[label][np.random.choice(50, 1)][0]))
        vec_targets.append(torch.Tensor([label]))
    vecs = torch.stack(vecs)
    vec_targets = torch.stack(vec_targets).squeeze(1)
    # import pdb
    # pdb.set_trace()
    return vecs, vec_targets


def initial_d_train(save=True):
    print("Initial Discriminator Training Start!")

    for epoch in range(INITIAL_D_EPOCHS):
        print('[INITAL D TRAINING] EPOCH %d STARTED' % (epoch + 1))

        _loss_cls = 0.
        for batch_idx, (inputs, featmaps, targets) in enumerate(tqdm(train_loader)):
            featmaps, targets = featmaps.to(device), targets.to(device)

            initial_d_optimizer.zero_grad()
            feats, logits_cls, _ = disc(featmaps)

            # Only train discriminator on classification
            # No adversarial training here
            loss_cls = cls_criterion(logits_cls, targets.long())
            loss_cls.backward()
            initial_d_optimizer.step()

            _loss_cls += loss_cls.item()
            if batch_idx % 100 == 99:    # print every 100 mini-batches
                print('[INITIAL D TRAINING] EPOCH %d, MINI-BATCH %5d LR: %f loss: %.5f' %
                      (epoch + 1, batch_idx + 1, get_lr(initial_d_optimizer), _loss_cls / 100))
                _loss_cls = 0.0

        intial_d_scheduler.step()
    print("Finished training initial D.")

    if save:
        torch.save({
            'model_state_dict': disc.state_dict(),
            'optimizer_state_dict': initial_d_optimizer.state_dict(),
        }, INIT_D_MODEL_CHECKPOINT_PATH)
        print("Initial D model saved.")

    # Check initial discriminator classification accuracy
    INITIAL_D_ACC = get_disc_cls_acc(disc, test_loader)
    print("INITIAL D ACCURACY:", INITIAL_D_ACC)
    disc.train()


def load_init_d(path=INIT_D_MODEL_CHECKPOINT_PATH):
    checkpoint = torch.load(path)
    disc.load_state_dict(checkpoint['model_state_dict'])
    initial_d_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def load_class_feats_vectors(path=CLASS_FEATS_VECTOR_PATH):
    global class_feats_vectors
    global class_feats_counts

    checkpoint = torch.load(path)
    class_feats_vectors = checkpoint['class_feats_vectors']
    class_feats_counts = checkpoint['class_feats_counts']


def adversarial_train():
    for epoch in range(ADV_EPOCHS):
        _loss_g, _loss_cls_gen, _loss_adv_gen = 0., 0., 0.
        _loss_d, _loss_cls, _loss_adv = 0., 0., 0.

        for batch_idx, (inputs, featmaps, targets) in enumerate(tqdm(test_loader)):
            inputs, featmaps, targets = inputs.to(
                device), featmaps.to(device), targets.to(device)

            ''' Optimize Discriminator:  max log(B(I)) + log(1 - D(G(u | m_i, covmat_i))) '''
            optimizer_d.zero_grad()

            feats, logits_cls, p_adv = disc(featmaps)

            ''' Classification loss '''
            loss_cls = cls_criterion(logits_cls, targets.long())
            _loss_cls += loss_cls.item()

            ''' Adversarial loss '''
            # Real
            # ------------------
            disc_real_loss = adversarial_loss(
                p_adv, True, criterion=adv_criterion)
            # Average confidence of selecting a particular class (for printing purposes only)
            disc_real_loss_avg = disc_real_loss.mean().item()

            # Fake
            # ------------------
            # Generate examples
            sample_feats_vec, sample_targets = sample_vecs(inputs.shape[0])
            sample_feats_vec = sample_feats_vec.to(device)

            gen_feats_maps = gen(sample_feats_vec)

            gen_feats, gen_logits_cls, gen_logits_adv = disc(
                gen_feats_maps.detach())

            disc_fake_loss = adversarial_loss(
                gen_logits_adv, False, criterion=adv_criterion)
            disc_fake_loss_avg = disc_fake_loss.mean().item()

            loss_adv = disc_real_loss + disc_fake_loss
            _loss_adv += loss_adv.item()

            ''' Optimize parameters '''
            loss_adv.backward()
            optimizer_d.step()

            # Optimize Generator: max log(D(G(u | m_i, covmat_i)))
            gen.zero_grad()
            gen_feats, gen_logits_cls, gen_logits_adv = disc(gen_feats_maps)
            gen_loss = adversarial_loss(
                gen_logits_adv, False, criterion=adv_criterion)


if __name__ == '__main__':
    if os.path.exists(INIT_D_MODEL_CHECKPOINT_PATH):
        load_init_d()
    else:
        initial_d_train()

    if os.path.exists(CLASS_FEATS_VECTOR_PATH):
        load_class_feats_vectors()
    else:
        save_class_feats_vectors()

    print(class_feats_vectors)
    print(class_feats_counts)
    print(sample_vecs(10))

    # adversarial_train()
