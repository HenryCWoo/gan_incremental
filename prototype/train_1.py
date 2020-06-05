import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from generator import generator
from discriminator import discriminator
from feat_vecs_loader import FeatVecsDataset
from cifar10_featmap import CIFAR10_featmap as CIFAR10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

# Set Seeds
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed=123)
torch.manual_seed(456)
torch.cuda.manual_seed_all(789)

# Prefer GPU
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Save paths
attempt = 1
CLS_MODEL_CHECKPOINT_PATH = './saved_models/cls_model_norm.sav'
ADV_MODEL_CHECKPOINT_PATH = './saved_models/adv_model_v%s.sav' % attempt
CLASS_FEAT_OLD_VECS_PATH = './saved_models/class_feat_vecs/class_feat_old_vecs_norm.sav'
CLASS_FEAT_NEW_VECS_PATH = './saved_models/class_feat_vecs/class_feat_new_vecs_v%s.sav' % attempt

# Hyperparameters
CLASSES = 10
BATCH_SIZE = 100

CLS_LR = 0.001  # Retrain / Initialize training on D for X classes
CLS_EPOCHS = 16  # TODO: Increase classifier epochs when performing actual experiments

GEN_LR = 0.0002
DISC_LR = 0.05
ADV_EPOCHS = 90

# Labels for real and fake
REAL_LABEL = 0.1  # TODO: Try out 0.9 and 0.1 for less confident adversary
FAKE_LABEL = 0.9


class Prototype():
    def __init__(self, optimizer='SGD'):
        # Models
        self.disc = discriminator().to(device)
        self.gen = generator(deconv=False).to(device)

        # Loss functions
        self.cls_criterion = nn.CrossEntropyLoss()
        self.adv_criterion = nn.BCEWithLogitsLoss()

        # Replay buffer
        self.train_class_feat_vecs = []
        self.test_class_feat_vecs = []

        # Data loaders
        self.train_loader = None
        self.test_loader = None
        self._init_data_loaders()

        # Optimizers & Schedulers
        self._init_cls_optim()
        self._init_gen_optim(optimizer=optimizer)
        self._init_disc_optim(optimizer=optimizer)

    def _init_data_loaders(self):
        # Dataloader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        trainset = CIFAR10(root='../data', train=True,
                                download=True, transform=transform)
        testset = CIFAR10(root='../data', train=False,
                               download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                                        shuffle=True, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                                       shuffle=False, num_workers=0)

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _init_feat_vecs_loaders(self):
        self.train_feat_vecs_set = FeatVecsDataset(self.train_class_feat_vecs)
        self.train_feat_vecs_loader = torch.utils.data.DataLoader(
            self.train_feat_vecs_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        self.test_feat_vecs_set = FeatVecsDataset(self.test_class_feat_vecs)
        self.test_feat_vecs_loader = torch.utils.data.DataLoader(
            self.test_feat_vecs_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    def _init_cls_optim(self):
        self.cls_optimizer = optim.Adam(
            self.disc.parameters(), lr=CLS_LR, betas=(0.5, 0.999))
        self.cls_scheduler = optim.lr_scheduler.StepLR(
            self.cls_optimizer, step_size=8, gamma=0.1)

    def _init_gen_optim(self, optimizer='SGD'):
        if optimizer == 'SGD':
            self.optimizer_g = optim.SGD(
                self.gen.parameters(), lr=GEN_LR, momentum=0.5, weight_decay=0.00001)
            self.scheduler_g = optim.lr_scheduler.StepLR(
                self.optimizer_g, step_size=40, gamma=0.1)
        elif optimizer == 'Adam':
            self.optimizer_g = optim.Adam(
                self.gen.parameters(), lr=GEN_LR, betas=(0.5, 0.999))
            self.scheduler_g = optim.lr_scheduler.StepLR(
                self.optimizer_g, step_size=40, gamma=0.1)
        else:
            raise NotImplementedError(
                'Current optimizer options are: {SGD, Adam}')

    def _init_disc_optim(self, optimizer='SGD'):
        if optimizer == 'SGD':
            self.optimizer_d = optim.SGD(
                self.disc.parameters(), lr=DISC_LR, momentum=0.9, weight_decay=0.00001)
            self.scheduler_d = optim.lr_scheduler.StepLR(
                self.optimizer_d, step_size=40, gamma=0.1)
        elif optimizer == 'Adam':
            self.optimizer_d = optim.Adam(
                self.disc.parameters(), lr=GEN_LR, betas=(0.5, 0.999))
            self.scheduler_d = optim.lr_scheduler.StepLR(
                self.optimizer_d, step_size=40, gamma=0.1)
        else:
            raise NotImplementedError(
                'Current optimizer options are: {SGD, Adam}')

    def _get_disc_cls_acc(self, dataloader=None):
        if dataloader is None:
            dataloader = self.test_loader

        # self.disc.eval()  # Turn off training mode.
        correct = 0.
        total = 0.

        print("*** Evaluating discriminator classification on base-net feature maps.")
        with torch.no_grad():
            for _, data in enumerate(tqdm(dataloader)):
                featmaps, labels = data[1].to(device), data[2].to(
                    device)  # data[1] = featmaps | data[2] = label

                _, logits_cls, _ = self.disc(featmaps)
                _, predicted = torch.max(logits_cls.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return ('%.2f%%' % (100.0 * (correct / total)))

    def _get_disc_cls_acc_gen(self, dataloader=None):
        if dataloader is None:
            dataloader = self.test_feat_vecs_loader

        # self.disc.eval()  # Turn off training mode.
        # self.gen.eval()

        correct = 0.
        total = 0.

        print("*** Evaluating discriminator classification on generated feature maps.")
        with torch.no_grad():
            for _, (feat_vecs, labels) in enumerate(tqdm(dataloader)):
                feat_vecs, labels = feat_vecs.to(
                    device, dtype=torch.float), labels.to(device)

                gen_feat_maps = self.gen(feat_vecs)

                _, logits_cls, _ = self.disc(gen_feat_maps)
                _, predicted = torch.max(logits_cls.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # TODO: Accuracies seems to be printing to only the 2nd decimal?!
        return ('%.2f%%' % (100.0 * (correct / total)))

    def _get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def _adversarial_loss(self, outputs, is_real, criterion=nn.BCEWithLogitsLoss(), target_real_label=REAL_LABEL, target_fake_label=FAKE_LABEL):
        real_label = torch.tensor(target_real_label)
        fake_label = torch.tensor(target_fake_label)

        labels = (real_label if is_real else fake_label).expand_as(
            outputs).to(device)
        loss = criterion(outputs, labels)
        return loss

    def _save_class_feat_vecs(self, path, save=True):
        # Save features for inputs to generator
        print("*** Saving feature vectors for inputs to generator")
        self.disc.eval()

        # Reset buffer
        self.train_class_feat_vecs = [[] for _ in range(CLASSES)]

        for batch_idx, (inputs, featmaps, targets) in enumerate(tqdm(self.train_loader)):
            featmaps, targets = featmaps.to(device), targets.to(device)
            feats, _, _ = self.disc(featmaps)

            for i, target in enumerate(targets.detach().cpu().numpy().astype(int)):
                self.train_class_feat_vecs[target].append(
                    feats[i].detach().cpu().numpy())
        self.train_class_feat_vecs = np.array(self.train_class_feat_vecs)

        self.test_class_feat_vecs = [[] for _ in range(CLASSES)]
        for batch_idx, (inputs, featmaps, targets) in enumerate(tqdm(self.test_loader)):
            featmaps, targets = featmaps.to(device), targets.to(device)
            feats, _, _ = self.disc(featmaps)

            for i, target in enumerate(targets.detach().cpu().numpy().astype(int)):
                self.test_class_feat_vecs[target].append(
                    feats[i].detach().cpu().numpy())
        self.test_class_feat_vecs = np.array(self.test_class_feat_vecs)

        if save:
            torch.save({
                'train_class_feat_vecs': self.train_class_feat_vecs,
                'test_class_feat_vecs': self.test_class_feat_vecs
            }, path)
            print("*** Features vectors saved.")

        self._init_feat_vecs_loaders()

    def _load_cls(self, path=CLS_MODEL_CHECKPOINT_PATH):
        checkpoint = torch.load(path)
        self.disc.load_state_dict(checkpoint['disc_state_dict'])
        self.cls_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def _load_adv(self, path=ADV_MODEL_CHECKPOINT_PATH):
        checkpoint = torch.load(path)
        self.disc.load_state_dict(checkpoint['disc_state_dict'])
        self.gen.load_state_dict(checkpoint['gen_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])

    def _load_class_feat_vecs(self, path):
        checkpoint = torch.load(path)
        self.train_class_feat_vecs = checkpoint['train_class_feat_vecs']
        self.test_class_feat_vecs = checkpoint['test_class_feat_vecs']

        # Reload data loader to be comprised of the freshly loaded feature vectors
        self._init_feat_vecs_loaders()

    def visualize_featmap(self, featmaps, feat_vec, reconstr_feat_vec, gen_image, out_dir=None, show=False, num_examples=10):
        feat_vec = feat_vec.view(32, 16)
        reconstr_feat_vec = reconstr_feat_vec.view(32, 16)

        plt.subplot(3, num_examples, num_examples + 1)
        plt.imshow(
            (feat_vec.detach().cpu().numpy()+1.)/2)
        plt.axis('off')

        plt.subplot(3, num_examples, num_examples + 2)
        plt.imshow(
            (reconstr_feat_vec.detach().cpu().numpy()+1.)/2)
        plt.axis('off')

        for j in range(num_examples):
            plt.subplot(3, num_examples, j+1)
            plt.imshow(
                (featmaps[j].detach().cpu().numpy()+1.)/2)
            plt.axis('off')

            plt.subplot(3, num_examples, (num_examples*2)+j+1)
            plt.imshow(
                (gen_image[j].detach().cpu().numpy()+1.)/2)
            plt.axis('off')
    #         plt.imshow((gen_image.detach().cpu().numpy().transpose(1,2,0)+1.)/2)
        if out_dir is not None:
            plt.savefig(out_dir)
        if show:
            plt.show()

    def _train_cls(self, is_initial=False):
        if is_initial:
            print("*** Initial Discriminator Classification Training...")
        else:
            print("*** Discriminator Classification Training...")

        # TODO: Add training for class_feat_vecs if not initial training

        # Reset optimizer and scheduler
        self._init_cls_optim()

        self.disc.train()

        for epoch in range(CLS_EPOCHS):
            _loss_cls = 0.
            for batch_idx, (inputs, featmaps, targets) in enumerate(tqdm(self.train_loader)):
                featmaps, targets = featmaps.to(device), targets.to(device)

                self.cls_optimizer.zero_grad()
                feats, logits_cls, _ = self.disc(featmaps)

                # Only train discriminator on classification
                # No adversarial training here
                loss_cls = self.cls_criterion(logits_cls, targets.long())
                loss_cls.backward()
                self.cls_optimizer.step()

                _loss_cls += loss_cls.item()
                if batch_idx % 100 == 99:    # print every 100 mini-batches
                    print('\n\n[CLS TRAINING] EPOCH %d, MINI-BATCH %5d LR: %f loss: %.5f' %
                          (epoch + 1, batch_idx + 1, self._get_lr(self.cls_optimizer), _loss_cls / 100))
                    _loss_cls = 0.0

            self.cls_scheduler.step()
        print("*** Finished training discriminator for classification.")

        torch.save({
            'disc_state_dict': self.disc.state_dict(),
            'optimizer_state_dict': self.cls_optimizer.state_dict(),
        }, CLS_MODEL_CHECKPOINT_PATH)
        print("*** Discriminator model saved.")

    def _train_adv(self, visualize=True):
        print("*** Adversarial Training...")
        self.disc.train()
        self.gen.train()

        for epoch in range(ADV_EPOCHS):
            _loss_g, _loss_cls_gen, _loss_adv_gen, _reconstr_loss = 0., 0., 0., 0.
            _loss_d, _loss_cls_real, _loss_adv_real = 0., 0., 0.
            _overall_g_loss, _overall_d_loss = 0., 0.

            feat_vec_it = iter(self.train_feat_vecs_loader)

            for batch_idx, (inputs, featmaps, targets) in enumerate(tqdm(self.train_loader)):
                inputs, featmaps, targets = inputs.to(
                    device), featmaps.to(device), targets.to(device)

                # Samples of fake feature vectors (inputs for generator)
                sample_feat_vecs, sample_targets = next(feat_vec_it)
                sample_feat_vecs, sample_targets = sample_feat_vecs.to(
                    device, dtype=torch.float), sample_targets.to(device)

                # Generate examples
                gen_feat_maps = self.gen(sample_feat_vecs)

                # ================================================================================
                #  ===== Optimize Discriminator:  max log(B(I)) + log(1 - D(G(u | m_i, covmat_i)))
                # ================================================================================
                self.optimizer_d.zero_grad()

                feats, logits_cls, p_adv = self.disc(featmaps)
                gen_feats, gen_logits_cls, gen_logits_adv = self.disc(
                    gen_feat_maps.detach())

                ''' Classification loss '''
                real_loss_cls = self.cls_criterion(logits_cls, targets.long())
                _loss_cls_real += real_loss_cls.item()

                ''' Adversarial loss '''
                # Real
                disc_real_loss = self._adversarial_loss(
                    p_adv, True, criterion=self.adv_criterion)
                # Average confidence of selecting a particular class (for printing purposes only)
                _loss_adv_real = disc_real_loss.item()

                # Fake
                disc_fake_loss = self._adversarial_loss(
                    gen_logits_adv, False, criterion=self.adv_criterion)
                _loss_adv_gen = disc_fake_loss.item()

                # Total Adversarial Loss
                total_adv_loss = disc_real_loss + disc_fake_loss
                _loss_d += total_adv_loss.item()

                ''' Reconstruction Loss '''
                # See https://pytorch.org/docs/stable/nn.html#cosineembeddingloss for details
                y = torch.ones(
                    sample_feat_vecs.shape[0], requires_grad=False).to(device)
                reconstr_loss = nn.CosineEmbeddingLoss()(sample_feat_vecs, gen_feats, y)
                _reconstr_loss += reconstr_loss.item()

                ''' Overall Loss and Optimization '''
                loss_d = total_adv_loss + real_loss_cls + reconstr_loss
                _overall_d_loss += loss_d

                loss_d.backward()
                self.optimizer_d.step()

                # ==========================================================
                # ===== Optimize Generator: max log(D(G(u | m_i, covmat_i)))
                # ==========================================================
                # TODO: Wasserstein loss?
                self.optimizer_g.zero_grad()
                gen_feats, gen_logits_cls, gen_logits_adv = self.disc(
                    gen_feat_maps)

                ''' Classification loss '''
                gen_loss_cls = self.cls_criterion(
                    gen_logits_cls, sample_targets.long())
                _loss_cls_gen += gen_loss_cls.item()

                ''' Adversarial loss '''
                gen_loss = self._adversarial_loss(
                    gen_logits_adv, True, criterion=self.adv_criterion)  # Real is set to true because we are encouraging the GAN to appear real
                _loss_g += gen_loss.item()

                ''' Overall Loss and Optimization '''
                total_gen_loss = gen_loss + gen_loss_cls
                _overall_g_loss += total_gen_loss

                total_gen_loss.backward()
                self.optimizer_g.step()

                if batch_idx % 100 == 99:    # print every 100 mini-batches
                    print('\n\n[ADVERSARIAL TRAINING] EPOCH %d, MINI-BATCH %5d\ngen_loss     : %.5f disc_loss    : %.5f \ncls_gen_loss : %.5f cls_real_loss: %.5f \nadv_gen_loss : %.5f adv_real_loss: %.5f\nreconstr_loss: %.5f\noverall_g_loss: %.5f overall_d_loss  : %.5f\n' %
                          (epoch + 1, batch_idx + 1, _loss_g / 100, _loss_d / 100, _loss_cls_gen / 100, _loss_cls_real / 100, _loss_adv_gen / 100, _loss_adv_real / 100, _reconstr_loss / 100, _overall_g_loss / 100, _overall_d_loss / 100))
                    _loss_g, _loss_cls_gen, _loss_adv_gen, _reconstr_loss = 0., 0., 0., 0.
                    _loss_d, _loss_cls_real, _loss_adv_real = 0., 0., 0.
                    _overall_g_loss, _overall_d_loss = 0., 0.

                if batch_idx % 250 == 249:  # print every 250 mini-batches
                    # Visually inspect feature maps
                    if visualize:
                        self.visualize_featmap(
                            featmaps[0], sample_feat_vecs[0], gen_feats[0], gen_feat_maps[0], out_dir='./visual_featmaps/' + str(attempt) + '/epoch' + str(epoch+1) + '_batch_' + str(batch_idx+1) + ".jpg")

            self.scheduler_g.step()
            self.scheduler_d.step()

            if epoch % 5 == 4:
                torch.save({
                    'disc_state_dict': self.disc.state_dict(),
                    'gen_state_dict': self.gen.state_dict(),
                    'optimizer_d_state_dict': self.optimizer_d.state_dict(),
                    'optimizer_g_state_dict': self.optimizer_g.state_dict(),
                    'scheduler_d_state_dict': self.scheduler_d.state_dict(),
                    'scheduler_g_state_dict': self.scheduler_g.state_dict(),
                    'epoch': epoch
                }, ADV_MODEL_CHECKPOINT_PATH)
                print("Model saved.")

                print("BASE-NET FEATURE MAPS CLASSIFICATION ACCURACY:",
                      self._get_disc_cls_acc())

                print("DISC FEATURE VECTORS CLASSIFICATION ACCURACY:",
                      self._get_disc_cls_acc_gen())

                self.disc.train()
                self.gen.train()

    def train(self):
        # TODO: Outer loop for each additional class incrementally added

        # ===== ALGORITHM
        # 1. Train D using X_0
        if os.path.exists(CLS_MODEL_CHECKPOINT_PATH):
            self._load_cls()
            print('*** Loaded classification model (Discriminator).')
        else:
            self._train_cls(is_initial=True)

        # Check initial discriminator classification accuracy
        print("BASE-NET FEATURE MAPS INITIAL CLASSIFICATION ACCURACY:",
              self._get_disc_cls_acc())

        # 2. <Skipped> Calculate statistics {m, covmat}_0 of normalized embeddings for initial classes

        # 3. Sample from statistics and get {u}_0 (instead employ direct-way instead of statistic-way)
        if os.path.exists(CLASS_FEAT_OLD_VECS_PATH):
            self._load_class_feat_vecs(CLASS_FEAT_OLD_VECS_PATH)
            print('*** Loaded existing class feature vectors.')
        else:
            self._save_class_feat_vecs(CLASS_FEAT_OLD_VECS_PATH)

        # 4. Alternately train G and D in an adversarial way using {X_0, {u}_0} and overall loss function
        if os.path.exists(ADV_MODEL_CHECKPOINT_PATH):
            self._load_adv()
            print('*** Loaded adversary model (Discriminator and Generator).')
        else:
            self._train_adv()

        # 5. Update {m, covmat}_0, {u}_0
        self._save_class_feat_vecs(CLASS_FEAT_NEW_VECS_PATH)

        # Check discriminator classification accuracy
        print("BASE-NET FEATURE MAPS CLASSIFICATION ACCURACY:",
              self._get_disc_cls_acc())

        print("DISC FEATURE VECTORS CLASSIFICATION ACCURACY:",
              self._get_disc_cls_acc_gen())


if __name__ == '__main__':
    model = Prototype(optimizer='Adam')
    model.train()
