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
np.random.seed(seed=123)
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Prefer GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Save paths
attempt = 1
CLS_MODEL_CHECKPOINT_PATH = './saved_models/cls_model_v%s.sav' % attempt
ADV_MODEL_CHECKPOINT_PATH = './saved_models/adv_model_v%s.sav' % attempt
CLASS_FEAT_VECS_PATH = './saved_models/class_feat_vecs/class_feat_vecs_v%s.sav' % attempt

# Hyperparameters
CLASSES = 10
BATCH_SIZE = 100

CLS_LR = 0.001  # Train D on initial X classes
CLS_EPOCHS = 16

ADV_LR = 0.0002
ADV_EPOCHS = 20

# Labels for real and fake
REAL_LABEL = 1.0
FAKE_LABEL = 1.0 - REAL_LABEL


class Prototype():
    def __init__(self):
        # Models
        self.disc = discriminator().to(device)
        self.gen = generator(deconv=False).to(device)

        # Data loaders
        self._init_data_loaders()

        # Loss functions
        self.cls_criterion = nn.CrossEntropyLoss()
        self.adv_criterion = nn.BCEWithLogitsLoss()

        # Replay buffer
        self.class_feat_vecs = np.zeros((CLASSES, 5000, 512))
        self.class_feat_counts = [0] * CLASSES

        # Optimizers & Schedulers
        self._init_cls_optim_sched()
        self._init_gen_optim()
        self._init_disc_optim()

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

    def _init_cls_optim_sched(self):
        self.cls_optimizer = optim.Adam(
            self.disc.parameters(), lr=CLS_LR, betas=(0.5, 0.999))
        self.cls_scheduler = optim.lr_scheduler.StepLR(
            self.cls_optimizer, step_size=8, gamma=0.5)

    def _init_gen_optim(self):
        self.optimizer_g = optim.Adam(
            self.gen.parameters(), lr=ADV_LR, betas=(0.5, 0.999))

    def _init_disc_optim(self):
        self.optimizer_d = optim.Adam(
            self.disc.parameters(), lr=ADV_LR, betas=(0.5, 0.999))

    def _get_disc_cls_acc(self):
        '''
        Param #loader must be set to testloader or trainloader
        '''
        self.disc.eval()  # Turn off training mode.
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.test_loader:
                featmaps, labels = data[1].to(device), data[2].to(
                    device)  # data[1] = featmaps | data[2] = label

                _, logits_cls, _ = self.disc(featmaps)
                _, predicted = torch.max(logits_cls.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return ('%.2f%%' % (100.0 * (correct / total)))

    def _get_disc_cls_acc_gen(self):
        '''
        Param #loader must be set to testloader or trainloader
        '''
        self.disc.eval()  # Turn off training mode.
        correct = 0
        total = 0

        with torch.no_grad():
            # correct = 0
            # total = 0
            # for label in range(CLASSES):
            #     samples = torch.Tensor(
            #         self.class_feat_vecs[label][:100]).to(device)
            #     labels = torch.Tensor([label for _ in range(100)]).to(device)

            # gen_feats_maps = self.gen(samples.unsqueeze(2).unsqueeze(3))
            #     _, logits_cls, _ = self.disc(gen_feats_maps)
            #     _, predicted = torch.max(logits_cls.data, 1)

            #     total += labels.size(0)
            #     correct += (predicted == labels).sum().item()

            correct = 0
            total = 0
            for _ in range(20):
                samples, labels = self._sample_vecs(100)
                samples, labels = samples.to(device), labels.to(device)

                gen_feats_maps = self.gen(samples.unsqueeze(2).unsqueeze(3))
                _, logits_cls, _ = self.disc(gen_feats_maps)
                _, predicted = torch.max(logits_cls.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return ('%.2f%%' % (100.0 * (correct / total)))

    def _get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def _adversarial_loss(self, outputs, is_real, criterion=nn.BCELoss(), target_real_label=REAL_LABEL, target_fake_label=FAKE_LABEL):
        real_label = torch.tensor(target_real_label)
        fake_label = torch.tensor(target_fake_label)

        labels = (real_label if is_real else fake_label).expand_as(
            outputs).to(device)
        loss = criterion(outputs, labels)
        return loss

    def _save_class_feat_vecs(self, save=True):
        # Save features for inputs to generator
        print("*** Saving feature vectors for inputs to generator")

        # Reset buffer
        self.class_feat_vecs = np.zeros((CLASSES, 5000, 512))
        self.class_feat_counts = [0] * CLASSES

        for batch_idx, (inputs, featmaps, targets) in enumerate(tqdm(self.train_loader)):
            featmaps, targets = featmaps.to(device), targets.to(device)
            feats, _, _ = self.disc(featmaps)

            for i, target in enumerate(targets.detach().cpu().numpy().astype(int)):
                self.class_feat_vecs[target, self.class_feat_counts[target]] = feats[i].detach(
                ).cpu().numpy()
                self.class_feat_counts[target] += 1
        self.class_feat_vecs = np.array(self.class_feat_vecs)

        if save:
            torch.save({
                'class_feat_vecs': self.class_feat_vecs,
                'class_feat_counts': self.class_feat_counts
            }, CLASS_FEAT_VECS_PATH)
            print("*** Features vectors saved.")

    def _sample_vecs(self, batch_size):
        vecs = []
        vec_targets = []
        for _ in range(batch_size):
            label = np.random.randint(10)
            vecs.append(torch.Tensor(self.class_feat_vecs[label][np.random.choice(
                self.class_feat_vecs[label].shape[0], 1)][0]))
            # vecs.append(torch.Tensor(self.class_feat_vecs[label][np.random.choice(50, 1)][0]))
            vec_targets.append(torch.Tensor([label]))
        vecs = torch.stack(vecs)
        vec_targets = torch.stack(vec_targets).squeeze(1)
        # import pdb
        # pdb.set_trace()
        return vecs, vec_targets

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

    def _load_class_feat_vecs(self, path=CLASS_FEAT_VECS_PATH):
        checkpoint = torch.load(path)
        self.class_feat_vecs = checkpoint['class_feat_vecs']
        self.class_feat_counts = checkpoint['class_feat_counts']

    def _train_cls(self, is_initial=False):
        if is_initial:
            print("*** Initial Discriminator Classification Training...")
        else:
            print("*** Discriminator Classification Training...")

        # TODO: Add training for class_feat_vecs if not initial training

        # Reset optimizer and scheduler
        self._init_cls_optim_sched()

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

        self.disc.train()

    def _train_adv(self):
        print("*** Adversarial Training...")

        self.disc.train()
        self.gen.train()

        for epoch in range(ADV_EPOCHS):
            _loss_g, _loss_cls_gen, _loss_adv_gen, _reconstr_loss = 0., 0., 0., 0.
            _loss_d, _loss_cls_real, _loss_adv_real = 0., 0., 0.
            _overall_loss = 0.

            for batch_idx, (inputs, featmaps, targets) in enumerate(tqdm(self.train_loader)):
                inputs, featmaps, targets = inputs.to(
                    device), featmaps.to(device), targets.to(device)

                # Samples of fake feature vectors (inputs for generator)
                sample_feats_vec, sample_targets = self._sample_vecs(
                    inputs.shape[0])
                sample_feats_vec, sample_targets = sample_feats_vec.to(
                    device), sample_targets.to(device)

                # Generate examples
                gen_feats_maps = self.gen(
                    sample_feats_vec.unsqueeze(2).unsqueeze(3))

                # ================================================================================
                #  ===== Optimize Discriminator:  max log(B(I)) + log(1 - D(G(u | m_i, covmat_i)))
                # ================================================================================
                self.optimizer_d.zero_grad()

                feats, logits_cls, p_adv = self.disc(featmaps)
                gen_feats, gen_logits_cls, gen_logits_adv = self.disc(
                    gen_feats_maps.detach())

                ''' Classification loss '''
                real_loss_cls = self.cls_criterion(logits_cls, targets.long())
                gen_loss_cls = self.cls_criterion(
                    gen_logits_cls, sample_targets.long())
                _loss_cls_real += real_loss_cls.item()
                _loss_cls_gen += gen_loss_cls.item()

                total_cls_loss = real_loss_cls + gen_loss_cls

                ''' Adversarial loss '''
                # Real
                # ------------------
                disc_real_loss = self._adversarial_loss(
                    p_adv, True, criterion=self.adv_criterion)
                # Average confidence of selecting a particular class (for printing purposes only)
                _loss_adv_real = disc_real_loss.mean().item()

                # Fake
                # ------------------
                disc_fake_loss = self._adversarial_loss(
                    gen_logits_adv, False, criterion=self.adv_criterion)
                _loss_adv_gen = disc_fake_loss.mean().item()

                # Total Adversarial Loss
                total_adv_loss = disc_real_loss + disc_fake_loss
                _loss_d += total_adv_loss.item()

                ''' Reconstruction Loss '''
                # See https://pytorch.org/docs/stable/nn.html#cosineembeddingloss for details
                y = torch.ones(sample_feats_vec.shape[0]).to(device)
                reconstr_loss = nn.CosineEmbeddingLoss()(sample_feats_vec, gen_feats, y)
                _reconstr_loss += reconstr_loss.mean().item()

                ''' Overall Loss and Optimization '''
                loss_d = total_adv_loss + total_cls_loss + reconstr_loss
                _overall_loss += loss_d

                loss_d.backward()
                self.optimizer_d.step()

                # ==========================================================
                # ===== Optimize Generator: max log(D(G(u | m_i, covmat_i)))
                # ==========================================================
                # TODO: Include classification loss into generator loss
                self.gen.zero_grad()
                gen_feats, gen_logits_cls, gen_logits_adv = self.disc(
                    gen_feats_maps)
                gen_loss = self._adversarial_loss(
                    gen_logits_adv, True, criterion=self.adv_criterion)  # Real is set to true because we are encouraging the GAN to appear real

                _loss_g += gen_loss.mean().item()

                gen_loss.backward()
                self.optimizer_g.step()

                if batch_idx % 100 == 99:    # print every 100 mini-batches
                    print('\n\n[ADVERSARIAL TRAINING] EPOCH %d, MINI-BATCH %5d\ngen_loss     : %.5f disc_loss    : %.5f \ncls_gen_loss : %.5f cls_real_loss: %.5f \nadv_gen_loss : %.5f adv_real_loss: %.5f\nreconstr_loss: %.5f\noverall_loss  : %.5f\n' %
                          (epoch + 1, batch_idx + 1, _loss_g / 100, _loss_d / 100, _loss_cls_gen / 100, _loss_cls_real / 100, _loss_adv_gen / 100, _loss_adv_real / 100, _reconstr_loss / 100, _overall_loss / 100))
                    _loss_g, _loss_cls_gen, _loss_adv_gen, _reconstr_loss = 0., 0., 0., 0.
                    _loss_d, _loss_cls_real, _loss_adv_real = 0., 0., 0.
                    _overall_loss = 0.

            if epoch % 5 == 4:
                torch.save({
                    'disc_state_dict': self.disc.state_dict(),
                    'gen_state_dict': self.gen.state_dict(),
                    'optimizer_d_state_dict': self.optimizer_d.state_dict(),
                    'optimizer_g_state_dict': self.optimizer_g.state_dict(),
                    'epoch': epoch
                }, ADV_MODEL_CHECKPOINT_PATH)

    def train(self):
        # TODO: Outer loop for each additional class incrementally added
        # self._load_adv()
        # self._load_class_feat_vecs()
        # print(self._get_disc_cls_acc_gen())

        # self._load_cls()
        # print(self._get_disc_cls_acc())

        # ===== ALGORITHM
        # 1. Train D using X_0
        if os.path.exists(CLS_MODEL_CHECKPOINT_PATH):
            self._load_cls()
            print('*** Loaded classification model (Discriminator).')
        else:
            self._train_cls(is_initial=True)

        # Check initial discriminator classification accuracy
        disc_acc = self._get_disc_cls_acc()
        print("CLASSIFICATION ACCURACY:", disc_acc)

        # 2. <Skipped> Calculate statistics {m, covmat}_0 of normalized embeddings for initial classes

        # 3. Sample from statistics and get {u}_0 (instead employ direct-way instead of statistic-way)
        if os.path.exists(CLASS_FEAT_VECS_PATH):
            self._load_class_feat_vecs()
            print('*** Loaded existing class feature vectors.')
        else:
            self._save_class_feat_vecs()

        # 4. Alternately train G and D in an adversarial way using {X_0, {u}_0} and overall loss function
        self._train_adv()

        # 5. Update {m, covmat}_0, {u}_0
        self._save_class_feat_vecs()

        # Check discriminator classification accuracy
        print("BASE-NET FEATURE MAPS CLASSIFICATION ACCURACY:",
              self._get_disc_cls_acc())
        print("DISC FEATURE VECTORS CLASSIFICATION ACCURACY:",
              self._get_disc_cls_acc_gen())


if __name__ == '__main__':
    model = Prototype()
    model.train()
