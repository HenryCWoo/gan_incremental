import os
import yaml
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

EXPERIMENTS_PATH = './experiments'


class Prototype():
    def __init__(self, exp_no, disc_type='no_resnet', gen_type='no_bn', optimizer='sgd', device=0, classes_count=10,
                 batch_size=128, cls_lr=1e-3, cls_epochs=16, gen_lr=1e-4, disc_lr=5e-4, adv_epochs=100, loss='minimax',
                 l2_norm=False, latent_vec_recon_coeff=10, latent_var_recon_coeff=100):
        # Initialize folders
        self._init_dirs(exp_no)

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.classes_count = classes_count

        # Models
        self.disc = discriminator(disc_type, l2_norm=l2_norm).to(self.device)
        self.gen = generator(gen_type, deconv=False).to(self.device)

        # Hyperparameters
        self.disc_type = disc_type
        self.gen_type = gen_type
        self.optim = optimizer

        self.batch_size = batch_size
        self.cls_lr = cls_lr
        self.cls_epochs = cls_epochs
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.adv_epochs = adv_epochs

        # Labels for real and fake
        if loss == 'wass':
            self.real_label = 1
            self.fake_label = -1
        else:
            self.real_label = 1
            self.fake_label = 0

        # Loss functions
        self.loss = loss
        self.latent_var_recon_coeff = latent_var_recon_coeff
        self.latent_vec_recon_coeff = latent_vec_recon_coeff
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

        # Track adv learning progress
        self.gen_loss_plot, self.cls_gen_loss_plot, self.adv_gen_loss_plot, self.overall_g_loss_plot = [], [], [], []
        self.disc_loss_plot, self.cls_real_loss_plot, self.adv_real_loss_plot, self.overall_d_loss_plot = [], [], [], []
        self.latent_var_loss_plot, self.latent_vec_loss_plot = [], []

    def _init_dirs(self, exp_no):
        def mkdir(path):
            if not os.path.exists(path):
                os.mkdir(path)

        # Save paths
        self.BASE_PATH = EXPERIMENTS_PATH + '/%s' % exp_no

        self.SAVE_MODEL_PATH = self.BASE_PATH + '/saved_models'
        self.SAVE_FEATS_PATH = self.BASE_PATH + '/visual_feats'
        self.SAVE_PLOTS_PATH = self.BASE_PATH + '/plots'

        self.CLS_MODEL_CHECKPOINT_PATH = self.SAVE_MODEL_PATH + '/cls_model.sav'
        self.ADV_MODEL_CHECKPOINT_PATH = self.SAVE_MODEL_PATH + '/adv_model.sav'
        self.CLASS_FEAT_OLD_VECS_PATH = self.SAVE_MODEL_PATH + \
            '/class_feat_old_vecs_norm.sav'
        self.CLASS_FEAT_NEW_VECS_PATH = self.SAVE_MODEL_PATH + '/class_feat_new_vecs.sav'

        mkdir(self.BASE_PATH)
        mkdir(self.SAVE_MODEL_PATH)
        mkdir(self.SAVE_FEATS_PATH)
        mkdir(self.SAVE_PLOTS_PATH)

    def _update_learning_adv_prog(self, losses):
        self.gen_loss_plot.append(losses[0])
        self.cls_gen_loss_plot.append(losses[1])
        self.adv_gen_loss_plot.append(losses[2])
        self.overall_g_loss_plot.append(losses[3])
        self.disc_loss_plot.append(losses[4])
        self.cls_real_loss_plot.append(losses[5])
        self.adv_real_loss_plot.append(losses[6])
        self.overall_d_loss_plot.append(losses[7])
        self.latent_var_loss_plot.append(losses[8])
        self.latent_vec_loss_plot.append(losses[9])

    def _save_adv_progress(self, path=None):
        if path is None:
            path = self.SAVE_PLOTS_PATH
        plt.figure()
        plt.plot(self.gen_loss_plot, label='gen_loss')
        plt.plot(self.disc_loss_plot, label='disc_loss')
        plt.legend()
        plt.savefig(path + '/gen_disc_loss.jpg')
        plt.close()

        plt.figure()
        plt.plot(self.cls_gen_loss_plot, label='cls_gen_loss')
        plt.plot(self.cls_real_loss_plot, label='cls_real_loss')
        plt.legend()
        plt.savefig(path + '/cls_loss.jpg')
        plt.close()

        plt.figure()
        plt.plot(self.adv_gen_loss_plot, label='adv_gen_loss')
        plt.plot(self.adv_real_loss_plot, label='adv_real_loss')
        plt.legend()
        plt.savefig(path + '/adv_loss.jpg')
        plt.close()

        plt.figure()
        plt.plot(self.overall_g_loss_plot, label='overall_g_loss')
        plt.plot(self.overall_d_loss_plot, label='overall_d_loss')
        plt.legend()
        plt.savefig(path + '/overall_loss.jpg')
        plt.close()

        plt.figure()
        plt.plot(self.latent_var_loss_plot,
                 label='latent_var_recon_loss')
        plt.legend()
        plt.savefig(path + '/latent_var_recon_loss.jpg')
        plt.close()

        plt.figure()
        plt.plot(self.latent_vec_loss_plot,
                 label='latent_vec_recon_loss')
        plt.legend()
        plt.savefig(path + '/latent_vec_recon_loss.jpg')
        plt.close()

        plt.figure()
        plt.plot(self.gen_loss_plot, label='gen_loss')
        plt.plot(self.disc_loss_plot, label='disc_loss')

        plt.plot(self.cls_gen_loss_plot, label='cls_gen_loss')
        plt.plot(self.cls_real_loss_plot, label='cls_real_loss')

        plt.plot(self.adv_gen_loss_plot, label='adv_gen_loss')
        plt.plot(self.adv_real_loss_plot, label='adv_real_loss')

        plt.plot(self.overall_g_loss_plot, label='overall_g_loss')
        plt.plot(self.overall_d_loss_plot, label='overall_d_loss')

        plt.plot(self.latent_var_loss_plot,
                 label='latent_var_recon_loss')
        plt.plot(self.latent_vec_loss_plot,
                 label='latent_vec_recon_loss')
        plt.legend()
        plt.savefig(path + '/combined.jpg')
        plt.close()

    def visualize_featmap(self, feat_maps, feat_vec, reconstr_feat_vec, gen_image, out_dir=None, show=False, num_examples=10):
        if out_dir is None:
            out_dir = self.SAVE_FEATS_PATH

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
                (feat_maps[j].detach().cpu().numpy()+1.)/2)
            plt.axis('off')

            plt.subplot(3, num_examples, (num_examples*2)+j+1)
            plt.imshow(
                (gen_image[j].detach().cpu().numpy()+1.)/2)
            plt.axis('off')
        # plt.imshow((gen_image.detach().cpu().numpy().transpose(1,2,0)+1.)/2)
        if out_dir is not None:
            plt.savefig(out_dir)
        if show:
            plt.show()
        plt.close()

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

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=0)

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _init_feat_vecs_loaders(self):
        self.train_feat_vecs_set = FeatVecsDataset(self.train_class_feat_vecs)
        self.train_feat_vecs_loader = torch.utils.data.DataLoader(
            self.train_feat_vecs_set, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.test_feat_vecs_set = FeatVecsDataset(self.test_class_feat_vecs)
        self.test_feat_vecs_loader = torch.utils.data.DataLoader(
            self.test_feat_vecs_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def _init_cls_optim(self):
        self.cls_optimizer = optim.Adam(
            self.disc.parameters(), lr=self.cls_lr, betas=(0.5, 0.999))
        self.cls_scheduler = optim.lr_scheduler.StepLR(
            self.cls_optimizer, step_size=8, gamma=0.1)

    def _init_gen_optim(self, optimizer='sgd'):
        if optimizer == 'sgd':
            self.optimizer_g = optim.SGD(
                self.gen.parameters(), lr=self.gen_lr, momentum=0.5, weight_decay=0.00001)
        elif optimizer == 'adam':
            self.optimizer_g = optim.Adam(
                self.gen.parameters(), lr=self.gen_lr, betas=(0.5, 0.999))
        elif optimizer == 'rmsprop':
            self.optimizer_g = optim.RMSprop(
                self.gen.parameters(), lr=self.gen_lr)
        else:
            raise NotImplementedError(
                'Current optimizer options are: {SGD, Adam}')
        self.scheduler_g = optim.lr_scheduler.StepLR(
            self.optimizer_g, step_size=50, gamma=0.6)

    def _init_disc_optim(self, optimizer='sgd'):
        if optimizer == 'sgd':
            self.optimizer_d = optim.SGD(
                self.disc.parameters(), lr=self.disc_lr, momentum=0.9, weight_decay=0.00001)
        elif optimizer == 'adam':
            self.optimizer_d = optim.Adam(
                self.disc.parameters(), lr=self.disc_lr, betas=(0.5, 0.999))
        elif optimizer == 'rmsprop':
            self.optimizer_d = optim.RMSprop(
                self.disc.parameters(), lr=self.disc_lr)
        else:
            raise NotImplementedError(
                'Current optimizer options are: {SGD, Adam}')
        self.scheduler_d = optim.lr_scheduler.StepLR(
            self.optimizer_d, step_size=50, gamma=0.6)

    def _get_disc_cls_acc(self, dataloader=None):
        if dataloader is None:
            dataloader = self.test_loader

        self.disc.eval()  # Turn off training mode.

        correct = 0.
        total = 0.

        print("*** Evaluating discriminator classification on base-net feature maps.")
        with torch.no_grad():
            for _, data in enumerate(tqdm(dataloader)):
                feat_maps, labels = data[1].to(self.device), data[2].to(
                    self.device)  # data[1] = feat_maps | data[2] = label

                _, logits_cls, _ = self.disc(feat_maps)
                _, predicted = torch.max(logits_cls.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return (correct / total)
        # return ('%.2f%%' % (100.0 * (correct / total)))

    def _get_disc_cls_acc_gen(self, dataloader=None):
        if dataloader is None:
            dataloader = self.test_feat_vecs_loader

        self.disc.eval()  # Turn off training mode.
        self.gen.eval()

        correct = 0.
        total = 0.

        print("*** Evaluating discriminator classification on generated feature maps.")
        with torch.no_grad():
            for _, (feat_vecs, labels) in enumerate(tqdm(dataloader)):
                feat_vecs, labels = feat_vecs.to(
                    self.device, dtype=torch.float), labels.to(self.device)

                gen_feat_maps = self.gen(feat_vecs)

                _, logits_cls, _ = self.disc(gen_feat_maps)
                _, predicted = torch.max(logits_cls.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return (correct / total)
        # return ('%.2f%%' % (100.0 * (correct / total)))

    def _get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def _adversarial_loss(self, outputs, is_real, criterion=nn.BCEWithLogitsLoss()):
        real_label = torch.tensor(self.real_label).type_as(outputs)
        fake_label = torch.tensor(self.fake_label).type_as(outputs)

        labels = (real_label if is_real else fake_label).expand_as(
            outputs).to(self.device)
        loss = criterion(outputs, labels)
        return loss

    def _save_class_feat_vecs(self, path, save=True):
        # Save features for inputs to generator
        print("*** Saving feature vectors for inputs to generator")
        self.disc.eval()

        # Reset buffer
        self.train_class_feat_vecs = [[] for _ in range(self.classes_count)]

        for batch_idx, (inputs, feat_maps, targets) in enumerate(tqdm(self.train_loader)):
            feat_maps, targets = feat_maps.to(
                self.device), targets.to(self.device)
            feats, _, _ = self.disc(feat_maps)

            for i, target in enumerate(targets.detach().cpu().numpy().astype(int)):
                self.train_class_feat_vecs[target].append(
                    feats[i].detach().cpu().numpy())
        self.train_class_feat_vecs = np.array(self.train_class_feat_vecs)

        self.test_class_feat_vecs = [[] for _ in range(self.classes_count)]
        for batch_idx, (inputs, feat_maps, targets) in enumerate(tqdm(self.test_loader)):
            feat_maps, targets = feat_maps.to(
                self.device), targets.to(self.device)
            feats, _, _ = self.disc(feat_maps)

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

    def _load_cls(self, path=None):
        if path is None:
            path = self.CLS_MODEL_CHECKPOINT_PATH
        checkpoint = torch.load(path)
        self.disc.load_state_dict(checkpoint['disc_state_dict'])
        self.cls_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def _load_adv(self, path=None, load_disc=True, load_gen=True, load_opt_d=True, load_opt_g=True):
        if path is None:
            path = self.ADV_MODEL_CHECKPOINT_PATH
        checkpoint = torch.load(path)

        # User can choose to use the saved discriminator or train a new one (used in rev_train_cls)
        if load_disc:
            self.disc.load_state_dict(checkpoint['disc_state_dict'])

        if load_opt_d:
            self.optimizer_d.load_state_dict(
                checkpoint['optimizer_d_state_dict'])
            self.scheduler_d.load_state_dict(
                checkpoint['scheduler_d_state_dict'])

        if load_gen:
            self.gen.load_state_dict(checkpoint['gen_state_dict'])

        if load_opt_g:
            self.optimizer_g.load_state_dict(
                checkpoint['optimizer_g_state_dict'])
            self.scheduler_g.load_state_dict(
                checkpoint['scheduler_g_state_dict'])

        (self.gen_loss_plot, self.cls_gen_loss_plot, self.adv_gen_loss_plot, self.overall_g_loss_plot,
            self.disc_loss_plot, self.cls_real_loss_plot, self.adv_real_loss_plot, self.overall_d_loss_plot,
            self.latent_var_loss_plot, self.latent_vec_loss_plot) = checkpoint['loss_progress']

        return checkpoint['epoch']

    def _load_class_feat_vecs(self, path):
        checkpoint = torch.load(path)
        self.train_class_feat_vecs = checkpoint['train_class_feat_vecs']
        self.test_class_feat_vecs = checkpoint['test_class_feat_vecs']

        # Reload data loader to be comprised of the freshly loaded feature vectors
        self._init_feat_vecs_loaders()

    def _train_cls(self, is_initial=False):
        if is_initial:
            print("*** Initial Discriminator Classification Training...")
        else:
            print("*** Discriminator Classification Training...")

        # TODO: Add training for class_feat_vecs if not initial training

        # Reset optimizer and scheduler
        self._init_cls_optim()

        self.disc.train()

        for epoch in range(self.cls_epochs):
            _loss_cls = 0.
            for batch_idx, (inputs, feat_maps, targets) in enumerate(tqdm(self.train_loader)):
                feat_maps, targets = feat_maps.to(
                    self.device), targets.to(self.device)

                self.cls_optimizer.zero_grad()
                feats, logits_cls, _ = self.disc(feat_maps)

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
        }, self.CLS_MODEL_CHECKPOINT_PATH)
        print("*** Discriminator model saved.")

    def _train_adv(self, visualize=True):
        print("*** Adversarial Training...")

        cur_epoch = 0
        # Pick up from where you left off if there's more to be trained
        if os.path.exists(self.ADV_MODEL_CHECKPOINT_PATH):
            cur_epoch = self._load_adv()
            print('*** Loaded adversary model (Discriminator and Generator).')

        self.disc.train()
        self.gen.train()

        for epoch in range(cur_epoch, self.adv_epochs):
            losses = {
                'gen_loss': 0., 'disc_loss': 0.,
                'cls_gen_loss': 0., 'cls_real_loss': 0.,
                'adv_gen_loss': 0., 'adv_real_loss': 0.,
                'latent_var_recon_loss': 0., 'latent_vec_recon_loss': 0.,
                'overall_g_loss': 0.,  'overall_d_loss': 0.
            }

            feat_vec_it = iter(self.train_feat_vecs_loader)

            for batch_idx, (inputs, feat_maps, targets) in enumerate(tqdm(self.train_loader)):
                inputs, feat_maps, targets = inputs.to(
                    self.device), feat_maps.to(self.device), targets.to(self.device)

                # Samples of fake feature vectors (inputs for generator)
                sample_feat_vecs, gen_targets = next(feat_vec_it)
                sample_feat_vecs, gen_targets = sample_feat_vecs.to(
                    self.device, dtype=torch.float), gen_targets.to(self.device)

                # Generate examples
                gen_feat_maps = self.gen(sample_feat_vecs)

                # ================================================================================
                #  ===== Optimize Discriminator:  max log(B(I)) + log(1 - D(G(u | m_i, covmat_i)))
                # ================================================================================
                self.optimizer_d.zero_grad()

                feats, logits_cls, p_adv = self.disc(feat_maps)
                gen_feats, gen_logits_cls, gen_adv = self.disc(
                    gen_feat_maps.detach())

                ''' Classification loss '''
                real_loss_cls = self.cls_criterion(logits_cls, targets.long())
                losses['cls_real_loss'] += real_loss_cls.item()

                ''' Adversarial loss '''
                # Real
                disc_real_loss = self._adversarial_loss(
                    p_adv, True, criterion=self.adv_criterion)
                # Average confidence of selecting a particular class (for printing purposes only)
                losses['adv_real_loss'] = disc_real_loss.item()

                # Fake
                disc_fake_loss = self._adversarial_loss(
                    gen_adv, False, criterion=self.adv_criterion)
                losses['adv_gen_loss'] = disc_fake_loss.item()

                # Total Adversarial Loss
                total_adv_loss = disc_real_loss + disc_fake_loss
                losses['disc_loss'] += total_adv_loss.item()

                ''' Overall Loss and Optimization '''
                loss_d = total_adv_loss + real_loss_cls
                losses['overall_d_loss'] += loss_d

                loss_d.backward()
                self.optimizer_d.step()

                # ==========================================================
                # ===== Optimize Generator: max log(D(G(u | m_i, covmat_i)))
                # ==========================================================
                self.optimizer_g.zero_grad()
                gen_feats, gen_logits_cls, gen_adv = self.disc(gen_feat_maps)

                ''' Classification loss '''
                gen_loss_cls = self.cls_criterion(
                    gen_logits_cls, gen_targets.long())
                losses['cls_gen_loss'] += gen_loss_cls.item()

                ''' Adversarial loss '''
                gen_loss = self._adversarial_loss(
                    gen_adv, True, criterion=self.adv_criterion)  # Real is set to true because we are encouraging the GAN to appear real
                losses['gen_loss'] += gen_loss.item()

                ''' Latent Variable Reconstruction Loss '''
                # See https://pytorch.org/docs/stable/nn.html#cosineembeddingloss for details
                y = torch.ones(
                    sample_feat_vecs.shape[0], requires_grad=False).to(self.device)
                latent_var_recon_loss = nn.CosineEmbeddingLoss()(
                    sample_feat_vecs, gen_feats, y) * self.latent_var_recon_coeff
                losses['latent_var_recon_loss'] += latent_var_recon_loss.item()

                ''' Latent Vector Reconstruction Loss '''
                latent_vec_recon_loss = nn.MSELoss()(gen_feat_maps, feat_maps) * \
                    self.latent_vec_recon_coeff
                losses['latent_vec_recon_loss'] += latent_vec_recon_loss.item()

                ''' Overall Loss and Optimization '''
                total_gen_loss = gen_loss + gen_loss_cls + \
                    latent_var_recon_loss + latent_vec_recon_loss
                losses['overall_g_loss'] += total_gen_loss

                total_gen_loss.backward()
                self.optimizer_g.step()

                if batch_idx % 100 == 99:    # print every 100 mini-batches
                    print('\n\n[ADVERSARIAL TRAINING] EPOCH %d, MINI-BATCH %5d\ngen_loss     : %.5f disc_loss    : %.5f \ncls_gen_loss : %.5f cls_real_loss: %.5f \nadv_gen_loss : %.5f adv_real_loss: %.5f\nlatent_var_recon_loss: %.5f latent_vec_recon_loss: %.5f\noverall_g_loss: %.5f overall_d_loss  : %.5f\ng_lr     : %.8f d_lr     : %.8f\n' %
                          (epoch + 1, batch_idx + 1,
                           losses['gen_loss'] / 100, losses['disc_loss'] / 100,
                           losses['cls_gen_loss'] /
                           100, losses['cls_real_loss'] / 100,
                           losses['adv_gen_loss'] /
                           100, losses['adv_real_loss'] / 100,
                           losses['latent_var_recon_loss'] / 100,
                           losses['latent_vec_recon_loss'] / 100,
                           losses['overall_g_loss'] /
                           100, losses['overall_d_loss'] / 100,
                           self._get_lr(self.optimizer_g), self._get_lr(self.optimizer_d)))

                    # Store information for plotting
                    self._update_learning_adv_prog([losses['gen_loss'] / 100,
                                                    losses['cls_gen_loss'] / 100,
                                                    losses['adv_gen_loss'] / 100,
                                                    losses['overall_g_loss'] / 100,
                                                    losses['disc_loss'] / 100,
                                                    losses['cls_real_loss'] / 100,
                                                    losses['adv_real_loss'] / 100,
                                                    losses['overall_d_loss'] / 100,
                                                    losses['latent_var_recon_loss'] / 100,
                                                    losses['latent_vec_recon_loss'] / 100])

                    losses = {
                        'gen_loss': 0., 'disc_loss': 0.,
                        'cls_gen_loss': 0., 'cls_real_loss': 0.,
                        'adv_gen_loss': 0., 'adv_real_loss': 0.,
                        'latent_var_recon_loss': 0., 'latent_vec_recon_loss': 0.,
                        'overall_g_loss': 0.,  'overall_d_loss': 0.
                    }

                if batch_idx % 250 == 249:  # print every 250 mini-batches
                    # Visually inspect feature maps
                    if visualize:
                        self.visualize_featmap(
                            feat_maps[0],
                            sample_feat_vecs[0],
                            gen_feats[0],
                            gen_feat_maps[0],
                            out_dir=self.SAVE_FEATS_PATH + '/epoch' + str(epoch+1) + '_batch_' + str(batch_idx+1) + ".jpg")

            self.scheduler_g.step()
            self.scheduler_d.step()

            self._save_adv_progress()

            torch.save({
                'disc_state_dict': self.disc.state_dict(),
                'gen_state_dict': self.gen.state_dict(),
                'optimizer_d_state_dict': self.optimizer_d.state_dict(),
                'optimizer_g_state_dict': self.optimizer_g.state_dict(),
                'scheduler_d_state_dict': self.scheduler_d.state_dict(),
                'scheduler_g_state_dict': self.scheduler_g.state_dict(),
                'epoch': epoch,
                'loss_progress': (self.gen_loss_plot, self.cls_gen_loss_plot, self.adv_gen_loss_plot, self.overall_g_loss_plot,
                                  self.disc_loss_plot, self.cls_real_loss_plot, self.adv_real_loss_plot, self.overall_d_loss_plot,
                                  self.latent_var_loss_plot, self.latent_vec_loss_plot)
            }, self.ADV_MODEL_CHECKPOINT_PATH)
            print("Model saved.")

            # Test accuracy every-so-often
            if epoch % 5 == 4:
                print("BASE-NET FEATURE MAPS CLASSIFICATION ACCURACY:",
                      self._get_disc_cls_acc())

                print("DISC FEATURE VECTORS CLASSIFICATION ACCURACY:",
                      self._get_disc_cls_acc_gen())

                self.disc.train()
                self.gen.train()

    def _train_adv_wass(self, visualize=True):
        print("*** Wasserstein Adversarial Training...")

        cur_epoch = 0
        # Pick up from where you left off if there's more to be trained
        if os.path.exists(self.ADV_MODEL_CHECKPOINT_PATH):
            cur_epoch = self._load_adv()
            print('*** Loaded adversary model (Discriminator and Generator).')

        self.disc.train()
        self.gen.train()

        for epoch in range(cur_epoch, self.adv_epochs):
            losses = {
                'gen_loss': 0., 'disc_loss': 0.,
                'cls_gen_loss': 0., 'cls_real_loss': 0.,
                'adv_gen_loss': 0., 'adv_real_loss': 0.,
                'latent_var_recon_loss': 0., 'latent_vec_recon_loss': 0.,
                'overall_g_loss': 0.,  'overall_d_loss': 0.
            }

            feat_map_it = iter(self.train_loader)
            feat_vec_it = iter(self.train_feat_vecs_loader)

            batch_idx = 0
            while batch_idx < len(feat_map_it):

                # Update critic more times than the generator
                critic_iter_count = 0
                while critic_iter_count < 5 and batch_idx < len(feat_map_it):
                    critic_iter_count += 1
                    inputs, feat_maps, targets = next(feat_map_it)
                    sample_feat_vecs, gen_targets = next(feat_vec_it)
                    batch_idx += 1  # Counts how many more batches were seen

                    inputs, feat_maps, targets = inputs.to(
                        self.device), feat_maps.to(self.device), targets.to(self.device)
                    # Samples of fake feature vectors (inputs for generator)
                    sample_feat_vecs, gen_targets = sample_feat_vecs.to(
                        self.device, dtype=torch.float), gen_targets.to(self.device)

                    # Generate examples
                    gen_feat_maps = self.gen(sample_feat_vecs)

                    # ==========================================================
                    #  ===== Optimize Critic: max D(x) - D(G(u | m_i, covmat_i))
                    # ==========================================================
                    self.optimizer_d.zero_grad()

                    feats, logits_cls, p_adv = self.disc(feat_maps)
                    gen_feats, gen_logits_cls, gen_adv = self.disc(
                        gen_feat_maps.detach())

                    ''' Classification loss '''
                    real_loss_cls = self.cls_criterion(
                        logits_cls, targets.long())
                    losses['cls_real_loss'] += real_loss_cls.item()

                    ''' Classification loss '''
                    gen_loss_cls = self.cls_criterion(
                        gen_logits_cls, gen_targets.long())
                    losses['cls_gen_loss'] += gen_loss_cls.item()

                    total_cls_loss = real_loss_cls + gen_loss_cls

                    ''' Adversarial loss '''
                    # Real
                    disc_real_loss = -torch.mean(p_adv)
                    losses['adv_real_loss'] += disc_real_loss.item()

                    # Fake
                    disc_fake_loss = torch.mean(gen_adv)
                    losses['adv_gen_loss'] += disc_fake_loss.item()

                    total_adv_loss = disc_real_loss + disc_fake_loss
                    losses['disc_loss'] += total_adv_loss.item()

                    ''' Overall Loss and Optimization '''
                    loss_d = total_adv_loss + total_cls_loss
                    losses['overall_d_loss'] += loss_d

                    loss_d.backward()
                    self.optimizer_d.step()

                    # Weight clipping
                    for p in self.disc.parameters():
                        p.data.clamp_(-0.01, 0.01)

                # ==========================================================
                # ===== Optimize Generator: max log(D(G(u | m_i, covmat_i)))
                # ==========================================================
                self.optimizer_g.zero_grad()
                gen_feats, gen_logits_cls, gen_adv = self.disc(
                    gen_feat_maps)

                ''' Adversarial loss '''
                gen_loss = -torch.mean(F.relu(gen_adv))
                losses['gen_loss'] += gen_loss.item()

                ''' Latent Variable Reconstruction Loss '''
                # See https://pytorch.org/docs/stable/nn.html#cosineembeddingloss for details
                y = torch.ones(
                    sample_feat_vecs.shape[0], requires_grad=False).to(self.device)
                latent_var_recon_loss = nn.CosineEmbeddingLoss()(
                    sample_feat_vecs, gen_feats, y)
                losses['latent_var_recon_loss'] += latent_var_recon_loss.item()

                ''' Latent Vector Reconstruction Loss '''
                latent_vec_recon_loss = nn.MSELoss()(gen_feat_maps, feat_maps) * \
                    self.latent_vec_recon_coeff
                losses['latent_vec_recon_loss'] += latent_vec_recon_loss.item()

                ''' Overall Loss and Optimization '''
                total_gen_loss = gen_loss + \
                    latent_var_recon_loss + latent_vec_recon_loss
                losses['overall_g_loss'] += total_gen_loss

                total_gen_loss.backward()
                self.optimizer_g.step()

                mod_mini_batch = 50
                if batch_idx % mod_mini_batch == 0:    # print every mod_mini_batch mini-batches
                    print('\n\n[ADVERSARIAL TRAINING] EPOCH %d, MINI-BATCH %5d\ngen_loss     : %.5f disc_loss    : %.5f \ncls_gen_loss : %.5f cls_real_loss: %.5f \nadv_gen_loss : %.5f adv_real_loss: %.5f\nlatent_var_recon_loss: %.5f latent_vec_recon_loss: %.5f\noverall_g_loss: %.5f overall_d_loss  : %.5f\ng_lr     : %.5f d_lr     : %.5f\n' %
                          (epoch + 1, batch_idx + 1, losses['gen_loss'] / mod_mini_batch, losses['disc_loss'] / (critic_iter_count * mod_mini_batch), losses['cls_gen_loss'] / mod_mini_batch, losses['cls_real_loss'] / (critic_iter_count * mod_mini_batch), losses['adv_gen_loss'] / mod_mini_batch, losses['adv_real_loss'] /
                           mod_mini_batch, losses['latent_var_recon_loss'] / mod_mini_batch, losses['latent_vec_recon_loss'] / mod_mini_batch, losses['overall_g_loss'] / mod_mini_batch, losses['overall_d_loss'] / (critic_iter_count * mod_mini_batch), self._get_lr(self.optimizer_g), self._get_lr(self.optimizer_d)))

                    # Store information for plotting
                    self._update_learning_adv_prog([losses['gen_loss'] / mod_mini_batch,
                                                    losses['cls_gen_loss'] /
                                                    mod_mini_batch,
                                                    losses['adv_gen_loss'] /
                                                    mod_mini_batch,
                                                    losses['overall_g_loss'] /
                                                    mod_mini_batch,
                                                    losses['disc_loss'] /
                                                    mod_mini_batch,
                                                    losses['cls_real_loss'] /
                                                    mod_mini_batch,
                                                    losses['adv_real_loss'] /
                                                    mod_mini_batch,
                                                    losses['overall_d_loss'] /
                                                    mod_mini_batch,
                                                    losses['latent_var_recon_loss'] /
                                                    mod_mini_batch,
                                                    losses['latent_vec_recon_loss'] / mod_mini_batch])

                    losses = {
                        'gen_loss': 0., 'disc_loss': 0.,
                        'cls_gen_loss': 0., 'cls_real_loss': 0.,
                        'adv_gen_loss': 0., 'adv_real_loss': 0.,
                        'latent_var_recon_loss': 0., 'latent_vec_recon_loss': 0.,
                        'overall_g_loss': 0.,  'overall_d_loss': 0.
                    }

                if batch_idx % 190 == 0:  # print every 250 mini-batches
                    # Visually inspect feature maps
                    if visualize:
                        self.visualize_featmap(
                            feat_maps[0],
                            sample_feat_vecs[0],
                            gen_feats[0],
                            gen_feat_maps[0],
                            out_dir=self.SAVE_FEATS_PATH + '/epoch' + str(epoch+1) + '_batch_' + str(batch_idx+1) + ".jpg")

            self.scheduler_g.step()
            self.scheduler_d.step()

            self._save_adv_progress()

            if epoch % 5 == 4:
                torch.save({
                    'disc_state_dict': self.disc.state_dict(),
                    'gen_state_dict': self.gen.state_dict(),
                    'optimizer_d_state_dict': self.optimizer_d.state_dict(),
                    'optimizer_g_state_dict': self.optimizer_g.state_dict(),
                    'scheduler_d_state_dict': self.scheduler_d.state_dict(),
                    'scheduler_g_state_dict': self.scheduler_g.state_dict(),
                    'epoch': epoch,
                    'loss_progress': (self.gen_loss_plot, self.cls_gen_loss_plot, self.adv_gen_loss_plot, self.overall_g_loss_plot,
                                      self.disc_loss_plot, self.cls_real_loss_plot, self.adv_real_loss_plot, self.overall_d_loss_plot,
                                      self.latent_var_loss_plot, self.latent_vec_loss_plot)
                }, self.ADV_MODEL_CHECKPOINT_PATH)
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
        if os.path.exists(self.CLS_MODEL_CHECKPOINT_PATH):
            self._load_cls()
            print('*** Loaded classification model (Discriminator).')
        else:
            self._train_cls(is_initial=True)

        # Check initial discriminator classification accuracy
        print("BASE-NET FEATURE MAPS INITIAL CLASSIFICATION ACCURACY:",
              self._get_disc_cls_acc())

        # 2. <Skipped> Calculate statistics {m, covmat}_0 of normalized embeddings for initial classes

        # 3. Sample from statistics and get {u}_0 (instead employ direct-way instead of statistic-way)
        if os.path.exists(self.CLASS_FEAT_OLD_VECS_PATH):
            self._load_class_feat_vecs(self.CLASS_FEAT_OLD_VECS_PATH)
            print('*** Loaded existing class feature vectors.')
        else:
            self._save_class_feat_vecs(self.CLASS_FEAT_OLD_VECS_PATH)

        # 4. Alternately train G and D in an adversarial way using {X_0, {u}_0} and overall loss function
        if self.loss == 'wass':
            self._train_adv_wass()
        else:
            self._train_adv()

        # # 5. (TODO: Worry about further steps from here during incremental learning) Update {m, covmat}_0, {u}_0
        # self._save_class_feat_vecs(CLASS_FEAT_NEW_VECS_PATH)

        # Check and log discriminator classification accuracy
        bnet_acc = self._get_disc_cls_acc()
        feat_vec_acc = self._get_disc_cls_acc_gen()
        print("BASE-NET FEATURE MAPS CLASSIFICATION ACCURACY:", bnet_acc)
        print("DISC FEATURE VECTORS CLASSIFICATION ACCURACY:", feat_vec_acc)

        acc_data = {
            'accuracies': {
                'bnet_cls_acc': bnet_acc,
                'gen_cls_acc': feat_vec_acc
            }
        }
        yaml_path = os.path.join(self.BASE_PATH, 'data.yml')
        with open(yaml_path, 'r') as yamlfile:
            cur_yaml = yaml.safe_load(yamlfile)
            cur_yaml.update(acc_data)

        if cur_yaml:
            with open(yaml_path, 'w') as yamlfile:
                yaml.safe_dump(cur_yaml, yamlfile)  # Also note the safe_dump

    def rev_train_cls(self):
        # Get generator from adversarial training
        self._load_adv(load_disc=False, load_opt_d=False)
        yaml_path = os.path.join(self.BASE_PATH, 'data.yml')
        with open(yaml_path, 'r') as yamlfile:
            cur_yaml = yaml.safe_load(yamlfile)
            print('Test BNet acc:', cur_yaml['accuracies']['bnet_cls_acc'])
            print('Test feat vecs acc:', cur_yaml['accuracies']['gen_cls_acc'])
        self._load_class_feat_vecs(self.CLASS_FEAT_OLD_VECS_PATH)

        self.gen.eval()
        self.disc.train()

        for epoch in range(self.cls_epochs):
            _loss_cls = 0.
            for batch_idx, (feat_vec, targets) in enumerate(tqdm(self.train_feat_vecs_loader)):
                feat_vec, targets = feat_vec.to(
                    self.device), targets.to(self.device)

                self.cls_optimizer.zero_grad()
                gen_feat_maps = self.gen(feat_vec)
                feats, logits_cls, _ = self.disc(gen_feat_maps.detach())

                # Only train discriminator on classification
                # No adversarial training here
                loss_cls = self.cls_criterion(logits_cls, targets.long())
                loss_cls.backward()
                self.cls_optimizer.step()

                _loss_cls += loss_cls.item()
                if batch_idx % 100 == 99:    # print every 100 mini-batches
                    print('\n\n[REV CLS TRAINING] EPOCH %d, MINI-BATCH %5d LR: %f loss: %.5f' %
                          (epoch + 1, batch_idx + 1, self._get_lr(self.cls_optimizer), _loss_cls / 100))
                    _loss_cls = 0.0

            # self.cls_optimizer.step()

            print("BNET FEATURE MAP CLS ACC:", self._get_disc_cls_acc())
            print("DISC FEATURE VECTORS CLS ACC:",
                  self._get_disc_cls_acc_gen())

        print("*** Finished training discriminator for classification.")

        # Check and log discriminator classification accuracy
        bnet_acc = self._get_disc_cls_acc()
        feat_vec_acc = self._get_disc_cls_acc_gen()
        print("BASE-NET FEATURE MAPS CLS ACC:", bnet_acc)
        print("DISC FEATURE VECTORS CLS ACC:", feat_vec_acc)

        acc_data = {
            'rev_train_cls': {
                'rev_bnet_cls_acc': bnet_acc,
                'rev_gen_cls_acc': feat_vec_acc,
                'cls_lr': self.cls_lr,
                'disc': self.disc_type
            }
        }
        yaml_path = os.path.join(self.BASE_PATH, 'data.yml')
        with open(yaml_path, 'r') as yamlfile:
            cur_yaml = yaml.safe_load(yamlfile)
            cur_yaml.update(acc_data)

        if cur_yaml:
            with open(yaml_path, 'w') as yamlfile:
                yaml.safe_dump(cur_yaml, yamlfile)  # Also note the safe_dump
