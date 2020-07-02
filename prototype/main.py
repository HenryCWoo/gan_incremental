import os
import yaml
import argparse
from train_embed import Prototype, EXPERIMENTS_PATH
# from train import Prototype, EXPERIMENTS_PATH

parser = argparse.ArgumentParser()

# Model
parser.add_argument("--disc", type=str, default="no_resnet",
                    help="type of discriminator | options=[resnet, resnet_large, no_resnet]")
parser.add_argument("--gen", type=str, default="no_bn",
                    help="type of generator | options=[no_bn, bn]")
parser.add_argument("--deconv", action="store_true", default=False)

# Training
parser.add_argument("--cls_epochs", type=int, default=16,
                    help="number of epochs of classifier training")
parser.add_argument("--adv_epochs", type=int, default=40,
                    help="number of epochs of adversarial training")
parser.add_argument("--batch_size", type=int, default=128,
                    help="size of the batches")
parser.add_argument("--cls_lr", type=float, default=1e-3,
                    help="learning rate")
parser.add_argument("--gen_lr", type=float, default=1e-4,
                    help="learning rate")
parser.add_argument("--disc_lr", type=float, default=5e-4,
                    help="learning rate")
parser.add_argument("--optimizer", type=str, default="adam",
                    help="options=[sgd, adam, rmsprop]")
parser.add_argument("--loss", type=str, default="minimax",
                    help="options=[minimax, wass]")
parser.add_argument("--l2_norm", action="store_true", default=False,
                    help="Add l2 normalization layer in discriminator.")
parser.add_argument("--latent_var_recon_coeff", type=int, default=1,
                    help="Coefficient multiplied to latent vector / discriminator output reconstruction loss.")
# parser.add_argument("--scheduler", action="store_true", default=False)

# Misc
parser.add_argument("--rev_train_cls", action="store_true", default=False)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--exp_no', type=int, default=-1)
parser.add_argument('--note', type=str,
                    help="Add notes to describe experiment")

args = parser.parse_args()


def save_args(args, path):
    data = dict()

    data['device'] = args.device

    data['model'] = dict()
    model_dict = data['model']
    model_dict['disc'] = args.disc
    model_dict['optim'] = args.optimizer
    model_dict['deconv'] = args.deconv
    model_dict['gen'] = args.gen

    data['training'] = dict()
    training_dict = data['training']
    training_dict['batch_size'] = args.batch_size
    training_dict['adv_epochs'] = args.adv_epochs
    training_dict['cls_epochs'] = args.cls_epochs
    training_dict['cls_lr'] = args.cls_lr
    training_dict['gen_lr'] = args.gen_lr
    training_dict['disc_lr'] = args.disc_lr
    training_dict['loss'] = args.loss
    training_dict['l2_norm'] = args.l2_norm
    training_dict['latent_var_recon_coeff'] = args.latent_var_recon_coeff

    data['note'] = args.note

    with open(os.path.join(path, 'data.yml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def load_args(yaml_path):
    with open(yaml_path, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

        model_dict = data['model']
        args.deconv = model_dict['deconv']
        args.disc = model_dict['disc']  # Should use the same discriminator
        args.gen = model_dict['gen']

        training_dict = data['training']
        args.batch_size = training_dict['batch_size']
        if args.adv_epochs < training_dict['adv_epochs']:
            args.adv_epochs = training_dict['adv_epochs']
        else:
            data['training']['adv_epochs'] = args.adv_epochs

        args.cls_epochs = training_dict['cls_epochs']
        args.disc_lr = training_dict['disc_lr']
        args.gen_lr = training_dict['gen_lr']
        args.loss = training_dict['loss']
        args.l2_norm = training_dict['l2_norm']
        args.latent_var_recon_coeff = training_dict['latent_var_recon_coeff']

        if not args.rev_train_cls:
            args.optimizer = model_dict['optim']
            args.cls_lr = training_dict['cls_lr']

        if args.note and data['note'] is not None:
            args.note = data['note'] + ' | ' + args.note  # Append notes
        elif args.note and data['note'] is None:
            data['note'] = args.note

    # Update data if there were any changes like adding more epochs
    with open(yaml_path, 'w') as yamlfile:
        yaml.dump(data, yamlfile)


def increm_experiment_dir(args):
    ''' Increment and save args '''
    exp_nums = []
    for exp_num in os.listdir(EXPERIMENTS_PATH):
        exp_nums.append(int(exp_num))
    if len(exp_nums) == 0:
        next_exp = 0
    else:
        next_exp = max(exp_nums) + 1
    next_exp_dir = os.path.join(EXPERIMENTS_PATH, str(next_exp))
    if not os.path.exists(next_exp_dir):
        os.mkdir(next_exp_dir)
    return next_exp


if __name__ == '__main__':
    # Prepare path to save experiments
    if not os.path.exists(EXPERIMENTS_PATH):
        os.mkdir(EXPERIMENTS_PATH)

    if args.exp_no == -1:  # Create new experiment
        exp_no = increm_experiment_dir(args)
        # Save hyperparameters
        next_exp_dir = os.path.join(EXPERIMENTS_PATH, str(exp_no))
        save_args(args, next_exp_dir)
    else:   # Continue existing experiment
        if not os.path.exists(os.path.join(EXPERIMENTS_PATH, str(args.exp_no))):
            raise NotADirectoryError('Experiment number not found.')
        exp_no = args.exp_no
        load_args(os.path.join(EXPERIMENTS_PATH, str(exp_no), 'data.yml'))

    print('[PARAMETERS]', args)

    model = Prototype(
        exp_no,
        disc_type=args.disc,
        gen_type=args.gen,
        optimizer=args.optimizer,
        device=args.device,
        classes_count=10,  # TODO: currently we stay at 10 classes bc no incremental learning
        batch_size=args.batch_size,
        cls_lr=args.cls_lr,
        cls_epochs=args.cls_epochs,
        gen_lr=args.gen_lr,
        disc_lr=args.disc_lr,
        adv_epochs=args.adv_epochs,
        loss=args.loss,
        l2_norm=args.l2_norm,
        latent_var_recon_coeff=args.latent_var_recon_coeff
    )

    if args.rev_train_cls:
        model.rev_train_cls()
    else:
        model.train()
