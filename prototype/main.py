import os
import yaml
import argparse
from train import Prototype
from train import EXPERIMENTS_PATH

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
parser.add_argument("--optimizer", type=str, default="sgd",
                    help="options=[sgd, adam, rmsprop]")
parser.add_argument("--loss", type=str, default="minimax",
                    help="options=[minimax, wass]")
parser.add_argument("--l2_norm", action="store_true", default=False,
                    help="Add l2 normalization layer in discriminator.")
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

    data['note'] = args.note

    with open(os.path.join(path, 'data.yml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


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

    if args.exp_no == -1:
        exp_no = increm_experiment_dir(args)
    else:
        if not os.path.exists(os.path.join(EXPERIMENTS_PATH, str(args.exp_no))):
            raise NotADirectoryError('Experiment number not found.')
        exp_no = args.exp_no

    # Save hyperparameters
    next_exp_dir = os.path.join(EXPERIMENTS_PATH, str(exp_no))
    save_args(args, next_exp_dir)

    model = Prototype(exp_no,
                      disc_type=args.disc,
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
                      l2_norm=args.l2_norm)

    if args.rev_train_cls:
        model.rev_train_cls()
    else:
        model.train()
