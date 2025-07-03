# Common
import os
import logging
import warnings
import argparse
import numpy as np
from tqdm import tqdm
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# my module
from dataset.semkitti_trainset import SemanticKITTI
from utils.config import ConfigDs as cfg
from utils.metric import compute_acc, IoUCalculator
from network.RandLANet import Network
from network.loss_func1 import compute_loss


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='/media/jtyang/backup3/model/checkpoint92.tar', help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default=6, help='Batch Size during training [default: 5]')
parser.add_argument('--val_batch_size', type=int, default=20, help='Batch Size during training [default: 30]')
parser.add_argument('--num_workers', type=int, default=14, help='Number of workers [default: 5]')

parser.add_argument('--clustering_items_per_class', type=int, default=100)
parser.add_argument('--clustering_start_iter', type=int, default=5)
parser.add_argument('--clustering_update_mu_iter', type=int, default=2)
parser.add_argument('--clustering_momentum', type=float, default=0.5)

parser.add_argument('--margin', type=float, default=2.0)
parser.add_argument('--feature_store_update_iter_iteraval', type=int, default=2000)
parser.add_argument('--loss_weight_clustering', type=float, default=0.01)
FLAGS = parser.parse_args()


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Trainer:
    def __init__(self):
        # Init Logging
        if not os.path.exists(FLAGS.log_dir):
            os.mkdir(FLAGS.log_dir)
        self.log_dir = FLAGS.log_dir
        log_fname = os.path.join(FLAGS.log_dir, 'log_train92.txt')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger("Trainer")
        # tensorboard writer
        self.tf_writer = SummaryWriter(self.log_dir)
        # get_dataset & dataloader
        train_dataset = SemanticKITTI('training')
        val_dataset = SemanticKITTI('validation')
        self.train_dataset = train_dataset

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            worker_init_fn=my_worker_init_fn,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=FLAGS.val_batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            worker_init_fn=my_worker_init_fn,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True
        )
        # Network & Optimizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        class_weights = torch.from_numpy(train_dataset.get_class_weight()).squeeze().float().cuda()
        # print(class_weights.shape)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.net = Network(cfg)
        self.net.to(device)

        # Load the Adam optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.learning_rate,weight_decay=1e-5)      #加入weight decay
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)
        # Load module
        self.highest_val_iou = 0
        self.start_epoch = 0

        CHECKPOINT_PATH = FLAGS.checkpoint_path
        # if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            # checkpoint = torch.load(CHECKPOINT_PATH)
            # self.net.load_state_dict(checkpoint['model_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # self.start_epoch = checkpoint['epoch']
            # print('load checkpoint')


        self.logger.info('')
        if torch.cuda.device_count() > 1:
            self.logger.info("Let's use %d GPUs!" % (torch.cuda.device_count()))
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.net = nn.DataParallel(self.net)
        # self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.iter=0

    def train_one_epoch(self):
        # print('*'*80)
        self.net.train()  # set model to training mode
        tqdm_loader = tqdm(self.train_loader, total=len(self.train_loader))
        iou_calc = IoUCalculator(cfg)
        for batch_idx, batch_data in enumerate(tqdm_loader):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(cfg.num_layers):
                        batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                else:
                    batch_data[key] = batch_data[key].cuda(non_blocking=True)

            self.optimizer.zero_grad()
            # Forward pass
            torch.cuda.synchronize()
            # print(batch_data['xyz'].shape)
            end_points = self.net(batch_data)
            end_points = compute_loss(end_points, self.train_dataset, self.criterion)
            loss=end_points['cls_loss']
            iou_calc.add_data(end_points)
            self.iter = self.iter + 1
            if batch_idx % 50 == 0:
                acc, end_points = compute_acc(end_points)
                mean_iou, iou_list = iou_calc.compute_iou()
                print(acc)
                print(loss)
                print(mean_iou)
                print('lr:', self.optimizer.state_dict()['param_groups'][0]['lr'])
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

    def train(self):
        for epoch in range(self.start_epoch, FLAGS.max_epoch):
            # torch.cuda.empty_cache()
            self.cur_epoch = epoch
            self.logger.info('**** EPOCH %03d ****' % (epoch))
            self.train_one_epoch()
            self.logger.info('**** EVAL EPOCH %03d ****' % (epoch))
            # torch.cuda.empty_cache()
            mean_iou = self.validate()
            # Save best checkpoint
            if mean_iou > self.highest_val_iou:
                self.highest_val_iou = mean_iou
                checkpoint_file = os.path.join(self.log_dir, 'checkpoint92.tar')
                self.save_checkpoint(checkpoint_file)
                print('save checkpoint')

    def validate(self):
        self.net.eval()  # set model to eval mode (for bn and dp)
        iou_calc = IoUCalculator(cfg)

        tqdm_loader = tqdm(self.val_loader, total=len(self.val_loader))
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm_loader):
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(cfg.num_layers):
                            batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                    else:
                        batch_data[key] = batch_data[key].cuda(non_blocking=True)

                # Forward pass
                torch.cuda.synchronize()
                end_points = self.net(batch_data)

                end_points = compute_loss(end_points, self.train_dataset, self.criterion)
                iou_calc.add_data(end_points)
                if batch_idx % 50 == 0:
                    acc, end_points = compute_acc(end_points)
                    iou, _ = iou_calc.compute_iou()
                    print(acc)
                    print(iou)

        mean_iou, iou_list = iou_calc.compute_iou()
        self.logger.info('mean IoU:{:.1f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        self.logger.info(s)
        print(mean_iou)
        return mean_iou

    def save_checkpoint(self, fname):
        save_dict = {
            'epoch': self.cur_epoch+1,  # after training one epoch, the start_epoch should be epoch+1
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        # with nn.DataParallel() the net is added as a submodule of DataParallel
        try:
            save_dict['model_state_dict'] = self.net.module.state_dict()
        except AttributeError:
            save_dict['model_state_dict'] = self.net.state_dict()
        torch.save(save_dict, fname)


def main():
    trainer = Trainer()
    trainer.train()



if __name__ == '__main__':
    main()