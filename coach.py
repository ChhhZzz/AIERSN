import os
import random
from shutil import copyfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from configs import data_configs
from criteria import id_loss, l2_loss, moco_loss, grad_loss
from criteria.lpips.lpips import LPIPS
from datasets.images_dataset import ImagesDataset
from models.main_network import styleSketch
from torch.utils.data import DataLoader
from utils.common import count_parameters, toogle_grad
from torchvision.utils import save_image
import torch_fidelity

from torch.cuda.amp import autocast, GradScaler


class Coach:
    def __init__(self, opts):
        self.opts = opts
        self.device = "cuda:0"
        self.opts.device = self.device
        torch.backends.cudnn.benchmark = True
        self.global_step = 0
        self.best_val_loss = None

        # Fix random seed
        SEED = 2107
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        print(opts)

        # init_dataset
        self.configure_path()
        train_set, test_set, test_real_set = self.configure_datasets()
        self.train_loader = DataLoader(train_set, batch_size=opts.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(test_set, batch_size=opts.test_batch_size, shuffle=False, drop_last=False)
        self.test_real_loader = DataLoader(test_real_set, batch_size=opts.test_batch_size, shuffle=False, drop_last=False)

        # init_network
        self.net = styleSketch(self.opts)
        if self.opts.resume_training:
            print(f'load weights from {self.opts.checkpoint_path}')
            self.global_step = self.opts.resume_iter
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.net.spatialNet.load_state_dict(ckpt['model'])
        self.net.to(self.device)

        # Log some information about network on terminal
        print("Number of parameters:")
        print("==> spatialNet: ", count_parameters(self.net.spatialNet))
        print("==> Generator: ", count_parameters(self.net.Generator))
        print("==> Discriminator: ", count_parameters(self.net.discriminator))
        # print("==> w_encoder: ", count_parameters(self.net.w_encoder))

        # init_optimizer
        self.spatial_optimizer = self.configure_spatial_optimizers()
        self.discriminator_optimizer = self.configure_discriminator_optimizers()

        # Initialize loss
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type=self.opts.lpips_type).to(self.opts.device).eval()
        if self.opts.gl_lambda > 0:
            self.grad_loss = grad_loss.GradientLoss()

        self.logger = SummaryWriter(log_dir=self.opts.exp_dir)

    def train_tst(self):
        if os.path.exists(self.opts.exp_dir):
            os.makedirs(self.opts.exp_dir + '/train')
            os.makedirs(self.opts.exp_dir + '/tst')
            os.makedirs(self.opts.exp_dir + '/tst_real')
            os.makedirs(self.opts.exp_dir + '/fid_img')
        self.toogle_mode("train")
        all_loss_dict = {}
        while True:
            for idx, (img, w_GT) in enumerate(self.train_loader):
                img, w_GT = img.to(self.opts.device), w_GT.to(self.opts.device)

                self.use_adv_loss = self.global_step >= self.opts.step_to_add_adversarial_loss

                # forward
                GT, Revised, img9 = self.net.forward(img, w_GT)  # img w_GT only for condition

                # backward ===== Update G  ============
                loss, loss_dict = self.calc_loss(GT, Revised, img9)
                # if self.opts.gl_lambda > 0 and self.opts.use_real:
                #     real_rec = self.net.real_forward(img)
                #     loss_real = self.cal_real_loss(real_rec, GT[0][-1])
                #     loss += loss_real
                #     loss_dict['train_real'] = loss_real

                self.spatial_optimizer.zero_grad()
                loss.backward()
                self.spatial_optimizer.step()

                if self.use_adv_loss:
                    # ===== Update D ============
                    toogle_grad(self.net.discriminator, True)
                    d_loss, d_loss_dict = self.calc_discriminator_loss(Revised[0][-1].detach(), GT[0][-1].detach())
                    self.discriminator_optimizer.zero_grad()
                    d_loss.backward()
                    self.discriminator_optimizer.step()
                    loss_dict = {**loss_dict, **d_loss_dict}

                    # R1 Regularization
                    if self.opts.d_r1_gamma > 0:
                        if self.global_step % self.opts.d_reg_every == 0:
                            d_r1_loss, d_r1_loss_dict = self.calc_discriminator_r1_loss(GT[0][-1].detach())
                            self.discriminator_optimizer.zero_grad()
                            d_r1_loss.backward()
                            self.discriminator_optimizer.step()
                            loss_dict = {**loss_dict, **d_r1_loss_dict}
                    toogle_grad(self.net.discriminator, False)


                # update loss dict
                for key in loss_dict:
                    if key not in all_loss_dict:
                        all_loss_dict[key] = []
                    all_loss_dict[key].append(loss_dict[key])

                # save image
                if self.global_step % 100 == 0:
                    if self.opts.condition == 'img':
                        save_image(torch.cat([Revised[0][-1], GT[0][-1]], dim=0), self.opts.exp_dir + '/train/%d.jpg' % (self.global_step),
                                   normalize=True, nrow=self.opts.batch_size)
                    else:
                        save_image(torch.cat([img, Revised[0][-1], GT[0][-1]], dim=0), self.opts.exp_dir + '/train/%d.jpg' % (self.global_step),
                                   normalize=True, nrow=self.opts.batch_size)

                # save loss
                if self.global_step % 100 == 0:
                    self.log_loss(loss_dict, prefix='train')

                # Validation
                if self.global_step % self.opts.val_iter == 0 or self.global_step == self.opts.max_steps:
                    GT = Revised = None
                    loss = None
                    self.validate()
                    self.validate_real()

                # save model
                if self.global_step % self.opts.save_iter == 0:
                    torch.save({'model': self.net.spatialNet.state_dict()}, self.opts.exp_dir + f'/iteration_{self.global_step}.pth')

                if self.global_step == self.opts.max_steps:
                    print("OMG, finished training!")
                    break

                if self.global_step % 100 == 0:
                    print(f'\r{self.global_step}: {loss_dict}', end='')

                self.global_step += 1

            if self.global_step >= self.opts.max_steps:
                print("OMG, finished training!")
                break


    def validate(self):
        self.toogle_mode("eval")
        all_loss_dict = {}
        all_loss_dict_14 = {}
        with torch.no_grad():
            for idx, (img, w_GT) in enumerate(self.test_loader):
                img, w_GT = img.to(self.opts.device), w_GT.to(self.opts.device)

                GT_img_list, rev14_img_list, rand14_img, rev9_img_list, rand9_img, rev9_f_list, rev14_f_list, GT_f_list = self.net.validate(img, w_GT)
                if self.opts.condition == 'img':
                    save_image(torch.cat([GT_img_list[-1], rev14_img_list[-1], rand14_img, rev9_img_list[-1], rand9_img], dim=0), self.opts.exp_dir + '/tst/%d_%d.jpg'
                               % (idx, self.global_step), normalize=True, nrow=self.opts.test_batch_size)
                else:
                    save_image(torch.cat([img, GT_img_list[-1], rev9_img_list[-1], rand9_img], dim=0), self.opts.exp_dir + '/tst/%d_%d.jpg'
                               % (idx, self.global_step), normalize=True, nrow=self.opts.test_batch_size)
                    self.save_fid_img(rev9_img_list[-1], idx)

                # rev_img_list, rev_f_list, GT_img_list, GT_f_list= self.net.validate(sketch, w_GT)
                # save_image(torch.cat([sketch, rev_img_list[-1], GT_img_list[-1]], dim=0), 'output/tst/%d_%d.jpg' % (idx, self.global_step), normalize=True, nrow=self.opts.test_batch_size)

                loss, loss_dict = self.calc_loss((GT_img_list, GT_f_list), (rev9_img_list, rev9_f_list))
                loss14, loss_dict14 = self.calc_loss((GT_img_list, GT_f_list), (rev14_img_list, rev14_f_list))
                # update loss dict
                for key in loss_dict:
                    if key not in all_loss_dict:
                        all_loss_dict[key] = 0
                    all_loss_dict[key] += loss_dict[key]

                for key in loss_dict14:
                    if key not in all_loss_dict_14:
                        all_loss_dict_14[key] = 0
                    all_loss_dict_14[key] += loss_dict14[key]
        if self.opts.condition != 'img':
            all_loss_dict['fid'] = self.cal_fid('tst')
        self.log_loss(all_loss_dict, prefix='test_9')
        if self.opts.condition == 'img':
            self.log_loss(all_loss_dict_14, prefix='test_14')
        self.toogle_mode("train")

    def validate_real(self):
        self.toogle_mode("eval")
        all_loss_dict = {}
        with torch.no_grad():
            for idx, img in enumerate(self.test_real_loader):
                img = img.to(self.opts.device)
                use_w = [1,1,1,1,1,1,1,1,1,0,0,0,0]
                spatial_res, _ = self.net.spatialNet(img, use_w)

                w_avg = self.net.latent_avg.unsqueeze(0).unsqueeze(0).repeat(self.opts.test_batch_size, 14, 1)
                z1 = torch.randn([self.opts.test_batch_size, self.net.Generator.z_dim], device='cuda')
                w_rand = self.net.Generator.mapping(z1, None, truncation_psi=1, truncation_cutoff=None)

                rec_img, _ = self.net.Generator.synthesis(w_avg, spatial_res, noise_mode="const")
                if self.opts.condition == 'img':
                    mix_layer = 7
                else:
                    mix_layer = self.opts.mix_layer
                rand_img, _ = self.net.Generator.synthesis(torch.cat([w_avg[:,:mix_layer,:], w_rand[:,mix_layer:,:]], dim=1), spatial_res, noise_mode="const")

                save_image(torch.cat((img, rec_img[-1], rand_img[-1]), dim=0), self.opts.exp_dir + '/tst_real/%d_%d.jpg' % (idx, self.global_step), normalize=True, nrow=self.opts.test_batch_size)
                if self.opts.condition != 'img':
                    self.save_fid_img(rec_img[-1], idx)

                loss_l2 = l2_loss.l2_loss(rec_img[-1], img)
                loss_lpips = self.lpips_loss(rec_img[-1], img)

                if all_loss_dict == {}:
                    all_loss_dict['loss_l2'] = loss_l2
                    all_loss_dict['loss_lpips'] = loss_lpips
                else:
                    all_loss_dict['loss_l2'] += loss_l2
                    all_loss_dict['loss_lpips'] += loss_lpips

        if self.best_val_loss == None:
            self.best_val_loss = all_loss_dict['loss_l2']
        elif all_loss_dict['loss_l2'] < self.best_val_loss:
            self.best_val_loss = all_loss_dict['loss_l2']
            torch.save({'model': self.net.spatialNet.state_dict()}, self.opts.exp_dir + f'/best_iteration_{self.global_step}.pth')  # save best model
        if self.opts.condition != 'img':
            all_loss_dict['fid'] = self.cal_fid('real')
        self.log_loss(all_loss_dict, prefix='test_real_9')
        self.toogle_mode("train")

    def save_fid_img(self, img_batch, idx):
        for i in range(len(img_batch)):
            save_image(img_batch[i], self.opts.exp_dir + f'/fid_img/{str(self.opts.test_batch_size * idx + i).zfill(5)}.jpg', normalize=True)

    def cal_fid(self, type):
        if type == 'real':
            GT_path = self.dataset_args['test_root'] + '/img'
        elif type == 'tst':
            GT_path = self.dataset_args['test_root_real'] + '/img'
        gen_path = self.opts.exp_dir + '/fid_img'
        metric_scores_dict = torch_fidelity.calculate_metrics(
            input1=GT_path,
            input2=gen_path,
            cuda=True,
            batch_size=1,
            fid=True,
            kid=False,
            verbose=False,
            save_cpu_ram=True
        )
        fid_score = metric_scores_dict["frechet_inception_distance"]
        return fid_score

    def log_loss(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def configure_path(self):
        self.dataset_args = data_configs.DATASETS[self.opts.dataset_type]

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception(f"{self.opts.dataset_type} is not a valid dataset_type")
        print(f"Loading dataset for {self.opts.dataset_type}")
        transforms_dict = self.dataset_args["transforms"](self.opts).get_transforms()
        train_dataset = ImagesDataset(
            data_path=self.dataset_args["train_root"],
            transform=transforms_dict["transform_gt_train"],
            opts=self.opts,
        )
        test_dataset = ImagesDataset(
            data_path=self.dataset_args["test_root"],
            transform=transforms_dict["transform_gt_train"],
            opts=self.opts,
        )
        # real_opts = self.opts
        # real_opts.condition = 'img'
        test_dataset_real = ImagesDataset(
            data_path=self.dataset_args["test_root_real"],
            transform=transforms_dict["transform_gt_train"],
            opts=self.opts,
            use_w=False,
        )
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        print(f"Number of test real samples: {len(test_dataset)}")
        return train_dataset, test_dataset, test_dataset_real

    def configure_spatial_optimizers(self):
        params = list(self.net.spatialNet.parameters())
        if self.opts.encoder_optim_name == "adam":
            optimizer = torch.optim.Adam(params, lr=self.opts.encoder_learning_rate)
        # elif self.opts.encoder_optim_name == "ranger":
        #     optimizer = Ranger(params, lr=self.opts.encoder_learning_rate)
        else:
            raise Exception(f"{self.opts.encoder_optim_name} optimizer is not defined.")
        return optimizer

    def configure_discriminator_optimizers(self):
        params = list(self.net.discriminator.parameters())
        if self.opts.discriminator_optim_name == "adam":
            optimizer = torch.optim.Adam(params, lr=self.opts.discriminator_learning_rate)
        # elif self.opts.discriminator_optim_name == "ranger":
        #     optimizer = Ranger(params, lr=self.opts.discriminator_learning_rate)
        else:
            raise Exception(f"{self.opts.discriminator_optim_name} optimizer is not defined.")
        return optimizer

    def toogle_mode(self, mode="train"):
        if mode == "train":
            self.net.spatialNet.train()
            self.net.discriminator.train()
        else:
            self.net.spatialNet.eval()
            self.net.discriminator.eval()

    def g_nonsaturating_loss(self, fake_preds):
        loss = F.softplus(-fake_preds).mean()
        return loss

    def calc_loss(self, GT, Revised, img9=None):
        GT_img_list, GT_f_list = GT
        rev_img_list, rev_f_list = Revised

        loss_dict = {}
        loss = 0.0

        # Adversarial loss
        if self.use_adv_loss:
            # Loss G
            fake_preds = self.net.discriminator(rev_img_list[-1], None)
            loss_G_adv = self.g_nonsaturating_loss(fake_preds)
            loss_dict["loss_G_adv"] = float(loss_G_adv)
            loss += loss_G_adv * self.opts.adv_lambda

        # L2 loss
        loss_dict["loss_l2_img"] = 0
        loss_dict["loss_l2_f"] = 0
        for GT_img, rev_img in zip(GT_img_list, rev_img_list):
            if self.opts.l2_lambda > 0:
                loss_l2 = l2_loss.l2_loss(rev_img, GT_img)
                if GT_img.shape[-1] == 256:
                    loss_dict["loss_l2_last_img"] = float(loss_l2)
                else:
                    loss_dict["loss_l2_img"] += float(loss_l2)
                loss += loss_l2 * self.opts.l2_lambda

        for GT_f, rev_f in zip(GT_f_list, rev_f_list):
            if self.opts.l2_lambda > 0:
                loss_l2 = l2_loss.l2_loss(rev_f, GT_f)
                # loss_l2 = -torch.cosine_similarity(rev_f, GT_f).mean()
                loss_dict["loss_l2_f"] += float(loss_l2)
                loss += loss_l2 * self.opts.l2_lambda

        # LPIPS loss
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(GT_img_list[-1], rev_img_list[-1])
            loss_dict["loss_lpips"] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda


        # grad loss
        if self.opts.gl_lambda > 0:
            loss_gl = self.grad_loss(GT_img_list[-1], rev_img_list[-1])
            loss_dict["loss_grad"] = float(loss_gl)
            loss += loss_gl * self.opts.gl_lambda

        # img9 loss
        if img9 != None:
            loss_l2_9 = l2_loss.l2_loss(GT_img_list[-1], img9)
            loss_dict["loss_l2_9"] = float(loss_l2_9)
            loss += loss_l2_9 * self.opts.l2_lambda

            loss_lpips_9 = self.lpips_loss(GT_img_list[-1], img9)
            loss_dict["loss_lpips_9"] = float(loss_lpips_9)
            loss += loss_lpips_9 * self.opts.lpips_lambda

            if self.opts.gl_lambda > 0:
                loss_gl_9 = self.grad_loss(GT_img_list[-1], img9)
                loss_dict["loss_grad_9"] = float(loss_gl_9)
                loss += loss_gl_9 * self.opts.gl_lambda

        loss_dict["loss"] = float(loss)

        return loss, loss_dict

    def cal_real_loss(self, img, GT):
        return self.opts.gl_lambda * self.grad_loss(img, GT)

    def d_logistic_loss(self, real_preds, fake_preds):
        real_loss = F.softplus(-real_preds)
        fake_loss = F.softplus(fake_preds)

        return (real_loss.mean() + fake_loss.mean()) / 2

    def calc_discriminator_loss(self, generated_images, real_images):
        loss_dict = {}
        fake_preds = self.net.discriminator(generated_images, None)
        real_preds = self.net.discriminator(real_images, None)
        loss = self.d_logistic_loss(real_preds, fake_preds)
        loss_dict["loss_D"] = float(loss)
        return loss, loss_dict

    def d_r1_loss(self, real_pred, real_img):
        (grad_real,) = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def calc_discriminator_r1_loss(self, real_images):
        loss_dict = {}
        real_images.requires_grad = True
        real_preds = self.net.discriminator(real_images, None)
        real_preds = real_preds.view(real_images.size(0), -1)
        real_preds = real_preds.mean(dim=1).unsqueeze(1)
        r1_loss = self.d_r1_loss(real_preds, real_images)
        loss_D_R1 = self.opts.d_r1_gamma / 2 * r1_loss * self.opts.d_reg_every + 0 * real_preds[0]
        loss_dict["loss_D_r1_reg"] = float(loss_D_R1)
        return loss_D_R1, loss_dict

    def print_metrics(self, metrics_dict, prefix):
        print(f"Metrics for {prefix}, step {self.global_step}")
        for key, value in metrics_dict.items():
            print(f"\t{key} = ", value)

    def __get_save_dict(self, loss_dict):
        save_dict = {"state_dict": self.net.state_dict(), "opts": vars(self.opts)}

        if self.opts.save_checkpoint_for_resuming_training:
            save_dict["encoder_optimizer"] = self.encoder_optimizer.state_dict()
            save_dict["discriminator_optimizer"] = self.discriminator_optimizer.state_dict()
            save_dict["loss_dict"] = loss_dict

        return save_dict
