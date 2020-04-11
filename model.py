import itertools
import functools

import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from arch import define_Gen, define_Dis, set_grad
from torch.optim import lr_scheduler
import torchvision
import vgg

print_msg = 500
'''
Class for CycleGAN with train() as a member function

'''
class cycleGAN(object):
    def __init__(self,args):

        # Define the network 
        #####################################################
        self.Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)

        utils.print_networks([self.Gab,self.Gba,self.Da,self.Db], ['Gab','Gba','Da','Db'])

        # Define Loss criterias

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # Optimizers
        #####################################################
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(),self.Gba.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(),self.Db.parameters()), lr=args.lr, betas=(0.5, 0.999))


        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0

    def train(self,args):
        # For transforming the input image
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((args.load_height,args.load_width)),
             transforms.RandomCrop((args.crop_height,args.crop_width)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        dataset_dirs = utils.get_traindata_link(args.dataset_dir)

        # Pytorch dataloader
        a_loader = torch.utils.data.DataLoader(dsets.ImageFolder(dataset_dirs['trainA'], transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)
        b_loader = torch.utils.data.DataLoader(dsets.ImageFolder(dataset_dirs['trainB'], transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        max_len = max(len(a_loader), len(b_loader))

        steps = 0
        for epoch in range(self.start_epoch, args.epochs):
            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            a_it = iter(a_loader)
            b_it = iter(b_loader)

            for i in range(max_len):
                # step
                # step = epoch * min(len(a_loader), len(b_loader)) + i + 1
                try:
                    a_real = next(a_it)[0]
                except:
                    a_it = iter(a_loader)

                try:
                    b_real = next(b_it)[0]
                except:
                    b_it = iter(b_loader)

                # Generator Computations
                ##################################################

                set_grad([self.Da, self.Db], False)
                self.g_optimizer.zero_grad()

                a_real = Variable(a_real)
                b_real = Variable(b_real)
                a_real, b_real = utils.cuda([a_real, b_real])

                # Forward pass through generators
                ##################################################
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                # Identity losses
                ###################################################
                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)

                # lamda = 1.75E+12
                lamda = args.lamda * args.idt_coef
                a_idt_loss = self.L1(a_idt, a_real) * lamda
                b_idt_loss = self.L1(b_idt, b_real) * lamda

                a_real_features = vgg.get_features(a_real)
                b_real_features = vgg.get_features(b_real)
                a_fake_features = vgg.get_features(a_fake)
                b_fake_features = vgg.get_features(b_fake)

                # Content losses
                # content_loss_weight = 1.50
                # content_loss_weight = 1
                # a_content_loss = vgg.get_content_loss(b_fake_features, a_real_features) * content_loss_weight
                # b_content_loss = vgg.get_content_loss(a_fake_features, b_real_features) * content_loss_weight

                # style losse
                # style_loss_weight = 3.00E+05
                # style_loss_weight = 1

                # a_style_loss = vgg.get_style_loss(a_fake_features, a_real_features) * style_loss_weight
                # b_style_loss = vgg.get_style_loss(b_fake_features, b_real_features) * style_loss_weight

                # Adversarial losses
                ###################################################
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)

                real_label = utils.cuda(Variable(torch.ones(a_fake_dis.size())))

                # gen_loss_weight = 4.50E+08
                gen_loss_weight = 1
                a_gen_loss = self.MSE(a_fake_dis, real_label) * gen_loss_weight
                b_gen_loss = self.MSE(b_fake_dis, real_label) * gen_loss_weight

                # Cycle consistency losses
                ###################################################
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda
                # lamda = 3.50E+12
                # a_cycle_loss = self.L1(a_recon, a_real) * lamda
                # b_cycle_loss = self.L1(b_recon, b_real) * lamda

                # gen_loss = a_gen_loss + b_gen_loss +\
                #            a_cycle_loss + b_cycle_loss +\
                #            a_style_loss + b_style_loss +\
                #            a_content_loss + b_content_loss +\
                #            a_idt_loss + b_idt_loss

                # # Total generators losses
                # ###################################################
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss
                # # Update generators
                # ###################################################
                gen_loss.backward()
                self.g_optimizer.step()
                #
                #
                # Discriminator Computations
                #################################################

                set_grad([self.Da, self.Db], True)
                self.d_optimizer.zero_grad()

                # Sample from history of generated images
                #################################################
                a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = utils.cuda([a_fake, b_fake])

                # Forward pass through discriminators
                #################################################
                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(a_fake)
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(b_fake)
                real_label = utils.cuda(Variable(torch.ones(a_real_dis.size())))
                fake_label = utils.cuda(Variable(torch.zeros(a_fake_dis.size())))

                # Discriminator losses
                ##################################################
                a_dis_real_loss = self.MSE(a_real_dis, real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)

                # Total discriminators losses
                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss)*0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss)*0.5

                # Update discriminators
                ##################################################
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()

                steps += 1
                if steps % print_msg == 0:
                    print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                                                (epoch, i + 1, max(len(a_loader), len(b_loader)),
                                                                gen_loss, a_dis_loss+b_dis_loss))

            # Override the latest checkpoint
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()



