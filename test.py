import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from arch import define_Gen, define_Dis

def test(args):

    transform = transforms.Compose(
        [transforms.Resize((args.crop_height,args.crop_width)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset_dirs = utils.get_testdata_link(args.dataset_dir)

    a_test_data = dsets.ImageFolder(dataset_dirs['testA'], transform=transform)
    b_test_data = dsets.ImageFolder(dataset_dirs['testB'], transform=transform)


    a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                          use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)
    Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                          use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)

    utils.print_networks([Gab,Gba], ['Gab','Gba'])

    try:
        ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
        if len(args.gpu_ids) > 0:
            print(args.gpu_ids)
            Gab.load_state_dict(ckpt['Gab'])
            Gba.load_state_dict(ckpt['Gba'])
        else:
            new_state_dict = {}
            for k, v in ckpt['Gab'].items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            Gab.load_state_dict(new_state_dict)

            new_state_dict = {}
            for k, v in ckpt['Gba'].items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            Gba.load_state_dict(new_state_dict)
    except:
        print(' [*] No checkpoint!')


    """ run """
    a_it = iter(a_test_loader)
    b_it = iter(b_test_loader)
    for i in range(len(a_test_loader)):
        try:
            a_real_test = Variable(next(a_it)[0], requires_grad=True)
        except:
            a_it = iter(a_test_loader)

        try:
            b_real_test = Variable(next(b_it)[0], requires_grad=True)
        except:
            b_it = iter(b_test_loader)

        print(a_real_test)
        return

        a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
        print(a_real_test.shape)

        Gab.eval()
        Gba.eval()

        with torch.no_grad():
            a_fake_test = Gab(b_real_test)
            b_fake_test = Gba(a_real_test)
            a_recon_test = Gab(b_fake_test)
            b_recon_test = Gba(a_fake_test)

        pic = (torch.cat([a_real_test, b_fake_test, a_recon_test, b_real_test, a_fake_test, b_recon_test], dim=0).data + 1) / 2.0

        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        path = str.format("{}/sample-{}.jpg", args.results_dir, i)
        torchvision.utils.save_image(pic, path, nrow=3)

