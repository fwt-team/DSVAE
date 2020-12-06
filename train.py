# -*- coding: utf-8 -*-
try:
    import os
    import argparse
    from itertools import chain
    from PIL import Image, ImageChops, ImageEnhance

    import torch
    import torch.nn as nn
    import numpy as np
    import torch.nn.functional as F

    from torch.optim.lr_scheduler import StepLR
    from tqdm import tqdm
    from sklearn.metrics import normalized_mutual_info_score as NMI

    from vmfmix.vmf import VMFMixture
    from dsvae.datasets import dataset_list, get_dataloader
    from dsvae.config import RUNS_DIR, DATASETS_DIR, DEVICE, DATA_PARAMS
    from dsvae.model import Decoder, VMFMM, Encoder
    from dsvae.utils import save_images, cluster_acc, init_weights
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default="dsvae", help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=150, type=int, help="Number of epochs")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,
                        help="Dataset name")
    parser.add_argument("-v", "--version_name", dest="version_name", default="1")
    args = parser.parse_args()

    run_name = args.run_name
    dataset_name = args.dataset_name

    # make directory
    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name, 'VMFMM_V{}'.format(args.version_name))
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    imgs_dir = os.path.join(run_dir, 'images')
    models_dir = os.path.join(run_dir, 'models')
    log_path = os.path.join(run_dir, 'logs')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # -----train-----
    n_cluster = 10
    n_epochs = args.n_epochs
    b1 = 0.3
    b2 = 0.5
    decay = 2.5 * 1e-5
    print('b1: {}, b2: {}, decay: {}'.format(b1, b2, decay))

    data_params = DATA_PARAMS[dataset_name]
    pretrain_batch_size, train_batch_size, latent_dim, picture_size, cshape, data_size, \
    pre_epoch, pre_lr, train_lr = data_params
    print('params: {}'.format(data_params))

    # test detail var
    test_batch_size = 10000

    # net
    decoder = Decoder(latent_dim=latent_dim, x_shape=picture_size, cshape=cshape)
    vmfmm = VMFMM(n_cluster=n_cluster, n_features=latent_dim)
    encoder = Encoder(input_channels=picture_size[0], output_channels=latent_dim, cshape=cshape)

    xe_loss = nn.BCELoss(reduction="sum")
    xe_logit_loss = nn.BCEWithLogitsLoss(reduction='sum')

    # set device: cuda or cpu
    decoder.to(DEVICE)
    encoder.to(DEVICE)
    vmfmm.to(DEVICE)
    xe_loss.to(DEVICE)
    xe_logit_loss.to(DEVICE)

    # optimization
    dec_enc_ops = torch.optim.Adam(chain(
        decoder.parameters(),
        encoder.parameters(),
    ), lr=pre_lr, betas=(0.5, 0.99), weight_decay=decay)
    dec_enc_vmfmm_ops = torch.optim.Adam(chain(
        decoder.parameters(),
        encoder.parameters(),
    ), lr=train_lr, betas=(b1, b2))
    vmf_ops = torch.optim.Adam(chain(
        vmfmm.parameters()
    ), lr=train_lr * 0.1, betas=(b1, b2))

    lr_s = StepLR(dec_enc_vmfmm_ops, step_size=10, gamma=0.8)
    dataloader = get_dataloader(dataset_path=data_dir, dataset_name=dataset_name,
                                batch_size=pretrain_batch_size, train=True)
    test_dataloader = get_dataloader(dataset_path=data_dir, dataset_name=dataset_name,
                                     batch_size=test_batch_size, train=False)

    # =============================================================== #
    # ==========================pretraining========================== #
    # =============================================================== #
    pre_train_path = os.path.join(models_dir, 'pre_train')
    if not os.path.exists(pre_train_path):

        print('Pretraining......')
        epoch_bar = tqdm(range(pre_epoch))
        for _ in epoch_bar:
            L = 0
            for index, (x, y) in enumerate(dataloader):
                x = x.to(DEVICE)
                _, z, _ = encoder(x)
                x_ = decoder(z)
                loss = xe_loss(x_, x) * x.size(1) / pretrain_batch_size

                L += loss.detach().cpu().numpy()

                dec_enc_ops.zero_grad()
                loss.backward()
                dec_enc_ops.step()

            epoch_bar.write('Loss={:.4f}'.format(L / len(dataloader)))

        # encoder.k.weight.data = torch.mean(encoder.mu.weight.data, dim=0, keepdim=True)
        # encoder.k.bias.data = torch.mean(encoder.mu.bias.data, dim=0, keepdim=True)

        _vmfmm = VMFMixture(n_cluster=n_cluster, max_iter=100)
        Z = []
        Y = []
        with torch.no_grad():
            for index, (x, y) in enumerate(dataloader):
                x = x.to(DEVICE)
                _, z, _ = encoder(x)
                Z.append(z)
                Y.append(y)

        Z = torch.cat(Z, 0).detach().cpu().numpy()
        Y = torch.cat(Y, 0).detach().numpy()

        _vmfmm.fit(Z)
        pre = _vmfmm.predict(Z)
        print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

        vmfmm.pi_.data = torch.from_numpy(_vmfmm.pi).to(DEVICE).float()
        vmfmm.mu_c.data = torch.from_numpy(_vmfmm.xi).to(DEVICE).float()
        vmfmm.k_c.data = torch.from_numpy(_vmfmm.k).to(DEVICE).float()

        os.makedirs(pre_train_path, exist_ok=True)
        torch.save(encoder.state_dict(), os.path.join(pre_train_path, 'enc.pkl'))
        torch.save(decoder.state_dict(), os.path.join(pre_train_path, 'dec.pkl'))
        torch.save(vmfmm.state_dict(), os.path.join(pre_train_path, 'vmfmm.pkl'))

    else:
        print("load pretrain model...")
        decoder.load_state_dict(torch.load(os.path.join(pre_train_path, "dec.pkl"), map_location=DEVICE))
        encoder.load_state_dict(torch.load(os.path.join(pre_train_path, "enc.pkl"), map_location=DEVICE))
        vmfmm.load_state_dict(torch.load(os.path.join(pre_train_path, "vmfmm.pkl"), map_location=DEVICE))

    # =============================================================== #
    # ====================check the cheekpoint model================= #
    # =============================================================== #
    file_list = os.listdir(models_dir)
    max_dir_index = 0  # init max dir index val
    for file in file_list:
        # judge file is dir
        if os.path.isdir(os.path.join(models_dir, file)) and file.split("_")[0] == "cheekpoint":
            index = int(file.split("_")[1])
            if index > max_dir_index:
                max_dir_index = index

    if max_dir_index > 0:
        print("have cheekpoint file %d" % max_dir_index)
        # load cheekpoint file
        cheek_path = os.path.join(models_dir, "cheekpoint_{}".format(max_dir_index))
        decoder.load_state_dict(torch.load(os.path.join(cheek_path, "dec.pkl")))
        encoder.load_state_dict(torch.load(os.path.join(cheek_path, "enc.pkl")))
        vmfmm.load_state_dict(torch.load(os.path.join(cheek_path, "vmfmm.pkl")))

    dataloader = get_dataloader(dataset_path=data_dir, dataset_name=dataset_name,
                                batch_size=train_batch_size, train=True)
    # =============================================================== #
    # ============================training=========================== #
    # =============================================================== #
    epoch_bar = tqdm(range(max_dir_index, n_epochs))
    best_score = 0
    best_epoch = 0
    best_dec = None
    best_encoder = None
    best_vmf = None
    offset_value = 3.5
    for epoch in epoch_bar:
        g_t_loss = 0
        for index, (real_images, target) in enumerate(dataloader):
            enhance_images = []
            for item in real_images:
                enhance_image = Image.fromarray((item.squeeze(0) * 255).numpy().astype(np.uint8))

                enhance_image = enhance_image.rotate(np.random.random() * 50 - 25)
                offset1 = np.random.randint(-offset_value, offset_value)
                offset2 = np.random.randint(-offset_value, offset_value)
                enhance_image = ImageChops.offset(enhance_image, offset1, offset2)
                enhance_image = ImageEnhance.Contrast(enhance_image).enhance(1.2)
                enhance_images.append(np.array(enhance_image) / 255.0)
            enhance_images = torch.tensor(enhance_images, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            real_images = real_images.to(DEVICE)

            decoder.train()
            vmfmm.train()
            encoder.train()
            dec_enc_vmfmm_ops.zero_grad()
            vmf_ops.zero_grad()

            z, mu, k = encoder(real_images)
            z_e, _, _ = encoder(enhance_images)

            en_loss = nn.MSELoss(reduction='sum')(z, z_e).to(DEVICE) / train_batch_size - torch.mean(torch.sum(z_e * z, dim=1))
            fake_images = decoder(z)
            fake_images1 = decoder(mu)

            rec_loss = xe_loss(fake_images, real_images) * real_images.size(1) / train_batch_size
            rec_loss1 = xe_logit_loss(fake_images, fake_images1) * real_images.size(1) / train_batch_size
            # train decoder, encoder and vmfmm
            g_loss = vmfmm.vmfmm_Loss(z, mu, k) + 0.25 * rec_loss + 0.35 * rec_loss1 + 20 * en_loss

            g_loss.backward()

            nn.utils.clip_grad_norm_(chain(
                vmfmm.parameters(),
                encoder.parameters(),
                decoder.parameters(),
            ), 5)

            dec_enc_vmfmm_ops.step()
            vmf_ops.step()

            g_t_loss += g_loss
        vmfmm.mu_c.data = vmfmm.mu_c.data / torch.norm(vmfmm.mu_c.data, dim=1, keepdim=True)
        lr_s.step()
        # save cheekpoint model
        if (epoch + 1) % 20 == 0:
            cheek_path = os.path.join(models_dir, "cheekpoint_{}".format(epoch))
            os.makedirs(cheek_path, exist_ok=True)
            torch.save(decoder.state_dict(), os.path.join(cheek_path, 'dec.pkl'))
            torch.save(encoder.state_dict(), os.path.join(cheek_path, 'enc.pkl'))
            torch.save(vmfmm.state_dict(), os.path.join(cheek_path, 'vmfmm.pkl'))

        # =============================================================== #
        # ==============================test============================= #
        # =============================================================== #
        decoder.eval()
        encoder.eval()
        vmfmm.eval()

        with torch.no_grad():
            _data, _target = next(iter(test_dataloader))
            _target = _target.numpy()
            _data = _data.to(DEVICE)
            _z, _, _ = encoder(_data)
            _pred = vmfmm.predict(_z)
            _acc = cluster_acc(_pred, _target)[0] * 100
            _nmi = NMI(_pred, _target)

            if best_score < _acc:
                best_score = _acc
                best_epoch = epoch
                best_dec = decoder.state_dict()
                best_encoder = encoder.state_dict()
                best_vmf = vmfmm.state_dict()

            stack_images = None
            for k in range(n_cluster):

                z = vmfmm.sample_by_k(k)
                fake_images = decoder(z)

                if stack_images is None:
                    stack_images = fake_images[:n_cluster].data.cpu().numpy()
                else:
                    stack_images = np.vstack((stack_images, fake_images[:n_cluster].data.cpu().numpy()))
            stack_images = torch.from_numpy(stack_images)
            save_images(stack_images, imgs_dir, 'test_dec_{}'.format(epoch), nrow=n_cluster)

            logger = open(os.path.join(log_path, "log.txt"), 'a')
            logger.write(
                "[VAE-vMFMM]: epoch: {}, g_loss: {}, acc: {}%, nmi: {}\n".format(epoch, g_t_loss / len(dataloader), _acc, _nmi)
            )
            logger.close()
            print("[VAE-vMFMM]: epoch: {}, g_loss: {}, acc: {}%, nmi: {}".format(epoch, g_t_loss / len(dataloader), _acc, _nmi))

    print('best score is: {}, iteration is: {}'.format(best_score, best_epoch))
    print('save model......')
    torch.save(best_dec, os.path.join(models_dir, 'dec.pkl'))
    torch.save(best_encoder, os. path.join(models_dir, 'enc.pkl'))
    torch.save(best_vmf, os.path.join(models_dir, 'vmfmm.pkl'))


if __name__ == '__main__':
    main()
