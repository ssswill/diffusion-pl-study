

import pytorch_lightning as pl
from argparse import ArgumentParser


from pytorch_lightning import Trainer


from model import MInterface
from data import DInterface



def main(args):
    pl.seed_everything(args.seed)
    data_module = DInterface(**vars(args))

    model = MInterface(**vars(args))

    trainer = Trainer.from_argparse_args(args, gpus=1, )

    trainer.test(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()

    # dataset Info
    parser.add_argument('--dataset', default='inpaint_dataset_sr', type=str)
    parser.add_argument('--data_root', default="E:\\data\\celeba_hq_256\\test2", type=str)
    parser.add_argument('--save_dir', default="lightning_logs_test\\", type=str)

    # dataloader Info
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dict', default='weights/sr_diffusion.pth', type=str)
    parser.add_argument('--save_dict', default='weights/sr_diffusion.pth', type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # model Info
    parser.add_argument('--model_name', default='palette', type=str)
    parser.add_argument('--init_type', default='kaiming', type=str)
    parser.add_argument('--network_struct', default='sr3', type=str)
    parser.add_argument('--loss', default='mse', type=str)
    parser.add_argument('--sample_num', default=8, type=int)
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')

    parser.add_argument('--ema_scheduler', default=True, type=bool)
    parser.add_argument('--ema_start', default=1, type=int)
    parser.add_argument('--ema_decay', default=0.9999, type=float)

    # model LR
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr_scheduler', default='step', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # model network struct(unet)
    parser.add_argument('--in_channel', default=6, type=int)
    parser.add_argument('--out_channel', default=3, type=int)
    parser.add_argument('--inner_channel', default=64, type=int)
    parser.add_argument('--channel_mults', default=[1, 2, 4, 8, 8], type=list)
    parser.add_argument('--attn_res', default=[8], type=list)
    parser.add_argument('--res_blocks', default=3, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--image_size', default=128, type=list)
    parser.add_argument('--norm_groups', default=32, type=int)

    # model network struct(unet) beta scheduler
    parser.add_argument('--beta_schedule', default='linear', type=str)
    parser.add_argument('--n_timestep', default=1000, type=int)
    parser.add_argument('--linear_start', default=1e-6, type=float)
    parser.add_argument('--linear_end', default=0.001, type=float)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    parser.set_defaults(max_epochs=50)

    args = parser.parse_args()

    main(args)
