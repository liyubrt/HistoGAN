import os
import logging
from datetime import datetime, timedelta, date
from torch._C import device
from tqdm import tqdm
from typing import Any, AnyStr, Dict, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from torchvision import transforms
from torch.utils.data import DataLoader
from dl.utils.io_utils import get_current_rank


from ReHistoGAN import recoloringTrainer
from rehistoGAN import get_args
import utils.pyramid_upsampling as upsampling
from histogram_classes.RGBuvHistBlock import RGBuvHistBlock
from jupiter_data_convert import JupiterData, convert_to_time_in_a_day


class JupiterDataPL(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.source_dataset = None
        self.source_df = None

    def setup(self, stage: str = "fit") -> None:
        if stage == "test":
            self.source_df = pd.read_csv(self.args.source_csv, low_memory=False)
            # self.source_df['datetime'] = self.source_df.collected_on.apply(lambda x: convert_to_time_in_a_day(x))
            self.source_dataset = JupiterData(self.args.source_dir, self.source_df)
            logging.info(f'source df size: {len(self.source_df)}')

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.source_dataset,
            batch_size=8,
            num_workers=2,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
        )

class JupiterDataConverter(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rank = get_current_rank()
        self.init_datasets()
        self.load_model()
        self.prepare_for_HistoGAN()

    def init_datasets(self):
        self.target_df = pd.read_csv(self.args.target_csv, low_memory=False)
        self.target_df['datetime'] = self.target_df.collected_on.apply(lambda x: convert_to_time_in_a_day(x))
        self.target_df['index'] = self.target_df.index
        self.target_dataset = JupiterData(self.args.target_dir, self.target_df, input_size=[])
        self.num_targets = self.args.num_targets  # number of targets to transfer to for each source image
        self.save_dir = self.args.save_dir
        logging.info(f'target df size: {len(self.target_df)}')

    def load_model(self):
        self.model = recoloringTrainer(
            self.args.name,
            self.args.results_dir,
            self.args.models_dir,
            batch_size=self.args.batch_size,
            gradient_accumulate_every=self.args.gradient_accumulate_every,
            image_size=self.args.image_size,
            network_capacity=self.args.network_capacity,
            transparent=self.args.transparent,
            lr=self.args.learning_rate,
            num_workers=self.args.num_workers,
            save_every=self.args.save_every,
            trunc_psi=self.args.trunc_psi,
            fp16=self.args.fp16,
            fq_layers=self.args.fq_layers,
            fq_dict_size=self.args.fq_dict_size,
            attn_layers=self.args.attn_layers,
            hist_insz=self.args.hist_insz,
            hist_bin=self.args.hist_bin,
            hist_sigma=self.args.hist_sigma,
            hist_resizing=self.args.hist_resizing,
            hist_method=self.args.hist_method,
            rec_loss=self.args.rec_loss,
            fixed_gan_weights=self.args.fixed_gan_weights,
            skip_conn_to_GAN=self.args.skip_conn_to_GAN,
            initialize_gan=self.args.initialize_gan,
            variance_loss=self.args.variance_loss,
            internal_hist=self.args.internal_hist,
            change_hyperparameters=self.args.change_hyperparameters,
            change_hyperparameters_after=self.args.change_hyperparameters_after
        )
        self.model.load(self.args.load_from)

    def prepare_for_HistoGAN(self):
        self.resizing_mode = 'upscaling'  # upscaling, downscaling, none
        self.base_transform = transforms.Compose([transforms.ToTensor()])  # used for transforming target
        self.image_size = 256  # resize to 256x256 before feeding to the model
        self.image_transform = transforms.Compose([
            # transforms.Lambda(convert_transparent_to_rgb),
            # transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),  # note slightly different from Image.resize
            # transforms.Lambda(expand_greyscale(3))
        ])
        self.histblock = RGBuvHistBlock(insz=self.args.hist_insz, h=self.args.hist_bin,
                                        resizing=self.args.hist_resizing, method=self.args.hist_method,
                                        sigma=self.args.hist_sigma,
                                        device=f'cuda:{self.rank}')

    def find_target_styles(self, source_datetime) -> pd.DataFrame:
        # binary search to find targets
        min_delta = timedelta(seconds=5)  # ten seconds range
        delta = timedelta(seconds=12*60*60)  # one day
        last_matched_rows = pd.DataFrame(data={})
        while delta > min_delta:
            matched_rows = self.target_df[(self.target_df.datetime >= source_datetime - delta) & 
                                          (self.target_df.datetime < source_datetime + delta)]
            if len(matched_rows) >= self.num_targets:
                last_matched_rows = matched_rows
                delta /= 2
            else:
                break
        return last_matched_rows.sample(n=self.num_targets).index.to_list()

    def process_image(self, image, style):
        # process image
        _, height, width = image.shape
        reference = image.cpu().numpy().transpose((1, 2, 0))
        image = torch.unsqueeze(self.image_transform(image), dim=0)
        # process hist
        style = torch.unsqueeze(self.base_transform(style), dim=0).to(device=image.device)
        h = self.histblock(style)

        # generate result
        result = self.model.evaluate('', image_batch=image,
                            hist_batch=h,
                            resizing=self.resizing_mode,
                            resizing_method=self.args.upsampling_method,
                            swapping_levels=self.args.swapping_levels,
                            pyramid_levels=self.args.pyramid_levels,
                            level_blending=self.args.level_blending,
                            original_size=[width, height],
                            original_image=reference,  # not used
                            input_image_name=reference,
                            save_input=False,
                            post_recoloring=self.args.post_recoloring)
        # upsample image
        result = upsampling.pyramid_upsampling(
            result, reference, levels=self.args.pyramid_levels,
            swapping_levels=self.args.swapping_levels, blending=self.args.level_blending, to_tensor=False)
        return result

    def test_step(self, batch, batch_idx):
        images = torch.permute(batch["image"], (0, 3, 1, 2))  # NHWC to NCHW
        depths = batch["depth"]
        ids = batch["id"]
        collected_ons = batch["collected_on"]

        for image, depth, id, collected_on in zip(images, depths, ids, collected_ons):
            sample = {"depth": depth.cpu().numpy(), "target_ids":[]}
            datetime = convert_to_time_in_a_day(collected_on)
            target_row_idxs = self.find_target_styles(datetime)
            for t_i in target_row_idxs:
                t_sample = self.target_dataset[t_i]
                result = self.process_image(image, t_sample["image"]).astype(np.float32)
                sample["target_ids"].append(t_sample["id"])
                sample[t_sample["id"]] = result
            sample_save_dir = os.path.join(self.save_dir, id)
            os.makedirs(sample_save_dir, exist_ok=True)
            sample_save_path = os.path.join(sample_save_dir, "color_transfer_output.npz")
            np.savez_compressed(sample_save_path, **sample)
        
        if (1+batch_idx) % 200 == 0:
            logging.info(f'processed {1+batch_idx} batches')

if __name__ == '__main__':
    args = get_args()
    args.source_dir = '/home/bluerivertech/li.yu/data/Jupiter_train_v4_53_missing_human_relabeled'
    args.source_csv = '/home/bluerivertech/li.yu/data/Jupiter_train_v4_53_missing_human_relabeled/master_annotations_untouched.csv'
    args.target_dir = '/home/bluerivertech/li.yu/data/Jupiter_train_v4_53_heavy_dust_relabeled'
    args.target_csv = '/home/bluerivertech/li.yu/data/Jupiter_train_v4_53_heavy_dust_relabeled/master_annotations.csv'
    args.num_targets = 2
    args.save_dir = '/home/bluerivertech/li.yu/data/Jupiter_train_v4_53_missing_human_relabeled/processed_color_transfer/images'
    os.makedirs(args.save_dir, exist_ok=True)

    jupiter_data_converter = JupiterDataConverter(args)
    source_data_module = JupiterDataPL(args)
    num_gpus = 2
    trainer = pl.Trainer(
        devices=num_gpus,
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        accelerator="gpu",
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )
    trainer.test(jupiter_data_converter, datamodule=source_data_module)