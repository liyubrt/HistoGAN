import os
from datetime import datetime, timedelta, date
from tqdm import tqdm
from typing import Any, AnyStr, Dict, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dl.dataset.dataframe_value_constants import DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH
from dl.utils import io_utils
from dl.utils.config import DEFAULT_NORMALIZATION_PARAMS, LEFT, POINT_CLOUD, MAX_DEPTH
from dl.utils.image_transforms import resize_image, resize_depth, depth_from_point_cloud


from ReHistoGAN import recoloringTrainer
from rehistoGAN import get_args
import utils.pyramid_upsampling as upsampling
from histogram_classes.RGBuvHistBlock import RGBuvHistBlock


class JupiterData(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir,
            df,
            normalization_params: str = DEFAULT_NORMALIZATION_PARAMS,
            input_size: Union[Tuple, List] = (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
    ) -> None:
        self.data_dir = data_dir
        self.df = df
        self.normalization_params = normalization_params
        self.input_size = input_size

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        if self.data_dir is not None:
            stereo_data_sample = np.load(
                os.path.join(self.data_dir, df_row.stereo_pipeline_npz_save_path)
            )
        else:
            stereo_data_sample = np.load(df_row.stereo_pipeline_npz_save_path)
        image = io_utils.normalize_image(
            stereo_data_sample[LEFT],
            df_row.hdr_mode,
            self.normalization_params,
            return_8_bit=False,
        )
        depth = depth_from_point_cloud(
            stereo_data_sample[POINT_CLOUD],
            clip_and_normalize=True,
            max_depth=MAX_DEPTH,
            make_3d=True,
        )
        if len(self.input_size) > 0:
            H, W = self.input_size
            image = resize_image(image, H, W)
            depth = resize_depth(depth, H, W)
        return {"image": image, "depth": depth, "id": df_row.id, "collected_on": df_row.collected_on}

    def __len__(self) -> int:
        return len(self.df)


def convert_to_time_in_a_day(x):
    if len(x) == 19:  # 1900-01-01 15:45:30
        return datetime.strptime(str(x)[11:], "%H:%M:%S")
    else:  # 1900-01-01 15:45:30.115
        return datetime.strptime(str(x)[11:], "%H:%M:%S.%f")


class JupiterDataConverter:
    def __init__(self, args):
        self.args = args
        self.device = torch.cuda.current_device()
        self.init_datasets()
        self.load_model()
        self.prepare_for_HistoGAN()

    def init_datasets(self):
        self.source_df = pd.read_csv(self.args.source_csv, low_memory=False)
        # self.source_df['datetime'] = self.source_df.collected_on.apply(lambda x: convert_to_time_in_a_day(x))
        self.source_dataset = JupiterData(self.args.source_dir, self.source_df)
        self.target_df = pd.read_csv(self.args.target_csv, low_memory=False)
        self.target_df['datetime'] = self.target_df.collected_on.apply(lambda x: convert_to_time_in_a_day(x))
        self.target_df['index'] = self.target_df.index
        self.target_dataset = JupiterData(self.args.target_dir, self.target_df)
        self.num_targets = self.args.num_targets  # number of targets to transfer to for each source image
        self.save_dir = self.args.save_dir
        print(f'source df size: {len(self.source_df)}, target df size: {len(self.target_df)}')

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
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),  # note slightly different from Image.resize
            # transforms.Lambda(expand_greyscale(3))
        ])
        self.histblock = RGBuvHistBlock(insz=self.args.hist_insz, h=self.args.hist_bin,
                                        resizing=self.args.hist_resizing, method=self.args.hist_method,
                                        sigma=self.args.hist_sigma,
                                        device=self.device)

    def find_target_styles(self, source_datetime) -> pd.DataFrame:
        # source_datetime = source_df_row.datetime
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
        height, width, _ = image.shape
        reference = image.copy()
        image = torch.unsqueeze(self.image_transform(image), dim=0).to(device=self.device)
        # process hist
        style = torch.unsqueeze(self.base_transform(style), dim=0).to(device=self.device)
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

    def run(self):
        for sample in tqdm(self.source_dataset, total=len(self.source_dataset)):
            image = sample["image"]
            s_id = sample["id"]
            s_datetime = convert_to_time_in_a_day(sample["collected_on"])

            del sample["image"]
            del sample["depth"]
            del sample["id"]
            del sample["collected_on"]
            sample["target_ids"] = []
            target_row_idxs = self.find_target_styles(s_datetime)
            for t_i in target_row_idxs:
                t_sample = self.target_dataset[t_i]
                result = self.process_image(image, t_sample["image"]).astype(np.float32)
                sample["target_ids"].append(t_sample["id"])
                sample[t_sample["id"]] = result
            sample_save_dir = os.path.join(self.save_dir, s_id)
            os.makedirs(sample_save_dir, exist_ok=True)
            sample_save_path = os.path.join(sample_save_dir, "color_transfer_output.npz")
            np.savez_compressed(sample_save_path, **sample)
            # break


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
    jupiter_data_converter.run()