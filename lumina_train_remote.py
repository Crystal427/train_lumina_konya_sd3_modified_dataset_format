# training with captions

# Swap blocks between CPU and GPU:
# This implementation is inspired by and based on the work of 2kpr.
# Many thanks to 2kpr for the original concept and implementation of memory-efficient offloading.
# The original idea has been adapted and extended to fit the current project's needs.

# Key features:
# - CPU offloading during forward and backward passes
# - Use of fused optimizer and grad_hook for efficient gradient processing
# - Per-block fused optimizer instances

import argparse
import copy
import math
import os
from multiprocessing import Value
import toml

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from library import (
    deepspeed_utils,
    lumina_train_util,
    lumina_util,
    strategy_base,
    strategy_lumina,
    sai_model_spec
)
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler

import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util

# import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.custom_train_functions import apply_masked_loss, add_custom_train_arguments


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    # sdxl_train_util.verify_sdxl_training_args(args)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    # temporary: backward compatibility for deprecated options. remove in the future
    if not args.skip_cache_check:
        args.skip_cache_check = args.skip_latents_validity_check

    # assert (
    #     not args.weighted_captions
    # ), "weighted_captions is not supported currently / weighted_captionsは現在サポートされていません"
    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning(
            "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled / cache_text_encoder_outputs_to_diskが有効になっているため、cache_text_encoder_outputsも有効になります"
        )
        args.cache_text_encoder_outputs = True

    if args.cpu_offload_checkpointing and not args.gradient_checkpointing:
        logger.warning(
            "cpu_offload_checkpointing is enabled, so gradient_checkpointing is also enabled / cpu_offload_checkpointingが有効になっているため、gradient_checkpointingも有効になります"
        )
        args.gradient_checkpointing = True

    # assert (
    #     args.blocks_to_swap is None or args.blocks_to_swap == 0
    # ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing / blocks_to_swapはcpu_offload_checkpointingと併用できません"

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
    if args.cache_latents:
        latents_caching_strategy = strategy_lumina.LuminaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(
            ConfigSanitizer(True, True, args.masked_loss, True)
        )
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = (
            config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        )
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = (
        train_dataset_group if args.max_data_loader_n_workers == 0 else None
    )
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(16)  # TODO これでいいか確認

    if args.debug_dataset:
        if args.cache_text_encoder_outputs:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
                strategy_lumina.LuminaTextEncoderOutputsCachingStrategy(
                    args.cache_text_encoder_outputs_to_disk,
                    args.text_encoder_batch_size,
                    args.skip_cache_check,
                    False,
                )
            )
        strategy_base.TokenizeStrategy.set_strategy(
            strategy_lumina.LuminaTokenizeStrategy(args.system_prompt)
        )

        train_dataset_group.set_current_strategies()
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # モデルを読み込む

    # load VAE for caching latents
    ae = None
    if cache_latents:
        ae = lumina_util.load_ae(
            args.ae, weight_dtype, "cpu", args.disable_mmap_load_safetensors
        )
        ae.to(accelerator.device, dtype=weight_dtype)
        ae.requires_grad_(False)
        ae.eval()

        train_dataset_group.new_cache_latents(ae, accelerator)

        ae.to("cpu")  # if no sampling, vae can be deleted
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # prepare tokenize strategy
    if args.gemma2_max_token_length is None:
        gemma2_max_token_length = 256
    else:
        gemma2_max_token_length = args.gemma2_max_token_length

    lumina_tokenize_strategy = strategy_lumina.LuminaTokenizeStrategy(
        args.system_prompt, gemma2_max_token_length
    )
    strategy_base.TokenizeStrategy.set_strategy(lumina_tokenize_strategy)

    # load gemma2 for caching text encoder outputs
    gemma2 = lumina_util.load_gemma2(
        args.gemma2, weight_dtype, "cpu", args.disable_mmap_load_safetensors
    )
    gemma2.eval()
    gemma2.requires_grad_(False)

    text_encoding_strategy = strategy_lumina.LuminaTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # cache text encoder outputs
    sample_prompts_te_outputs = None
    if args.cache_text_encoder_outputs:
        # Text Encodes are eval and no grad here
        gemma2.to(accelerator.device)

        text_encoder_caching_strategy = (
            strategy_lumina.LuminaTextEncoderOutpu