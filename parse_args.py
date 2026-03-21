import configargparse
import argparse
def parse_args(input_args=None):
    parser = configargparse.ArgumentParser(description="Simple example of a Neural Gaffer training script.")
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="kxic/zero123-xl",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--specific_object",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--cond_lighting_index",
        type=int,
        default=0,
    )
    
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--lighting_per_view",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--total_view",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="zero123-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    parser.add_argument(
        "--training_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=80000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="unconditional guidance scale, if guidance_scale>1.0, do_classifier_free_guidance"
    )

    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.05,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800"
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=4000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--best_checkpoint_until_step",
        type=int,
        default=72000,
        help="Continuously refresh a dedicated best-validation checkpoint only up to this optimization step.",
    )
    parser.add_argument(
        "--non_finite_early_stop_patience",
        type=int,
        default=3,
        help="Stop training after this many skipped non-finite optimization steps.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        # 训练时每隔多少步记录一次 loss / 权重统计到 wandb。
        # 数值越小，日志越密；但也会带来更高的记录开销。
        help="Log training metrics every N optimization steps.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",    # log_image currently only for wandb
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", default=True, help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        default=True,
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    # =======================
    # 高光相关 loss 参数
    # =======================
    # 当前默认策略:
    # 1. 标准 diffusion loss 继续作为主损失
    # 2. 默认开启 image-space 高光辅助约束
    # 3. latent/noise-space 的高光重加权默认关闭
    parser.add_argument(
        "--use_highlight_weighted_loss",
        action="store_true",
        default=False,
        # 主开关:
        # - True: 启用高光区域加权
        # - False: 使用普通的全图均匀 MSE
        help="Whether to upweight highlight regions directly in latent/noise-space diffusion loss."
    )
    parser.add_argument(
        "--use_image_space_highlight_loss",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "y", "on"),
        default=True,
        help="Whether to apply an auxiliary highlight-aware constraint in image space. Enabled by default."
    )
    parser.add_argument(
        "--image_space_constraint_weight",
        type=float,
        default=0.1,
        help="Overall weight applied to the image-space highlight auxiliary loss."
    )
    parser.add_argument(
        "--highlight_loss_weight",
        type=float,
        default=1.0,
        # 高光区域的额外权重。
        # 最终普通区域约为 1，高光区域约为 1 + highlight_loss_weight * score
        help="Extra spatial weight applied to highlight regions for highlight-aware losses."
    )
    parser.add_argument(
        "--highlight_threshold",
        type=float,
        default=0.8,
        # 亮度阈值。
        # 越高表示只把最亮的部分视为高光。
        help="Luminance threshold used to detect highlights from the target image."
    )
    parser.add_argument(
        "--highlight_gamma",
        type=float,
        default=2.0,
        # 软权重模式下的指数参数。
        # gamma 越大，真正非常亮的区域会被强调得更明显。
        help="Exponent used by soft highlight weighting. Larger values focus more on strong highlights."
    )
    parser.add_argument(
        "--highlight_soft_weighting",
        action="store_true",
        # 是否使用平滑的连续权重，而不是简单的 0/1 二值 mask。
        help="Use a soft highlight score instead of a binary threshold mask."
    )
    parser.add_argument(
        "--highlight_use_quantile_threshold",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "y", "on"),
        default=False,
        help="Adapt the highlight threshold per image from foreground luminance quantiles instead of using only a fixed threshold.",
    )
    parser.add_argument(
        "--highlight_quantile",
        type=float,
        default=0.95,
        help="Foreground luminance quantile used to derive the adaptive highlight threshold when quantile thresholding is enabled.",
    )
    parser.add_argument(
        "--highlight_min_threshold",
        type=float,
        default=0.6,
        help="Lower clamp for the adaptive highlight threshold to avoid overly broad highlight regions.",
    )
    parser.add_argument(
        "--highlight_max_threshold",
        type=float,
        default=0.95,
        help="Upper clamp for the adaptive highlight threshold so area-light highlights remain sufficiently wide.",
    )
    parser.add_argument(
        "--foreground_background_threshold",
        type=float,
        default=0.98,
        help="Pixels with all RGB channels close to 1 above this threshold are treated as white background and excluded from highlight-related masks and losses.",
    )
    parser.add_argument(
        "--random_lighting_condition_prob",
        type=float,
        default=0.1,
        help="Probability of replacing the training condition image with the random area-light rendering.",
    )

    parser.add_argument(
        "--train_img_dir",
        type=str,
        default='/scratch/datasets/hj453/objaverse-rendering/filtered_V2/rendered_images_resized',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--train_lighting_dir",
        type=str,
        default='/scratch/datasets/hj453/objaverse-rendering/filtered_V2/preprocessed_environment_resized/',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--val_img_dir",
        type=str,
        default='./Neural_Gaffer/preprocessed_data',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--val_lighting_dir",
        type=str,
        default="./preprocessed_lighting_data",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--initial_validation_step",
        type=int,
        default=-1,
        help="Run initial validation at this step (set to -1 to disable).",
    )
    parser.add_argument(
        "--early_stop_on_validation_collapse",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "y", "on"),
        default=True,
        help="Whether to stop training if validation collapses after the best-checkpoint window.",
    )
    parser.add_argument(
        "--collapse_psnr_threshold",
        type=float,
        default=5.0,
        help="Absolute mean held-out PSNR threshold used to detect catastrophic post-window collapse.",
    )
    parser.add_argument(
        "--collapse_relative_psnr_ratio",
        type=float,
        default=0.25,
        help="Relative mean-PSNR threshold versus the best pre-window checkpoint used for collapse detection.",
    )

    parser.add_argument(
        "--num_validation_batches",
        type=int,
        default=120*24*80,
        # default=24,
        help=(
            "Number of batches to use for validation. If `None`, use all batches."
        ),
    )

    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_neural_gaffer_private",
        help=(
            "Fallback wandb project name. Training will first try to reuse the most recent local wandb project,"
            " and only fall back to this value if no previous project can be inferred."
        ),
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help=(
            "Optional explicit wandb run name. If not provided, training will build a descriptive name from the"
            " active loss settings plus a timestamp."
        ),
    )
    parser.add_argument(
        "--wandb_run_note",
        type=str,
        default=None,
        help=(
            "Optional short note appended to the auto-generated wandb run name, for example ablation, threshold_tune,"
            " or better_mask_vis."
        ),
    )

    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        help=(
            "A parameter that controls if the metrics should be computed during validation. If `False`, the metrics will not be computed."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images."
        )

    return args
