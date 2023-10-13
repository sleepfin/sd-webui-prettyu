# 经验：
# 1. 总训练步长一般控制在1500-5000，人物训练一般需要更少的step。（除非network_dim特别大，需要增加步长）
# 2. network_dim越大，可以学习越复杂的特征，有助于学习细节，需要更长的训练步数，但也容易过拟合。在XL下建议更大的dim。训练画风建议使用128，模型大小144MB。训练人物建议使用64
# 3. network_alpha建议设置为network_dim的一半
# 4. learning_rate建议不设置，而是设置unet_lr=1e-4，text_encoder_lr=1e-5（通常时unet_lr的1/2或1/10）。并且不要设置--network_train_unet_only才能生效
# 5. scheduler建议使用cosine_with_restarts，重启次数一般是1,4,8，取决与训练步长（步长越大，可以用更大的重启次数）
# 6. optimizer_type，显存允许的情况下保持默认
# 7. cache_latents，不影响精度的情况下减小缓存（因为VAE不需要训练），与random_crop/color_aug互斥
# 8. Loss一般要下降到0.01以下，如0.008，大概率才是训练的比较好了
# 9. 如果使用了LoCon或LoHa，增加的network_args参数中，建议conv_dim取值为network_dim的一半，conv_alpha取值为conv_dim的一半
# 10. 如果用到了dyLora，network_args的unit设置为conv_dim的1/4

# 常用参数
# 1. XL+Lora
# --network_module="networks.lora"
# --network_dim=64
# --network_alpha=32

# 2. XL+LoCon
# --network_module="networks.lora"
# --network_dim=64
# --network_alpha=32
# --network_args conv_dim=32 conv_alpha=16

# 3. XL+dyLoCon
# --network_module="networks.dylora"
# --network_dim=64
# --network_alpha=32
# --network_args conv_dim=32 conv_alpha=16 unit=8


# 参考文档：
# https://www.bilibili.com/read/cv22793033/

/home/zzy2/workspace/stable-diffusion-webui/venv/bin/python /home/zzy2/workspace/sd-scripts/train_network.py \
--pretrained_model_name_or_path=/home/zzy2/workspace/stable-diffusion-webui/models/Stable-diffusion/majicmixRealistic_betterV2V25.safetensors \
--train_data_dir=/home/zzy2/workspace/sd-webui-prettyu/prettyu/lora/data/train_zhangzhenyu/ \
--output_dir=/home/zzy2/workspace/sd-webui-prettyu/prettyu/lora/models/zzy_retouch_aug3_lora_sd15_v7/ \
--prior_loss_weight=1.0 \
--resolution="512,512" \
--train_batch_size=2 \
--unet_lr=1e-4 \
--text_encoder_lr=1e-5 \
--max_train_epochs=20 \
--xformers \
--save_every_n_epochs=20 \
--save_model_as="safetensors" \
--clip_skip=2 \
--seed=42 \
--no_half_vae \
--network_module="networks.lora" \
--network_dim=64 \
--network_alpha=32 \
--cache_latents \
--lr_scheduler=cosine_with_restarts \
--lr_scheduler_num_cycles=1

# 因为要对比最后几个checkpoint，cosine函数训练到2/3的epoch时，lr降低到75%水位。最好不要restart
# --network_args conv_dim=32 conv_alpha=16 \
# --learning_rate=1e-4 \
# --network_train_unet_only \
# --color_aug \
# --use_8bit_adam \ 
# --mixed_precision="fp16" \

