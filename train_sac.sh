export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

# Train SAC: Soft Actor-Critic without safety constraints
# Key parameters:
#   --train_sac True          : Enable SAC mode (instead of RIS)
#   --safety False           : Disable safety constraints (pure SAC)
#
# --exp_name if the folder exists, weights will be loaded from the folder
python ris_train_polamp_env.py --eval_freq 50000 --exp_name polamp_env_sac \
                               --dataset cross_dataset_test_level_2 \
                               --train_sac True \
                               --safety False \
                               --not_visual_validation \
                               --plot_trajectory \
                               #--with_noise \
                               #--obs_noise_std 0.1 \
                               #--action_noise_std 0.1
