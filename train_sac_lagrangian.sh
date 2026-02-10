export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

# Train SAC-Lagrangian: SAC with safety constraints using Lagrangian multiplier
# Key parameters:
#   --train_sac True          : Enable SAC mode (instead of RIS)
#   --safety True            : Enable safety constraints (Lagrangian multiplier)
#   --cost_limit 5.0         : Cost limit for safety constraint
#   --update_lambda 1000     : Frequency of lambda (Lagrangian multiplier) updates
#   --lambda_initialization 1.0 : Initial value for lambda
#
# --exp_name if the folder exists, weights will be loaded from the folder
python ris_train_polamp_env.py --eval_freq 50000 --exp_name polamp_env_sac_lagrangian \
                               --dataset cross_dataset_test_level_2 \
                               --train_sac True \
                               --safety True \
                               --cost_limit 5.0 \
                               --update_lambda 1000 \
                               --lambda_initialization 1.0 \
                               --not_visual_validation \
                               --plot_trajectory \
                               #--with_noise \
                               #--obs_noise_std 0.1 \
                               #--action_noise_std 0.1
