export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

# Resume from checkpoint: add --load_weights
python ris_train_polamp_env.py --eval_freq 50000 --exp_name polamp_env_ex_6 \
                               --dataset cross_dataset_test_level_2 \
                               --not_visual_validation \
                               --plot_trajectory \
                                --with_noise \
                                --obs_noise_std 0.1\
                                --action_noise_std 0.1
