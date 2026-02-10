
export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

python ris_validate_polamp_env.py --exp_name final_weights \
                                  --dataset cross_dataset_test_level_2 \
                                  --not_visual_validation \
                                  --plot_trajectory \
                                  #--with_noise \
                                  #--obs_noise_std 0.1 \
                                  #--action_noise_std 0.1 \