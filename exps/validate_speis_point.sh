export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

# Fix compatibility: use stable-baselines3 2.0.1 which works with gymnasium 0.27.1
pip uninstall -y stable-baselines3 gymnasium 2>/dev/null || true
pip install 'stable-baselines3==2.0.1' --quiet 2>/dev/null || true
pip install 'gymnasium==0.27.1' --quiet 2>/dev/null || true
pip install 'shimmy>=0.2.1' --quiet 2>/dev/null || true

cd ../mfnlc
python exps/train/obstacle/ris/point.py --validate \
                                        --total_timesteps 1 \
                                        --validate_freq 1 \
                                        --load_model --load_model_folder idhvhftt \
                                        --add_subgoal_reinforce_sg_num 0 \
                                        --validate_video_idx 4 \
                                        --obs_noise_std 0.1 \
                                        --action_noise_std 0.1 \
                                        #--validate_subgoal_video 