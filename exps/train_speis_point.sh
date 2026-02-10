export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

# Fix compatibility: use stable-baselines3 2.0.1 which works with gymnasium 0.27.1
#pip uninstall -y stable-baselines3 gymnasium 2>/dev/null || true
#pip install 'stable-baselines3==2.0.1' --quiet 2>/dev/null || true
#pip install 'gymnasium==0.27.1' --quiet 2>/dev/null || true
#pip install 'shimmy>=0.2.1' --quiet 2>/dev/null || true

cd ../mfnlc
python exps/train/obstacle/ris/point.py --obs_noise_std 0.05 --action_noise_std 0.05