# Download weights
```commandline
sh_utils
sh get_weights_safety_ris.sh
```

# Docker setup
```commandline
cd docker
sh build.sh
sh start.sh
```

# Environment
## create dataset
```commandline
python utilite_cross_dataset.py
```

# Validation
## dataset 1
```commandline
sh validate_safety_ris_dataset_1.sh
```
## dataset 2
```commandline
sh validate_safety_ris_dataset_2.sh
```

# Train
## RIS (Risk-Informed Subgoal)
```commandline
sh train_safety_ris.sh
```

## SAC-Lagrangian (SAC with safety constraints)
```commandline
sh train_sac_lagrangian.sh
```

Key parameters for SAC-Lagrangian:
- `--train_sac True`: Enable SAC mode (instead of RIS)
- `--safety True`: Enable safety constraints (Lagrangian multiplier)
- `--cost_limit 5.0`: Cost limit for safety constraint
- `--update_lambda 1000`: Frequency of lambda (Lagrangian multiplier) updates
- `--lambda_initialization 1.0`: Initial value for lambda
