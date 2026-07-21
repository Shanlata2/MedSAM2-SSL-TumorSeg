import torch
from nnunetv2.run.run_training import get_trainer_from_args

def main(dataset, config, fold, trainer_name, logfile):
    tr = get_trainer_from_args(dataset, config, fold, trainer_name, device=torch.device('cuda'))
    tr.run_training()
    tr.perform_actual_validation()

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], None)