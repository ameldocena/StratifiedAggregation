from arguments import Arguments
from nets import Cifar10CNN
from nets import FashionMNISTCNN
import os
import torch
from loguru import logger
# Unmodified from original work
if __name__ == '__main__':
    args = Arguments(logger)
    if not os.path.exists(args.get_default_model_folder_path()):
        os.mkdir(args.get_default_model_folder_path())

    # ---------------------------------
    # ----------- Cifar10CNN ----------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar10CNN.model")
    torch.save(Cifar10CNN().state_dict(), full_save_path)

    # ---------------------------------
    # -------- FashionMNISTCNN --------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "FashionMNISTCNN.model")
    torch.save(FashionMNISTCNN().state_dict(), full_save_path)