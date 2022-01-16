import subprocess
import sys
from saving_utils import get_class_label_from_num

dataset = sys.argv[1]
source = sys.argv[2]
target = sys.argv[3]

for upload in [1.0, 0.7]:
    for download in [1.0, 0.7]:
        for dist in ["IID", "Non-IID"]:
            for initialization in ["Default", "Randomized"]:
                with open("experimental_param_notes.txt", "r") as f:
                    run = True
                    for line in f.readlines():
                        if f"{source}, {target}, {get_class_label_from_num(dataset, source)}, {get_class_label_from_num(dataset, target)}, {upload}, {download}, FedAvg, {initialization}, {dist}" in line:
                            print("These settings have already been run... Continuing...")
                            run = False
                            break
                        else:
                            ix = line.split(',')[0]
                if run:
                    subprocess.Popen(f"python3 federated_training_wrapped.py {int(ix)+1} {upload} {download} {dataset} {target} {source} {initialization} {dist}")
                    subprocess.Popen(f"rsync -a --delete empty_dir/ {int(ix)+1}_models/")
