import os
import pdb
import shutil
import pandas as pd
from datasets import Dataset, load_dataset

audio_dir = "./data/Patient_sil_trim_16k_normed_5_snr_40/"
# split_files = {"train": "data/Patient_sil_trim_16k_normed_5_snr_40/train.csv", 
#                "test": "data/Patient_sil_trim_16k_normed_5_snr_40/test.csv",
#                "dev": "data/Patient_sil_trim_16k_normed_5_snr_40/dev.csv"}
src_dataset = load_dataset("audiofolder", data_dir=audio_dir, split="train")
pdb.set_trace()
def train_dev_test_split(
    dataset: Dataset, dev_rate=0.1, test_rate=0.1, seed=1, metadata_output=False, root_dir=None
):
    """
    input: dataset
    dev_rate,
    test_rate
    seed
    -------
    Output:
    dataset_dict{"train", "dev", "test"}
    """
    train_dev_test = dataset.train_test_split(test_size=test_rate, seed=seed)
    test = train_dev_test["test"]
    train_dev = train_dev_test["train"]

    if len(train_dev) <= int(len(dataset) * dev_rate):
        train = Dataset.from_dict({"audio": [], "transcription": []})
        dev = train_dev
    else:
        train_dev = train_dev.train_test_split(
            test_size=int(len(dataset) * dev_rate), seed=seed
        )
        train = train_dev["train"]
        dev = train_dev["test"]
    
    train_size = len(train)
    dev_size = len(dev)
    test_size = len(test)
    
    print(f"Train Size: {len(train)}")
    print(f"Dev Size: {len(dev)}")
    print(f"Test Size: {len(test)}")
    import pdb
    if metadata_output:
        pdb.set_trace()
        train_df = pd.DateFrame(train)
        dev_df = pd.DataFrame(dev)
        test_df = pd.DataFrame(test)

    try:
         os.path.exists(root_dir)
    except:
        raise FileNotFoundError
    
    # Create directories for train, dev, and test data
    import pdb
    if not os.path.exists(f'{root_dir}/train'):
        os.makedirs(f'{root_dir}/train')
    if not os.path.exists(f'{root_dir}/dev'):
        os.makedirs(f'{root_dir}/dev')
    if not os.path.exists(f'{root_dir}/test'):
        os.makedirs(f'{root_dir}/test')

    pdb.set_trace()
    train_df.to_csv(f'{root_dir}/train/metadata.csv', index=False)

    dev_df.to_csv(f'{root_dir}/dev/metadata.csv', index=False)

    test_df.to_csv(f'{root_dir}/test/metadata.csv', index=False)

    return train, dev, test

train, dev, test = train_dev_test_split(src_dataset, dev_rate=0.1, test_rate=0.1, seed=1, metadata_output=True, root_dir=audio_dir)

pdb.set_trace()