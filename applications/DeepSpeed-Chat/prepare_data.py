from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from datasets.download import DownloadConfig
import transformers
import sys

transformers.logging.set_verbosity_debug()

def download_model(model_path):
    AutoTokenizer.from_pretrained(model_path, resume_download=True)
    AutoModel.from_pretrained(model_path, resume_download=True)

def download_dataset(dataset_path, name):
    while True:
        try:
            load_dataset(dataset_path, name, download_config=DownloadConfig(resume_download=True), verification_mode='all_checks')
            return
        except ConnectionError as e:
            print(f'Exception {e} when load dataset {dataset_path}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage:\n\t{sys.argv[0]} m model_path|d dataset_path')
        exit(1)

    if sys.argv[1] == 'm':
        print(f'Start to cache model_path = {sys.argv[2]}')
        download_model(sys.argv[2])
        print(f'Finish cache model_path = {sys.argv[2]}')
    else:
        print(f'Start to cache dataset_path = {sys.argv[2]}')
        download_dataset(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        print(f'Finish cache dataset_path = {sys.argv[2]}')
