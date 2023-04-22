from transformers import AutoModel, AutoTokenizer
import sys

def download_model(model_path):
    AutoTokenizer.from_pretrained(model_path)
    AutoModel.from_pretrained(model_path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage:\n\t{sys.argv[0]} model_path')
        exit(1)
    print(f'Start to cache model_path = {sys.argv[1]}')
    download_model(sys.argv[1])
    print(f'Finish cache model_path = {sys.argv[1]}')
