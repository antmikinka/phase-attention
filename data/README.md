Byte-level LM data (not committed):

    curl -L -o data/tinystories_valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt

The training script splits this 19 MB file 95/5 into train and val.
