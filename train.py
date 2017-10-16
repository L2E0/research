from colorization_model import ColorizationModel
import argparse
from data_gen import xygen, batchgen, epochgen, count_file, chunk
def get_args():
    parser = argparse.ArgumentParser(description="colorization")
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-r', '--resume', dest="resume", action="store_true")
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('-c', '--category', type=str, default="grass")
    parser.add_argument('-m', '--mode', type=str, default='train')
    parser.add_argument('-s', '--step_size', type=int, default=100)
    args = parser.parse_args()
    return args

def train_data(category, batch_size):
    path = 'train_' + category

    gen = xygen(path, horizontal_flip=True,
                vertical_flip=True,
                rotation_range=180,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.3)

def val_data(category):
    val_path = 'valid_' + category
    val_size = count_file(val_path)
    val_gen = chunk(xygen(val_path), val_size)


if __name__  == "__main__":
    args = get_args()
    f = open('epoch.txt', 'r')
    offset = int(next(f)) if args.resume else 0
    model = ColorizationModel()
    if args.mode == "train":
        if offset != 0:
            model.load_weights()
            model.train(args.batch, args.step_size, args.epochs, args.category, offset)
    elif args.mode == "predict":
        model.predict(args.category, offset)
    else:
        print("^^;")
