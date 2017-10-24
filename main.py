import argparse
import gan_model
import hsv_model
from data_gen import xygen, batchgen, epochgen, count_file, chunk
def get_args():
    parser = argparse.ArgumentParser(description="colorization")
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-r', '--resume', dest="resume", action="store_true")
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('-c', '--category', type=str, default="grass")
    parser.add_argument('-m', '--mode', type=str, default='train')
    parser.add_argument('-s', '--step', type=int, default=100)
    parser.add_argument('-p', '--pattern', type=str, default='hsv')
    args = parser.parse_args()
    return args

def train_data(category):
    path = 'train_' + category

    gen = xygen(path, horizontal_flip=True,
                vertical_flip=True,
                rotation_range=180,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.3)

    val_path = 'valid_' + category
    val_size = count_file(val_path)
    val_gen = chunk(xygen(val_path), val_size)

    return gen, val_gen

def train(model, offset, resume=False, batch=32, step=100, epochs=100, category='grass', mode='train'):
    if offset != 0:
        model.load_weights()
    xygen, val_gen = train_data(category)
    model.train(category, xygen, val_gen, batch, step, epochs, offset)

def predict(model, offset, category):
    model.load_weights()
    model.predict(category, offset)

def read_offset(file):
    f = open(file, 'r')
    return int(next(f))

if __name__  == "__main__":
    args = get_args()
    model = gan_model.ColorizationModel() if args.pattern=='gan' else hsv_model.ColorizationModel()
    offset = read_offset('epoch_%s.txt' % (args.pattern)) if args.resume or args.mode=='predict' else 0
    if args.mode == 'train':
        train(model, offset, vars(args))
    elif args.mode =='predict':
        predict(model, offset, args.category)
    else:
        print("^^;")
