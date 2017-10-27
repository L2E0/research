import gan_model
import hsv_model
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="colorization")
    parser.add_argument('-p', '--pattern', type=str, default='hsv')
    args = parser.parse_args()
    return args

def summary(pat):
    model = gan_model.ColorizationModel() if pat == 'gan' else hsv_model.ColorizationModel()
    model.summary()

if __name__ == "__main__":
    args = get_args()
    summary(args.pattern)
