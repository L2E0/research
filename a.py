import argparse
import gan_model
import hsv_model
from main import *

args = get_args()
for i in range(100):
    try: 
        model = gan_model.ColorizationModel() if args.pattern=='gan' else hsv_model.ColorizationModel()
        offset = read_offset('%s.txt' % (args.pattern)) if args.resume or args.mode=='predict' else 0
        train(model, offset, **vars(args))
    except:
        pass
