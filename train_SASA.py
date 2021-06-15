import logging
import warnings
import argparse
import datetime
from fastai.vision.all import *
from SASA_Model import resnet50

CLASSES = 10

def run_training(ds, lr, epochs, output, num_heads, kernel_size, img_size, name):      
    print("Setup model..")
    model = resnet50(num_classes=CLASSES, attention=[False,False,True,True], num_heads=num_heads, kernel_size=kernel_size, image_size=img_size)
    
    print("built model")  
    print("Saving files to", output)

    learn = Learner(ds, model, opt_func=Adam, metrics=accuracy, 
                    cbs=[
                        CSVLogger(fname = './logs/attention/imagenette{}_SASA_{}.csv'.format(TIME, name), append=True),
                        SaveModelCallback(monitor='accuracy', fname= output + 'imagenette{}_SASA_{}'.format(TIME, name))
                    ]
                   )
    learn.fit(n_epoch=epochs, lr=lr, cbs=ReduceLROnPlateau(monitor='accuracy', min_delta=0.001, patience=10, factor=3, min_lr=1e-8)) 
    
        
def prepare_data(img_size, batch_size):
    path = "../../../data/imagenette2-320/"
    item_tfms=RandomResizedCrop(size=img_size, min_scale=0.8)
    batch_tfms=[*aug_transforms(flip_vert=False, max_lighting=0.2, max_rotate=15., max_zoom=1.1, max_warp=0.2, p_affine=0.75, p_lighting=0.75), 
                Normalize.from_stats(*imagenet_stats)]
    dls = ImageDataLoaders.from_folder(path, valid='val', item_tfms=item_tfms, batch_tfms=batch_tfms, bs=batch_size)  
    return dls
      
    
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--img_size', default=224, type=int,
                        help='Image size to which input images should be scaled.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch Size for training.')
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='Learning rate for training model.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Epoch number for training model.')
    parser.add_argument('--output', default="../my_saved_models/attention/", type=str,
                        help='Where to output all stuff')
    parser.add_argument('--num_heads', default=8, type=int,
                        help='Number of heads in attention layers.')
    parser.add_argument('--kernel_size', default=11, type=int,
                        help='Kernel size in attention layers.')
    parser.add_argument('--name', default="", type=str,
                        help='Add a name ending to saved model and log file.')
    args = parser.parse_args()
    
    print(args)
    global TIME
    TIME = str(datetime.now())[:-7].replace(" ", "_")
    print("Starting at " + TIME)

    ds = prepare_data(args.img_size, args.batch_size)
    print("Prepared Data")
    
    run_training(ds, args.lr, args.epochs, args.output, args.num_heads, args.kernel_size, args.img_size, args.name)

if __name__ == '__main__':
    main()