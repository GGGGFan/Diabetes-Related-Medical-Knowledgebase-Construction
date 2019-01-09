import sys
sys.path.insert(0, 'code/')
from process_data import *
from utils import *
from val import *
from bilstm_crf_model import *
import argparse
from keras.callbacks import ModelCheckpoint
from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(2)

if __name__ == '__main__':
    #Preprocessing
    data_makeup()
    print("############## Ready! ###############")
    get_train_data(data_dir='data/train/')
    
    parser = argparse.ArgumentParser(description='TRAIN')
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--embed', type=int, default=300)
    parser.add_argument('--units', type=int, default=300)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    #parser.add_argument('--save', type=str)
    parser.add_argument('--batch', type=int, default=64)
    
    args = parser.parse_args()
    gpu_config(args.gpu)
    
    # Build model
    model, (train_x, train_y), (test_x, test_y) = bilstm_crf_model.create_model(args.embed, args.units)
    
    # only get the best single model
    filepath = 'model.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    # Train
    model.fit(train_x, train_y, batch_size=args.batch, epochs=args.epoch, validation_data=[test_x, test_y], callbacks=[checkpoint])
    
    # Predict
    model_dir = 'model.h5'
    model, (word2idx, chunk_tags) = bilstm_crf_model.create_model(args.embed, args.units, train=False)
    model.load_weights(model_dir)
    
    test_dir = 'data/ruijin_round1_test_b_20181112/'
    submit_dir = 'submit/'
    test(test_dir, submit_dir, model, word2idx, chunk_tags)
