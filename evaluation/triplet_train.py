import os
import sys
import pickle
import librosa
import tempfile
import itertools
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from scipy.spatial.distance import cosine as cosine_distance


import triplet_network

SR = 16000
BATCH_SIZE = 16
DATA_DIR = '../'
MFCCT_DIR = './mfcc-t'
MFCCV_DIR = './mfcc-v'

def load_audio_mfcc(path):
    a, _ = librosa.load(path, sr=SR)
    return librosa.feature.mfcc(a, sr=SR, n_mfcc=13, hop_length=500)[1:].T


def main():
    model, backbone = triplet_network.build_model(num_features=12)
    
    ta, tb ,tc = triplet_network.all_data(os.path.join(DATA_DIR, 'triplets_train'), load_fn=load_audio_mfcc)
    t_size = ta.shape[0]
    train_triplet = {'anchor':ta, 'positive':tb, 'negative':tc}
    
    i_len = len(str(t_size - 1))
    print('here')
    with open(os.path.join(MFCCT_DIR, 'train_list'), 'w') as f_list:
        for i in tqdm(range(t_size)):
            paths = []
            for name in ['anchor', 'positive', 'negative']:
                example = train_triplet[name][i]
                path = '{}_{}.npy'.format(str(i).zfill(i_len), name)
                np.save(os.path.join(MFCCT_DIR, path), example, allow_pickle=False)
                paths.append(path)
            print(*paths, sep='\t', file=f_list)

    va, vb ,vc = triplet_network.all_data(os.path.join(DATA_DIR, 'triplets_val'), load_fn=load_audio_mfcc)
    v_size = va.shape[0]
    val_triplet = {'anchor':va, 'positive':vb, 'negative':vc}

    i_len = len(str(v_size - 1))
    print('there')
    with open(os.path.join(MFCCV_DIR, 'val_list'), 'w') as f_list:
        for i in tqdm(range(v_size)):
            paths = []
            for name in ['anchor', 'positive', 'negative']:
                example = val_triplet[name][i]
                path = '{}_{}.npy'.format(str(i).zfill(i_len), name)
                np.save(os.path.join(MFCCV_DIR, path), example, allow_pickle=False)
                paths.append(path)
            print(*paths, sep='\t', file=f_list) 

    del ta, tb, tc, va, vb, vc           
   

    train_loader, steps_per_epoch = triplet_network.data_loader(os.path.join(MFCCT_DIR, 'train_list'), load_fn=np.load, batch_size=BATCH_SIZE, shuffle=True, repeat=True)
    val_loader, val_dataset_size = triplet_network.data_loader(os.path.join(MFCCV_DIR, 'val_list'), load_fn=np.load, batch_size=1)
    val_data = list(val_loader())
    

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer, loss=triplet_network.triplet_hinge_loss)
    for i in range(10):
        pred = model.predict(iter(val_data)).reshape(-1, 2)
        print('Accuracy:', np.mean(pred.argmax(axis=1) == 0))
        print(pred[:4])

        model.fit(train_loader(),
            epochs=1,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
        )

    pred = model.predict(iter(val_data)).reshape(-1, 2)
    print('Accuracy:', np.mean(pred.argmax(axis=1) == 0))
    print(len(pred))
    print(pred.shape)

    model.save_weights('checkpoint.ckpt')

if __name__ == "__main__":
    main()