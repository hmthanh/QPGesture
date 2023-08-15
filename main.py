import numpy as np
import os
import scipy.io.wavfile as wavfile

import pdb

import numpy as np
import random
import os
import argparse
import numpy as np
# from data_processing import load_train_db, load_test_db, calc_data_stats
# from control import create_control_filter
# from utils import normalize_data, normalize_feat
from tqdm import tqdm
# from scipy import spatial
from sklearn.metrics.pairwise import paired_distances
# from visualization import generate_seq_videos
# from constant import UPPERBODY_PARENT, NUM_AUDIO_FEAT, NUM_BODY_FEAT, \
#     NUM_MFCC_FEAT, NUM_JOINTS, STEP_SZ, WAV_TEST_SIZE, num_frames_code, num_frames, codebook_size, NUM_AUDIO_FEAT_FRAMES
import Levenshtein
import torch

seed_value = 123456
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# **************************** Constant ****************************

SR = 16000
WAV_TEST_SIZE = 409600

UPPERBODY_PARENT = np.array([1, 11, 1, 2, 3, 1, 5, 6, 10, 10, 10, 10, 1, 13, 13, 14, 15, 13, 17, 18, 13, 20, \
                             21, 13, 23, 24, 13, 26, 27, 16, 19, 22, 25, 28, 34, 34, 35, 36, 34, 38, 39, 34, 41, 42, 34, 44, 45, 34, \
                             47, 48, 37, 40, 43, 46, 49])

PARWISE_ORDER = [10, 11, 1, 12, 0, 2, 3, 4, 5, 6, 7, 8, 9] + list(range(13, 61))

FILTER_SMOOTH_STD = 1.5

NUM_AUDIO_FEAT_FRAMES = 6  # 6 or 8

# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
BODY_FEAT_IDX = [0, 8, 9, 12, 13]  # 'RightArm': 8, 'RightForeArm': 9, 'LeftArm': 12, 'LeftForeArm': 13, 'Spine3': 3?, 'Spine': 0
# 'RightArm': 8, 'RightForeArm': 9, 'LeftArm': 12, 'LeftForeArm': 13, 'Spine3': 3?, 'Spine': 0

NUM_MFCC_FEAT = 13  # notice that 14th always be zero nearly
NUM_AUDIO_FEAT = 13 * 8  # 14*8
NUM_BODY_FEAT = 144 + 36  # 3*8*4 -> 9*4*4
NUM_JOINTS = 135  # 3*55 -> 9 * 15
STEP_SZ = 4  # 8 -> 30

FRAME_INTERVAL = 4  # 2, 4, 8

BATCH_SIZE = 100
LR = 1e-4
MAX_ITERS = 300000  # 300000 -> 30
BURNIN_ITER = 10000  # 10000 -> 1

WEIGHT_GEN = 1
WEIGHT_RECON = 0.1
LAMBDA_GP = 100
GEN_HOP = 5

num_frames = 240
num_frames_code = 30
codebook_size = 512
# **************************** Constant ****************************

# **************************** ARGS ****************************
args = argparse.Namespace(
    train_database="./data/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_txt.npz",
    train_codebook="./data/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_code.npz",
    train_wavlm="./data/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_WavLM.npz",
    train_wavvq="./data/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_WavVQ.npz",
    codebook_signature="./data/BEAT/BEAT_output_60fps_rotation/code.npz",
    test_data="./data/BEAT/speaker_10_state_0/speaker_10_state_0_test_240_txt.npz",
    test_wavlm="./data/BEAT/speaker_10_state_0/speaker_10_state_0_test_240_WavLM.npz",
    test_wavvq="./data/Example1/ZeroEGGS_cut/wavvq_240.npz",
    out_knn_filename="./codebook/Speech2GestureMatching/output/result.npz",
    out_video_path="./codebook/Speech2GestureMatching/output/output_video_folder/",
    desired_k=0,
    fake=False,
    out_fake_knn_filename="./codebook/Speech2GestureMatching/output/knn_pred.npz",
    max_frames=0
)


# **************************** ARGS ****************************

# **************************** Utils ****************************
def normalize_data(data, mean, std):
    return (data - mean) / (std + 1E-8)


def inv_normalize_data(data, mean, std):
    return data * std + mean


def generate_wavfile(audio, audio_path):
    def pad_audio(audio, num_audio_samples):
        if len(audio) > num_audio_samples:
            audio = audio[:num_audio_samples]
        elif len(audio) < num_audio_samples:
            audio = np.pad(audio, [0, num_audio_samples - len(audio)], mode='constant', constant_values=0)
        return audio

    if not (os.path.exists(audio_path)):
        os.makedirs(audio_path)

    for i in range(len(audio)):
        wavfile.write(audio_path + str(i) + '.wav', SR, pad_audio(audio[i], WAV_TEST_SIZE))


def normalize_feat(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std = np.clip(std, a_min=1E-8, a_max=None)
    norm_data = (data - mean) / std
    return norm_data, mean, std


# **************************** Utils ****************************

# **************************** Data processing ****************************
def load_train_db(data_file):
    print('read training dataset...')
    data = np.load(data_file)

    # mfcc shape: (num_seq, num_frames=64, num_feat=NUM_MFCC_FEAT)
    # motion shape: (num_seq, num_frames=64, num_feat=NUM_JOINTS)

    mfcc = data['mfcc'][:, :, :NUM_MFCC_FEAT]
    motion = data['body']  # (934, 64, 165)

    n_b, n_t = motion.shape[0], motion.shape[1]  # num_seq, 64
    n_mfcc_feat = NUM_MFCC_FEAT

    # motion = motion.reshape((n_b, n_t, 9, -1))  # (934, 64, 9, 55)
    # motion = motion.transpose((0, 1, 3, 2))  # (934, 64, 55, 9)

    motion = motion.reshape((n_b, n_t, -1, 9))

    # Thumb, Index, Middle, Ring, Pinky (https://ts1.cn.mm.bing.net/th/id/R-C.c2b42cc4bc071eff96d9ad041349d1af?rik=HYG%2fn%2bgJX1Pxgg&riu=http%3a%2f%2fastrogurukul.com%2fweb%2fwp-content%2fuploads%2f2013%2f02%2ffnger-names.jpg&ehk=D%2biw5zn33M%2f8xUEeWTLsMXJ7%2b8siGDki%2fLYIBspncCI%3d&risl=&pid=ImgRaw&r=0&sres=1&sresct=1)
    # [3, 4, 6, 7, 14, 20, 35, 41] -> right_elbow, right_wrist, left_elbow, left_wrist, right_index_0, right_little_0, left_index_0, left_little_0

    slc_body_kpts = np.take(motion, BODY_FEAT_IDX, axis=2)  # (934, 64, 4, 9)
    slc_body_kpts = slc_body_kpts.reshape((n_b, n_t, -1))  # (934, 64, 36)

    body_feat = np.zeros((n_b, n_t, NUM_BODY_FEAT_FRAMES, 9 * len(BODY_FEAT_IDX)))  # (934, 64, 4, 36)

    for i in range(NUM_BODY_FEAT_FRAMES):  # 4
        post_pad = np.zeros((n_b, i * FRAME_INTERVAL, 9 * len(BODY_FEAT_IDX)))  # (934, i * 2, 3*8)
        body_feat[:, :, i, :] = np.concatenate((slc_body_kpts[:, (i * FRAME_INTERVAL):], post_pad), axis=1)
        # [(934, 64, 3*8), (934, 0, 3*8)] [(934, 62, 3*8), (934, 2, 3*8)]

    body_feat = body_feat.reshape((n_b, n_t, -1))
    audio_feat = np.zeros((n_b, n_t, NUM_AUDIO_FEAT_FRAMES, n_mfcc_feat))

    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((n_b, i * FRAME_INTERVAL, n_mfcc_feat))
        audio_feat[:, :, i, :] = np.concatenate((mfcc[:, (i * FRAME_INTERVAL):, :n_mfcc_feat], post_pad), axis=1)

    audio_feat = audio_feat.reshape((n_b, n_t, -1))

    features = np.concatenate((audio_feat, body_feat), axis=2)

    motion = motion.reshape((n_b, n_t, -1))

    features = features.transpose((0, 2, 1))
    mfcc = mfcc.transpose((0, 2, 1))
    motion = motion.transpose((0, 2, 1))

    # features shape: (num_seq, num_feat=(NUM_AUDIO_FEAT+NUM_BODY_FEAT), num_frames=64)
    # mfcc shape: (num_seq, num_feat=NUM_MFCC_FEAT, num_frames=64)
    # motion shape: (num_seq, num_feat=NUM_JOINTS, num_frames=64)

    return features.astype(np.float32), mfcc.astype(np.float32), motion.astype(np.float32)


def load_test_db(data_file):
    # mfcc shape: (num_seq, num_frames=64, num_feat=NUM_MFCC_FEAT)
    # audio shape: (num_seq, num_feat=WAV_TEST_SIZE)
    print('read testing dataset...')
    data = np.load(data_file)
    mfcc = data['mfcc'][:, :, :NUM_MFCC_FEAT]
    # audio = data['wav']

    n_b, n_t = mfcc.shape[0], mfcc.shape[1]
    n_mfcc_feat = NUM_MFCC_FEAT

    audio_feat = np.zeros((n_b, n_t, NUM_AUDIO_FEAT_FRAMES, n_mfcc_feat))

    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((n_b, i * FRAME_INTERVAL, n_mfcc_feat))
        audio_feat[:, :, i, :] = np.concatenate((mfcc[:, (i * FRAME_INTERVAL):, :n_mfcc_feat], post_pad), axis=1)

    audio_feat = audio_feat.reshape((n_b, n_t, -1))
    features = audio_feat.transpose((0, 2, 1))

    # mfcc = mfcc.reshape(-1, num_frames, mfcc.shape[-1])     # 20221012

    mfcc = mfcc.transpose((0, 2, 1))

    # features shape: (num_seq, num_feat=(NUM_AUDIO_FEAT), num_frames=64)
    # mfcc shape: (num_seq, num_feat=NUM_MFCC_FEAT, num_frames=64)
    # audio shape: (num_seq, num_feat=WAV_TEST_SIZE)

    return features.astype(np.float32), mfcc.astype(np.float32), None


def prep_train_resync_data(train_path):
    print('read training resync dataset...')
    training_data = np.load(train_path)  # 'body', 'face', 'wav', 'imgs'

    # mfcc_train shape          : (num_seq, num_frame=64, num_feat=NUM_MFCC_FEAT)
    # gesture_knn_train shape   : (num_seq, num_frame=64, num_feat=NUM_JOINTS)
    # gesture_real_train shape  : (num_seq, num_frame=64, num_feat=NUM_JOINTS)

    mfcc_train, gesture_real_train = training_data['mfcc'][:, :, :NUM_MFCC_FEAT], training_data['body']  # ?, ?, (4748, 64, 165)

    gesture_knn_train = gesture_real_train.copy().reshape(-1, gesture_real_train.shape[-1])
    np.random.shuffle(gesture_knn_train)
    gesture_knn_train = gesture_knn_train.reshape(gesture_real_train.shape[0], gesture_real_train.shape[1], gesture_real_train.shape[2])

    # mfcc_train = np.random.rand(2, 112 + 96, 14)
    # gesture_knn_train = np.random.rand(2, 112 + 96, 165)
    # gesture_real_train = training_data['body']

    # mfcc_train shape          : (num_seq, num_feat=NUM_MFCC_FEAT, num_frame=64)
    # gesture_knn_train shape   : (num_seq, num_feat=NUM_JOINTS, num_frame=64)
    # gesture_real_train shape  : (num_seq, num_feat=NUM_JOINTS, num_frame=64)

    mfcc_train = mfcc_train.transpose((0, 2, 1))
    gesture_knn_train = gesture_knn_train.transpose((0, 2, 1))
    gesture_real_train = gesture_real_train.transpose((0, 2, 1))

    # gesture_knn_train shape: (num_seq, num_feat, num_frame)
    # gesture_real_train shape: (num_seq, num_feat, num_frame)

    # mfcc_mean shape: (1, num_feat=NUM_MFCC_FEAT, 1)
    # mfcc_std shape: (1, num_feat=NUM_MFCC_FEAT, 1)

    # gesture_knn_mean shape: (1, num_feat=NUM_JOINTS, 1)
    # gesture_knn_std shape: (1, num_feat=NUM_JOINTS, 1)

    mfcc_mean, mfcc_std, gesture_knn_mean, gesture_knn_std = calc_data_stats(
        mfcc_train.transpose((0, 2, 1)),
        gesture_knn_train.transpose((0, 2, 1))
    )

    # gesture_real_mean shape: (1, num_feat=NUM_JOINTS, 1)
    # gesture_real_std shape: (1, num_feat=NUM_JOINTS, 1)

    _, _, gesture_real_mean, gesture_real_std = calc_data_stats(
        mfcc_train.transpose((0, 2, 1)),
        gesture_real_train.transpose((0, 2, 1))
    )

    # shuffle the sequences
    train_len = mfcc_train.shape[0]
    rand_idx = np.arange(train_len)
    np.random.shuffle(rand_idx)

    mfcc_train = mfcc_train[rand_idx]
    gesture_knn_train = gesture_knn_train[rand_idx]
    gesture_real_train = gesture_real_train[rand_idx]

    # normalize
    mfcc_train = normalize_data(mfcc_train, mfcc_mean, mfcc_std)

    motion_knn_train = normalize_data(gesture_knn_train, gesture_knn_mean, gesture_knn_std)
    x_knn_train = np.concatenate((mfcc_train, motion_knn_train), axis=1)
    motion_real_train = normalize_data(gesture_real_train, gesture_real_mean, gesture_real_std)
    x_real_train = np.concatenate((mfcc_train, motion_real_train), axis=1)

    x_knn_train = x_knn_train[0::FRAME_INTERVAL]
    x_real_train = x_real_train[0::FRAME_INTERVAL]

    # x_knn_train shape: (num_seq, num_feat=(NUM_MFCC_FEAT+NUM_BODY_FEAT), num_frames=64)
    # x_real_train shape: (num_seq, num_feat=(NUM_MFCC_FEAT+NUM_BODY_FEAT), num_frames=64)

    return torch.tensor(x_knn_train), torch.tensor(x_real_train)


def calc_data_stats(feat1, feat2=None):
    feat1_mean = np.expand_dims(feat1.mean(axis=(1, 0)), axis=(0, -1))
    feat1_std = np.expand_dims(feat1.std(axis=(1, 0)), axis=(0, -1))

    if feat2 is None:
        return feat1_mean, feat1_std

    else:
        feat2_mean = np.expand_dims(feat2.mean(axis=(1, 0)), axis=(0, -1))
        feat2_std = np.expand_dims(feat2.std(axis=(1, 0)), axis=(0, -1))
        return feat1_mean, feat1_std, feat2_mean, feat2_std


def convert_abswise_to_parwise(motion):  # (num_seq, num_feat=NUM_JOINTS, num_frames=64)
    motion = motion.transpose((0, 2, 1))  # -> (num_seq, 64, NUM_JOINTS)
    N, T, _ = motion.shape

    motion = motion.reshape((N, T, -1, 3))  # -> (num_seq, 64, NUM_JOINTS//3, 3)
    motion = motion - motion[:, :, UPPERBODY_PARENT]

    motion = motion.reshape((N, T, -1))  # -> (num_seq, 64, NUM_JOINTS)

    return motion.transpose((0, 2, 1))  # -> (num_seq, NUM_JOINTS, 64)


def load_db_codebook(data_file, codepath, test_data_path, train_wavlm, test_wavlm, train_wavvq, test_wavvq):
    import torch.nn.functional as F
    print('read training dataset...')
    data = np.load(data_file)
    mfcc = data['mfcc'][:, :, :NUM_MFCC_FEAT]
    energy = np.expand_dims(data['energy'], axis=-1)  # (n, 240, 1)
    pitch = np.expand_dims(data['pitch'], axis=-1)
    volume = np.expand_dims(data['volume'], axis=-1)
    speech_features = np.concatenate((energy, pitch, volume), axis=2)
    code = np.load(codepath)['code']

    audio_feat = np.zeros((mfcc.shape[0], mfcc.shape[1], NUM_AUDIO_FEAT_FRAMES, NUM_MFCC_FEAT))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((mfcc.shape[0], i * FRAME_INTERVAL, NUM_MFCC_FEAT))
        audio_feat[:, :, i, :] = np.concatenate((mfcc[:, (i * FRAME_INTERVAL):], post_pad), axis=1)
    train_feat = audio_feat.reshape((mfcc.shape[0], mfcc.shape[1], -1))

    speech_features_feat = np.zeros((speech_features.shape[0], speech_features.shape[1], NUM_AUDIO_FEAT_FRAMES, 3))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((speech_features.shape[0], i * FRAME_INTERVAL, 3))
        speech_features_feat[:, :, i, :] = np.concatenate((speech_features[:, (i * FRAME_INTERVAL):], post_pad), axis=1)
    train_speech_features_feat = speech_features_feat.reshape((speech_features.shape[0], speech_features.shape[1], -1))

    print('mfcc shape', mfcc.shape)
    print('code shape', code.shape)
    print('train_feat shape', train_feat.shape)
    print('speech_features shape', speech_features.shape)
    print('train speech features feat shape', train_speech_features_feat.shape)

    print('\nread testing dataset...')
    test_data = np.load(test_data_path)
    test_mfcc = test_data['mfcc'][:, :, :NUM_MFCC_FEAT]
    test_energy = np.expand_dims(test_data['energy'], axis=-1)
    test_pitch = np.expand_dims(test_data['pitch'], axis=-1)
    test_volume = np.expand_dims(test_data['volume'], axis=-1)
    test_speech_features = np.concatenate((test_energy, test_pitch, test_volume), axis=2)

    test_audio_feat = np.zeros((test_mfcc.shape[0], test_mfcc.shape[1], NUM_AUDIO_FEAT_FRAMES, NUM_MFCC_FEAT))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((test_mfcc.shape[0], i * FRAME_INTERVAL, NUM_MFCC_FEAT))
        test_audio_feat[:, :, i, :] = np.concatenate((test_mfcc[:, (i * FRAME_INTERVAL):], post_pad), axis=1)
    test_feat = test_audio_feat.reshape((test_mfcc.shape[0], test_mfcc.shape[1], -1))

    speech_features_feat = np.zeros((test_speech_features.shape[0], test_speech_features.shape[1], NUM_AUDIO_FEAT_FRAMES, 3))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((test_speech_features.shape[0], i * FRAME_INTERVAL, 3))
        speech_features_feat[:, :, i, :] = np.concatenate((test_speech_features[:, (i * FRAME_INTERVAL):], post_pad), axis=1)
    test_speech_features_feat = speech_features_feat.reshape((test_speech_features.shape[0], test_speech_features.shape[1], -1))

    print('test_mfcc shape: ', test_mfcc.shape)
    print('test_feat shape: ', test_feat.shape)
    print('test_speech_features shape: ', test_speech_features.shape)
    print('test speech features feat shape: ', test_speech_features_feat.shape)

    # debugs
    # train_wavlm = np.zeros((2, 2, 2))
    # test_wavlm = np.zeros((2, 2, 2))
    # train_wavlm_feat = np.zeros((2, 2, 2))
    # test_wavlm_feat = np.zeros((2, 2, 2))
    # train_wavlm_interpolate = np.zeros((2, 2, 2))
    # test_wavlm_interpolate = np.zeros((2, 2, 2))

    train_wavlm = np.load(train_wavlm)['wavlm']
    test_wavlm = np.load(test_wavlm)['wavlm']

    nums_wavlm_frames = train_wavlm.shape[1]
    new_wavlm_frames = nums_wavlm_frames // code.shape[-1] * code.shape[-1]
    train_wavlm_interpolate = F.interpolate(torch.from_numpy(train_wavlm).transpose(1, 2), size=new_wavlm_frames, align_corners=True, mode='linear').transpose(1, 2).numpy()
    test_wavlm_interpolate = F.interpolate(torch.from_numpy(test_wavlm).transpose(1, 2), size=new_wavlm_frames, align_corners=True, mode='linear').transpose(1, 2).numpy()
    print("train_wavlm.shape: ", train_wavlm.shape)
    print("test_wavlm.shape: ", test_wavlm.shape)
    print("train_wavlm_interpolate.shape: ", train_wavlm_interpolate.shape)
    print("test_wavlm_interpolate.shape: ", test_wavlm_interpolate.shape)

    train_wavlm_feat = np.zeros((train_wavlm_interpolate.shape[0], train_wavlm_interpolate.shape[1], NUM_AUDIO_FEAT_FRAMES, train_wavlm_interpolate.shape[-1]))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((train_wavlm_interpolate.shape[0], i * (FRAME_INTERVAL - 2), train_wavlm_interpolate.shape[-1]))
        train_wavlm_feat[:, :, i, :] = np.concatenate((train_wavlm_interpolate[:, (i * (FRAME_INTERVAL - 2)):], post_pad), axis=1)
    train_wavlm_feat = train_wavlm_feat.reshape((train_wavlm_interpolate.shape[0], train_wavlm_interpolate.shape[1], -1))

    test_wavlm_feat = np.zeros((test_wavlm_interpolate.shape[0], test_wavlm_interpolate.shape[1], NUM_AUDIO_FEAT_FRAMES, test_wavlm_interpolate.shape[-1]))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((test_wavlm_interpolate.shape[0], i * (FRAME_INTERVAL - 2), test_wavlm_interpolate.shape[-1]))
        test_wavlm_feat[:, :, i, :] = np.concatenate((test_wavlm_interpolate[:, (i * (FRAME_INTERVAL - 2)):], post_pad), axis=1)
    test_wavlm_feat = test_wavlm_feat.reshape((test_wavlm_interpolate.shape[0], test_wavlm_interpolate.shape[1], -1))

    print('train_wavlm_feat shape: ', train_wavlm_feat.shape)
    print('test_wavlm_feat shape: ', test_wavlm_feat.shape)

    train_wavvq = np.load(train_wavvq)['wavvq']
    test_wavvq = np.load(test_wavvq)['wavvq']

    FRAME_INTERVAL_vq = train_wavvq.shape[1] / num_frames_code

    '''
    train_wavvq_feat = np.zeros((train_wavvq.shape[0], train_wavvq.shape[1], NUM_AUDIO_FEAT_FRAMES, train_wavvq.shape[-1]))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((train_wavvq.shape[0], int(i * FRAME_INTERVAL_vq), train_wavvq.shape[-1]))
        train_wavvq_feat[:, :, i, :] = np.concatenate((train_wavvq[:, int(i * FRAME_INTERVAL_vq):], post_pad), axis=1)
    train_wavvq_feat = train_wavvq_feat.reshape((train_wavvq.shape[0], train_wavvq.shape[1], -1))

    test_wavvq_feat = np.zeros((test_wavvq.shape[0], test_wavvq.shape[1], NUM_AUDIO_FEAT_FRAMES, test_wavvq.shape[-1]))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((test_wavvq.shape[0], int(i * FRAME_INTERVAL_vq), test_wavvq.shape[-1]))
        test_wavvq_feat[:, :, i, :] = np.concatenate((test_wavvq[:, int(i * FRAME_INTERVAL_vq):], post_pad), axis=1)
    test_wavvq_feat = test_wavvq_feat.reshape((test_wavvq.shape[0], test_wavvq.shape[1], -1))
    '''
    # 20221101
    train_wavvq_feat = np.zeros((train_wavvq.shape[0], train_wavvq.shape[1], NUM_AUDIO_FEAT_FRAMES, train_wavvq.shape[-1]))
    for i in range(0, NUM_AUDIO_FEAT_FRAMES):
        pre_pad_len = int((NUM_AUDIO_FEAT_FRAMES - i - 1) * FRAME_INTERVAL_vq)
        post_pad_len = int(i * FRAME_INTERVAL_vq)
        pre_pad = np.zeros((train_wavvq.shape[0], pre_pad_len, train_wavvq.shape[-1]))
        post_pad = np.zeros((train_wavvq.shape[0], post_pad_len, train_wavvq.shape[-1]))
        train_wavvq_feat[:, :, i, :] = np.concatenate((pre_pad, train_wavvq[:, :(train_wavvq.shape[1] - pre_pad_len)]), axis=1)

    train_wavvq_feat1 = train_wavvq_feat.reshape((train_wavvq.shape[0], train_wavvq.shape[1], -1))

    train_wavvq_feat = np.zeros((train_wavvq.shape[0], train_wavvq.shape[1], NUM_AUDIO_FEAT_FRAMES, train_wavvq.shape[-1]))
    for i in range(0, NUM_AUDIO_FEAT_FRAMES):
        pre_pad_len = int((NUM_AUDIO_FEAT_FRAMES - i - 1) * FRAME_INTERVAL_vq)
        post_pad_len = int(i * FRAME_INTERVAL_vq)
        pre_pad = np.zeros((train_wavvq.shape[0], pre_pad_len, train_wavvq.shape[-1]))
        post_pad = np.zeros((train_wavvq.shape[0], post_pad_len, train_wavvq.shape[-1]))
        train_wavvq_feat[:, :, i, :] = np.concatenate((train_wavvq[:, post_pad_len:], post_pad), axis=1)
    train_wavvq_feat = np.delete(train_wavvq_feat, 0, axis=2)
    train_wavvq_feat2 = train_wavvq_feat.reshape((train_wavvq.shape[0], train_wavvq.shape[1], -1))

    train_wavvq_feat = np.concatenate((train_wavvq_feat1, train_wavvq_feat2), axis=-1)

    test_wavvq_feat = np.zeros((test_wavvq.shape[0], test_wavvq.shape[1], NUM_AUDIO_FEAT_FRAMES, test_wavvq.shape[-1]))
    for i in range(0, NUM_AUDIO_FEAT_FRAMES):
        pre_pad_len = int((NUM_AUDIO_FEAT_FRAMES - i - 1) * FRAME_INTERVAL_vq)
        pre_pad = np.zeros((test_wavvq.shape[0], pre_pad_len, test_wavvq.shape[-1]))
        test_wavvq_feat[:, :, i, :] = np.concatenate((pre_pad, test_wavvq[:, :(test_wavvq.shape[1] - pre_pad_len)]), axis=1)

    test_wavvq_feat1 = test_wavvq_feat.reshape((test_wavvq.shape[0], test_wavvq.shape[1], -1))

    test_wavvq_feat = np.zeros((test_wavvq.shape[0], test_wavvq.shape[1], NUM_AUDIO_FEAT_FRAMES, test_wavvq.shape[-1]))
    for i in range(0, NUM_AUDIO_FEAT_FRAMES):
        post_pad_len = int(i * FRAME_INTERVAL_vq)
        post_pad = np.zeros((test_wavvq.shape[0], post_pad_len, test_wavvq.shape[-1]))
        test_wavvq_feat[:, :, i, :] = np.concatenate((test_wavvq[:, post_pad_len:], post_pad), axis=1)
    test_wavvq_feat = np.delete(test_wavvq_feat, 0, axis=2)
    test_wavvq_feat2 = test_wavvq_feat.reshape((test_wavvq.shape[0], test_wavvq.shape[1], -1))

    test_wavvq_feat = np.concatenate((test_wavvq_feat1, test_wavvq_feat2), axis=-1)

    print('train_wavvq_feat shape: ', train_wavvq_feat.shape)
    print('test_wavvq_feat shape: ', test_wavvq_feat.shape)

    train_phase = np.load(data_file, allow_pickle=True)['phase']  # n, len, 4 (1 * 8 * 1)
    test_phase = np.load(test_data_path, allow_pickle=True)['phase']

    train_context = np.load(data_file)['context'].squeeze(2)  # n, len, 384
    test_context = np.load(test_data_path)['context'].squeeze(2)  # n, len, 384

    return mfcc.transpose((0, 2, 1)), code, test_mfcc.transpose((0, 2, 1)), \
        train_feat.transpose((0, 2, 1)), test_feat.transpose((0, 2, 1)), \
        train_wavlm_interpolate.transpose((0, 2, 1)), test_wavlm_interpolate.transpose((0, 2, 1)), \
        train_wavlm_feat.transpose((0, 2, 1)), test_wavlm_feat.transpose((0, 2, 1)), \
        speech_features.transpose((0, 2, 1)), test_speech_features.transpose((0, 2, 1)), \
        train_speech_features_feat.transpose((0, 2, 1)), test_speech_features_feat.transpose((0, 2, 1)), \
        train_wavvq_feat.transpose((0, 2, 1)), test_wavvq_feat.transpose((0, 2, 1)), \
        train_phase.transpose((0, 2, 1)), test_phase.transpose((0, 2, 1)), \
        train_context.transpose((0, 2, 1)), test_context.transpose((0, 2, 1))


# **************************** Data processing ****************************


# **************************** load_db_codebook ****************************
def create_control_filter(feat_train, control_type=None, n_aud_feat=NUM_AUDIO_FEAT):
    num_seq, _, num_frm = feat_train.shape
    control_mask = np.ones((num_seq, num_frm))
    num_joint_feat = len(BODY_FEAT_IDX)

    if control_type is None:
        return control_mask

    body_feat = feat_train.transpose((0, 2, 1))
    body_feat = body_feat[:, :, n_aud_feat:]
    body_feat = body_feat.reshape((num_seq, num_frm, NUM_BODY_FEAT_FRAMES, -1))
    body_feat = body_feat.reshape((num_seq, num_frm, NUM_BODY_FEAT_FRAMES, num_joint_feat, 3))

    # the index of the left wrist joint in BODY_FEAT_IDX is 3
    lwrist_height = body_feat[:, :, 0, 3, 1]

    height_list = []
    for i in range(feat_train.shape[0]):
        for j in range(0, feat_train.shape[2]):
            height_list.append(lwrist_height[i, j] * -1)

    quantile_high = np.quantile(height_list, 0.85)
    quantile_low = np.quantile(height_list, 0.15)

    if control_type == "hand_high":
        for i in range(num_seq):
            for j in range(num_frm):
                # mask out frames where hand height is below threshold
                # the y-axis is multiplied by -1 to flip it upward
                if lwrist_height[i, j] * -1 < quantile_high:
                    control_mask[i, j] = 0
    elif control_type == "hand_low":
        for i in range(num_seq):
            for j in range(num_frm):
                # mask out frames where hand height is above threshold
                # the y-axis is multiplied by -1 to flip it upward
                if lwrist_height[i, j] * -1 > quantile_low:
                    control_mask[i, j] = 0
    else:
        control_mask = np.ones((num_seq, num_frm))

    return control_mask


# **************************** load_db_codebook ****************************


# **************************** create_control_filter ****************************
def wavvq_distances(ls1, ls2, mode='sum'):
    if mode == 'sum':
        def ls2str(ls):
            ls = ls.reshape(NUM_AUDIO_FEAT_FRAMES, -1).transpose()  # (NUM_AUDIO_FEAT_FRAMES, groups=2)
            str1 = ''.join([chr(int(i)) for i in ls[0]])
            str2 = ''.join([chr(int(i)) for i in ls[1]])
            return str1, str2

        ls1_group1_str, ls1_group2_str = ls2str(ls1)
        ls2_group1_str, ls2_group2_str = ls2str(ls2)

        return Levenshtein.distance(ls1_group1_str, ls2_group1_str) + Levenshtein.distance(ls1_group2_str, ls2_group2_str)

    elif mode == 'combine':
        def ls2str(ls):
            ls = ls.reshape(-1, 2).transpose()  # (NUM_AUDIO_FEAT_FRAMES * 2, groups=2)
            ls = ls[0] * 320 + ls[1]
            str = ''.join([chr(int(i)) for i in ls])
            return str

        ls1_str = ls2str(ls1)
        ls2_str = ls2str(ls2)

        return Levenshtein.distance(ls1_str, ls2_str)


# **************************** create_control_filter ****************************

# **************************** wavvq_distances ****************************
def wavvq_distances(ls1, ls2, mode='sum'):
    if mode == 'sum':
        def ls2str(ls):
            ls = ls.reshape(NUM_AUDIO_FEAT_FRAMES, -1).transpose()  # (NUM_AUDIO_FEAT_FRAMES, groups=2)
            str1 = ''.join([chr(int(i)) for i in ls[0]])
            str2 = ''.join([chr(int(i)) for i in ls[1]])
            return str1, str2

        ls1_group1_str, ls1_group2_str = ls2str(ls1)
        ls2_group1_str, ls2_group2_str = ls2str(ls2)

        return Levenshtein.distance(ls1_group1_str, ls2_group1_str) + Levenshtein.distance(ls1_group2_str, ls2_group2_str)

    elif mode == 'combine':
        def ls2str(ls):
            ls = ls.reshape(-1, 2).transpose()  # (NUM_AUDIO_FEAT_FRAMES * 2, groups=2)
            ls = ls[0] * 320 + ls[1]
            str = ''.join([chr(int(i)) for i in ls])
            return str

        ls1_str = ls2str(ls1)
        ls2_str = ls2str(ls2)

        return Levenshtein.distance(ls1_str, ls2_str)


# **************************** wavvq_distances ****************************


# **************************** GestureKNN ****************************
class GestureKNN(object):
    def __init__(self, feat_train, motn_train, control_mask, n_aud_feat=112, n_body_feat=96, n_joints=165, step_sz=8):
        super(GestureKNN, self).__init__()

        # feat_train shape    : (num_seq, num_frames, (n_aud_feat + n_body_feat))
        # control_mask shape  : (num_seq, num_frames)
        # motn_train shape    : (num_seq, num_frames, n_joints)

        self.n_aud_feat = n_aud_feat
        self.n_body_feat = n_body_feat
        self.n_joints = n_joints
        self.step_sz = step_sz

        self.feat_train = feat_train
        self.motn_train = motn_train

        self.control_mask = control_mask
        self.n_db_seq = feat_train.shape[0]
        self.n_db_frm = feat_train.shape[1]

    def init_frame(self):
        init_seq = np.random.randint(0, self.n_db_seq)
        init_frm = np.random.randint(0, self.n_db_frm)

        while self.control_mask[init_seq, init_frm] != 1:
            init_seq = np.random.randint(0, self.n_db_seq)
            init_frm = np.random.randint(0, self.n_db_frm)

        return init_seq, init_frm

    def search_motion(self, feat_test, desired_k):
        # feat_test shape    : (self.n_aud_feat, num_frames)), (112, 64)

        n_frames = feat_test.shape[-1]  # 64
        feat_test = np.concatenate((feat_test[:, 0:1], feat_test), axis=1)  # (112, 1+64)
        pose_feat = np.zeros((self.n_body_feat, feat_test.shape[1]))  # (96， 1+64)
        feat_test = np.concatenate((feat_test, pose_feat), axis=0)  # (96+112, 1+64)

        # initialize pose feature
        init_seq, init_frm = self.init_frame()  # 21, 147
        feat_test[self.n_aud_feat:, 0] = self.feat_train[init_seq, init_frm, self.n_aud_feat:]  # (96, )
        pred_motion = np.zeros((self.n_joints, n_frames + 1))  # (165, 1+64)

        # start from j = 1 (j = 0 is just a duplicate of j = 1 so that we can initialize pose feature)
        j = 1
        while j < n_frames:
            pos_dist_cands, pose_cands, feat_cands = self.search_pose_cands(feat_test[self.n_aud_feat:, j - 1])
            # search_pose_cands( (96,) ) -> (num_seq,) (num_seq, 165, 8) (num_seq, 208, 8)

            n_retained = pos_dist_cands.shape[0]  # (num_seq, )

            # compute distance between audio pose feature and the pre-selected feature candidates
            audio_test_feat = feat_test[:self.n_aud_feat, j]  # (112, )

            aud_dist_cands = []
            for k in range(n_retained):
                # audio_sim_score = spatial.distance.cosine(audio_test_feat, feat_cands[k, :self.n_aud_feat, 0])
                # This library is not precise enough, if the input is the same two 1D matrices, the output is a number of order 1e-8 instead of 0, which will lead to incorrect sorting from smallest to largest.

                audio_sim_score = paired_distances([audio_test_feat], [feat_cands[k, :self.n_aud_feat, 0]], metric='cosine')[0]
                aud_dist_cands.append(audio_sim_score)

            # len(aud_dist_cands) = num_seq
            pos_score = np.array(pos_dist_cands).argsort().argsort()
            aud_score = np.array(aud_dist_cands).argsort().argsort()

            combined_score = pos_score + aud_score
            combined_sorted_idx = np.argsort(combined_score).tolist()  # len=num_seq

            feat_cands = feat_cands[combined_sorted_idx]  # (num_seq, 208, 8)
            pose_cands = pose_cands[combined_sorted_idx]  # (num_seq, 165, 8)

            feat_test[self.n_aud_feat:, j:(j + self.step_sz)] = feat_cands[desired_k, self.n_aud_feat:, :self.step_sz]  # (96, 8)
            pred_motion[:, j:(j + self.step_sz)] = pose_cands[desired_k, :, :self.step_sz]  # (165, 8)

            j += self.step_sz

        # pred_motion shape    : (self.n_joints, num_frames))
        return pred_motion[:, 1:]

    def search_pose_cands(self, body_test_feat):
        pos_dist_cands = []
        pose_cands = []
        feat_cands = []

        for k in range(self.feat_train.shape[0]):  # num_seq
            if self.control_mask[k].sum() == 0:
                continue

            body_dist_list = []
            body_train_feat = self.feat_train[k, :, self.n_aud_feat:]  # (num_seq, 64, 112+96) -> (64, 96)

            for l in range(body_train_feat.shape[0]):  # num_frames
                body_dist = np.linalg.norm(body_test_feat - body_train_feat[l])  # for every frame
                body_dist_list.append(body_dist)

            sorted_idx_list = np.argsort(body_dist_list)

            pose_cand_ctr = 0
            pose_cand_found = False

            while pose_cand_ctr < len(sorted_idx_list) - 1:  # for every frame
                f = sorted_idx_list[pose_cand_ctr]  # index
                d = body_dist_list[f]  # distance

                pose_cand_ctr += 1

                # skip the same sequence
                if d == 0.:
                    continue

                # skip frames with padded features
                if f > self.n_db_frm - self.step_sz:  # num_frames-8
                    continue

                # check if control condition is satisfied, self.control_mask: default ones like (num_seq, num_frames)
                if (self.control_mask[k, f] + self.control_mask[k, f + self.step_sz - 1]) != 2:
                    continue
                else:
                    pose_cand_found = True
                    break

            if pose_cand_found == False:
                continue

            # feat_cand shape: (num_feat_dim, step_sz)
            # pose_cand shape: (num_feat_dim, step_sz)
            feat_cand = self.feat_train[k, f:(f + self.step_sz), :].transpose()  # (8, 112+96).transpose()
            pose_cand = self.motn_train[k, f:(f + self.step_sz), :].transpose()  # (8, 165).transpose()

            pos_dist_cands.append(d)
            pose_cands.append(pose_cand)
            feat_cands.append(feat_cand)

        pos_dist_cands = np.array(pos_dist_cands)
        pose_cands = np.array(pose_cands)
        feat_cands = np.array(feat_cands)

        return pos_dist_cands, pose_cands, feat_cands

    def search_fake_motion(self, feat_test, desired_k):
        # feat_test shape    : (self.n_aud_feat, num_frames)), (112, 64)

        n_frames = feat_test.shape[-1]  # 64
        pose_feat = np.zeros((self.n_body_feat, feat_test.shape[1]))  # (96， 1+64)
        feat_test = np.concatenate((feat_test, pose_feat), axis=0)  # (96+112, 1+64)

        # initialize pose feature
        pred_motion = np.zeros((self.n_joints, n_frames))  # (165, 1+64)

        # start from j = 1 (j = 0 is just a duplicate of j = 1 so that we can initialize pose feature)
        j = 0
        while j < n_frames:
            pos_dist_cands, pose_cands = self.search_fake_pose_cands(feat_test[:self.n_aud_feat, j])  # 20221010

            pos_score = np.array(pos_dist_cands).argsort().argsort()

            combined_sorted_idx = np.argsort(pos_score).tolist()  # len=num_seq

            pose_cands = pose_cands[combined_sorted_idx]  # (num_seq, 165, 8)

            pred_motion[:, j:(j + self.step_sz)] = pose_cands[desired_k, :, :self.step_sz]  # (165, 8)

            j += self.step_sz

        # pred_motion shape    : (self.n_joints, num_frames))
        return pred_motion

    def search_fake_pose_cands(self, body_test_feat):
        pos_dist_cands = []
        pose_cands = []

        for k in range(self.feat_train.shape[0]):  # num_seq
            if self.control_mask[k].sum() == 0:
                continue

            body_dist_list = []
            body_train_feat = self.feat_train[k, :, :self.n_aud_feat]  # (num_seq, 64, 112+96) -> (64, 96)

            for l in range(body_train_feat.shape[0]):  # num_frames
                body_dist = paired_distances([body_test_feat], [body_train_feat[l]], metric='cosine')[0]
                body_dist_list.append(body_dist)

            sorted_idx_list = np.argsort(body_dist_list)

            pose_cand_ctr = 0
            pose_cand_found = False

            while pose_cand_ctr < len(sorted_idx_list) - 1:  # for every frame
                f = sorted_idx_list[pose_cand_ctr]  # index
                d = body_dist_list[f]  # distance

                pose_cand_ctr += 1

                # skip the same sequence
                if d == 0.:
                    continue

                # skip frames with padded features
                if f > self.n_db_frm - self.step_sz:  # num_frames-8
                    continue

                # check if control condition is satisfied, self.control_mask: default ones like (num_seq, num_frames)
                if (self.control_mask[k, f] + self.control_mask[k, f + self.step_sz - 1]) != 2:
                    continue
                else:
                    pose_cand_found = True
                    break

            if pose_cand_found == False:
                continue

            # pose_cand shape: (num_feat_dim, step_sz)
            pose_cand = self.motn_train[k, f:(f + self.step_sz), :].transpose()  # (8, 165).transpose()

            pos_dist_cands.append(d)
            pose_cands.append(pose_cand)

        pos_dist_cands = np.array(pos_dist_cands)
        pose_cands = np.array(pose_cands)

        return pos_dist_cands, pose_cands


# **************************** GestureKNN ****************************


# **************************** predict_gesture_from_audio ****************************
def predict_gesture_from_audio(feat_train, pose_train, feat_test, control_mask, data_stats, \
                               k=0, n_aud_feat=112, n_body_feat=96, n_joints=165, step_sz=8, frames=0):
    # feat_train shape: (num_seq, num_feat=(NUM_AUDIO_FEAT+NUM_BODY_FEAT), num_frames)
    # pose_train shape: (num_seq, num_feat=NUM_JOINTS, num_frames)
    # feat_test shape: (num_seq, num_feat=NUM_AUDIO_FEAT, num_frames)
    # control_mask shape: (num_seq, num_frames)

    feat_mean = data_stats['feat_mean']
    feat_std = data_stats['feat_std']

    aud_mean_test = feat_mean[:, :n_aud_feat]
    aud_std_test = feat_std[:, :n_aud_feat]

    norm_feat_test = normalize_data(feat_test, aud_mean_test, aud_std_test)
    norm_feat_train = normalize_data(feat_train, feat_mean, feat_std)
    norm_feat_train = norm_feat_train.transpose((0, 2, 1))
    pose_train = pose_train.transpose((0, 2, 1))

    n_test_seq = frames if frames != 0 else feat_test.shape[0]
    print('init knn...')
    gesture_knn = GestureKNN(feat_train=norm_feat_train,
                             motn_train=pose_train,
                             control_mask=control_mask,
                             n_aud_feat=n_aud_feat,
                             n_body_feat=n_body_feat,
                             n_joints=n_joints,
                             step_sz=step_sz)

    motion_output = []

    print('begin search...')
    desired_k = np.random.choice(15, n_test_seq, p=[0.5, 0.5 / 14, 0.5 / 14, 0.5 / 14, 0.5 / 14, 0.5 / 14, 0.5 / 14, 0.5 / 14, 0.5 / 14,
                                                    0.5 / 14, 0.5 / 14, 0.5 / 14, 0.5 / 14, 0.5 / 14, 0.5 / 14])
    for i in tqdm(range(0, n_test_seq)):
        # pred_motion shape    : (NUM_JOINTS, num_frames))
        if args.fake:
            pred_motion = gesture_knn.search_fake_motion(feat_test=norm_feat_test[i], desired_k=desired_k[i])
        else:
            pred_motion = gesture_knn.search_motion(feat_test=norm_feat_test[i], desired_k=k)
        motion_output.append(pred_motion)

    # motion_output shape    : (num_seqs, num_feat=NUM_JOINTS, num_frames))
    return np.array(motion_output)


# **************************** predict_gesture_from_audio ****************************

# **************************** wavvq_distances ****************************
def wavvq_distances(ls1, ls2, mode='sum'):
    if mode == 'sum':
        def ls2str(ls):
            ls = ls.reshape(NUM_AUDIO_FEAT_FRAMES, -1).transpose()  # (NUM_AUDIO_FEAT_FRAMES, groups=2)
            str1 = ''.join([chr(int(i)) for i in ls[0]])
            str2 = ''.join([chr(int(i)) for i in ls[1]])
            return str1, str2

        ls1_group1_str, ls1_group2_str = ls2str(ls1)
        ls2_group1_str, ls2_group2_str = ls2str(ls2)

        return Levenshtein.distance(ls1_group1_str, ls2_group1_str) + Levenshtein.distance(ls1_group2_str, ls2_group2_str)

    elif mode == 'combine':
        def ls2str(ls):
                ls = ls.reshape(-1, 2).transpose()      # (NUM_AUDIO_FEAT_FRAMES * 2, groups=2)
                ls = ls[0] * 320 + ls[1]
                str = ''.join([chr(int(i)) for i in ls])
                return str

        ls1_str = ls2str(ls1)
        ls2_str = ls2str(ls2)

        return Levenshtein.distance(ls1_str, ls2_str)
# **************************** wavvq_distances ****************************

# **************************** load_train_db ****************************

# **************************** load_train_db ****************************


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_feats, _, train_motion = load_train_db(args.train_database)

    train_mfcc, train_code, test_mfcc, train_feat, test_feat, train_wavlm, test_wavlm, train_wavlm_feat, \
        test_wavlm_feat, speech_features, test_speech_features, train_speech_features_feat, test_speech_features_feat, \
        train_wavvq_feat, test_wavvq_feat, train_phase, test_phase, train_context, test_context \
        = load_db_codebook(
        args.train_database, args.train_codebook, args.test_data, args.train_wavlm, args.test_wavlm, args.train_wavvq, args.test_wavvq)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
