import tensorflow as tf
import numpy as np

import soundfile as sf
import librosa
#from modules import config as cfg

def load_audio(file_path):
    """Load audio file"""
    audio, sample_rate = sf.read(file_path)
    if sample_rate != 32000:
        audio = librosa.resample(audio.T, orig_sr=sample_rate, target_sr=32000)
        audio = audio.T
    return np.array(audio)  # tf.squeeze(audio)

@tf.function
def normalize_audio(audio, norm_factor):
    """Normalize the audio at the peak values used in Perch model training"""
    audio = tf.identity(audio)
    audio -= tf.reduce_mean(audio, axis=-1, keepdims=True)
    peak_norm = tf.reduce_max(tf.abs(audio), axis=-1, keepdims=True)
    audio = tf.where(peak_norm > 0.0, audio / peak_norm, audio)
    audio = audio * norm_factor
    return audio

@tf.function
def frame_audio(
                audio_array: np.ndarray,
                window_size_s: float = 5.0,
                hop_size_s: float = 5.0,
                sample_rate=32000) -> np.ndarray:
    """Framing audio for inference"""
    if window_size_s is None or window_size_s < 0:
        return audio_array[tf.newaxis, :]  # np.newaxis

    frame_length = int(window_size_s * sample_rate)
    hop_length = int(hop_size_s * sample_rate)

    num_frames = int(tf.math.ceil(tf.shape(audio_array)[0] / frame_length))

    framed_audio = tf.signal.frame(
        audio_array, frame_length, hop_length, pad_end=False, pad_value=0.0
    )
    # if the last frame of audio is shorter than frame_length pad it by concatenating the frame multiple times
    if tf.shape(framed_audio)[0] < num_frames:
        tail = audio_array[((num_frames - 1) * frame_length) :]
        num_repeats = int(tf.math.ceil(frame_length / tf.shape(tail)[0]))  # np.ciel
        last_audio_frame = tf.tile(tail, [num_repeats])[tf.newaxis, :frame_length]
        framed_audio = tf.concat([framed_audio, last_audio_frame], 0)

    return framed_audio  # tf.squeeze(framed_audio, axis=0)

def load_and_preprocess(file_path, e_model):
    audio = load_audio(file_path)  
    #if len(audio.shape) < 2:
    #    audio = audio[np.newaxis,]
    audio = tf.cast(audio, tf.float32)
    normalized_audio = normalize_audio(audio, 0.25)
    framed_audio = frame_audio(normalized_audio)
    if len(framed_audio.shape) > 2:
        framed_audio = tf.squeeze(framed_audio)
    e = e_model.infer_tf(framed_audio)
    return e["embedding"]

def flatten_pred_list(nested_list):
    flat_list = []
    for i, l in enumerate(nested_list):
        for e in l:
            flat_list.append(e.numpy().item())
    return flat_list

def get_classifier_predictions(embeddings, classifier):
    logits = list(map(classifier, embeddings))
    preds = list(map(tf.sigmoid, logits))
    return flatten_pred_list(preds)