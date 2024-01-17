import os
import numpy as np
import librosa
from numpy.fft import fft
from copy import copy
from sklearn.metrics import confusion_matrix, classification_report

TRAIN_PATH = './train/'

MALE_FREQ_RANGE = (55, 160)
FEMALE_FREQ_RANGE = (170, 275)
HPS_TIME_FRAME = 3
HPS_LOOP_LIMIT = 5
DEBUG_OUTPUT_ENABLED = False
REPORT_ENABLED = True

def harmonic_product_spectrum(sample_rate, audio_data, hps_time=HPS_TIME_FRAME):
    hps_time = min(hps_time, len(audio_data) / sample_rate)
    mid_point = len(audio_data) // 2
    start_index = max(0, mid_point - int(hps_time / 2 * sample_rate))
    end_index = min(len(audio_data) - 1, mid_point + int(hps_time / 2 * sample_rate))

    data_voice = audio_data[start_index:end_index]
    part_len = int(sample_rate)
    parts = [data_voice[i * part_len:(i + 1) * part_len] for i in range(int(hps_time))]

    result_parts = []
    for data in parts:
        if len(data) == 0:
            continue
        window = np.hamming(len(data))
        data_windowed = data * window
        fft_values = abs(fft(data_windowed)) / sample_rate
        hps_result = copy(fft_values)
        for i in range(2, HPS_LOOP_LIMIT):
            downsampled = copy(fft_values[::i])
            hps_result = hps_result[:len(downsampled)] * downsampled
        result_parts.append(hps_result)

    aggregate_result = sum(result_parts)

    male_score = np.sum(aggregate_result[MALE_FREQ_RANGE[0]:MALE_FREQ_RANGE[1]])
    female_score = np.sum(aggregate_result[FEMALE_FREQ_RANGE[0]:FEMALE_FREQ_RANGE[1]])

    return "M" if male_score > female_score else "K"


def load_audios(directory):
    male_audios, female_audios = [], []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                gender = file.split("_")[1].replace(".wav", "")
                if gender == "K":  # Assuming "K" stands for female
                    female_audios.append(file)
                elif gender == "M":
                    male_audios.append(file)
    return np.array(male_audios), np.array(female_audios)


def gender_decoder(data):
    if data == "M":
        return "Male"
    elif data == "K":
        return "Female"


if __name__ == "__main__":
    actual_genders = []
    predicted_genders = []

    male_audios, female_audios = load_audios(TRAIN_PATH)

    for male_audio in male_audios:
        try:
            audio_data, sample_rate = librosa.load(TRAIN_PATH + male_audio, sr=None)
            gender_prediction = harmonic_product_spectrum(sample_rate, audio_data)
            actual_genders.append("M")
            predicted_genders.append(gender_prediction)
            if DEBUG_OUTPUT_ENABLED:
                print(f"Audio: {male_audio}, Predicted Gender: {gender_decoder((gender_prediction))}, where it was Male")
            else:
                print(gender_prediction)
        except Exception as e:
            if DEBUG_OUTPUT_ENABLED:
                print(f"Error processing {male_audio}: {e}")

    for female_audio in female_audios:
        try:
            audio_data, sample_rate = librosa.load(TRAIN_PATH + female_audio, sr=None)
            gender_prediction = harmonic_product_spectrum(sample_rate, audio_data)
            actual_genders.append("K")
            predicted_genders.append(gender_prediction)
            if DEBUG_OUTPUT_ENABLED:
                print(f"Audio: {female_audio}, Predicted Gender: {gender_decoder((gender_prediction))}, where it was Female")
            else:
                print(gender_prediction)
        except Exception as e:
            if DEBUG_OUTPUT_ENABLED:
                print(f"Error processing {female_audio}: {e}")

    if REPORT_ENABLED:
        print("\nConfusion Matrix:")
        print(confusion_matrix(actual_genders, predicted_genders))
        print("\nClassification Report:")
        print(classification_report(actual_genders, predicted_genders))