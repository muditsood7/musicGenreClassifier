import os
import librosa

DATASET_PATH = "Data"
JSON_PATH = "Data.json"

SAMPLE_RATE=22050
DURATION=30
SAMPLES_PER_TRACK=SAMPLE_RATE*DURATION

def save_MFCC(dataset_path, json_path, n_MFCC=13, n_fft=2048, hop_length=512, num_segments=5):

    # dictionary for storing data
    data = {
        "mapping": ["classical", "blues"],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

    # loop through all the genres in the dataset
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that the loop is not at the root level
        if dirpath is not dataset_path:
            # save the semantic label
            dirpath_components = dirpath.split("/") # genre/blues => ['genre', 'blues']
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)

            # process files for a genre

            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                #process segments, extracting the MFCCs and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc([start_sample, finish_sample],
                                                sr=SAMPLE_RATE,
                                                n_fft=n_fft,
                                                n_mfcc=n_MFCC,
                                                hop_length=hop_length)