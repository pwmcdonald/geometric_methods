import numpy as np
from sklearn.preprocessing import MinMaxScaler
from noisereduce import reduce_noise
import numpy as np
from fdasrsf.time_warping import pairwise_align_functions
from fdasrsf.utility_functions import warp_f_gamma
from os import listdir
from pydub import AudioSegment
from sqlite3 import connect
import enums
from math import sqrt


# finite difference used to approximate integration
FINITE_DIFFERENCE = 6.25042370008813e-05

# only accounts for every nth time step in integration to speed things up
TIME_STEP_JUMP = 1

# constants for spectrograms/calculations
TIME_INTERVAL = 0.01
TIME_UPPER_BOUND = 1.01

FREQ_INTERVAL = 100
FREQ_UPPER_BOUND = 8100

# degree of polynomial fit for smoothing
SMOOTHING_DEGREE = 18

# square root of -1
i = 1j

# silence threshold values for each speaker
silence_threshold_dict = {
    "FR01_1": -35.0,
    "FR01_2": -35.0,
    "FR01_3": -35.0,
    "FR01_4": -35.0,  # NOTE: Based on first three words for this speaker
    "FR01_5": -35.0,  # NOTE: Based on first three words for this speaker
    "FR01_6": -35.0,  # NOTE: Based on first three words for this speaker
    "FR01_7": -35.0,  # NOTE: Based on first three words for this speaker
    "FR01_8": -35.0,  # NOTE: Based on first three words for this speaker
    "FR01_9": -35.0,  # NOTE: Based on first three words for this speaker
    "FR01_10": -35.0,  # NOTE: Based on first three words for this speaker
    "FR02_1": -35.0,
    "FR02_2": -35.0,
    "FR02_3": -35.0,
    "FR02_4": -35.0,  # NOTE: Based on first three words for this speaker
    "FR02_5": -35.0,  # NOTE: Based on first three words for this speaker
    "FR02_6": -35.0,  # NOTE: Based on first three words for this speaker
    "FR02_7": -35.0,  # NOTE: Based on first three words for this speaker
    "FR02_8": -35.0,  # NOTE: Based on first three words for this speaker
    "FR02_9": -35.0,  # NOTE: Based on first three words for this speaker
    "FR02_10": -35.0,  # NOTE: Based on first three words for this speaker
    "FR03_1": -35.0,
    "FR03_2": -35.0,
    "FR03_3": -35.0,
    "FR03_4": -35.0,  # NOTE: Based on first three words for this speaker
    "FR03_5": -35.0,  # NOTE: Based on first three words for this speaker
    "FR03_6": -35.0,  # NOTE: Based on first three words for this speaker
    "FR03_7": -35.0,  # NOTE: Based on first three words for this speaker
    "FR03_8": -35.0,  # NOTE: Based on first three words for this speaker
    "FR03_9": -35.0,  # NOTE: Based on first three words for this speaker
    "FR03_10": -35.0,  # NOTE: Based on first three words for this speaker
    "FR05_1": -35.0,
    "FR05_2": -35.0,
    "FR05_3": -35.0,  # NOTE: Based on first three words for this speaker
    "FR05_4": -35.0,  # NOTE: Based on first three words for this speaker
    "FR05_5": -35.0,  # NOTE: Based on first three words for this speaker
    "FR05_6": -35.0,  # NOTE: Based on first three words for this speaker
    "FR05_7": -35.0,  # NOTE: Based on first three words for this speaker
    "FR05_8": -35.0,  # NOTE: Based on first three words for this speaker
    "FR05_9": -35.0,  # NOTE: Based on first three words for this speaker
    "FR05_10": -35.0,  # NOTE: Based on first three words for this speaker
    "FR06_1": -35.0,
    "FR06_2": -35.0,  # NOTE: Based on rest of French words
    "FR06_3": -35.0,  # NOTE: Based on rest of French words
    "FR06_4": -35.0,  # NOTE: Based on rest of French words
    "FR06_5": -35.0,  # NOTE: Based on rest of French words
    "FR06_6": -35.0,  # NOTE: Based on rest of French words
    "FR06_7": -35.0,  # NOTE: Based on rest of French words
    "FR06_8": -35.0,  # NOTE: Based on rest of French words
    "FR06_9": -35.0,  # NOTE: Based on rest of French words
    "FR06_10": -35.0,  # NOTE: Based on rest of French words
    "FR07_1": -35.0,
    "FR07_2": -35.0,  # NOTE: Based on rest of French words
    "FR07_3": -35.0,  # NOTE: Based on rest of French words
    "FR07_4": -35.0,  # NOTE: Based on rest of French words
    "FR07_5": -35.0,  # NOTE: Based on rest of French words
    "FR07_6": -35.0,  # NOTE: Based on rest of French words
    "FR07_7": -35.0,  # NOTE: Based on rest of French words
    "FR07_8": -35.0,  # NOTE: Based on rest of French words
    "FR07_9": -35.0,  # NOTE: Based on rest of French words
    "FR07_10": -35.0,  # NOTE: Based on rest of French words
    "FR08_1": -35.0,
    "FR08_2": -35.0,
    "FR08_3": -30.0,
    "FR08_4": -35.0,
    "FR08_5": -35.0,
    "FR08_6": -35.0,  # NOTE: Based on rest of French words
    "FR08_7": -35.0,  # NOTE: Based on rest of French words
    "FR08_8": -35.0,  # NOTE: Based on rest of French words
    "FR08_9": -35.0,  # NOTE: Based on rest of French words
    "FR08_10": -35.0,  # NOTE: Based on rest of French words
    "IT01_1": -34.0,
    "IT01_2": -45.0,
    "IT01_3": -36.0,
    "IT01_4": -35.0,
    "IT01_5": -35.0,
    "IT01_6": -35.0,
    "IT01_7": -35.0,
    "IT01_8": -35.0,
    "IT01_9": -32.0,
    "IT01_10": -35.0,
    "IT02_1": -30.0,
    "IT02_2": -40.0,
    "IT02_3": -35.0,
    "IT02_4": -35.0,
    "IT02_5": -50.0,
    "IT02_6": -50.0,
    "IT02_7": -50.0,
    "IT02_8": -35.0,
    "IT02_9": -40.0,
    "IT02_10": -40.0,
    "IT03_1": -40.0,
    "IT03_2": -40.0,
    "IT03_3": -40.0,
    "IT03_4": -40.0,  # NOTE: Based on first three words for this speaker
    "IT03_5": -40.0,  # NOTE: Based on first three words for this speaker
    "IT03_6": -40.0,  # NOTE: Based on first three words for this speaker
    "IT03_7": -40.0,  # NOTE: Based on first three words for this speaker
    "IT03_8": -40.0,  # NOTE: Based on first three words for this speaker
    "IT03_9": -40.0,  # NOTE: Based on first three words for this speaker
    "IT03_10": -40.0,  # NOTE: Based on first three words for this speaker
    "IT04_1": -40.0,
    "IT04_2": -40.0,
    "IT04_3": -40.0,
    "IT04_4": -40.0,  # NOTE: Based on first three words for this speaker
    "IT04_5": -40.0,  # NOTE: Based on first three words for this speaker
    "IT04_6": -40.0,  # NOTE: Based on first three words for this speaker
    "IT04_7": -40.0,  # NOTE: Based on first three words for this speaker
    "IT04_8": -40.0,  # NOTE: Based on first three words for this speaker
    "IT04_9": -40.0,  # NOTE: Based on first three words for this speaker
    "IT04_10": -40.0,  # NOTE: Based on first three words for this speaker
    "IT05_1": -40.0,
    "IT05_2": -40.0,
    "IT05_3": -40.0,
    "IT05_4": -40.0,  # NOTE: Based on first three words for this speaker
    "IT05_5": -40.0,  # NOTE: Based on first three words for this speaker
    "IT05_6": -40.0,  # NOTE: Based on first three words for this speaker
    "IT05_7": -40.0,  # NOTE: Based on first three words for this speaker
    "IT05_8": -40.0,  # NOTE: Based on first three words for this speaker
    "IT05_9": -40.0,  # NOTE: Based on first three words for this speaker
    "IT05_10": -40.0,  # NOTE: Based on first three words for this speaker
    "PO01_1": -40.0,
    "PO01_2": -40.0,
    "PO01_3": -40.0,
    "PO01_4": -40.0,  # NOTE: Based on first three words for this speaker
    "PO01_5": -40.0,  # NOTE: Based on first three words for this speaker
    "PO01_6": -40.0,  # NOTE: Based on first three words for this speaker
    "PO01_7": -40.0,  # NOTE: Based on first three words for this speaker
    "PO01_8": -40.0,  # NOTE: Based on first three words for this speaker
    "PO01_9": -40.0,  # NOTE: Based on first three words for this speaker
    "PO01_10": -40.0,  # NOTE: Based on first three words for this speaker
    "PO02_1": -30.0,
    "PO02_2": -40.0,
    "PO02_3": -30.0,
    "PO02_4": -30.0,
    "PO02_5": -40.0,
    "PO02_6": -40.0,
    "PO02_7": -40.0,
    "PO02_8": -40.0,
    "PO02_9": -40.0,  # NOTE: Based on last few
    "PO02_10": -40.0,  # NOTE: Based on last few
    "PO03_1": -30.0,
    "PO03_2": -30.0,
    "PO03_3": -30.0,
    "PO03_4": -30.0,
    "PO03_5": -30.0,  # NOTE: Based on first four words for this speaker
    "PO03_6": -30.0,  # NOTE: Based on first four words for this speaker
    "PO03_7": -30.0,  # NOTE: Based on first four words for this speaker
    "PO03_8": -30.0,  # NOTE: Based on first four words for this speaker
    "PO03_9": -30.0,  # NOTE: Based on first four words for this speaker
    "PO03_10": -30.0,  # NOTE: Based on first four words for this speaker
    "PO04_2": -40.0,
    "PO04_3": -40.0,
    "PO04_4": -40.0,
    "PO04_5": -40.0,
    "PO04_6": -40.0,  # NOTE: Based on first four words for this speaker
    "PO04_7": -40.0,  # NOTE: Based on first four words for this speaker
    "PO04_8": -40.0,  # NOTE: Based on first four words for this speaker
    "PO04_9": -40.0,  # NOTE: Based on first four words for this speaker
    "PO04_10": -40.0,  # NOTE: Based on first four words for this speaker
    "PO05_1": -40.0,
    "PO05_2": -40.0,
    "PO05_3": -40.0,
    "PO05_4": -40.0,
    "PO05_5": -40.0,  # NOTE: Based on first four words for this speaker
    "PO05_6": -40.0,  # NOTE: Based on first four words for this speaker
    "PO05_7": -40.0,  # NOTE: Based on first four words for this speaker
    "PO05_8": -40.0,  # NOTE: Based on first four words for this speaker
    "PO05_9": -40.0,  # NOTE: Based on first four words for this speaker
    "PO05_10": -40.0,  # NOTE: Based on first four words for this speaker
    "PO06_1": -40.0,
    "PO06_2": -40.0,
    "PO06_7": -40.0,
    "PO06_8": -40.0,
    "PO06_9": -40.0,  # NOTE: Based on first four words for this speaker
    "PO06_10": -40.0,  # NOTE: Based on first four words for this speaker
    "SA01_1": -40.0,
    "SA01_2": -40.0,
    "SA01_3": -40.0,
    "SA01_4": -40.0,
    "SA01_5": -40.0,  # NOTE: Based on first four words for this speaker
    "SA01_6": -40.0,  # NOTE: Based on first four words for this speaker
    "SA01_7": -40.0,  # NOTE: Based on first four words for this speaker
    "SA01_8": -40.0,  # NOTE: Based on first four words for this speaker
    "SA01_9": -40.0,  # NOTE: Based on first four words for this speaker
    "SA01_10": -40.0,  # NOTE: Based on first four words for this speaker
    "SA02_1": -50.0,
    "SA02_2": -50.0,
    "SA02_3": -50.0,
    "SA02_4": -50.0,
    "SA02_5": -50.0,  # NOTE: Based on first four words for this speaker
    "SA02_6": -50.0,  # NOTE: Based on first four words for this speaker
    "SA02_7": -50.0,  # NOTE: Based on first four words for this speaker
    "SA02_8": -50.0,  # NOTE: Based on first four words for this speaker
    "SA02_9": -50.0,  # NOTE: Based on first four words for this speaker
    "SA02_10": -50.0,  # NOTE: Based on first four words for this speaker
    "SA03_1": -50.0,
    "SA03_2": -50.0,
    "SA03_3": -50.0,
    "SA03_4": -50.0,
    "SA03_5": -50.0,  # NOTE: Based on first four words for this speaker
    "SA03_6": -50.0,  # NOTE: Based on first four words for this speaker
    "SA03_7": -50.0,  # NOTE: Based on first four words for this speaker
    "SA03_8": -50.0,  # NOTE: Based on first four words for this speaker
    "SA03_9": -50.0,  # NOTE: Based on first four words for this speaker
    "SA03_10": -50.0,  # NOTE: Based on first four words for this speaker
    "SA04_1": -50.0,
    "SA04_2": -50.0,
    "SA04_3": -50.0,
    "SA04_4": -50.0,
    "SA04_5": -50.0,  # NOTE: Based on first four words for this speaker
    "SA04_6": -50.0,  # NOTE: Based on first four words for this speaker
    "SA04_7": -50.0,  # NOTE: Based on first four words for this speaker
    "SA04_8": -50.0,  # NOTE: Based on first four words for this speaker
    "SA04_9": -50.0,  # NOTE: Based on first four words for this speaker
    "SA04_10": -50.0,  # NOTE: Based on first four words for this speaker
    "SA05_1": -50.0,
    "SA05_2": -50.0,
    "SA05_3": -50.0,
    "SA05_4": -50.0,
    "SA05_5": -50.0,
    "SA05_6": -50.0,
    "SA05_7": -50.0,
    "SA05_8": -50.0,
    "SA05_9": -50.0,
    "SA05_10": -50.0,
    "SI01_2": -40.0,
    "SI01_6": -40.0,
    "SI01_7": -40.0,
    "SI02_1": -40.0,
    "SI02_2": -40.0,
    "SI02_3": -50.0,
    "SI02_4": -50.0,
    "SI02_5": -50.0,  # NOTE: Based on last couple words for this speaker
    "SI02_6": -50.0,  # NOTE: Based on last couple words for this speaker
    "SI02_7": -50.0,  # NOTE: Based on last couple words for this speaker
    "SI02_8": -50.0,  # NOTE: Based on last couple words for this speaker
    "SI02_9": -50.0,  # NOTE: Based on last couple words for this speaker
    "SI02_10": -50.0,  # NOTE: Based on last couple words for this speaker
    "SI03_1": -50.0,
    "SI03_2": -50.0,
    "SI03_3": -50.0,
    "SI03_4": -50.0,
    "SI03_5": -50.0,  # NOTE: Based on first four words for this speaker
    "SI03_6": -50.0,  # NOTE: Based on first four words for this speaker
    "SI03_7": -50.0,  # NOTE: Based on first four words for this speaker
    "SI03_8": -50.0,  # NOTE: Based on first four words for this speaker
    "SI03_9": -50.0,  # NOTE: Based on first four words for this speaker
    "SI03_10": -50.0,  # NOTE: Based on first four words for this speaker
    "SI04_2": -50.0,
    "SI04_4": -50.0,
    "SI04_5": -50.0,  # NOTE: Based on other two words for this speaker
    "SI05_1": -50.0,
    "SI05_2": -50.0,
    "SI05_3": -50.0,  # NOTE: Based on first two words for this speaker
    "SI05_4": -50.0,  # NOTE: Based on first two words for this speaker
    "SI05_5": -50.0,  # NOTE: Based on first two words for this speaker
    "SI05_6": -50.0,  # NOTE: Based on first two words for this speaker
    "SI05_7": -50.0,  # NOTE: Based on first two words for this speaker
    "SI05_8": -50.0,  # NOTE: Based on first two words for this speaker
    "SI05_9": -50.0,  # NOTE: Based on first two words for this speaker
    "SI05_10": -50.0,  # NOTE: Based on first two words for this speaker
    "SI06_1": -40.0,
    "SI06_2": -40.0,
    "SI06_3": -40.0,  # NOTE: Based on first two words for this speaker
    "SI06_4": -40.0,  # NOTE: Based on first two words for this speaker
    "SI06_5": -40.0,  # NOTE: Based on first two words for this speaker
    "SI06_6": -40.0,  # NOTE: Based on first two words for this speaker
    "SI06_7": -40.0,  # NOTE: Based on first two words for this speaker
    "SI06_8": -40.0,  # NOTE: Based on first two words for this speaker
    "SI06_9": -40.0,  # NOTE: Based on first two words for this speaker
    "SI06_10": -40.0,  # NOTE: Based on first two words for this speaker
    "SI07_1": -40.0,
    "SI07_2": -40.0,
    "SI07_3": -40.0,  # NOTE: Based on first two words for this speaker
    "SI07_4": -40.0,  # NOTE: Based on first two words for this speaker
    "SI07_5": -40.0,  # NOTE: Based on first two words for this speaker
    "SI07_6": -40.0,  # NOTE: Based on first two words for this speaker
    "SI07_7": -40.0,  # NOTE: Based on first two words for this speaker
    "SI07_8": -40.0,  # NOTE: Based on first two words for this speaker
    "SI07_9": -40.0,  # NOTE: Based on first two words for this speaker
    "SI07_10": -40.0,  # NOTE: Based on first two words for this speaker
}


def _get_TIME_INTERVAL() -> float:
    """Getter method for TIME_INTERVAL.

    Returns:
        float: TIME_INTERVAL.
    """
    return TIME_INTERVAL


def _get_TIME_UPPER_BOUND() -> float:
    """Getter method for TIME_UPPER_BOUND.

    Returns:
        float: TIME_UPPER_BOUND.
    """
    return TIME_UPPER_BOUND


def _get_FREQ_INTERVAL() -> float:
    """Getter method for FREQ_INTERVAL.

    Returns:
        float: FREQ_INTERVAL.
    """
    return FREQ_INTERVAL


def _get_FREQ_UPPER_BOUND() -> float:
    """Getter method for FREQ_UPPER_BOUND.

    Returns:
        float: FREQ_UPPER_BOUND.
    """
    return FREQ_UPPER_BOUND


def _psi(
    tau: float,
) -> float:
    """Define window function from Pigoli et al.'s ``The statistical
    analysis of acoustic phonetic data'' (2017).

    Args:
        tau (float): The input to the window function.

    Returns:
        float: The output of the window function.
    """
    return np.e ** (-0.5 * (tau / 0.005) ** 2)


def _X(
    sa: np.array,
    ts: np.array,
    omega: float,
    t: float,
) -> float:
    """Local Fourier transform at (angular) frequency omega, time t.

    Args:
        sa (np.array): The signal array in question.
        ts (np.array): The times in question.
        omega (float): The angular frequency.
        t (float): The time.

    Returns:
        float: The Fourier transform output.
    """
    return np.sum(
        [
            (sa[k] * _psi(ts[k] - t) * np.e ** (-i * omega * ts[k])) * FINITE_DIFFERENCE
            for k in range(0, len(ts), TIME_STEP_JUMP)
        ]
    )


def _W(
    sa: np.array,
    ts: np.array,
    omega: float,
    t: float,
) -> float:
    """Power spectral density at (angular) frequency omega, time t.

    Args:
        sa (np.array): The signal array in question.
        ts (np.array): The times in question.
        omega (float): The angular frequency.
        t (float): The time.

    Returns:
        float: The power spectral density output.
    """
    return 10 * np.log10(np.abs(_X(sa, ts, omega, t)) ** 2)


def _detect_leading_silence(
    sound: AudioSegment,
    silence_threshold: float,
    chunk_size: int = 10,
):
    """
    This code was found on StackOverflow @
    https://stackoverflow.com/questions/29547218/remove-silence-at-the-
    beginning-and-at-the-end-of-wave-files-with-pydub
    and potentially modified somewhat

    Finds the number of milliseconds into a file that noise begins; allows
    us to trim silence at the beginning and end.

    Args:
        sound (AudioSegment): The AudioSegment object corresponding to the
                              .wav file to be trimmed.
        silence_threshold (float): The decibel threshold considered "silence"
                                   in the trimming process.
        chunk_size (int): The number of milliseconds we window-shift by in
                          trimming the file.

    Returns:
        trim_ms (int): The number of milliseconds into the audio file that the silence ceases
    """
    trim_ms = 0

    assert chunk_size > 0  # to avoid infinite loop
    while sound[
        trim_ms : trim_ms + chunk_size
    ].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def _get_unaligned_wav_spectrogram(
    wav_path: str,
    silence_threshold: str,
) -> np.array:
    """
    Preprocesses a .wav file. Some of this approach mirrors that from
    https://learnpython.com/blog/plot-waveform-in-python/.

    Args:
        wav_path (str): The file path corresponding to the .wav file to be
                        converted to a log spectrogram array.
        silence_threshold (str): The key for the required silence threshold.

    Returns:
        np.array: NumPy array containing log spectrogram values.
    """
    # parse .wav file located @ wav_path
    sound = AudioSegment.from_file(wav_path, format="wav")
    sample_freq = sound.frame_rate

    # reduce noise in signal
    noisereduced_array = reduce_noise(sound.get_array_of_samples(), sample_freq)

    # reconstruct AudioSegment from array
    # pulled from https://stackoverflow.com/questions/35735497/how-to-create-a-
    # pydub-audiosegment-using-an-numpy-array
    sound = AudioSegment(
        noisereduced_array.tobytes(),
        frame_rate=sample_freq,
        sample_width=noisereduced_array.dtype.itemsize,
        channels=1,
    )

    # get trim points on audio file
    start_trim = _detect_leading_silence(sound, silence_threshold=silence_threshold)
    end_trim = _detect_leading_silence(
        sound.reverse(), silence_threshold=silence_threshold
    )

    # trim audio file
    duration = len(sound)
    trimmed_sound_array = sound[start_trim : duration - end_trim].get_array_of_samples()
    n_samples = len(trimmed_sound_array)
    times = np.linspace(0, n_samples / sample_freq, num=n_samples)

    # standardize timescale
    times = MinMaxScaler().fit_transform(times.reshape(-1, 1))
    times = times.reshape((len(times),))

    # construct array to contain log spectrogram values
    X = np.arange(0, TIME_UPPER_BOUND, TIME_INTERVAL)
    # below set this way to get square X/Y coords for surface plot
    Y = np.arange(0, FREQ_UPPER_BOUND, FREQ_INTERVAL)

    lenX = len(X)
    lenY = len(Y)

    Z = np.zeros((lenX, lenY))

    counter = 0

    for k in range(lenX):
        for p in range(lenY):
            counter += 1

            Z[k][p] = _W(trimmed_sound_array, times, Y[p], X[k])

            # shows progress as pipeline runs
            if counter % 1000 == 0:
                print("...")

    return Z


def _store_raw_spectrograms(
    cleaned_data_path: str,
    digit_bound: int = 10,
) -> None:
    """Computes raw spectrograms for all of the cleaned data and stores them in a database.

    Args:
        cleaned_data_path (str): The file path for the cleaned data.
        digit_bound (int): The largest digit spoken by any speaker in the data set. Defaults
                           to 10.

    Returns:
        None.
    """
    con = connect(f"{enums._DBType.RAW_SPECTROGRAM.value}.db")
    cur = con.cursor()

    for dgt in range(1, digit_bound + 1):
        print(f"NOW CALCULATING/STORING SPECTROGRAMS FOR {dgt} WORDS")
        partial_path = f"{cleaned_data_path}/{dgt}_words"
        dgt_words = listdir(partial_path)
        for fl in dgt_words:
            if fl != ".DS_Store":
                spkr = fl[:4]
                print(f"FILE: {spkr}_word{dgt}")
                raw_spect_arr = _get_unaligned_wav_spectrogram(
                    partial_path + "/" + fl, silence_threshold_dict[f"{spkr}_{dgt}"]
                )
                for t in range(0, int(TIME_UPPER_BOUND / TIME_INTERVAL)):
                    hz_values = raw_spect_arr[t]
                    insert_query_string = f"INSERT INTO {spkr}_word{dgt} VALUES ("
                    for hz in hz_values:
                        insert_query_string += f"{hz}, "
                    insert_query_string = insert_query_string[:-2] + ");"
                    cur.execute(insert_query_string)
                con.commit()
        print(f"DONE CALCULATING/STORING SPECTROGRAMS FOR {dgt} WORDS")
        print("")


def _store_pairwise_warping_function(
    dgt: int,
    speaker1: str,
    speaker2: str,
) -> None:
    """Stores the pairwise warping function for any two speakers saying
    a given word.

    Args:
        dgt (int): The digit being spoken by both speakers.
        speaker1 (str): The first speaker saying the word.
        speaker2 (str): The second speaker saying the word.

    Returns:
        None.
    """
    print(
        f"NOW CALCULATING/STORING PAIRWISE WARPING FUNCTION {speaker1}_{speaker2}_{dgt}"
    )
    raw_con = connect(f"{enums._DBType.RAW_SPECTROGRAM.value}.db")
    raw_cur = raw_con.cursor()
    warp_con = connect("pairwise_warping_function.db")
    warp_cur = warp_con.cursor()
    time = np.array(np.arange(0, TIME_UPPER_BOUND, TIME_INTERVAL))
    warping_function_array = np.zeros(
        (int(TIME_UPPER_BOUND / TIME_INTERVAL), int(FREQ_UPPER_BOUND / FREQ_INTERVAL))
    )

    for freq in np.arange(0, FREQ_UPPER_BOUND, FREQ_INTERVAL):
        f_a = np.array(
            raw_cur.execute(f"SELECT Hz{freq} FROM {speaker1}_word{dgt}").fetchall()
        ).reshape(-1)
        f_b = np.array(
            raw_cur.execute(f"SELECT Hz{freq} FROM {speaker2}_word{dgt}").fetchall()
        ).reshape(-1)
        _, warping_function, _ = pairwise_align_functions(f1=f_a, f2=f_b, time=time)
        freq_index = int(freq / FREQ_INTERVAL)
        warping_function_array[:, freq_index] = warping_function

    for t in range(0, int(TIME_UPPER_BOUND / TIME_INTERVAL)):
        warp_values = warping_function_array[t]
        insert_query_string = f"INSERT INTO {speaker1}_{speaker2}_{dgt} VALUES ("
        for val in warp_values:
            insert_query_string += f"{val}, "
        insert_query_string = insert_query_string[:-2] + ");"
        warp_cur.execute(insert_query_string)

    warp_con.commit()
    print(
        f"DONE CALCULATING/STORING PAIRWISE WARPING FUNCTION {speaker1}_{speaker2}_{dgt}"
    )
    print("")


def _store_global_inverse_warping_function(
    dgt: int,
    speaker: str,
    cleaned_data_path: str,
) -> None:
    """Calculates and stores the global inverse warping function for a given word
        pronounced by a given speaker.

    Args:
        dgt (int): The digit being spoken by the speaker.
        speaker (str): The speaker saying the word.
        cleaned_data_path (str): The file path of the folder containing the
                                 cleaned pipeline input.
    Returns:
        None.
    """
    print(
        f"NOW CALCULATING/STORING GLOBAL INVERSE WARPING FUNCTION {speaker}_word{dgt}"
    )
    speaker_a = speaker

    warp_con = connect("pairwise_warping_function.db")
    warp_cur = warp_con.cursor()

    global_function_inverse = np.zeros(
        (int(TIME_UPPER_BOUND / TIME_INTERVAL), int(FREQ_UPPER_BOUND / FREQ_INTERVAL))
    )
    total_speaker_count = 0

    for fl in listdir(f"{cleaned_data_path}/{dgt}_words"):
        if fl != ".DS_Store" and fl[:4] != speaker_a:
            total_speaker_count += 1
            speaker_b = fl[:4]
            pairwise_func = np.array(
                warp_cur.execute(
                    f"SELECT * FROM {speaker_a}_{speaker_b}_{dgt}"
                ).fetchall()
            )
            global_function_inverse += pairwise_func

    global_function_inverse /= total_speaker_count

    global_inverse_con = connect(
        f"{enums._DBType.GLOBAL_INVERSE_WARPING_FUNCTION.value}.db"
    )
    global_inverse_cur = global_inverse_con.cursor()

    for t in range(0, int(TIME_UPPER_BOUND / TIME_INTERVAL)):
        global_inverse_vals = global_function_inverse[t]
        insert_query_string = f"INSERT INTO {speaker_a}_word{dgt} VALUES ("
        for val in global_inverse_vals:
            insert_query_string += f"{val}, "
        insert_query_string = insert_query_string[:-2] + ");"
        global_inverse_cur.execute(insert_query_string)

    global_inverse_con.commit()
    print(
        f"DONE CALCULATING/STORING GLOBAL INVERSE WARPING FUNCTION {speaker}_word{dgt}"
    )
    print("")


def _time_align_raw_spectrograms(
    cleaned_data_path: str,
    digit_bound: int,
) -> None:
    """Time aligns raw spectrograms and stores the results.

    Args:
        cleaned_data_path (str): The file path of the folder containing the
                                 cleaned pipeline input.
        digit_bound (int): The largest digit spoken by any speaker in the
                           data set.

    Returns:
        None.
    """
    raw_con = connect(f"{enums._DBType.RAW_SPECTROGRAM.value}.db")
    raw_cur = raw_con.cursor()
    warp_con = connect(f"{enums._DBType.GLOBAL_INVERSE_WARPING_FUNCTION.value}.db")
    warp_cur = warp_con.cursor()
    align_con = connect(f"{enums._DBType.TIME_ALIGNED_SPECTROGRAM.value}.db")
    align_cur = align_con.cursor()
    time = np.arange(0, TIME_UPPER_BOUND, TIME_INTERVAL)
    for dgt in range(1, digit_bound + 1):
        print(f"NOW TIME ALIGNING/STORING SPECTROGRAMS FOR {dgt} WORDS")
        partial_path = f"{cleaned_data_path}/{dgt}_words"
        dgt_words = listdir(partial_path)
        for fl in dgt_words:
            if fl != ".DS_Store":
                spkr = fl[:4]
                raw_spect_arr = np.array(
                    raw_cur.execute(f"SELECT * FROM {spkr}_word{dgt}").fetchall()
                )
                warp_function_array = np.array(
                    warp_cur.execute(f"SELECT * FROM {spkr}_word{dgt}").fetchall()
                )
                warped_array = np.zeros(
                    (
                        int(TIME_UPPER_BOUND / TIME_INTERVAL),
                        int(FREQ_UPPER_BOUND / FREQ_INTERVAL),
                    )
                )
                for freq in range(0, int(FREQ_UPPER_BOUND / FREQ_INTERVAL)):
                    hz_values = raw_spect_arr[:, freq].reshape(-1)
                    warp_function_values = warp_function_array[:, freq].reshape(-1)
                    warped_values = warp_f_gamma(time, hz_values, warp_function_values)
                    warped_array[:, freq] = warped_values
                for t in range(0, int(TIME_UPPER_BOUND / TIME_INTERVAL)):
                    warped_values = warped_array[t]
                    insert_query_string = f"INSERT INTO {spkr}_word{dgt} VALUES ("
                    for val in warped_values:
                        insert_query_string += f"{val}, "
                    insert_query_string = insert_query_string[:-2] + ");"
                    align_cur.execute(insert_query_string)
        print(f"DONE TIME ALIGNING/STORING SPECTROGRAMS FOR {dgt} WORDS")
        print("")
    align_con.commit()


def _smooth_time_aligned_spectrograms(
    cleaned_data_path: str,
    digit_bound: int,
) -> None:
    """Smooths time-aligned spectrograms and stores the results.

    Args:
        cleaned_data_path (str): The file path of the folder containing the
                                 cleaned pipeline input.
        digit_bound (int): The largest digit spoken by any speaker in the
                           data set.

    Returns:
        None.
    """
    aligned_con = connect(f"{enums._DBType.TIME_ALIGNED_SPECTROGRAM.value}.db")
    aligned_cur = aligned_con.cursor()
    smooth_con = connect(f"{enums._DBType.TIME_ALIGNED_SMOOTHED_SPECTROGRAM.value}.db")
    smooth_cur = smooth_con.cursor()
    time = np.arange(0, TIME_UPPER_BOUND, TIME_INTERVAL)
    for dgt in range(1, digit_bound + 1):
        print(f"NOW SMOOTHING/STORING SPECTROGRAMS FOR {dgt} WORDS")
        partial_path = f"{cleaned_data_path}/{dgt}_words"
        dgt_words = listdir(partial_path)
        time = np.arange(0, TIME_UPPER_BOUND, TIME_INTERVAL)
        deg = SMOOTHING_DEGREE
        for fl in dgt_words:
            if fl != ".DS_Store":
                spkr = fl[:4]
                aligned_spec_arr = np.array(
                    aligned_cur.execute(f"SELECT * FROM {spkr}_word{dgt}").fetchall()
                )
                smoothed_array = np.zeros(
                    (
                        int(TIME_UPPER_BOUND / TIME_INTERVAL),
                        int(FREQ_UPPER_BOUND / FREQ_INTERVAL),
                    )
                )
                for freq in range(0, int(FREQ_UPPER_BOUND / FREQ_INTERVAL)):
                    hz_values = aligned_spec_arr[:, freq].reshape(-1)
                    p = np.polyfit(time, hz_values, deg)
                    for t in range(0, int(TIME_UPPER_BOUND / TIME_INTERVAL)):
                        smoothed_array[t][freq] = np.polyval(p, time[t])
                for t in range(0, int(TIME_UPPER_BOUND / TIME_INTERVAL)):
                    smoothed_values = smoothed_array[t]
                    insert_query_string = f"INSERT INTO {spkr}_word{dgt} VALUES ("
                    for val in smoothed_values:
                        insert_query_string += f"{val}, "
                    insert_query_string = insert_query_string[:-2] + ");"
                    smooth_cur.execute(insert_query_string)
        print(f"DONE SMOOTHING/STORING SPECTROGRAMS FOR {dgt} WORDS")
        print("")
    smooth_con.commit()


def _create_mean_spectrograms():
    """Creates mean spectrograms and stores the results.

    Args:
        None.

    Returns:
        None.
    """
    print("NOW CREATING MEAN SPECTROGRAMS")
    with connect(
        f"{enums._DBType.TIME_ALIGNED_SMOOTHED_SPECTROGRAM.value}.db"
    ) as tas_c:
        tas_cur = tas_c.cursor()
        for dgt in range(1, 11):
            for lang in [
                l.value
                for l in enums.RomanceLanguages
                if l.value != enums.RomanceLanguages._BRAZILIAN_PORTUGUESE.value
                and l.value != enums.RomanceLanguages._LUSITANIAN_PORTUGUESE.value
            ]:
                mean_spct = np.zeros(
                    (
                        int(TIME_UPPER_BOUND / TIME_INTERVAL),
                        int(FREQ_UPPER_BOUND / FREQ_INTERVAL),
                    )
                )
                counter = 0
                if lang == enums.RomanceLanguages.FRENCH.value:
                    for spkr in [s.value for s in enums.FrenchSpeakers]:
                        spct = np.array(
                            tas_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            mean_spct += spct
                            counter += 1
                elif lang == enums.RomanceLanguages.ITALIAN.value:
                    for spkr in [s.value for s in enums.ItalianSpeakers]:
                        spct = np.array(
                            tas_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            mean_spct += spct
                            counter += 1
                elif lang == enums.RomanceLanguages.PORTUGUESE.value:
                    for spkr in [
                        s.value for s in enums._BrazilianPortugueseSpeakers
                    ] + [s.value for s in enums._LusitanianPortugueseSpeakers]:
                        spct = np.array(
                            tas_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            mean_spct += spct
                            counter += 1
                elif lang == enums.RomanceLanguages.AMERICAN_SPANISH.value:
                    for spkr in [s.value for s in enums.AmericanSpanishSpeakers]:
                        spct = np.array(
                            tas_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            mean_spct += spct
                            counter += 1
                elif lang == enums.RomanceLanguages.IBERIAN_SPANISH.value:
                    for spkr in [s.value for s in enums.IberianSpanishSpeakers]:
                        spct = np.array(
                            tas_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            mean_spct += spct
                            counter += 1

                mean_spct /= counter

                with connect(f"{enums._DBType.MEAN_SPECTROGRAM.value}.db") as mean_c:
                    mean_cur = mean_c.cursor()
                    for t in range(0, int(TIME_UPPER_BOUND / TIME_INTERVAL)):
                        hz_values = mean_spct[t]
                        insert_query_string = f"INSERT INTO {lang}_{dgt} VALUES ("
                        for hz in hz_values:
                            insert_query_string += f"{hz}, "
                        insert_query_string = insert_query_string[:-2] + ");"
                        mean_cur.execute(insert_query_string)
                    mean_c.commit()
    print("DONE CREATING MEAN SPECTROGRAMS")


def _create_mean_residual_spectrograms():
    """Creates mean residual spectrograms and stores the results.

    Args:
        None.

    Returns:
        None.
    """
    print("NOW CREATING MEAN RESIDUAL SPECTROGRAMS")
    with connect(f"{enums._DBType.RESIDUAL_SPECTROGRAM.value}.db") as rs_c:
        rs_cur = rs_c.cursor()
        for lang in [
            l.value
            for l in enums.RomanceLanguages
            if l.value != enums.RomanceLanguages._BRAZILIAN_PORTUGUESE.value
            and l.value != enums.RomanceLanguages._LUSITANIAN_PORTUGUESE.value
        ]:
            res_mean_spct = np.zeros(
                (
                    int(TIME_UPPER_BOUND / TIME_INTERVAL),
                    int(FREQ_UPPER_BOUND / FREQ_INTERVAL),
                )
            )
            counter = 0
            if lang == enums.RomanceLanguages.FRENCH.value:
                for spkr in [s.value for s in enums.FrenchSpeakers]:
                    for dgt in range(1, 11):
                        spct = np.array(
                            rs_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            res_mean_spct += spct
                            counter += 1
            elif lang == enums.RomanceLanguages.ITALIAN.value:
                for spkr in [s.value for s in enums.ItalianSpeakers]:
                    for dgt in range(1, 11):
                        spct = np.array(
                            rs_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            res_mean_spct += spct
                            counter += 1
            elif lang == enums.RomanceLanguages.PORTUGUESE.value:
                for spkr in [s.value for s in enums._BrazilianPortugueseSpeakers] + [
                    s.value for s in enums._LusitanianPortugueseSpeakers
                ]:
                    for dgt in range(1, 11):
                        spct = np.array(
                            rs_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            res_mean_spct += spct
                            counter += 1
            elif lang == enums.RomanceLanguages.AMERICAN_SPANISH.value:
                for spkr in [s.value for s in enums.AmericanSpanishSpeakers]:
                    for dgt in range(1, 11):
                        spct = np.array(
                            rs_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            res_mean_spct += spct
                            counter += 1
            elif lang == enums.RomanceLanguages.IBERIAN_SPANISH.value:
                for spkr in [s.value for s in enums.IberianSpanishSpeakers]:
                    for dgt in range(1, 11):
                        spct = np.array(
                            rs_cur.execute(
                                f"SELECT * FROM {lang}{spkr}_word{dgt}"
                            ).fetchall()
                        )
                        if list(spct) != []:
                            res_mean_spct += spct
                            counter += 1

            # NOTE: taking the sample mean here
            res_mean_spct /= counter - 1

            with connect("mean_residual_spectrogram.db") as mean_c:
                mean_cur = mean_c.cursor()
                for t in range(0, int(TIME_UPPER_BOUND / TIME_INTERVAL)):
                    hz_values = res_mean_spct[t]
                    insert_query_string = f"INSERT INTO {lang} VALUES ("
                    for hz in hz_values:
                        insert_query_string += f"{hz}, "
                    insert_query_string = insert_query_string[:-2] + ");"
                    mean_cur.execute(insert_query_string)
                mean_c.commit()
    print("DONE CREATING MEAN RESIDUAL SPECTROGRAMS")


def _create_residual_spectrograms(
    cleaned_data_path: str,
    digit_bound: int,
) -> None:
    """Creates residual spectrograms and stores the results.

    Args:
        cleaned_data_path (str): The file path of the folder containing the
                                 cleaned pipeline input.
        digit_bound (int): The largest digit spoken by any speaker in the
                           data set.

    Returns:
        None.
    """
    tas_con = connect(enums._DBType.TIME_ALIGNED_SMOOTHED_SPECTROGRAM.value + ".db")
    tas_cur = tas_con.cursor()
    mean_con = connect(enums._DBType.MEAN_SPECTROGRAM.value + ".db")
    mean_cur = mean_con.cursor()
    res_con = connect(enums._DBType.RESIDUAL_SPECTROGRAM.value + ".db")
    res_cur = res_con.cursor()
    for dgt in range(1, digit_bound + 1):
        print(f"NOW CREATING RESIDUAL SPECTROGRAMS FOR {dgt} WORDS")
        partial_path = f"{cleaned_data_path}/{dgt}_words"
        dgt_words = listdir(partial_path)
        for fl in dgt_words:
            if fl != ".DS_Store":
                spkr = fl[:4]
                lang = fl[:2]
                tas_spect_arr = np.array(
                    tas_cur.execute(f"SELECT * FROM {spkr}_word{dgt}").fetchall()
                )
                mean_array = np.array(
                    mean_cur.execute(f"SELECT * FROM {lang}_{dgt}").fetchall()
                )
                residual_array = tas_spect_arr - mean_array
                for t in range(0, int(TIME_UPPER_BOUND / TIME_INTERVAL)):
                    residual_values = residual_array[t]
                    insert_query_string = f"INSERT INTO {spkr}_word{dgt} VALUES ("
                    for val in residual_values:
                        insert_query_string += f"{val}, "
                    insert_query_string = insert_query_string[:-2] + ");"
                    res_cur.execute(insert_query_string)
        print(f"DONE CREATING RESIDUAL SPECTROGRAMS FOR {dgt} WORDS")
        print("")
    res_con.commit()


def _integrate_residual_mean_difference_time(
    time_1: int,
    time_2: int,
    arr: np.array,
    mean_arr: np.array,
) -> float:
    """Helper function for calculating covariance.

    Args:
        time_1 (int): The first time index.
        time_2 (int): The second time index.
        arr (np.array): A spectrogram array.
        mean_arr (np.array): A mean spectrogram array.

    Returns:
        float: The result of the integration.
    """
    return sum(
        [
            (
                (arr[time_1][omega] - mean_arr[time_1][omega])
                * (arr[time_2][omega] - mean_arr[time_2][omega])
            )
            * 1  # delta in this case equals 1, just here for consistency
            for omega in range(int(FREQ_UPPER_BOUND / FREQ_INTERVAL))
        ]
    )


def _integrate_residual_mean_difference_freq(
    omega_1: int,
    omega_2: int,
    arr: np.array,
    mean_arr: np.array,
) -> float:
    """Helper function for calculating covariance.

    Args:
        omega_1 (int): The first frequency index.
        omega_2 (int): The second frequency index.
        arr (np.array): A spectrogram array.
        mean_arr (np.array): A mean spectrogram array.

    Returns:
        float: The result of the integration.
    """
    return sum(
        [
            (
                (arr[t][omega_1] - mean_arr[t][omega_1])
                * (arr[t][omega_2] - mean_arr[t][omega_2])
            )
            * 1  # delta in this case equals 1, just here for consistency
            for t in range(int(TIME_UPPER_BOUND / TIME_INTERVAL))
        ]
    )


def _calculate_covariance(
    language: enums.RomanceLanguages,
    cov_type="",
) -> None:
    """Calculates a covariance structure.

    Args:
        language (enums.RomanceLanguages): The language for which to calculate
                                           the covariance structure.
        cov_type (str, optional): The type of covariance to calculate. Defaults
                                  to "".

    Returns:
        None.
    """
    # type checks
    if type(language) != enums.RomanceLanguages:
        raise Exception(f"language: {language} is not a valid RomanceLanguages enum")

    if language == enums.RomanceLanguages.ITALIAN:
        speakers = enums.ItalianSpeakers
    elif language == enums.RomanceLanguages.FRENCH:
        speakers = enums.FrenchSpeakers
    elif language == enums.RomanceLanguages.PORTUGUESE:
        speakers = enums.PortugueseSpeakers
    elif language == enums.RomanceLanguages.AMERICAN_SPANISH:
        speakers = enums.AmericanSpanishSpeakers
    elif language == enums.RomanceLanguages.IBERIAN_SPANISH:
        speakers = enums.IberianSpanishSpeakers

    language = language.value

    if cov_type == "time":
        covariance = np.zeros(
            (
                int(TIME_UPPER_BOUND / TIME_INTERVAL),
                int(TIME_UPPER_BOUND / TIME_INTERVAL),
            )
        )
        UPPER_BOUND = TIME_UPPER_BOUND
        INTERVAL = TIME_INTERVAL
    elif cov_type == "freq":
        covariance = np.zeros(
            (
                int(FREQ_UPPER_BOUND / FREQ_INTERVAL),
                int(FREQ_UPPER_BOUND / FREQ_INTERVAL),
            )
        )
        UPPER_BOUND = FREQ_UPPER_BOUND
        INTERVAL = FREQ_INTERVAL
    else:
        raise Exception(
            f"cov_type: {cov_type} invalid. Choose either 'time' or 'freq'."
        )

    conn = connect("mean_residual_spectrogram.db")
    cur = conn.cursor()

    mean_arr = np.array(cur.execute(f"SELECT * FROM {language}").fetchall())

    conn = connect(f"{enums._DBType.RESIDUAL_SPECTROGRAM.value}.db")
    cur = conn.cursor()

    # Taking covariance over all speaker/word combos in a given language
    language_arrays = []

    for dgt in range(1, 11):
        for spkr in speakers:
            arr = np.array(
                cur.execute(
                    f"SELECT * FROM {language}{spkr.value}_word{dgt}"
                ).fetchall()
            )
            if arr.shape != (0,):
                language_arrays.append(arr)

    n_L = len(language_arrays)

    if cov_type == "time":
        for time_1 in np.arange(0, UPPER_BOUND, INTERVAL):
            for time_2 in np.arange(0, UPPER_BOUND, INTERVAL):
                cov = (1 / (n_L - 1)) * sum(
                    [
                        _integrate_residual_mean_difference_time(
                            int(time_1 / INTERVAL),
                            int(time_2 / INTERVAL),
                            arr,
                            mean_arr,
                        )
                        for arr in language_arrays
                    ]
                )

                covariance[int(time_1 / INTERVAL)][int(time_2 / INTERVAL)] = cov

    elif cov_type == "freq":
        for omega_1 in np.arange(0, UPPER_BOUND, INTERVAL):
            for omega_2 in np.arange(0, UPPER_BOUND, INTERVAL):
                cov = (1 / (n_L - 1)) * sum(
                    [
                        _integrate_residual_mean_difference_freq(
                            int(omega_1 / INTERVAL),
                            int(omega_2 / INTERVAL),
                            arr,
                            mean_arr,
                        )
                        for arr in language_arrays
                    ]
                )

                covariance[int(omega_1 / INTERVAL)][int(omega_2 / INTERVAL)] = cov

    covariance /= sqrt(np.trace(covariance))

    conn = connect(f"{cov_type}_covariance.db")
    cur = conn.cursor()

    for t in range(0, int(UPPER_BOUND / INTERVAL)):
        vals = covariance[t]
        insert_query_string = f"INSERT INTO {language} VALUES ("
        for val in vals:
            insert_query_string += f"{val}, "
        insert_query_string = insert_query_string[:-2] + ");"
        cur.execute(insert_query_string)
    conn.commit()


def _calculate_covariances() -> None:
    """Calculates the time and frequency covariance for all languages.

    Returns:
        None.
    """
    for lang in [
        l
        for l in enums.RomanceLanguages
        if l != enums.RomanceLanguages._BRAZILIAN_PORTUGUESE
        and l != enums.RomanceLanguages._LUSITANIAN_PORTUGUESE
    ]:
        print(f"NOW CALCULATING TIME COVARIANCE FOR LANGUAGE: {lang.value}")
        _calculate_covariance(language=lang, cov_type="time")
        print(f"DONE CALCULATING TIME COVARIANCE FOR LANGUAGE: {lang.value}")

        print("")

        print(f"NOW CALCULATING FREQ COVARIANCE FOR LANGUAGE: {lang.value}")
        _calculate_covariance(language=lang, cov_type="freq")
        print(f"DONE CALCULATING FREQ COVARIANCE FOR LANGUAGE: {lang.value}")

        print("")


def run_pipeline(
    cleaned_data_path: str,
    digit_bound: int,
) -> None:
    """Runs the preprocessing pipeline.

    Args:
        cleaned_data_path (str): The file path of the folder containing the
                                 cleaned pipeline input.
        digit_bound (int): The upper (inclusive) bound on the spoken digits
                           to be preprocessed. For instance, a digit_bound
                           of 7 would process all sound files in which the
                           speaker pronounces a word corresponding to a digit
                           1 through 7.

    Returns:
        None.
    """
    # checks that digit bound is valid
    if digit_bound not in range(1, 11):
        raise Exception(
            f"digit_bound: {digit_bound} is invalid. Must be an integer \
                          in the range [1, 10] (inclusive)."
        )

    print("")
    # calculate/store raw spectrograms
    _store_raw_spectrograms(
        cleaned_data_path=cleaned_data_path, digit_bound=digit_bound
    )

    # calculate/store pairwise warping functions for each pair of speakers of a given word
    for dgt in range(1, digit_bound + 1):
        dgt_files = listdir(f"{cleaned_data_path}/{dgt}_words")
        for fl_a in dgt_files:
            for fl_b in dgt_files:
                if fl_a != fl_b and fl_a != ".DS_Store" and fl_b != ".DS_Store":
                    speaker_1 = fl_a[:4]
                    speaker_2 = fl_b[:4]
                    _store_pairwise_warping_function(dgt, speaker_1, speaker_2)

    # calculate/store global inverse warping functions for each speaker/word pair
    for dgt in range(1, digit_bound + 1):
        dgt_files = listdir(f"{cleaned_data_path}/{dgt}_words")
        for fl in dgt_files:
            if fl != ".DS_Store":
                speaker = fl[:4]
                _store_global_inverse_warping_function(dgt, speaker, cleaned_data_path)

    # time align raw spectrograms
    _time_align_raw_spectrograms(cleaned_data_path, digit_bound)

    # smooth time aligned spectrograms
    _smooth_time_aligned_spectrograms(cleaned_data_path, digit_bound)
