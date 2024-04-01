import pipeline
import enums
from sqlite3 import connect
from os import listdir


TIME_INTERVAL = pipeline._get_TIME_INTERVAL()
TIME_UPPER_BOUND = pipeline._get_TIME_UPPER_BOUND()
FREQ_INTERVAL = pipeline._get_FREQ_INTERVAL()
FREQ_UPPER_BOUND = pipeline._get_FREQ_UPPER_BOUND()


def _construct_db(
    db_type: enums._DBType,
) -> None:
    """Constructs the specified database in the current directory.

    Args:
        db_name (enums._DBType): The type of database to construct.

    Return:
        None.
    """
    db_type = db_type.value
    con = connect(f"{db_type}.db")
    cur = con.cursor()

    column_str = "".join(
        [f"Hz{i}, " for i in range(0, FREQ_UPPER_BOUND, FREQ_INTERVAL)]
    )[:-2]
    for lang in [
        l.value
        for l in enums.RomanceLanguages
        if l.value != enums.RomanceLanguages._BRAZILIAN_PORTUGUESE.value
        and l.value != enums.RomanceLanguages._LUSITANIAN_PORTUGUESE.value
    ]:
        if lang == enums.RomanceLanguages.FRENCH.value:
            for spkr in [s.value for s in enums.FrenchSpeakers]:
                for dgt in range(1, 11):
                    cur.execute(f"CREATE TABLE {lang}{spkr}_word{dgt}({column_str})")
        elif lang == enums.RomanceLanguages.ITALIAN.value:
            for spkr in [s.value for s in enums.ItalianSpeakers]:
                for dgt in range(1, 11):
                    cur.execute(f"CREATE TABLE {lang}{spkr}_word{dgt}({column_str})")
        elif lang == enums.RomanceLanguages.PORTUGUESE.value:
            for spkr in [s.value for s in enums._BrazilianPortugueseSpeakers] + [
                s.value for s in enums._LusitanianPortugueseSpeakers
            ]:
                for dgt in range(1, 11):
                    cur.execute(f"CREATE TABLE {lang}{spkr}_word{dgt}({column_str})")
        elif lang == enums.RomanceLanguages.AMERICAN_SPANISH.value:
            for spkr in [s.value for s in enums.AmericanSpanishSpeakers]:
                for dgt in range(1, 11):
                    cur.execute(f"CREATE TABLE {lang}{spkr}_word{dgt}({column_str})")
        elif lang == enums.RomanceLanguages.IBERIAN_SPANISH.value:
            for spkr in [s.value for s in enums.IberianSpanishSpeakers]:
                for dgt in range(1, 11):
                    cur.execute(f"CREATE TABLE {lang}{spkr}_word{dgt}({column_str})")


def construct_raw_spectrogram_db() -> None:
    """Constructs a raw spectrogram database in the current directory.

    Args:
        None.

    Return:
        None.
    """
    _construct_db(db_type=enums._DBType.RAW_SPECTROGRAM)


def construct_pairwise_warping_function_db(
    cleaned_data_path: str,
) -> None:
    """Constructs a pairwise warping function database in the current directory.

    Args:
        cleaned_data_path (str): The file path where the cleaned data resides, as
        constructed by clean_data.

    Return:
        None.
    """
    con = connect(f"pairwise_warping_function.db")
    cur = con.cursor()
    column_str = "".join(
        [f"Hz{i}, " for i in range(0, FREQ_UPPER_BOUND, FREQ_INTERVAL)]
    )[:-2]

    for dgt in range(1, 11):
        dgt_words = listdir(f"{cleaned_data_path}/{dgt}_words")
        for fl_a in dgt_words:
            for fl_b in dgt_words:
                if fl_a != fl_b and fl_a != ".DS_Store" and fl_b != ".DS_Store":
                    i_speaker = fl_a[:4]
                    j_speaker = fl_b[:4]
                    cur.execute(
                        f"CREATE TABLE {i_speaker}_{j_speaker}_{dgt}({column_str})"
                    )


def construct_global_inverse_warping_function_db() -> None:
    """Constructs a global inverse warping function database in the current directory.

    Args:
        None.

    Return:
        None.
    """
    _construct_db(db_type=enums._DBType.GLOBAL_INVERSE_WARPING_FUNCTION)


def construct_time_aligned_spectrogram_db() -> None:
    """Constructs a time aligned spectrogram database in the current directory.

    Args:
        None.

    Return:
        None.
    """
    _construct_db(db_type=enums._DBType.TIME_ALIGNED_SPECTROGRAM)


def construct_time_aligned_smoothed_spectrogram_db() -> None:
    """Constructs a time aligned and smoothed spectrogram database in the current directory.

    Args:
        None.

    Return:
        None.
    """
    _construct_db(db_type=enums._DBType.TIME_ALIGNED_SMOOTHED_SPECTROGRAM)


def construct_mean_spectrogram_db() -> None:
    """Constructs a mean spectrogram database in the current directory.

    Args:
        None.

    Return:
        None.
    """
    con = connect("mean_spectrogram.db")
    cur = con.cursor()

    column_str = "".join(
        [f"Hz{i}, " for i in range(0, FREQ_UPPER_BOUND, FREQ_INTERVAL)]
    )[:-2]
    for lang in [
        l.value
        for l in enums.RomanceLanguages
        if l.value != enums.RomanceLanguages._BRAZILIAN_PORTUGUESE.value
        and l.value != enums.RomanceLanguages._LUSITANIAN_PORTUGUESE.value
    ]:
        for digit in range(1, 11):
            if lang == enums.RomanceLanguages.FRENCH.value:
                cur.execute(f"CREATE TABLE {lang}_{digit}({column_str})")
            elif lang == enums.RomanceLanguages.ITALIAN.value:
                cur.execute(f"CREATE TABLE {lang}_{digit}({column_str})")
            elif lang == enums.RomanceLanguages.PORTUGUESE.value:
                cur.execute(f"CREATE TABLE {lang}_{digit}({column_str})")
            elif lang == enums.RomanceLanguages.AMERICAN_SPANISH.value:
                cur.execute(f"CREATE TABLE {lang}_{digit}({column_str})")
            elif lang == enums.RomanceLanguages.IBERIAN_SPANISH.value:
                cur.execute(f"CREATE TABLE {lang}_{digit}({column_str})")


def construct_residual_spectrogram_db() -> None:
    """Constructs a residual spectrogram database in the current directory.

    Args:
        None.

    Return:
        None.
    """
    _construct_db(db_type=enums._DBType.RESIDUAL_SPECTROGRAM)


def construct_mean_residual_spectrogram_db() -> None:
    """Constructs a mean residual spectrogram database in the current directory.

    Args:
        None.

    Return:
        None.
    """
    con = connect("mean_residual_spectrogram.db")
    cur = con.cursor()

    column_str = "".join(
        [f"Hz{i}, " for i in range(0, FREQ_UPPER_BOUND, FREQ_INTERVAL)]
    )[:-2]
    for lang in [
        l.value
        for l in enums.RomanceLanguages
        if l.value != enums.RomanceLanguages._BRAZILIAN_PORTUGUESE.value
        and l.value != enums.RomanceLanguages._LUSITANIAN_PORTUGUESE.value
    ]:
        if lang == enums.RomanceLanguages.FRENCH.value:
            cur.execute(f"CREATE TABLE {lang}({column_str})")
        elif lang == enums.RomanceLanguages.ITALIAN.value:
            cur.execute(f"CREATE TABLE {lang}({column_str})")
        elif lang == enums.RomanceLanguages.PORTUGUESE.value:
            cur.execute(f"CREATE TABLE {lang}({column_str})")
        elif lang == enums.RomanceLanguages.AMERICAN_SPANISH.value:
            cur.execute(f"CREATE TABLE {lang}({column_str})")
        elif lang == enums.RomanceLanguages.IBERIAN_SPANISH.value:
            cur.execute(f"CREATE TABLE {lang}({column_str})")


def _construct_covariance_db(cov_type="") -> None:
    """Constructs a time covariance database in the current directory.

    Args:
        cov_type (str): The type of covariance to be calculated.

    Return:
        None.
    """
    if cov_type == "time":
        UPPER_BOUND = TIME_UPPER_BOUND
        INTERVAL = TIME_INTERVAL
    elif cov_type == "freq":
        UPPER_BOUND = FREQ_UPPER_BOUND
        INTERVAL = FREQ_INTERVAL
    else:
        raise Exception(
            f"cov_type: {cov_type} is invalid. Must be either 'time' or 'freq'."
        )

    con = connect(f"{cov_type}_covariance.db")
    cur = con.cursor()

    # Construct inverse warping function tables
    column_str = "".join([f"idx_{i}, " for i in range(int(UPPER_BOUND / INTERVAL))])[
        :-2
    ]
    for lang in [
        l.value
        for l in enums.RomanceLanguages
        if l.value != enums.RomanceLanguages._BRAZILIAN_PORTUGUESE.value
        and l.value != enums.RomanceLanguages._LUSITANIAN_PORTUGUESE.value
    ]:
        if lang == enums.RomanceLanguages.FRENCH.value:
            cur.execute(f"CREATE TABLE {lang}({column_str})")
        elif lang == enums.RomanceLanguages.ITALIAN.value:
            cur.execute(f"CREATE TABLE {lang}({column_str})")
        elif lang == enums.RomanceLanguages.PORTUGUESE.value:
            cur.execute(f"CREATE TABLE {lang}({column_str})")
        elif lang == enums.RomanceLanguages.AMERICAN_SPANISH.value:
            cur.execute(f"CREATE TABLE {lang}({column_str})")
        elif lang == enums.RomanceLanguages.IBERIAN_SPANISH.value:
            cur.execute(f"CREATE TABLE {lang}({column_str})")


def construct_time_covariance_db():
    """Constructs a time covariance database in the current directory.

    Args:
        None.

    Return:
        None.
    """
    _construct_covariance_db("time")


def construct_freq_covariance_db():
    """Constructs a frequency covariance database in the current directory.

    Args:
        None.

    Return:
        None.
    """
    _construct_covariance_db("freq")
