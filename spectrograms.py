from sqlite3 import connect, OperationalError
import autograd.numpy as np
import utils
import enums
from typing import Dict, List, Tuple, Callable


def get_raw_spectrogram_array(
    language: enums.RomanceLanguages,
    speaker: enums.Speakers,
    digit: int,
) -> np.array:
    """Gets the raw spectrogram array corresponding to the given speaker and digit.

    Args:
        language (enums.RomanceLanguages): The language corresponding to the desired spectrogram array.
        speaker (enums.Speakers): The speaker corresponding to the desired spectrogram array.
        digit (int): The spoken digit corresponding to the desired spectrogram array. Should be an
                     integer in the range [1, 10] (inclusive).

    Returns:
        np.array: A NumPy array containing the desired raw spectrogram array.
    """
    return utils._get_array(
        f"{enums._DBType.RAW_SPECTROGRAM.value}.db",
        language,
        speaker,
        digit,
    )


def get_time_aligned_spectrogram_array(
    language: enums.RomanceLanguages,
    speaker: enums.Speakers,
    digit: int,
) -> np.array:
    """Gets the time aligned spectrogram array corresponding to the given speaker and digit.

    Args:
        language (enums.RomanceLanguages): The language corresponding to the desired spectrogram array.
        speaker (enums.Speakers): The speaker corresponding to the desired spectrogram array.
        digit (int): The spoken digit corresponding to the desired spectrogram array. Should be an
                     integer in the range [1, 10] (inclusive).

    Returns:
        np.array: A NumPy array containing the desired time aligned spectrogram array.
    """
    return utils._get_array(
        f"{enums._DBType.TIME_ALIGNED_SPECTROGRAM.value}.db",
        language,
        speaker,
        digit,
    )


def get_time_aligned_smoothed_spectrogram_array(
    language: enums.RomanceLanguages,
    speaker: enums.Speakers,
    digit: int,
) -> np.array:
    """Gets the time aligned and smoothed spectrogram array corresponding to the given speaker and digit.

    Args:
        language (enums.RomanceLanguages): The language corresponding to the desired spectrogram array.
        speaker (enums.Speakers): The speaker corresponding to the desired spectrogram array.
        digit (int): The spoken digit corresponding to the desired spectrogram array. Should be an
                     integer 1 through 10 (inclusive).

    Returns:
        np.array: A NumPy array containing the desired time aligned and smoothed spectrogram array.
    """
    return utils._get_array(
        f"{enums._DBType.TIME_ALIGNED_SMOOTHED_SPECTROGRAM.value}.db",
        language,
        speaker,
        digit,
    )


def get_mean_spectrogram_array(
    language: enums.RomanceLanguages,
    digit: int,
):
    """Gets the mean spectrogram array corresponding to the given language and digit.

    Args:
        language (enums.RomanceLanguages): The language corresponding to the desired array.
                                           Should be a RomanceLanguages enum.
        digit (int): The digit corresponding to the desired array.

    Returns:
        np.array: A NumPy array containing the desired mean spectrogram array.
    """
    # type checks
    if type(language) != enums.RomanceLanguages:
        raise Exception(f"language: {language} is not a valid RomanceLanguages enum")

    language = language.value

    with connect(f"{enums._DBType.MEAN_SPECTROGRAM.value}.db") as c:
        cur = c.cursor()
        try:
            cur.execute(f"SELECT * FROM {language}_{digit}").fetchall()
        except OperationalError as oe:
            if "no such table" in str(oe):
                raise Exception(
                    f'table: "{language}_{digit}" does not exist in this table.'
                )
        arr = np.array(cur.execute(f"SELECT * FROM {language}_{digit}").fetchall())

    return arr


def get_residual_spectrogram_array(
    language: enums.RomanceLanguages,
    speaker: enums.Speakers,
    digit: int,
) -> np.array:
    """Gets the residual spectrogram array corresponding to the given speaker and digit.

    Args:
        language (enums.RomanceLanguages): The language corresponding to the desired spectrogram array.
        speaker (enums.Speakers): The speaker corresponding to the desired spectrogram array.
        digit (int): The spoken digit corresponding to the desired spectrogram array. Should be an
                     integer 1 through 10 (inclusive).

    Returns:
        np.array: A NumPy array containing the desired residual spectrogram array.
    """
    return utils._get_array(
        f"{enums._DBType.RESIDUAL_SPECTROGRAM.value}.db",
        language,
        speaker,
        digit,
    )


def get_mean_residual_spectrogram_array(
    language: enums.RomanceLanguages,
) -> np.array:
    """Gets the mean residual spectrogram array corresponding to the given speaker and digit.

    Args:
        language (RomanceLanguages): The language corresponding to the desired mean residual
                                     spectrogram array.

    Returns:
        np.array: A NumPy array containing the desired mean residual spectrogram array.
    """
    # type checks
    if type(language) != enums.RomanceLanguages:
        raise Exception(f"language: {language} is not a valid RomanceLanguages enum")

    language = language.value

    with connect("mean_residual_spectrogram.db") as c:
        cur = c.cursor()
        try:
            cur.execute(f"SELECT * FROM {language}").fetchall()
        except OperationalError as oe:
            if "no such table" in str(oe):
                raise Exception(f'language: "{language}" does not exist in this table.')
        arr = np.array(cur.execute(f"SELECT * FROM {language}").fetchall())

    return arr


def get_digit_spectrogram_arrays(
    digit: int,
    db_type: enums._DBType = enums._DBType.TIME_ALIGNED_SMOOTHED_SPECTROGRAM.value,
) -> List[Tuple[str, np.array]]:
    """Gets the spectrograms for all pronunciations of a given digit.

    Args:
        digit (int): The digit in question.
        db_type (enums._DBType, optional): The database from which to select the spectrograms. Defaults to
                                           enums._DBType.TIME_ALIGNED_SMOOTHED_SPECTROGRAM.value.

    Returns:
        List[Tuple[str, np.array]]: A list of tuples, where each tuple consists of a label and a
                                    spectrogram.
    """
    with connect(f"{db_type}.db") as c:
        cur = c.cursor()
        res = cur.execute(
            f"SELECT name FROM sqlite_schema WHERE type='table' \
              AND name NOT LIKE 'sqlite_%';"
        ).fetchall()

        arrs = [
            (
                x[0],
                utils._get_array(
                    db_str=f"{db_type}.db",
                    language=x[0][:2],
                    speaker=x[0][2:4],
                    digit=digit,
                    override=True,
                ),
            )
            for x in res
            # Generalizing the slice for digit 10
            if x[0][-((digit // 10) + 1) :] == f"{digit}"
        ]

        # NOTE: Below is a fix to get around some apparently missing data
        return [x for x in arrs if x[1] != np.array([])]


def get_digit_mean_spectrogram_array(
    digit: int,
) -> np.array:
    """Gets the mean spectrogram across all pronunciations, regardless of language, for a given digit.

    Args:
        digit (int): The digit corresponding to the mean being calculated.

    Returns:
        np.array: The mean spectrogram across all pronunciations, regardless of language, for a given digit.
    """
    arrs = [arr for _, arr in get_digit_spectrogram_arrays(digit=digit)]
    return sum(arrs) / len(arrs)


def test_covariance_distance(
    metric: Callable,
) -> Tuple[Dict[str, float]]:
    """TODO: Tests

    Args:
        metric (Callable): The distance metric to test.

    Returns:
        Tuple[Dict[str, float]]: A tuple of dictionaries containing the distances
        between the languages' time and frequency covariance structures.
    """
    dict_time = {
        enums.RomanceLanguages.FRENCH.value: {},
        enums.RomanceLanguages.ITALIAN.value: {},
        enums.RomanceLanguages.PORTUGUESE.value: {},
        enums.RomanceLanguages.AMERICAN_SPANISH.value: {},
        enums.RomanceLanguages.IBERIAN_SPANISH.value: {},
    }

    dict_freq = {
        enums.RomanceLanguages.FRENCH.value: {},
        enums.RomanceLanguages.ITALIAN.value: {},
        enums.RomanceLanguages.PORTUGUESE.value: {},
        enums.RomanceLanguages.AMERICAN_SPANISH.value: {},
        enums.RomanceLanguages.IBERIAN_SPANISH.value: {},
    }

    for lang1 in enums.RomanceLanguages:
        if (
            lang1 != enums.RomanceLanguages._BRAZILIAN_PORTUGUESE
            and lang1 != enums.RomanceLanguages._LUSITANIAN_PORTUGUESE
        ):
            for lang2 in enums.RomanceLanguages:
                if (
                    lang2 != enums.RomanceLanguages._BRAZILIAN_PORTUGUESE
                    and lang2 != enums.RomanceLanguages._LUSITANIAN_PORTUGUESE
                ):
                    dict_time[lang1.value][lang2.value] = metric(
                        lang1=lang1,
                        lang2=lang2,
                        cov_type=enums.CovarianceType.TIME,
                    )
                    dict_freq[lang1.value][lang2.value] = metric(
                        lang1=lang1,
                        lang2=lang2,
                        cov_type=enums.CovarianceType.FREQUENCY,
                    )

    return (dict_time, dict_freq)
