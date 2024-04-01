from sqlite3 import connect, OperationalError
import autograd.numpy as np
from typing import Dict, List, Tuple
import enums
import pymanopt
from scipy.linalg import sqrtm, eig, svd
from sklearn.preprocessing import StandardScaler
import pipeline
from scipy.spatial.distance import euclidean

TIME_STEPS = int(pipeline._get_TIME_UPPER_BOUND() / pipeline._get_TIME_INTERVAL())
FREQ_STEPS = int(pipeline._get_FREQ_UPPER_BOUND() / pipeline._get_FREQ_INTERVAL())

FINITE_DIFFERENCE_TIME = pipeline._get_TIME_INTERVAL()
FINITE_DIFFERENCE_FREQ = pipeline._get_FREQ_INTERVAL()

INTERP_NO = 14

# For each digit, the speakers in each language saying that digit
digit_speakers = {
    1: {
        enums.RomanceLanguages.FRENCH: [
            enums.FrenchSpeakers.SPEAKER_1,
            enums.FrenchSpeakers.SPEAKER_2,
            enums.FrenchSpeakers.SPEAKER_3,
            enums.FrenchSpeakers.SPEAKER_5,
            enums.FrenchSpeakers.SPEAKER_6,
            enums.FrenchSpeakers.SPEAKER_7,
            enums.FrenchSpeakers.SPEAKER_8,
        ],
        enums.RomanceLanguages.ITALIAN: [
            enums.ItalianSpeakers.SPEAKER_1,
            enums.ItalianSpeakers.SPEAKER_2,
            enums.ItalianSpeakers.SPEAKER_3,
            enums.ItalianSpeakers.SPEAKER_4,
            enums.ItalianSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.PORTUGUESE: [
            enums.PortugueseSpeakers.SPEAKER_1,
            enums.PortugueseSpeakers.SPEAKER_2,
            enums.PortugueseSpeakers.SPEAKER_3,
            enums.PortugueseSpeakers.SPEAKER_5,
            enums.PortugueseSpeakers.SPEAKER_6,
        ],
        enums.RomanceLanguages.AMERICAN_SPANISH: [
            enums.AmericanSpanishSpeakers.SPEAKER_1,
            enums.AmericanSpanishSpeakers.SPEAKER_2,
            enums.AmericanSpanishSpeakers.SPEAKER_3,
            enums.AmericanSpanishSpeakers.SPEAKER_4,
            enums.AmericanSpanishSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.IBERIAN_SPANISH: [
            enums.IberianSpanishSpeakers.SPEAKER_2,
            enums.IberianSpanishSpeakers.SPEAKER_3,
            enums.IberianSpanishSpeakers.SPEAKER_5,
            enums.IberianSpanishSpeakers.SPEAKER_6,
            enums.IberianSpanishSpeakers.SPEAKER_7,
        ],
    },
    2: {
        enums.RomanceLanguages.FRENCH: [
            enums.FrenchSpeakers.SPEAKER_1,
            enums.FrenchSpeakers.SPEAKER_2,
            enums.FrenchSpeakers.SPEAKER_3,
            enums.FrenchSpeakers.SPEAKER_5,
            enums.FrenchSpeakers.SPEAKER_6,
            enums.FrenchSpeakers.SPEAKER_8,
        ],
        enums.RomanceLanguages.ITALIAN: [
            enums.ItalianSpeakers.SPEAKER_1,
            enums.ItalianSpeakers.SPEAKER_2,
            enums.ItalianSpeakers.SPEAKER_3,
            enums.ItalianSpeakers.SPEAKER_4,
            enums.ItalianSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.PORTUGUESE: [
            enums.PortugueseSpeakers.SPEAKER_1,
            enums.PortugueseSpeakers.SPEAKER_2,
            enums.PortugueseSpeakers.SPEAKER_3,
            enums.PortugueseSpeakers.SPEAKER_4,
            enums.PortugueseSpeakers.SPEAKER_5,
            enums.PortugueseSpeakers.SPEAKER_6,
        ],
        enums.RomanceLanguages.AMERICAN_SPANISH: [
            enums.AmericanSpanishSpeakers.SPEAKER_1,
            enums.AmericanSpanishSpeakers.SPEAKER_2,
            enums.AmericanSpanishSpeakers.SPEAKER_3,
            enums.AmericanSpanishSpeakers.SPEAKER_4,
            enums.AmericanSpanishSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.IBERIAN_SPANISH: [
            enums.IberianSpanishSpeakers.SPEAKER_1,
            enums.IberianSpanishSpeakers.SPEAKER_2,
            enums.IberianSpanishSpeakers.SPEAKER_3,
            enums.IberianSpanishSpeakers.SPEAKER_4,
            enums.IberianSpanishSpeakers.SPEAKER_5,
            enums.IberianSpanishSpeakers.SPEAKER_6,
            enums.IberianSpanishSpeakers.SPEAKER_7,
        ],
    },
    3: {
        enums.RomanceLanguages.FRENCH: [
            enums.FrenchSpeakers.SPEAKER_1,
            enums.FrenchSpeakers.SPEAKER_2,
            enums.FrenchSpeakers.SPEAKER_3,
            enums.FrenchSpeakers.SPEAKER_5,
            enums.FrenchSpeakers.SPEAKER_6,
            enums.FrenchSpeakers.SPEAKER_7,
            enums.FrenchSpeakers.SPEAKER_8,
        ],
        enums.RomanceLanguages.ITALIAN: [
            enums.ItalianSpeakers.SPEAKER_1,
            enums.ItalianSpeakers.SPEAKER_2,
            enums.ItalianSpeakers.SPEAKER_3,
            enums.ItalianSpeakers.SPEAKER_4,
            enums.ItalianSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.PORTUGUESE: [
            enums.PortugueseSpeakers.SPEAKER_1,
            enums.PortugueseSpeakers.SPEAKER_2,
            enums.PortugueseSpeakers.SPEAKER_3,
            enums.PortugueseSpeakers.SPEAKER_4,
            enums.PortugueseSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.AMERICAN_SPANISH: [
            enums.AmericanSpanishSpeakers.SPEAKER_1,
            enums.AmericanSpanishSpeakers.SPEAKER_2,
            enums.AmericanSpanishSpeakers.SPEAKER_3,
            enums.AmericanSpanishSpeakers.SPEAKER_4,
            enums.AmericanSpanishSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.IBERIAN_SPANISH: [
            enums.IberianSpanishSpeakers.SPEAKER_2,
            enums.IberianSpanishSpeakers.SPEAKER_3,
            enums.IberianSpanishSpeakers.SPEAKER_5,
            enums.IberianSpanishSpeakers.SPEAKER_6,
            enums.IberianSpanishSpeakers.SPEAKER_7,
        ],
    },
    4: {
        enums.RomanceLanguages.FRENCH: [
            enums.FrenchSpeakers.SPEAKER_1,
            enums.FrenchSpeakers.SPEAKER_2,
            enums.FrenchSpeakers.SPEAKER_3,
            enums.FrenchSpeakers.SPEAKER_5,
            enums.FrenchSpeakers.SPEAKER_6,
            enums.FrenchSpeakers.SPEAKER_7,
            enums.FrenchSpeakers.SPEAKER_8,
        ],
        enums.RomanceLanguages.ITALIAN: [
            enums.ItalianSpeakers.SPEAKER_1,
            enums.ItalianSpeakers.SPEAKER_2,
            enums.ItalianSpeakers.SPEAKER_3,
            enums.ItalianSpeakers.SPEAKER_4,
            enums.ItalianSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.PORTUGUESE: [
            enums.PortugueseSpeakers.SPEAKER_1,
            enums.PortugueseSpeakers.SPEAKER_2,
            enums.PortugueseSpeakers.SPEAKER_3,
            enums.PortugueseSpeakers.SPEAKER_4,
            enums.PortugueseSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.AMERICAN_SPANISH: [
            enums.AmericanSpanishSpeakers.SPEAKER_1,
            enums.AmericanSpanishSpeakers.SPEAKER_2,
            enums.AmericanSpanishSpeakers.SPEAKER_3,
            enums.AmericanSpanishSpeakers.SPEAKER_4,
            enums.AmericanSpanishSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.IBERIAN_SPANISH: [
            enums.IberianSpanishSpeakers.SPEAKER_2,
            enums.IberianSpanishSpeakers.SPEAKER_3,
            enums.IberianSpanishSpeakers.SPEAKER_4,
            enums.IberianSpanishSpeakers.SPEAKER_5,
            enums.IberianSpanishSpeakers.SPEAKER_6,
            enums.IberianSpanishSpeakers.SPEAKER_7,
        ],
    },
    5: {
        enums.RomanceLanguages.FRENCH: [
            enums.FrenchSpeakers.SPEAKER_1,
            enums.FrenchSpeakers.SPEAKER_2,
            enums.FrenchSpeakers.SPEAKER_3,
            enums.FrenchSpeakers.SPEAKER_5,
            enums.FrenchSpeakers.SPEAKER_6,
            enums.FrenchSpeakers.SPEAKER_7,
            enums.FrenchSpeakers.SPEAKER_8,
        ],
        enums.RomanceLanguages.ITALIAN: [
            enums.ItalianSpeakers.SPEAKER_1,
            enums.ItalianSpeakers.SPEAKER_2,
            enums.ItalianSpeakers.SPEAKER_3,
            enums.ItalianSpeakers.SPEAKER_4,
            enums.ItalianSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.PORTUGUESE: [
            enums.PortugueseSpeakers.SPEAKER_1,
            enums.PortugueseSpeakers.SPEAKER_2,
            enums.PortugueseSpeakers.SPEAKER_3,
            enums.PortugueseSpeakers.SPEAKER_4,
            enums.PortugueseSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.AMERICAN_SPANISH: [
            enums.AmericanSpanishSpeakers.SPEAKER_1,
            enums.AmericanSpanishSpeakers.SPEAKER_2,
            enums.AmericanSpanishSpeakers.SPEAKER_3,
            enums.AmericanSpanishSpeakers.SPEAKER_4,
            enums.AmericanSpanishSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.IBERIAN_SPANISH: [
            enums.IberianSpanishSpeakers.SPEAKER_3,
            enums.IberianSpanishSpeakers.SPEAKER_4,
            enums.IberianSpanishSpeakers.SPEAKER_5,
            enums.IberianSpanishSpeakers.SPEAKER_6,
            enums.IberianSpanishSpeakers.SPEAKER_7,
        ],
    },
    6: {
        enums.RomanceLanguages.FRENCH: [
            enums.FrenchSpeakers.SPEAKER_1,
            enums.FrenchSpeakers.SPEAKER_2,
            enums.FrenchSpeakers.SPEAKER_3,
            enums.FrenchSpeakers.SPEAKER_5,
            enums.FrenchSpeakers.SPEAKER_6,
            enums.FrenchSpeakers.SPEAKER_7,
            enums.FrenchSpeakers.SPEAKER_8,
        ],
        enums.RomanceLanguages.ITALIAN: [
            enums.ItalianSpeakers.SPEAKER_1,
            enums.ItalianSpeakers.SPEAKER_2,
            enums.ItalianSpeakers.SPEAKER_3,
            enums.ItalianSpeakers.SPEAKER_4,
            enums.ItalianSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.PORTUGUESE: [
            enums.PortugueseSpeakers.SPEAKER_1,
            enums.PortugueseSpeakers.SPEAKER_2,
            enums.PortugueseSpeakers.SPEAKER_3,
            enums.PortugueseSpeakers.SPEAKER_4,
            enums.PortugueseSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.AMERICAN_SPANISH: [
            enums.AmericanSpanishSpeakers.SPEAKER_1,
            enums.AmericanSpanishSpeakers.SPEAKER_2,
            enums.AmericanSpanishSpeakers.SPEAKER_3,
            enums.AmericanSpanishSpeakers.SPEAKER_4,
            enums.AmericanSpanishSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.IBERIAN_SPANISH: [
            enums.IberianSpanishSpeakers.SPEAKER_1,
            enums.IberianSpanishSpeakers.SPEAKER_2,
            enums.IberianSpanishSpeakers.SPEAKER_3,
            enums.IberianSpanishSpeakers.SPEAKER_5,
            enums.IberianSpanishSpeakers.SPEAKER_6,
            enums.IberianSpanishSpeakers.SPEAKER_7,
        ],
    },
    7: {
        enums.RomanceLanguages.FRENCH: [
            enums.FrenchSpeakers.SPEAKER_1,
            enums.FrenchSpeakers.SPEAKER_2,
            enums.FrenchSpeakers.SPEAKER_3,
            enums.FrenchSpeakers.SPEAKER_5,
            enums.FrenchSpeakers.SPEAKER_6,
            enums.FrenchSpeakers.SPEAKER_7,
            enums.FrenchSpeakers.SPEAKER_8,
        ],
        enums.RomanceLanguages.ITALIAN: [
            enums.ItalianSpeakers.SPEAKER_1,
            enums.ItalianSpeakers.SPEAKER_2,
            enums.ItalianSpeakers.SPEAKER_3,
            enums.ItalianSpeakers.SPEAKER_4,
            enums.ItalianSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.PORTUGUESE: [
            enums.PortugueseSpeakers.SPEAKER_1,
            enums.PortugueseSpeakers.SPEAKER_2,
            enums.PortugueseSpeakers.SPEAKER_3,
            enums.PortugueseSpeakers.SPEAKER_4,
            enums.PortugueseSpeakers.SPEAKER_5,
            enums.PortugueseSpeakers.SPEAKER_6,
        ],
        enums.RomanceLanguages.AMERICAN_SPANISH: [
            enums.AmericanSpanishSpeakers.SPEAKER_1,
            enums.AmericanSpanishSpeakers.SPEAKER_2,
            enums.AmericanSpanishSpeakers.SPEAKER_4,
            enums.AmericanSpanishSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.IBERIAN_SPANISH: [
            enums.IberianSpanishSpeakers.SPEAKER_1,
            enums.IberianSpanishSpeakers.SPEAKER_2,
            enums.IberianSpanishSpeakers.SPEAKER_3,
            enums.IberianSpanishSpeakers.SPEAKER_5,
            enums.IberianSpanishSpeakers.SPEAKER_6,
            enums.IberianSpanishSpeakers.SPEAKER_7,
        ],
    },
    8: {
        enums.RomanceLanguages.FRENCH: [
            enums.FrenchSpeakers.SPEAKER_1,
            enums.FrenchSpeakers.SPEAKER_2,
            enums.FrenchSpeakers.SPEAKER_3,
            enums.FrenchSpeakers.SPEAKER_5,
            enums.FrenchSpeakers.SPEAKER_6,
            enums.FrenchSpeakers.SPEAKER_7,
            enums.FrenchSpeakers.SPEAKER_8,
        ],
        enums.RomanceLanguages.ITALIAN: [
            enums.ItalianSpeakers.SPEAKER_1,
            enums.ItalianSpeakers.SPEAKER_2,
            enums.ItalianSpeakers.SPEAKER_3,
            enums.ItalianSpeakers.SPEAKER_4,
            enums.ItalianSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.PORTUGUESE: [
            enums.PortugueseSpeakers.SPEAKER_1,
            enums.PortugueseSpeakers.SPEAKER_2,
            enums.PortugueseSpeakers.SPEAKER_3,
            enums.PortugueseSpeakers.SPEAKER_4,
            enums.PortugueseSpeakers.SPEAKER_5,
            enums.PortugueseSpeakers.SPEAKER_6,
        ],
        enums.RomanceLanguages.AMERICAN_SPANISH: [
            enums.AmericanSpanishSpeakers.SPEAKER_1,
            enums.AmericanSpanishSpeakers.SPEAKER_2,
            enums.AmericanSpanishSpeakers.SPEAKER_4,
            enums.AmericanSpanishSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.IBERIAN_SPANISH: [
            enums.IberianSpanishSpeakers.SPEAKER_2,
            enums.IberianSpanishSpeakers.SPEAKER_3,
            enums.IberianSpanishSpeakers.SPEAKER_5,
            enums.IberianSpanishSpeakers.SPEAKER_6,
            enums.IberianSpanishSpeakers.SPEAKER_7,
        ],
    },
    9: {
        enums.RomanceLanguages.FRENCH: [
            enums.FrenchSpeakers.SPEAKER_1,
            enums.FrenchSpeakers.SPEAKER_2,
            enums.FrenchSpeakers.SPEAKER_3,
            enums.FrenchSpeakers.SPEAKER_5,
            enums.FrenchSpeakers.SPEAKER_6,
            enums.FrenchSpeakers.SPEAKER_7,
            enums.FrenchSpeakers.SPEAKER_8,
        ],
        enums.RomanceLanguages.ITALIAN: [
            enums.ItalianSpeakers.SPEAKER_1,
            enums.ItalianSpeakers.SPEAKER_2,
            enums.ItalianSpeakers.SPEAKER_3,
            enums.ItalianSpeakers.SPEAKER_4,
            enums.ItalianSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.PORTUGUESE: [
            enums.PortugueseSpeakers.SPEAKER_1,
            enums.PortugueseSpeakers.SPEAKER_2,
            enums.PortugueseSpeakers.SPEAKER_3,
            enums.PortugueseSpeakers.SPEAKER_4,
            enums.PortugueseSpeakers.SPEAKER_5,
            enums.PortugueseSpeakers.SPEAKER_6,
        ],
        enums.RomanceLanguages.AMERICAN_SPANISH: [
            enums.AmericanSpanishSpeakers.SPEAKER_1,
            enums.AmericanSpanishSpeakers.SPEAKER_2,
            enums.AmericanSpanishSpeakers.SPEAKER_4,
            enums.AmericanSpanishSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.IBERIAN_SPANISH: [
            enums.IberianSpanishSpeakers.SPEAKER_2,
            enums.IberianSpanishSpeakers.SPEAKER_3,
            enums.IberianSpanishSpeakers.SPEAKER_5,
            enums.IberianSpanishSpeakers.SPEAKER_6,
            enums.IberianSpanishSpeakers.SPEAKER_7,
        ],
    },
    10: {
        enums.RomanceLanguages.FRENCH: [
            enums.FrenchSpeakers.SPEAKER_1,
            enums.FrenchSpeakers.SPEAKER_2,
            enums.FrenchSpeakers.SPEAKER_3,
            enums.FrenchSpeakers.SPEAKER_5,
            enums.FrenchSpeakers.SPEAKER_6,
            enums.FrenchSpeakers.SPEAKER_7,
            enums.FrenchSpeakers.SPEAKER_8,
        ],
        enums.RomanceLanguages.ITALIAN: [
            enums.ItalianSpeakers.SPEAKER_1,
            enums.ItalianSpeakers.SPEAKER_2,
            enums.ItalianSpeakers.SPEAKER_3,
            enums.ItalianSpeakers.SPEAKER_4,
            enums.ItalianSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.PORTUGUESE: [
            enums.PortugueseSpeakers.SPEAKER_1,
            enums.PortugueseSpeakers.SPEAKER_3,
            enums.PortugueseSpeakers.SPEAKER_4,
            enums.PortugueseSpeakers.SPEAKER_5,
            enums.PortugueseSpeakers.SPEAKER_6,
        ],
        enums.RomanceLanguages.AMERICAN_SPANISH: [
            enums.AmericanSpanishSpeakers.SPEAKER_1,
            enums.AmericanSpanishSpeakers.SPEAKER_2,
            enums.AmericanSpanishSpeakers.SPEAKER_4,
            enums.AmericanSpanishSpeakers.SPEAKER_5,
        ],
        enums.RomanceLanguages.IBERIAN_SPANISH: [
            enums.IberianSpanishSpeakers.SPEAKER_5,
            enums.IberianSpanishSpeakers.SPEAKER_6,
            enums.IberianSpanishSpeakers.SPEAKER_7,
        ],
    },
}

langs = [
    x.value
    for x in enums.RomanceLanguages
    if not (
        x == enums.RomanceLanguages._BRAZILIAN_PORTUGUESE
        or x == enums.RomanceLanguages._LUSITANIAN_PORTUGUESE
    )
]

colors = {
    "FR": "r",
    "IT": "b",
    "PO": "g",
    "SA": "y",
    "SI": "c",
    "cov_in": "fuchsia",
    "hyp_in": "blueviolet",
}


def _get_array(
    db_str: str,
    language: enums.RomanceLanguages,
    speaker: enums.Speakers,
    digit: int,
    override: bool = False,
) -> np.array:
    """Gets the spectrogram array corresponding to the given speaker and digit.

    Args:
        db_str (str): The string corresponding to the database being connected to.
        language (enums.RomanceLanguages): The language corresponding to the desired spectrogram array.
        speaker (Speakers): The speaker corresponding to the desired spectrogram array.
        digit (int): The spoken digit corresponding to the desired spectrogram array. Should be an
                     integer 1 through 10 (inclusive).
        override (bool): Whether to override error checks and parsing of enum values. Defaults to
                         False.

    Returns:
        np.array: A NumPy array containing the desired spectrogram array.
    """
    if not override:
        # type checks
        if type(language) != enums.RomanceLanguages:
            raise Exception(
                f"language: {language} is not a valid RomanceLanguages enum"
            )

        if (
            type(speaker) != enums.FrenchSpeakers
            and type(speaker) != enums.ItalianSpeakers
            and type(speaker) != enums.PortugueseSpeakers
            and type(speaker) != enums.AmericanSpanishSpeakers
            and type(speaker) != enums.IberianSpanishSpeakers
        ):
            raise Exception(f"speaker: {speaker} is not a valid ______Speakers enum")

        language = language.value
        speaker = speaker.value

    # error check ensuring valid digits (i.e., integers 1-10)
    if not (digit in range(1, 11)):
        raise Exception(f"digit: {digit} is not an integer 1 through 10.")

    with connect(db_str) as c:
        cur = c.cursor()
        try:
            cur.execute(f"SELECT * FROM {language}{speaker}_word{digit}").fetchall()
        # Error handling in case of bad language/speaker/digit combo
        except OperationalError as oe:
            if "no such table" in str(oe):
                raise Exception(
                    f'language: "{language}", speaker: "{speaker}", digit: "{digit}" triplet does not exist in this dataset.'
                )
        arr = np.array(
            cur.execute(f"SELECT * FROM {language}{speaker}_word{digit}").fetchall()
        )

    return arr


def _get_covariance(
    language: enums.RomanceLanguages,
    cov_type: str,
) -> np.array:
    """Returns the covariance structure for a given language.

    Args:
        language (RomanceLanguages): The language for which to get the covariance structure.
        cov_type (str): The type of the covariance structures in question.

    Raises:
        Exception: Errors out if an invalid language enum is passed.
        Exception: Errors out if the desired language covariance doesn't exist.

    Returns:
        np.array: A NumPy array containing the desired covariance structure.
    """
    # type checks
    if type(language) != enums.RomanceLanguages:
        raise Exception(f"language: {language} is not a valid RomanceLanguages enum")

    language = language.value

    with connect(f"{cov_type}_covariance.db") as c:
        cur = c.cursor()
        try:
            cur.execute(f"SELECT * FROM {language}").fetchall()
        except OperationalError as oe:
            if "no such table" in str(oe):
                raise Exception(
                    f'language: "{language}" does not exist in this database.'
                )

    conn = connect(f"{cov_type}_covariance.db")
    cur = conn.cursor()
    arr = np.array(cur.execute(f"SELECT * FROM {language}").fetchall())

    return arr


def _add_diagonal_epsilon(
    m: np.array,
    epsilon: float = 1e-10,
) -> np.array:
    """Adds a small value along the diagonal of a given square matrix.

    Args:
        m (np.array): The given matrix.
        epsilon (float, optional): The epsilon value to add. Defaults to 1e-10.

    Returns:
        np.array: The matrix with the value added along the diagonal.
    """
    n, _ = m.shape
    return m + epsilon * np.identity(n)


def _get_hs_norm(
    m: np.array,
) -> float:
    """Gets the Hilbert-Schmidt norm of a given matrix per the definition in Pigoli et al.'s
    ``Distances and inference for covariance operators'' (2014).

    Args:
        m (np.array): The given matrix.

    Returns:
        float: The Hilbert-Schmidt norm.
    """
    return np.trace(m.T @ m)


# NOTE: below function is being debugged
def _get_R(
    c1: np.array,
    c2: np.array,
) -> np.array:
    """Gets the unitary operator that minimizes the expression in the Procrustes distance definition
    from Pigoli et al.'s ``Distances and inference for covariance operators'' (2014).

    Args:
        c1 (np.array): The first covariance matrix.
        c2 (np.array): The second covariance matrix.

    Returns:
        np.array: A NumPy array representing the unitary operator.
    """
    c1_sqrt = sqrtm(c1)
    c2_sqrt = sqrtm(c2)

    dim, _ = c1_sqrt.shape

    manifold = pymanopt.manifolds.Stiefel(
        n=dim,
        p=dim,
    )

    @pymanopt.function.autograd(manifold)
    def cost(R):
        output = np.trace((c1_sqrt - c2_sqrt @ R).T @ (c1_sqrt - c2_sqrt @ R))
        if type(output) == np.complex128:
            return output
        else:
            print(type(output))
            return output._value

    problem = pymanopt.Problem(manifold, cost)

    optimizer = pymanopt.optimizers.SteepestDescent()
    result = optimizer.run(problem)

    return result.point


def _is_pos_def(
    m: np.array,
) -> bool:
    """Checks whether a given matrix is positive definite.

    Args:
        m (np.array): The given matrix.

    Returns:
        bool: Whether the matrix is positive definite.
    """
    vals, _ = eig(m)
    return False not in [i > 0 for i in vals]


def _is_symm(
    m: np.array,
) -> bool:
    """Checks whether a given matrix is symmetric.

    Args:
        m (np.array): The given matrix.

    Returns:
        bool: Whether the matrix is symmetric.
    """
    return np.allclose(m, m.T)


def _lower_rank_approx(
    m: np.array,
    n: int,
) -> np.array:
    """Returns a lower-rank approximation of a given matrix.

    Args:
        m (np.array): The given matrix.
        n (int): The dimension of the approximation.

    Returns:
        np.array: The lower-rank approximation of the matrix.
    """
    U, sigma, Vh = svd(m)

    S = np.zeros((m.shape[0], m.shape[1]))
    for i in range(n):
        S[i][i] = sigma[i]

    return U[:n, :] @ S @ Vh[:, :n]


def _standard_scale_array(
    arr: np.array,
) -> np.array:
    """Standard scales an array.

    Args:
        arr (np.array): The array to scale.

    Returns:
        np.array: The scaled array.
    """
    scaler = StandardScaler()
    scaler.fit(arr.flatten().reshape(-1, 1))
    scaled_arr = arr
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            scaled_arr[i, j] = scaler.transform(np.array([[arr[i, j]]]))

    return scaled_arr


def _poincare_distance_basic(
    u: np.array,
    v: np.array,
) -> float:
    """Finds the Poincare distance between two points in a Poincare disk. This implementation
    is drawn from the GitHub repo accompanying Klimovskaia et al.'s ``Poincare maps for analyzing
    complex hierarchies in single-cell data'' (2020).

    Args:
        u (np.array): The first point in the disk.
        v (np.array): The second point in the disk.

    Returns:
        float: The Poincare distance between the points.
    """
    squnorm = np.sum(u * u)
    sqvnorm = np.sum(v * v)
    sqdist = np.sum(np.power(u - v, 2))

    x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
    z = np.sqrt(np.power(x, 2) - 1)

    return np.log(x + z)


def _recover_theta(
    pt: np.array,
) -> float:
    """Recovers the theta parameter of a given 2D point's polar representation based
    on Euclidean coordinates.

    Args:
        pt (np.array): A 2D point.

    Returns:
        float: The recovered theta parameter.
    """
    x, y = pt

    theta = np.arctan(y / x)

    # If pt is in either of quadrants 2 or 3
    if x < 0:
        theta += np.pi

    # Convert to positive radians
    if theta < 0:
        theta += 2 * np.pi

    return theta


def _theta_cost_function(
    theta: np.array,
    offset: float,
    centroids: Dict[enums.RomanceLanguages, np.array],
    obj_r: Dict[enums.RomanceLanguages, float],
    obj_prior_theta: Dict[enums.RomanceLanguages, float],
) -> float:
    """The cost function minimized when aggregating over digit-wise Poincare disks.

    Args:
        theta (np.array): The free variable being optimized over.
        offset (float): An angular offset that centers the optimization about a particular
                        quadrant.
        centroids (Dict[enums.RomanceLanguages, np.array]): The language centroids with respect
                                                            to which the other disk is being
                                                            rotated.
        obj_r (Dict[enums.RomanceLanguages, float]): The r parameters of the polar representations
                                                     of the points in the disk being rotated.
        obj_prior_theta (Dict[enums.RomanceLanguages, float]): The theta parameters of the polar
                                                               representations of the points in the
                                                               disk being rotated.

    Returns:
        float: The cost of the rotation.
    """
    return sum(
        [
            _poincare_distance_basic(
                centroids[lang],
                np.array(
                    [
                        obj_r[lang]
                        * np.cos(obj_prior_theta[lang] + (offset + theta[0])),
                        obj_r[lang]
                        * np.sin(obj_prior_theta[lang] + (offset + theta[0])),
                    ]
                ),
            )
            ** 2
            for lang in langs
        ]
    )


def _interpolation_score(
    anchor_1: np.array,
    anchor_2: np.array,
    cov_interps: List[Tuple[np.array, np.array]],
    hyp_interps: np.array,
) -> float:
    """Returns the interpolation score for a covariance-based/hyperbolic interpolation pair

    Args:
        anchor_1 (np.array): The first endpoint of the interpolation.
        anchor_2 (np.array): The second endpoint of the interpolation.
        cov_interps (List[Tuple[np.array, np.array]]): A list of tuples, where each tuple
                                                       contains the embedded and
                                                       pre-embedded forms of the covariance-
                                                       based interpolations.
        hyp_interps (np.array): An array of hyperbolic interpolations.

    Returns:
        float: The interpolation score.
    """
    return (
        sum(
            [
                _poincare_distance_basic(cov_interps[i][0], hyp_interps[i]) ** 2
                for i in range(len(cov_interps))
            ]
        )
        / _poincare_distance_basic(anchor_1, anchor_2) ** 2
    )
