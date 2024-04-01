import spectrograms
import utils
import enums
import autograd.numpy as np
from scipy.linalg import svd, sqrtm, eig, inv
from typing import List, Tuple


def get_time_covariance(
    language: enums.RomanceLanguages,
) -> np.array:
    """Gets the time covariance structure corresponding to the given language.

    Args:
        language (enums.RomanceLanguages): The language corresponding to the
                                           desired covariance structure.

    Returns:
        np.array: A NumPy array containing the desired covariance structure.
    """
    return utils._get_covariance(language=language, cov_type="time")


def get_freq_covariance(
    language: enums.RomanceLanguages,
) -> np.array:
    """Gets the frequency covariance structure corresponding to the given language.

    Args:
        language (enums.RomanceLanguages): The language corresponding to the
                                           desired covariance structure.

    Returns:
        np.array: A NumPy array containing the desired covariance structure.
    """
    return utils._get_covariance(language=language, cov_type="freq")


def get_procrustes_distance_R_inf(
    lang1: enums.RomanceLanguages,
    lang2: enums.RomanceLanguages,
    cov_type: enums.CovarianceType,
) -> float:
    """Gets the Procrustes distance between two languages' covariance structures, where the
    distance is defined by minimizing the trace expression in Pigoli et al.'s ``Distances and
    inference for covariance operators'' (2014) over unitary operators.

    Args:
        lang1 (enums.RomanceLanguages): The first language.
        lang2 (enums.RomanceLanguages): The second language.
        cov_type (enums.CovarianceType): The type of the covariance structures in question.

    Returns:
        float: The Procrustes distance between the two languages' covariance structures.
    """
    # NOTE: the below implementation is currently broken
    raise NotImplementedError("Please use get_procrustes_distance_svd instead")

    if (
        cov_type != enums.CovarianceType.TIME
        and cov_type != enums.CovarianceType.FREQUENCY
    ):
        raise Exception(f"type: {cov_type} is invalid.")

    c1 = utils._get_covariance(language=lang1, cov_type=cov_type.value)
    c2 = utils._get_covariance(language=lang2, cov_type=cov_type.value)

    # R that minimizes ||c1^{1/2} - c2^{1/2}R||_{HS}^2
    R = utils._get_R(c1, c2)

    return utils._get_hs_norm(sqrtm(c1) - sqrtm(c2) @ R)


def get_procrustes_distance_svd(
    lang1: enums.RomanceLanguages,
    lang2: enums.RomanceLanguages,
    cov_type: enums.CovarianceType,
) -> float:
    """Gets the Procrustes distance between two languages' covariance structures, where the
    distance is defined as in Proposition 1 in (Pigoli et al., 2014).

    Args:
        lang1 (enums.RomanceLanguages): The first language.
        lang2 (enums.RomanceLanguages): The second language.
        cov_type (enums.CovarianceType): The type of the covariance structures in question.

    Returns:
        float: The Procrustes distance between the two languages' covariance structures.
    """
    if (
        cov_type != enums.CovarianceType.TIME
        and cov_type != enums.CovarianceType.FREQUENCY
    ):
        raise Exception(f"type: {cov_type} is invalid.")

    cov_type = cov_type.value

    c1 = utils._get_covariance(language=lang1, cov_type=cov_type)
    c2 = utils._get_covariance(language=lang2, cov_type=cov_type)

    # get l1, l2 via cholesky decomposition
    l1 = np.linalg.cholesky(utils._add_diagonal_epsilon(c1))
    l2 = np.linalg.cholesky(utils._add_diagonal_epsilon(c2))

    # get singular values of l2^T * l1
    _, sv, _ = svd(l2.T @ l1)

    return utils._get_hs_norm(l1) + utils._get_hs_norm(l2) - 2 * np.sum(sv)


def get_covariance_interpolations(
    lang1: enums.RomanceLanguages,
    lang2: enums.RomanceLanguages,
    cov_type: enums.CovarianceType,
    interp_no: int = 10,
) -> List[np.array]:
    """Gets the covariance structures along the geodesic between the covariance structures of the
    two given languages.

    Args:
        lang1 (RomanceLanguages): The first language.
        lang2 (RomanceLanguages): The second language.
        cov_type (CovarianceType): The type of the covariance structures in question.
        interp_no (int, optional): The number of interpolations to return. Defaults to 10.

    Returns:
        List[np.array]: A list of NumPy arrays representing covariance structures on the
                        geodesic.
    """
    # NOTE: the below implementation is currently broken
    raise NotImplementedError

    interp_vals = np.linspace(0, 1, interp_no)
    cov_type = cov_type.value

    c1 = utils._get_covariance(language=lang1, cov_type=cov_type)
    c2 = utils._get_covariance(language=lang2, cov_type=cov_type)

    c1_sqrt = sqrtm(c1)
    c2_sqrt = sqrtm(c2)

    # R that minimizes ||c1^{1/2} - c2^{1/2}R||_{HS}^2
    R = utils._get_R(c1, c2)

    f = lambda x: c1_sqrt + x * (c2_sqrt @ R - c1_sqrt)

    cov_interps = []

    for i in interp_vals:
        interp = f(i) @ f(i).T

        if not utils._is_pos_def(interp):
            interp = utils._add_diagonal_epsilon(interp)

        if not utils._is_symm(interp):
            raise Exception(f"Interpolation at step {i} not symmetric")

        cov_interps.append(interp)

    return cov_interps


def interspeaker_interp(
    lang1: enums.RomanceLanguages,
    lang2: enums.RomanceLanguages,
    speaker1: enums.Speakers,
    speaker2: enums.Speakers,
    digit: int,
    interp_no: int = utils.INTERP_NO,
    lmbda: float = 1e-8,
) -> List[Tuple[str, np.array]]:
    """Returns the interpolated spectrograms between two speakers.

    Args:
        lang1 (enums.RomanceLanguages): The language of the first speaker.
        lang2 (enums.RomanceLanguages): The language of the second speaker.
        speaker1 (enums.Speakers): The first speaker.
        speaker2 (enums.Speakers): The second speaker.
        digit (int): The digit being pronounced.
        interp_no (int, optional): The number of interpolations to produce, excluding
                                   the endpoints. Defaults to utils.INTERP_NO.
        lmbda (float, optional): The small value added along the diagonals of covariance
                                 structures to ensure positive-definiteness. Defaults to
                                 1e-8.

    Returns:
        List[np.array]: A list of tuples, where each tuple consists of a string label and
                        an interpolated spectrogram.
    """
    if digit not in [x for x in range(1, 11)]:
        raise Exception(f"digit must be an integer 1 through 10")

    if type(interp_no) != int:
        raise Exception(f"interp_no must be an integer")

    # get covariance structures
    cov_time_l1 = get_time_covariance(language=lang1)
    cov_time_l2 = get_time_covariance(language=lang2)

    cov_freq_l1 = get_freq_covariance(language=lang1)
    cov_freq_l2 = get_freq_covariance(language=lang2)

    time_dim, _ = cov_time_l1.shape
    freq_dim, _ = cov_freq_l1.shape

    time_lmbda = np.identity(time_dim) * lmbda
    freq_lmbda = np.identity(freq_dim) * lmbda

    cov_time_l1 += time_lmbda
    cov_time_l2 += time_lmbda

    cov_freq_l1 += freq_lmbda
    cov_freq_l2 += freq_lmbda

    # covariance structure square roots
    cov_time_l1_sqrt = sqrtm(cov_time_l1)
    cov_time_l2_sqrt = sqrtm(cov_time_l2)
    cov_freq_l1_sqrt = sqrtm(cov_freq_l1)
    cov_freq_l2_sqrt = sqrtm(cov_freq_l2)

    C_t = cov_time_l2_sqrt.T @ cov_time_l1_sqrt
    U_t, _, VT_t = np.linalg.svd(C_t)
    R_t = U_t @ VT_t

    C_f = cov_freq_l2_sqrt.T @ cov_freq_l1_sqrt
    U_f, _, VT_f = np.linalg.svd(C_f)
    R_f = U_f @ VT_f

    m1 = spectrograms.get_mean_spectrogram_array(
        language=lang1,
        digit=digit,
    )
    m2 = spectrograms.get_mean_spectrogram_array(
        language=lang2,
        digit=digit,
    )

    res1 = spectrograms.get_residual_spectrogram_array(
        language=lang1,
        speaker=speaker1,
        digit=digit,
    )

    res2 = spectrograms.get_residual_spectrogram_array(
        language=lang2,
        speaker=speaker2,
        digit=digit,
    )

    Res1x = (
        inv(cov_time_l1_sqrt).astype(float) @ res1 @ inv(cov_freq_l1_sqrt).astype(float)
    )

    Res2x = (
        inv(cov_time_l2_sqrt).astype(float) @ res2 @ inv(cov_freq_l2_sqrt).astype(float)
    )

    weights = np.linspace(0, 1, interp_no)

    # remove interpolations corresponding to the endpoints
    weights = weights[1:-1]

    spect_path = []

    for w in weights:
        time_interp = (
            (cov_time_l1_sqrt + w * (cov_time_l2_sqrt - cov_time_l1_sqrt @ R_t))
            @ (cov_time_l1_sqrt + w * (cov_time_l2_sqrt - cov_time_l1_sqrt @ R_t)).T
        ).astype(float)

        freq_interp = (
            (cov_freq_l1_sqrt + w * (cov_freq_l2_sqrt - cov_freq_l1_sqrt @ R_f))
            @ (cov_freq_l1_sqrt + w * (cov_freq_l2_sqrt - cov_freq_l1_sqrt @ R_f)).T
        ).astype(float)

        M = (1 - w) * m1 + w * m2

        Resx = (1 - w) * Res1x + w * Res2x

        Y = sqrtm(time_interp) @ Resx @ sqrtm(freq_interp) + M

        spect_path.append((f"in_{round(w, 3)}", Y.astype(float)))

    return spect_path
