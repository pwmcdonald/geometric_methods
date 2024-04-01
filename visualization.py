import embedding_scoring
import hyperbolic_analysis
from enums import RomanceLanguages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import utils
import pipeline
from enums import RomanceLanguages, CovarianceType
import matplotlib.patches as mpatches
from typing import Dict, List
from scipy.linalg import eig


def plot_spectrogram(
    spect_arr: np.array,
    rotate: bool = False,
) -> None:
    """This code was found on StackOverflow @
    https://stackoverflow.com/questions/71925324/matplotlib-3d-place-colorbar-into-z-axis
    and modified somewhat.

    Plots a spectrogram value array.

    Args:
        spect_arr (np.array): The spectrogram value array to plot.
        rotate (bool, optional): Whether or not the spectrogram surface is rotated
                                 90 degrees clockwise. Defaults to False.

    Returns:
        None.
    """
    plt.figure(figsize=(30, 20))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface
    X = np.arange(0, pipeline._get_TIME_UPPER_BOUND(), pipeline._get_TIME_INTERVAL())
    Y = np.arange(0, pipeline._get_FREQ_UPPER_BOUND(), pipeline._get_FREQ_INTERVAL())
    X, Y = np.meshgrid(X, Y)

    spect_arr = spect_arr.T

    # Apply rotation if requested
    if rotate:
        surf = ax.plot_surface(
            Y, X, spect_arr, cmap=cm.hsv_r, linewidth=0, antialiased=False
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Time (standardized)")
        ax.invert_yaxis()
        plt.title("Intensity (dB)")
    else:
        surf = ax.plot_surface(
            X, Y, spect_arr, cmap=cm.hsv_r, linewidth=0, antialiased=False
        )
        ax.set_xlabel("Time (standardized)")
        ax.set_ylabel("Frequency (Hz)")
        plt.title("Intensity (dB)")

    # Customize the axes
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter("{x:.02f}")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_time_covariance(
    cov_arr: np.array,
    rotate=False,
) -> None:
    """This code was found on StackOverflow @
    https://stackoverflow.com/questions/71925324/matplotlib-3d-place-colorbar-into-z-axis
    and modified somewhat.

    Plots a time covariance array.

    Args:
        cov_arr (np.array): The time covariance array to plot.
        rotate (bool, optional): Whether or not the covariance surface is rotated
                                 90 degrees clockwise. Defaults to False.

    Returns:
        None.
    """
    plt.figure(figsize=(30, 20))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface
    X = np.arange(0, pipeline._get_TIME_UPPER_BOUND(), pipeline._get_TIME_INTERVAL())
    Y = np.arange(0, pipeline._get_TIME_UPPER_BOUND(), pipeline._get_TIME_INTERVAL())
    X, Y = np.meshgrid(X, Y)

    cov_arr = cov_arr.T

    # Apply rotation if requested
    if rotate:
        surf = ax.plot_surface(
            X, Y, cov_arr, cmap=cm.hsv_r, linewidth=0, antialiased=False
        )
        ax.set_xlabel("Time (standardized)")
        ax.set_ylabel("Time (standardized)")
        plt.title("Covariance")
    else:
        surf = ax.plot_surface(
            Y, X, cov_arr, cmap=cm.hsv_r, linewidth=0, antialiased=False
        )
        ax.set_xlabel("Time (standardized)")
        ax.set_ylabel("Time (standardized)")
        ax.invert_xaxis()
        plt.title("Covariance")

    # Customize the axes
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter("{x:.02f}")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_freq_covariance(
    cov_arr: np.array,
    rotate=False,
) -> None:
    """This code was found on StackOverflow @
    https://stackoverflow.com/questions/71925324/matplotlib-3d-place-colorbar-into-z-axis
    and modified somewhat

    Plots a frequency covariance array.

    Args:
        cov_arr (numpy.array): The frequency covariance array to plot.
        rotate (bool) [Optional]: Whether or not the covariance surface is rotated
                                  90 degrees clockwise. Defaults to False.

    Returns:
        None.
    """
    plt.figure(figsize=(30, 20))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface
    X = np.arange(0, pipeline._get_FREQ_UPPER_BOUND(), pipeline._get_FREQ_INTERVAL())
    Y = np.arange(0, pipeline._get_FREQ_UPPER_BOUND(), pipeline._get_FREQ_INTERVAL())
    X, Y = np.meshgrid(X, Y)

    cov_arr = cov_arr.T

    # Apply rotation if requested
    if rotate:
        surf = ax.plot_surface(
            X, Y, cov_arr, cmap=cm.hsv_r, linewidth=0, antialiased=False
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Frequency (Hz)")
        plt.title("Covariance")
    else:
        surf = ax.plot_surface(
            Y, X, cov_arr, cmap=cm.hsv_r, linewidth=0, antialiased=False
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Frequency (Hz)")
        ax.invert_xaxis()
        plt.title("Covariance")

    # Customize the axes
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter("{x:.02f}")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_distances(
    time_dict: Dict[str, float],
    freq_dict: Dict[str, float],
    lang: RomanceLanguages,
) -> None:
    """Plots the distances between language-wise covariance operators.

    Args:
        time_dict (Dict[str, float]): A dictionary of time covariance distances as generated by
                                      analysis.py > test_covariance_distance.
        freq_dict (Dict[str, float]): A dictionary of frequency covariance distances as generated by
                                      analysis.py > test_covariance_distance.
        lang (RomanceLanguages): The language for which to plot distances.
    """
    # Code largely pulled from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

    time_dict = time_dict[lang.value]
    freq_dict = freq_dict[lang.value]

    lang_distances = {"Time": [], "Frequency": []}

    for l in utils.langs:
        lang_distances["Time"] = lang_distances["Time"] + [time_dict[l]]
        lang_distances["Frequency"] = lang_distances["Frequency"] + [freq_dict[l]]

    x = np.arange(len(utils.langs))  # the label locations
    width = 0.45  # the width of the bars
    multiplier = 0

    _, ax = plt.subplots(layout="constrained")

    for attribute, measurement in lang_distances.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=8)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Procrustes distance between languages")
    ax.set_title(f"{lang.value}")
    ax.set_xticks(x + width, utils.langs)
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, 250)

    plt.show()


def plot_eigval_decomposition(
    language: RomanceLanguages,
    cov_type: CovarianceType,
) -> None:
    """Plots the eigenvalue decomposition for a given covariance structure.

    Args:
        language (RomanceLanguages): The language for which the decomposition is plotted.
        cov_type (CovarianceType): The covariance type for which the decomposition is plotted.

    Returns:
        None.
    """
    cov = utils._get_covariance(language=language, cov_type=cov_type.value)

    eigvals, _ = eig(cov)

    _, ax = plt.subplots(layout="constrained")

    plt.bar(x=[k for k in range(1, len(eigvals) + 1)], height=eigvals)

    ax.set_ylabel("Eigenvalue magnitude")
    ax.set_title(f"{language.value}")
    ax.set_xlabel("k")

    plt.show


def plot_poincare_disk(
    embedding_dict: Dict[str],
    digit: int,
    k: int,
    legend: bool = True,
    interp_settings: hyperbolic_analysis.InterpSettings = None,
    point_labels: bool = False,
    file_path: str = None,
):
    """Plots embeddings on a Poincare disk.

    Args:
        embedding_dict (Dict[str]): A dictionary of embeddings as generated by hyperbolic_analysis >
                                    get_embeddings.
        digit (int): The digit depicted by the disk.
        k (int): The k value with which the disk's underlying kNN graph was constructed.
        legend (bool, optional): Whether to include a legend. Defaults to True.
        interp_settings (hyperbolic_analysis.InterpSettings, optional): Settings describing interpolation
                                                                        details; will plot interpolations if
                                                                        and only if this is not None. Defaults
                                                                        to None.
        point_labels (bool, optional): Whether to label individual points. Defaults to False.
        file_path (str, optional): Where to save the image of the disk; will only save the image if not None.
                                   Defaults to None.
    """
    ax = plt.gca()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    circle1 = plt.Circle((0, 0), 1, color="black", fill=False)
    ax.add_patch(circle1)

    ax.set_aspect("equal", adjustable="box")

    red_patch = mpatches.Patch(color=utils.colors["FR"], label="FR")
    blue_patch = mpatches.Patch(color=utils.colors["IT"], label="IT")
    green_patch = mpatches.Patch(color=utils.colors["PO"], label="PO")
    yellow_patch = mpatches.Patch(color=utils.colors["SA"], label="SA")
    cyan_patch = mpatches.Patch(color=utils.colors["SI"], label="SI")

    handles = [
        red_patch,
        blue_patch,
        green_patch,
        yellow_patch,
        cyan_patch,
    ]

    title_string = f"Poincare embeddings for digit {digit}; k={k}"
    plt.title(title_string)

    if interp_settings:
        fuchsia_path = mpatches.Patch(
            color=utils.colors["cov_in"], label="Covariance interp."
        )
        handles.append(fuchsia_path)

        aquamarine_path = mpatches.Patch(
            color=utils.colors["hyp_in"], label="Hyperbolic interp."
        )
        handles.append(aquamarine_path)

    if legend:
        plt.legend(
            handles=handles,
        )

    for lang in utils.langs:
        x = [pt[0][0] for pt in embedding_dict[lang]]
        y = [pt[0][1] for pt in embedding_dict[lang]]
        txt = [pt[2] for pt in embedding_dict[lang]]

        plt.scatter(
            x,
            y,
            color=utils.colors[lang],
            label=lang,
        )

        if point_labels:
            # NOTE: Below code from https://stackoverflow.com/questions/
            #       14432557/scatter-plot-with-different-text-at-each-data-point
            for i, txt in enumerate(txt):
                ax.annotate(txt, (x[i], y[i]))

    if interp_settings:
        anchor_1 = embedding_dict["anchor"][0][0]
        anchor_2 = embedding_dict["anchor"][1][0]

        circle_sp1 = plt.Circle(
            (anchor_1[0], anchor_1[1]),
            0.1,
            color="black",
            fill=False,
        )
        ax.add_patch(circle_sp1)

        circle_sp2 = plt.Circle(
            (anchor_2[0], anchor_2[1]),
            0.1,
            color="black",
            fill=False,
        )
        ax.add_patch(circle_sp2)

        cov_interps = embedding_dict["in"]
        hyp_interps = hyperbolic_analysis.poincare_linspace(
            u=anchor_1,
            v=anchor_2,
        )[
            1:-1
        ]  # Trim anchor points off of interpolations

        for intp in cov_interps:
            plt.scatter(
                [intp[0][0]],
                [intp[0][1]],
                color=utils.colors["cov_in"],
                label="Cov interp.",
            )

        for intp in hyp_interps:
            plt.scatter(
                [intp[0]],
                [intp[1]],
                color=utils.colors["hyp_in"],
                label="Hyp interp.",
            )

    if not file_path:
        plt.show()

    else:
        if interp_settings:
            plt.savefig(
                f"{file_path}/"
                + f"{interp_settings.lang1.value}{interp_settings.speaker1.value}->"
                + f"{interp_settings.lang2.value}{interp_settings.speaker2.value}_{digit}"
            )
        else:
            plt.savefig(f"{file_path}/{digit}")

    plt.clf()

    if interp_settings:
        interp_score = utils._interpolation_score(
            anchor_1=anchor_1,
            anchor_2=anchor_2,
            cov_interps=cov_interps,
            hyp_interps=hyp_interps,
        )

        coord_high = []
        coord_low = []

        for lang in utils.langs:
            embedding_list = embedding_dict[lang]
            for cl, ch, _ in embedding_list:
                coord_high.append(ch)
                coord_low.append(cl)

        # Handle interpolations too
        embedding_list = embedding_dict["in"]
        for cl, ch, txt in embedding_list:
            # Below statement excludes anchor points
            if txt == "":
                coord_high.append(ch)
                coord_low.append(cl)

        Qlocal, Qglobal, _ = embedding_scoring.get_quality_metrics(
            coord_high=coord_high,
            coord_low=coord_low,
            k_neighbours=k,
        )

        Qlocal = round(Qlocal, 4)
        Qglobal = round(Qglobal, 4)

        return (interp_score, Qlocal, Qglobal)


def plot_poincare_centroids(
    centroid_dict: Dict[str, np.array],
    digit: int,
    k: int,
    kernel_sigma: float,
    legend: bool = True,
) -> None:
    """Plots the per-language centroids in a Poincare disk.

    Args:
        centroid_dict (Dict[str, np.array]): A dictionary of languages and their
                                             centroids.
        digit (int): The digit depicted by the disk.
        k (int): The k value with which the disk's underlying kNN graph was constructed.
        kernel_sigma (float): The kernel_sigma value with which the disk's underlying kNN
                              graph was constructed.
        legend (bool, optional): Whether to include a legend. Defaults to True.
    """
    plt.title(
        f"Poincare embeddings for digit {digit}; k={k}, kernel_sigma={kernel_sigma}"
    )
    red_patch = mpatches.Patch(color=utils.colors["FR"], label="FR")
    blue_patch = mpatches.Patch(color=utils.colors["IT"], label="IT")
    green_patch = mpatches.Patch(color=utils.colors["PO"], label="PO")
    yellow_patch = mpatches.Patch(color=utils.colors["SA"], label="SA")
    cyan_patch = mpatches.Patch(color=utils.colors["SI"], label="SI")

    if legend:
        plt.legend(
            handles=[red_patch, blue_patch, green_patch, yellow_patch, cyan_patch]
        )

    ax = plt.gca()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    circle1 = plt.Circle((0, 0), 1, color="black", fill=False)
    ax.add_patch(circle1)

    ax.set_aspect("equal", adjustable="box")

    plt.scatter(
        centroid_dict["FR"][0],
        centroid_dict["FR"][1],
        color=utils.colors["FR"],
        label="FR",
    )
    plt.scatter(
        centroid_dict["IT"][0],
        centroid_dict["IT"][1],
        color=utils.colors["IT"],
        label="IT",
    )
    plt.scatter(
        centroid_dict["PO"][0],
        centroid_dict["PO"][1],
        color=utils.colors["PO"],
        label="PO",
    )
    plt.scatter(
        centroid_dict["SA"][0],
        centroid_dict["SA"][1],
        color=utils.colors["SA"],
        label="SA",
    )
    plt.scatter(
        centroid_dict["SI"][0],
        centroid_dict["SI"][1],
        color=utils.colors["SI"],
        label="SI",
    )

    plt.show()


def plot_aligned_digit_centroids(
    embedding_dict: Dict[str, List],
    k: int,
    legend: bool = True,
    radii: bool = False,
) -> None:
    """Aligns and plots the centroid for each language/digit pair in the same Poincare
    disk.

    Args:
        embedding_dict (Dict[str, List]): Embeddings returned by hyperbolic_analysis.py >
                                          align_all_digit_disks.

        k (int): The k value with which the disk's underlying kNN graph was constructed.
        legend (bool, optional): Whether to include a legend. Defaults to True.
        radii (bool, optional): Whether to include visual indicators of each language's
                                overall centroid. Defaults to False.

    Returns:
        None.
    """
    plt.title(f"Aligned embeddings; k={k}")
    red_patch = mpatches.Patch(color=utils.colors["FR"], label="FR")
    blue_patch = mpatches.Patch(color=utils.colors["IT"], label="IT")
    green_patch = mpatches.Patch(color=utils.colors["PO"], label="PO")
    yellow_patch = mpatches.Patch(color=utils.colors["SA"], label="SA")
    cyan_patch = mpatches.Patch(color=utils.colors["SI"], label="SI")

    alpha = 0.5

    if legend:
        plt.legend(
            handles=[red_patch, blue_patch, green_patch, yellow_patch, cyan_patch]
        )

    for lang in utils.langs:
        plt.scatter(
            [x[0][0] for x in embedding_dict[lang]],
            [x[0][1] for x in embedding_dict[lang]],
            color=utils.colors[lang],
            label=lang,
        )

    ax = plt.gca()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    circle1 = plt.Circle((0, 0), 1, color="black", fill=False)
    ax.add_patch(circle1)

    centroids = hyperbolic_analysis.get_poincare_centroids(
        embedding_dict=embedding_dict
    )

    if radii:
        # NOTE: radii dict defined as such to allow for, e.g. defining radii
        #       to correspond to point cluster diameter, etc.
        radii = {
            "FR": 0.07,
            "IT": 0.07,
            "PO": 0.07,
            "SA": 0.07,
            "SI": 0.07,
        }

        circle1 = plt.Circle(
            centroids["FR"],
            radii["FR"],
            alpha=alpha,
            color=utils.colors["FR"],
            fill=True,
        )
        ax.add_patch(circle1)

        circle2 = plt.Circle(
            centroids["IT"],
            radii["IT"],
            alpha=alpha,
            color=utils.colors["IT"],
            fill=True,
        )
        ax.add_patch(circle2)

        circle3 = plt.Circle(
            centroids["PO"],
            radii["PO"],
            alpha=alpha,
            color=utils.colors["PO"],
            fill=True,
        )
        ax.add_patch(circle3)

        circle4 = plt.Circle(
            centroids["SA"],
            radii["SA"],
            alpha=alpha,
            color=utils.colors["SA"],
            fill=True,
        )
        ax.add_patch(circle4)

        circle5 = plt.Circle(
            centroids["SI"],
            radii["SI"],
            alpha=alpha,
            color=utils.colors["SI"],
            fill=True,
        )
        ax.add_patch(circle5)

    ax.set_aspect("equal", adjustable="box")
    plt.show()
