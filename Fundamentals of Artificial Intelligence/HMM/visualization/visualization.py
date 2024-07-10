import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


def plot_weibull_density(k, lambda_, state_name, l, r, t):
    """
    Plot the density function of a Weibull distribution.

    Parameters:
    k (float): Shape parameter of the Weibull distribution.
    lambda_ (float): Scale parameter of the Weibull distribution.
    state_name (str): Name of the state for labeling purposes.
    a (int): left bound of the sample interval
    b (int): right bound of the sample interval
    t (int): Number of samples to generate. Default is 50. Must be non-negative.
    """
    x = np.linspace(l, r, t)
    y = weibull_min.pdf(x, k, scale=lambda_)

    plt.plot(x, y, label=f"{state_name} (k={k}, λ={lambda_})")


def visualize_weibull(l, r, typ, xlabel, title):
    """
    Visualize Weibull distribution with given parameters.

    Parameters:
    l (int): left bound of the sample interval
    r (int): right bound of the sample interval
    typ (dictionary): type of the matrix, could be hf, p5 or sh
    title (string): title could be hf, p5 or sh
    """

    # Plot the density functions for each state
    plt.figure(figsize=(8, 5))
    plot_weibull_density(
        typ["shape_flapping"], typ["scale_flapping"], "Flapping Fly", l, r, 1000
    )
    plot_weibull_density(
        typ["shape_soaring"], typ["scale_soaring"], "Soaring Fly", l, r, 1000
    )
    plot_weibull_density(typ["shape_water"], typ["scale_water"], "On Water", l, r, 1000)

    plt.title("Weibull Density Functions of " + title)
    plt.ylabel("Density")
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def draw_table(
        title,
        low_interval_str,
        high_interval_str,
        prob_flapping_l,
        prob_soaring_l,
        prob_water_l,
        prob_flapping_h,
        prob_soaring_h,
        prob_water_h,
):
    """
    Draw the table about the probabilities in different intervals regarding matric given

    """
    data = [
        [prob_flapping_l, prob_flapping_h],
        [prob_soaring_l, prob_soaring_h],
        [prob_water_l, prob_water_h],
    ]

    fig, ax = plt.subplots(figsize=(8, 2), gridspec_kw={"top": 0.8, "bottom": 0.2})
    ax.axis("off")
    table = ax.table(
        cellText=data,
        loc="center",
        cellLoc="center",
        colLabels=[low_interval_str, high_interval_str],
        rowLabels=["flapping fly", "soaring fly", "on water"],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    fig.suptitle(title, fontsize=14)

    table.scale(1.2, 2.2)
    plt.show()


def visualize_probability(obs_seq, ylabel, prob):
    """
    Visualize probabilities over time for three categories.

    Parameters:
    - obs_seq: Sequence of observations over time.
    - ylabel: Label for the y-axis in the plot.
    - prob: Matrix of probabilities (columns: flapping, soaring, on-water).

    Returns:
    None
    """
    plt.figure(figsize=(10, 5))
    timespan = np.arange(1, len(obs_seq) + 1)

    color_dict = {0: "blue", 1: "orange", 2: "green"}
    line_styles = ["-", "--", ":"]
    markers = ["o", "^", "s"]

    for i in range(prob.shape[1]):
        plt.plot(
            timespan,
            prob[:, i],
            color=color_dict[i],
            linestyle=line_styles[i],
            marker=markers[i],
        )

    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.legend(["flapping flight", "soaring flight", "on-water"], loc="upper right")
    plt.title("Probabilities Over Time", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def getImage(path, zoom=0.15):
    return OffsetImage(plt.imread(path), zoom=zoom)


def vis_traj(
        state_list_list, color_list, linestyle_list, legend_list, datapath="./data"
):
    """
    Visualize trajectories with associated images along the path.

    Parameters:
    - state_list_list: List of lists representing trajectories for different states.
    - color_list: List of colors for each trajectory.
    - linestyle_list: List of line styles for each trajectory.
    - legend_list: List of legend labels for each trajectory.
    - datapath: Path to the folder containing image files. Default is './data'.
    - figsize: Tuple specifying the width and height of the figure in inches. Default is (8, 5).
    - image_zoom: Zoom factor for the inserted images. Default is 0.1.

    Returns:
    None
    """

    pic_dict = {0: "flapping_fly", 1: "soaring_fly", 2: "on_water"}
    paths_list = []
    for i in range(len(state_list_list)):
        paths = [datapath + "/" + pic_dict[n] + ".png" for n in state_list_list[i]]
        paths_list.append(paths)
    x = [n for n in range(len(state_list_list[0]))]

    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.set_facecolor('#d3e9f8')
    for i in range(len(state_list_list)):
        ax.plot(x, state_list_list[i], color=color_list[i], linestyle=linestyle_list[i])
        for x0, y0, path in zip(x, state_list_list[i], paths_list[i]):
            ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
            ax.add_artist(ab)
    ax.invert_yaxis()
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["On water", "Soaring flight", "Flapping flight"])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlabel("Timesteps")
    plt.ylabel("States")
    plt.legend(legend_list, loc="upper left", fontsize=10)
    plt.tight_layout()
