import matplotlib.pyplot as plt


def plot_potential_comparison(
    to_plot,
    colors,
    figsize=(18, 10),
    log_scale=False,
    show=True,
    save=True,
    path="./out/plots/",
    file_name="potential_comparison",
):
    """Plot a comparison of demand response potentials"""
    fig, ax = plt.subplots(figsize=figsize)
    _ = to_plot.plot(kind="bar", stacked=True, color=colors, ax=ax)
    if log_scale:
        _ = ax.set_yscale("log")
    _ = plt.legend(bbox_to_anchor=[1.01, 1.01])
    _ = plt.tight_layout()
    _ = plt.xlabel("Studien")
    _ = plt.ylabel("Potenzial in MW")
    if save:
        plt.savefig(f"{path}{file_name}.png", dpi=300)
    if show:
        plt.show()

    plt.close()
