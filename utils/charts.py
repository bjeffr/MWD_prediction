import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
from utils.params import blue_main_color, label_color
from math import ceil


def custom_hist(data, color=blue_main_color, alpha=1, ylim=None):
    plt.hist(data, bins=20, color=color, alpha=alpha)
    ax = plt.gca()
    if ylim:
        ax.set_ylim(ylim)
        plt.tick_params(left=False)
        ax.set_yticklabels([])
    else:
        plt.ylabel("Frequency")
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    plt.tick_params(bottom=False)
    ax.set_xticklabels([])


def custom_boxplot(data, label):
    plt.boxplot(
        data,
        vert=False,
        widths=0.5,
        boxprops=dict(color=blue_main_color),
        whiskerprops=dict(color=blue_main_color),
        flierprops=dict(
            marker=".",
            markerfacecolor="black",
            markeredgecolor="none",
            alpha=0.2,
        ),
        medianprops=dict(color=plt.cm.seismic(0.92)),
    )
    plt.xlabel(label)
    plt.gca().yaxis.set_ticks([])


def dual_hist_box_plot(series1, series2, label1, label2):
    plt.figure(figsize=(8.5, 3.2))

    ax = plt.subplot2grid(shape=(4, 2), loc=(0, 0), rowspan=3)
    custom_hist(series1)
    plt.subplot2grid(shape=(4, 2), loc=(3, 0))
    custom_boxplot(series1, label=label1)

    if series2 is None:
        plt.subplot2grid(shape=(4, 2), loc=(0, 1), rowspan=3)
        plt.axis("off")
        plt.subplot2grid(shape=(4, 2), loc=(3, 1))
        plt.axis("off")
    else:
        plt.subplot2grid(shape=(4, 2), loc=(0, 1), rowspan=3)
        custom_hist(series2, ylim=ax.get_ylim())
        plt.subplot2grid(shape=(4, 2), loc=(3, 1))
        custom_boxplot(series2, label=label2)

    plt.subplots_adjust(hspace=0.08, wspace=0.04)


def datapoints_scatter(df, s=1.5, dual=False, legend=True, yaxis=True):
    plt.scatter(
        df["M_W_S"],
        df["PDI_S"],
        s=s,
        color=plt.cm.tab10(0),
        edgecolors="none",
        label="$M_w^s$, $PDI^s$",
    )
    plt.scatter(
        df["M_W_L"],
        df["PDI_L"],
        s=s,
        color=plt.cm.tab10(1),
        edgecolors="none",
        label="$M_w^l$, $PDI^l$",
    )
    plt.grid(b=False)

    ax = plt.gca()
    plt.xlabel("$M_w$ [$g/mol$]")
    if yaxis:
        plt.ylabel("$PDI$")
    else:
        ax.yaxis.set_ticks([])

    if dual:
        ax.xaxis.set_ticks(
            [2_000_000, 4_000_000, 6_000_000, 8_000_000, 10_000_000, 12_000_000]
        )
        ax.set_xmargin(0.02)
        ax.set_ymargin(0.03)
    else:
        ax.set_xmargin(0.01)
        ax.set_ymargin(0.017)

    if legend:
        leg = plt.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
            scatteryoffsets=[0.5],
            labelcolor=label_color,
            labelspacing=1,
            handletextpad=0.1,
        )
        for legobj in leg.legendHandles:
            legobj.set_sizes([40])


def calc_boxplot_vals(df):
    box_vals = []
    for column in df:
        series = df[column].values
        Q1, median, Q3 = np.percentile(series, [25, 50, 75])
        IQR = Q3 - Q1
        whisker_range_series = np.compress(
            np.logical_and(Q1 - 1.5 * IQR <= series, series <= Q3 + 1.5 * IQR),
            series,
        )
        whisker_high = np.max(whisker_range_series)
        whisker_low = np.min(whisker_range_series)
        box_vals.append([whisker_low, Q1, median, Q3, whisker_high])

    return np.array(box_vals)


def feature_attrs_iqr_plot(y1, y2):
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(7.5)

    freq_range = np.load("data/freq_range.npy", allow_pickle=False)
    plt.loglog(
        freq_range,
        y1.quantile(0.5),
        label="$G'$ Median",
        base=10,
        zorder=10,
        color=plt.cm.tab10(0),
    )
    ax.fill_between(
        freq_range,
        y1.quantile(0.75),
        y1.quantile(0.25),
        label="$G'$ IQR",
        color=plt.cm.tab10(0),
        alpha=0.25,
        edgecolor="None",
    )
    plt.loglog(
        freq_range,
        y2.quantile(0.5),
        label="$G''$ Median",
        base=10,
        zorder=10,
        color=plt.cm.tab10(1),
    )
    ax.fill_between(
        freq_range,
        y2.quantile(0.75),
        y2.quantile(0.25),
        label="$G''$ IQR",
        color=plt.cm.tab10(1),
        alpha=0.25,
        edgecolor="None",
    )
    plt.minorticks_off()
    plt.xlabel("Frequency [$s^{-1}$]")
    plt.ylabel("$G'$, $G''$ [Pa]")
    handles, _ = ax.get_legend_handles_labels()
    handles = [handles[0], handles[2], handles[1], handles[3]]
    plt.legend(
        handles=handles,
        loc="lower right",
        labelcolor=label_color,
        framealpha=1,
        edgecolor="None",
    )


def custom_log_boxplot(x, position, color):
    plt.boxplot(
        np.log10(x),
        positions=[position],
        widths=0.75,
        boxprops=dict(color=color),
        whiskerprops=dict(color=color),
        flierprops=dict(
            marker=".",
            zorder=1,
            markersize=1,
            markerfacecolor=color,
            markeredgecolor="none",
        ),
        medianprops=dict(color="black"),
    )


def log10format(x):
    return f"$10^{{{x}}}$" % x


def log_box_plot(y1, y2, yticks, annot, ylabel=True):
    custom_log_boxplot(y1, 0, plt.cm.tab10(0))
    custom_log_boxplot(y2, 1, plt.cm.tab10(1))

    for line in plt.gca().get_lines()[6::7]:
        xoffsets = line.get_xdata()
        line.set_xdata(
            xoffsets + np.random.uniform(-0.195, 0.188, xoffsets.size)
        )

    plt.gca().xaxis.tick_top()
    plt.gca().set_xticklabels([" $G'$", " $G''$"])
    plt.gca().tick_params(labelsize=12, length=0, axis="x")
    plt.grid(b=False, axis="x")
    plt.annotate(
        annot,
        xy=(0.5, -0.05),
        xycoords="axes fraction",
        ha="center",
        color=label_color,
        size=12,
    )
    plt.gca().set_yticks(yticks)
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(log10format))
    if ylabel:
        plt.ylabel("$G'$, $G''$ [Pa]")


def dual_log_boxplot(
    storage1, loss1, storage2, loss2, yticks1, yticks2, annot1, annot2
):
    fig, ax = plt.subplots()
    fig.set_figheight(6.5)
    fig.set_figwidth(8.5)

    plt.subplot2grid(shape=(1, 2), loc=(0, 0))
    log_box_plot(storage1, loss1, yticks1, annot1)

    plt.subplot2grid(shape=(1, 2), loc=(0, 1))
    log_box_plot(storage2, loss2, yticks2, annot2, ylabel=False)


def corr_plot(corr):
    plt.figure(figsize=(7.5, 6.13))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=False,
        xticklabels=[],
        yticklabels=[],
    ).collections[0].colorbar.set_label("Correlation", size=12)

    plt.annotate(
        "Frequency $G'$",
        xy=(-0.032, 0.75),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        color=label_color,
        size=12,
    )
    plt.annotate(
        "Frequency $G''$",
        xy=(-0.032, 0.25),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        color=label_color,
        size=12,
    )
    plt.annotate(
        "Frequency $G'$",
        xy=(0.25, 1.03),
        xycoords="axes fraction",
        ha="center",
        va="center",
        color=label_color,
        size=12,
    )
    plt.annotate(
        "Frequency $G''$",
        xy=(0.75, 1.03),
        xycoords="axes fraction",
        ha="center",
        va="center",
        color=label_color,
        size=12,
    )


def loss_plot(train, val):
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(7.5)

    plt.plot(train, color=mpl.cm.tab10(0), label="Training loss")
    plt.plot(val, color=mpl.cm.tab10(1), label="Validation loss")

    plt.legend(
        loc="upper right",
        labelcolor=label_color,
        framealpha=1,
        edgecolor="None",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")


def plot_freq_range_errors(errors, unimodal):
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(9.5)

    freq_range = np.load("data/freq_range.npy", allow_pickle=False)
    plt.xlim(freq_range[0], freq_range[-1])
    plt.ylim(-0.2, 15.2)
    plt.xscale("log")
    ax.yaxis.set_ticks([])
    ax.set_facecolor("white")
    cm = mpl.colors.LinearSegmentedColormap.from_list(
        "custom_cm", ["b", "#F4000B"]
    )

    i = 0
    min_error, max_error = errors.min(), errors.max()
    freq_ranges = np.load("logs/freq_ranges/freq_ranges.npy", allow_pickle=True)
    for y, row in enumerate(freq_ranges):
        for f_range in row:
            error = (errors[i] - min_error) / (max_error - min_error)
            xmin = max(float(f_range[0] / 70 + 0.008), 0.009)
            xmax = min(float(f_range[1] / 70 - 0.008), 0.991)
            plt.axhline(
                y=15 - y,
                xmin=xmin,
                xmax=xmax,
                color=cm(error),
                linewidth=6,
                solid_capstyle="round",
            )
            plt.annotate(
                f"{errors[i]:.1%}",
                xy=(np.mean([xmin, xmax]), (15 - y) / 15.4 + 0.023),
                xycoords="axes fraction",
                ha="center",
                va="bottom",
                color=label_color,
            )
            i += 1
    plt.xlabel("Frequency [$s^{-1}$]")

    cbar = plt.colorbar(
        mappable=mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=min_error, vmax=max_error), cmap=cm
        )
    )
    cbar.ax.yaxis.set_major_formatter(
        mpl.ticker.PercentFormatter(xmax=1, decimals=0)
    )
    if unimodal:
        cbar.ax.set_ylabel("Avg. MRE ($M_w$, $PDI$)")
    else:
        cbar.ax.set_ylabel("Avg. MRE ($M_w^s$, $PDI^s$, $M_w^l$, $PDI^l$)")


def df_errors(y_test, y_pred, is_pdi=False):
    errors = pd.DataFrame(
        list(zip(y_test, y_pred)), columns=["Target", "Prediction"]
    )
    errors["Absolute_Error"] = abs(errors["Prediction"] - errors["Target"])
    if is_pdi:
        errors["Relative_Error"] = errors["Absolute_Error"] / (
            errors["Target"] - 1
        )
    else:
        errors["Relative_Error"] = errors["Absolute_Error"] / errors["Target"]
    errors.sort_values("Target", inplace=True)
    errors.set_index("Target", inplace=True)

    return errors


def abs_error_plot(Y_test, Y_pred, labels):
    for i in range(Y_pred.shape[1]):
        if i in (0, 2):
            errors = df_errors(Y_test[:, i], Y_pred[:, i])
            plt.subplot2grid(
                shape=(ceil(Y_pred.shape[1] / 2), 2),
                loc=(int(i / 2), 0),
                colspan=1,
            )
            plt.scatter(
                errors.index, errors["Absolute_Error"], c=blue_main_color, s=0.1
            )
            plt.gca().yaxis.set_major_formatter(
                mpl.ticker.StrMethodFormatter("{x:,.0f}")
            )
            plt.title(
                f"MAE = {np.mean(errors.Absolute_Error):,.0f} $g/mol$",
                c=label_color,
                size=11,
            )
            plt.xlabel(labels[i])
            plt.ylabel("Absolute Error")

        elif i in (1, 3):
            errors = df_errors(Y_test[:, i], Y_pred[:, i], is_pdi=True)
            plt.subplot2grid(
                shape=(ceil(Y_pred.shape[1] / 2), 2),
                loc=(int(i / 2), 1),
                colspan=1,
            )
            plt.scatter(
                errors.index, errors["Absolute_Error"], c=blue_main_color, s=0.1
            )
            plt.title(
                f"MAE = {np.mean(errors.Absolute_Error):.3f}",
                c=label_color,
                size=11,
            )
            plt.xlabel(labels[i])

        elif i == 4:
            errors = df_errors(Y_test[:, i], Y_pred[:, i])
            plt.subplot2grid(shape=(3, 2), loc=(2, 0), colspan=1)
            plt.scatter(
                errors.index, errors["Absolute_Error"], c=blue_main_color, s=0.1
            )
            plt.title(
                f"MAE = {np.mean(errors.Absolute_Error):.3f}",
                c=label_color,
                size=11,
            )
            plt.xlabel(labels[i])
            plt.ylabel("Absolute Error")


def rel_error_plot(Y_test, Y_pred, labels):
    for i in range(Y_pred.shape[1]):
        if i in (0, 2):
            errors = df_errors(Y_test[:, i], Y_pred[:, i])
            plt.subplot2grid(
                shape=(ceil(Y_pred.shape[1] / 2), 2),
                loc=(int(i / 2), 0),
                colspan=1,
            )
            plt.scatter(
                errors.index, errors["Relative_Error"], c=blue_main_color, s=0.1
            )
            plt.gca().yaxis.set_major_formatter(
                mpl.ticker.StrMethodFormatter("{x:,.0%}")
            )
            plt.title(
                f"MRE = {np.mean(errors.Relative_Error):.2%}",
                c=label_color,
                size=11,
            )
            plt.xlabel(labels[i])
            plt.ylabel("Relative Error")

        elif i in (1, 3):
            errors = df_errors(Y_test[:, i], Y_pred[:, i], is_pdi=True)
            plt.subplot2grid(
                shape=(ceil(Y_pred.shape[1] / 2), 2),
                loc=(int(i / 2), 1),
                colspan=1,
            )
            plt.scatter(
                errors.index, errors["Relative_Error"], c=blue_main_color, s=0.1
            )
            plt.gca().yaxis.set_major_formatter(
                mpl.ticker.StrMethodFormatter("{x:,.0%}")
            )
            plt.title(
                f"MRE = {np.mean(errors.Relative_Error):.2%}",
                c=label_color,
                size=11,
            )
            plt.xlabel(labels[i])


def abs_vs_rel_error_plot(y_test, y_pred, label, is_pdi=False):
    metrics = pd.DataFrame(
        list(zip(y_test, y_pred)), columns=["Target", "Prediction"]
    )
    metrics["Absolute_Error"] = abs(metrics["Prediction"] - metrics["Target"])
    if is_pdi:
        metrics["Relative_Error"] = metrics["Absolute_Error"] / (
            metrics["Target"] - 1
        )
    else:
        metrics["Relative_Error"] = (
            metrics["Absolute_Error"] / metrics["Target"]
        )
    metrics.sort_values("Target", inplace=True)
    metrics.set_index("Target", inplace=True)

    fig, ax = plt.subplots()
    fig.set_figheight(2.5)
    fig.set_figwidth(7.5)

    plt.subplot2grid(shape=(1, 2), loc=(0, 0), colspan=1)
    plt.title("Absolute Error", color=label_color, size=12)
    plt.scatter(
        metrics.index, metrics["Absolute_Error"], c=blue_main_color, s=0.1
    )
    if not is_pdi:
        plt.gca().yaxis.set_major_formatter(
            mpl.ticker.StrMethodFormatter("{x:,.0f}")
        )
    plt.xlabel(label)
    plt.ylabel("Error")

    plt.subplot2grid(shape=(1, 2), loc=(0, 1), colspan=1)
    plt.title("Relative Error", color=label_color, size=12)
    plt.scatter(
        metrics.index, metrics["Relative_Error"], c=blue_main_color, s=0.1
    )
    plt.gca().yaxis.set_major_formatter(
        mpl.ticker.PercentFormatter(xmax=1, decimals=0)
    )
    plt.xlabel(label)
    plt.subplots_adjust(wspace=0.25)


def ds_size_rel_error_plot(rel_errors):
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(7.5)

    for i, column in enumerate(rel_errors.columns):
        plt.plot(
            rel_errors[column],
            linewidth=0.8,
            color=plt.cm.tab10(i),
            label=column,
        )

    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0%}"))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Relative Error")
    plt.legend(loc="upper right", framealpha=1, edgecolor="None")
