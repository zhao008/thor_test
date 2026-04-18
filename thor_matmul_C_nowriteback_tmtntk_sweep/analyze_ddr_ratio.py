#!/usr/bin/env python3

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PARSED_RESULTS_DIR = SCRIPT_DIR / "parsed_results"
os.environ.setdefault("MPLCONFIGDIR", str((PARSED_RESULTS_DIR / ".mplconfig").resolve()))
if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


TILE_COLUMNS = ["tile_m", "tile_n", "tile_k"]
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


class TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(output_dir, prefix):
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{prefix}_{timestamp}.log"
    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeWriter(original_stdout, log_file)
    sys.stderr = TeeWriter(original_stderr, log_file)
    return log_path, log_file, original_stdout, original_stderr


def teardown_logging(log_file, original_stdout, original_stderr):
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


def config_tuple_from_row(row):
    values = []
    for col in TILE_COLUMNS:
        value = row.get(col, np.nan)
        values.append(None if pd.isna(value) else int(value))
    return tuple(values)


def config_sort_key(config):
    return tuple(-1 if value is None else int(value) for value in config)


def format_tile_value(value):
    return "?" if value is None else str(int(value))


def format_tile_label(config):
    tile_m, tile_n, tile_k = config
    return (
        f"tm={format_tile_value(tile_m)}, "
        f"tn={format_tile_value(tile_n)}, "
        f"tk={format_tile_value(tile_k)}"
    )


def format_tile_short(config):
    tile_m, tile_n, tile_k = config
    return (
        f"tm{format_tile_value(tile_m)}_"
        f"tn{format_tile_value(tile_n)}_"
        f"tk{format_tile_value(tile_k)}"
    )


def collect_tile_configs(df):
    config_rows = df[TILE_COLUMNS].drop_duplicates().copy()
    for col in TILE_COLUMNS:
        config_rows[f"{col}_sort"] = config_rows[col].fillna(-1).astype(int)
    config_rows = config_rows.sort_values(
        [f"{col}_sort" for col in TILE_COLUMNS]
    )
    return [config_tuple_from_row(row) for _, row in config_rows.iterrows()]


def find_transition_points(df_subset, normal_threshold=1.05, inflection_threshold=1.1):
    last_normal = None
    normal_points = df_subset[df_subset["ddr_read_ratio"] <= normal_threshold]
    if not normal_points.empty:
        last_normal = normal_points.iloc[-1]

    first_inflection = None
    inflection_points = df_subset[df_subset["ddr_read_ratio"] > inflection_threshold]
    if not inflection_points.empty:
        first_inflection = inflection_points.iloc[0]

    return last_normal, first_inflection


def maybe_int(value):
    return None if value is None else int(value)


def analyze_ddr_ratios(parsed_results_dir):
    """Analyze DDR read ratios and generate plots for tm/tn/tk sweeps."""

    combined_csv = parsed_results_dir / "ncu_results_combined.csv"
    if not combined_csv.exists():
        print(f"Error: {combined_csv} not found")
        return 1

    df = pd.read_csv(combined_csv)
    for col in TILE_COLUMNS + ["K", "M", "N"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ddr_read_ratio_m"] = df["m_ddr_read_bytes_actual"] / df["ddr_read_bytes_theoretical"]
    df["ddr_read_ratio_n"] = df["n_ddr_read_bytes_actual"] / df["ddr_read_bytes_theoretical"]
    df["ddr_read_ratio"] = (df["ddr_read_ratio_m"] + df["ddr_read_ratio_n"]) / 2
    df["tile_label"] = df.apply(lambda row: format_tile_label(config_tuple_from_row(row)), axis=1)

    tile_configs = collect_tile_configs(df)
    k_values = sorted(df["K"].dropna().astype(int).unique().tolist())
    if not tile_configs or not k_values:
        print("Error: No valid tile configs or K values found in combined CSV")
        return 1

    print(f"Found tile configs: {[format_tile_label(config) for config in tile_configs]}")
    print(f"Found {len(k_values)} K values: {k_values}")
    print("\nAnalyzing DDR read ratios...\n")

    inflection_points = {}
    last_normal_points = {}

    for config in tile_configs:
        tile_label = format_tile_label(config)
        df_config = df[df["tile_label"] == tile_label]
        print(f"=== {tile_label} ===")

        for k in k_values:
            df_k = df_config[df_config["K"] == k].sort_values("M")
            if df_k.empty:
                print(f"K={k:4d}: No data")
                continue

            key = (config, k)
            last_normal, first_inflection = find_transition_points(df_k)

            if last_normal is not None:
                last_normal_points[key] = {
                    "tile_m": maybe_int(config[0]),
                    "tile_n": maybe_int(config[1]),
                    "tile_k": maybe_int(config[2]),
                    "tile_label": tile_label,
                    "K": int(k),
                    "M": int(last_normal["M"]),
                    "N": int(last_normal["N"]),
                    "ratio": float(last_normal["ddr_read_ratio"]),
                    "theoretical_bytes": float(last_normal["ddr_read_bytes_theoretical"]),
                    "theoretical_mb": float(last_normal["ddr_read_bytes_theoretical"]) / (1024 * 1024),
                }

            if first_inflection is not None:
                inflection_points[key] = {
                    "tile_m": maybe_int(config[0]),
                    "tile_n": maybe_int(config[1]),
                    "tile_k": maybe_int(config[2]),
                    "tile_label": tile_label,
                    "K": int(k),
                    "M": int(first_inflection["M"]),
                    "N": int(first_inflection["N"]),
                    "ratio": float(first_inflection["ddr_read_ratio"]),
                    "theoretical_bytes": float(first_inflection["ddr_read_bytes_theoretical"]),
                    "theoretical_mb": float(first_inflection["ddr_read_bytes_theoretical"]) / (1024 * 1024),
                }

            if key in last_normal_points and key in inflection_points:
                ln = last_normal_points[key]
                ip = inflection_points[key]
                print(
                    f"K={k:4d}: Last normal at M={ln['M']:4d}, ratio={ln['ratio']:.3f}, "
                    f"Size={ln['theoretical_mb']:.2f} MB | "
                    f"Jump to M={ip['M']:4d}, ratio={ip['ratio']:.3f}, "
                    f"Size={ip['theoretical_mb']:.2f} MB"
                )
            elif key in last_normal_points:
                ln = last_normal_points[key]
                print(
                    f"K={k:4d}: All normal, max at M={ln['M']:4d}, "
                    f"ratio={ln['ratio']:.3f}, Size={ln['theoretical_mb']:.2f} MB"
                )
            elif key in inflection_points:
                ip = inflection_points[key]
                print(
                    f"K={k:4d}: Inflection at M={ip['M']:4d}, N={ip['N']:4d}, "
                    f"ratio={ip['ratio']:.3f}, Size={ip['theoretical_mb']:.2f} MB"
                )
            else:
                print(f"K={k:4d}: No data")
        print()

    k_colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
    config_colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(tile_configs))))

    fig = plt.figure(figsize=(24, 7))

    ax1 = fig.add_subplot(131, projection="3d")
    for config_index, config in enumerate(tile_configs):
        tile_label = format_tile_label(config)
        marker = MARKERS[config_index % len(MARKERS)]
        df_config = df[df["tile_label"] == tile_label]
        for k_index, k in enumerate(k_values):
            df_k = df_config[df_config["K"] == k].sort_values("M")
            if df_k.empty:
                continue

            x = df_k["M"].values
            y = np.full_like(x, k)
            z = df_k["ddr_read_ratio"].values
            ax1.plot(
                x,
                y,
                z,
                color=k_colors[k_index],
                linewidth=1.5,
                marker=marker,
                markersize=3,
                alpha=0.95,
            )

            key = (config, k)
            if key in last_normal_points:
                ln = last_normal_points[key]
                ax1.scatter([ln["M"]], [k], [ln["ratio"]], color="green", s=80, marker="s", zorder=5)
            if key in inflection_points:
                ip = inflection_points[key]
                ax1.scatter([ip["M"]], [k], [ip["ratio"]], color="red", s=80, marker="o", zorder=5)

    xx, yy = np.meshgrid(sorted(df["M"].dropna().astype(int).unique()), k_values)
    zz = np.ones_like(xx)
    ax1.plot_surface(xx, yy, zz, alpha=0.15, color="gray")
    ax1.set_xlabel("M (Matrix Size)", fontsize=10)
    ax1.set_ylabel("K Value", fontsize=10)
    ax1.set_zlabel("DDR Read Ratio (Actual/Theoretical)", fontsize=10)
    ax1.set_title("3D View: DDR Read Ratio vs Matrix Size, K, and Tile Config", fontsize=12)
    ax1.view_init(elev=20, azim=45)

    ax2 = fig.add_subplot(132)
    for config_index, config in enumerate(tile_configs):
        tile_label = format_tile_label(config)
        marker = MARKERS[config_index % len(MARKERS)]
        df_config = df[df["tile_label"] == tile_label]
        for k_index, k in enumerate(k_values):
            df_k = df_config[df_config["K"] == k].sort_values("M")
            if df_k.empty:
                continue

            ax2.plot(
                df_k["M"],
                df_k["ddr_read_ratio"],
                color=k_colors[k_index],
                linewidth=1.5,
                marker=marker,
                markersize=4,
            )

            key = (config, k)
            if key in last_normal_points:
                ln = last_normal_points[key]
                ax2.scatter([ln["M"]], [ln["ratio"]], color="green", s=65, marker="s", zorder=5)
            if key in inflection_points:
                ip = inflection_points[key]
                ax2.scatter([ip["M"]], [ip["ratio"]], color="red", s=65, marker="*", zorder=5)

    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axhline(y=1.05, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axhline(y=1.1, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("M (Matrix Size)", fontsize=10)
    ax2.set_ylabel("DDR Read Ratio (Actual/Theoretical)", fontsize=10)
    ax2.set_title("2D View: DDR Read Ratio for All K / Tile Config Values", fontsize=12)
    ax2.grid(True, alpha=0.3)

    k_handles = [
        Line2D([0], [0], color=k_colors[index], linewidth=2, label=f"K={k}")
        for index, k in enumerate(k_values)
    ]
    tile_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=MARKERS[index % len(MARKERS)],
            linestyle="None",
            markersize=7,
            label=format_tile_short(config),
        )
        for index, config in enumerate(tile_configs)
    ]
    threshold_handles = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="Ideal (1.0x)"),
        Line2D([0], [0], color="green", linestyle="--", linewidth=1, label="Normal (1.05x)"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1, label="Inflection (1.1x)"),
    ]
    legend_k = ax2.legend(handles=k_handles, title="K", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    ax2.add_artist(legend_k)
    ax2.legend(
        handles=tile_handles + threshold_handles,
        title="Tile / Threshold",
        bbox_to_anchor=(1.02, 0.42),
        loc="upper left",
        fontsize=8,
    )

    ax3 = fig.add_subplot(133)
    for config_index, config in enumerate(tile_configs):
        line_color = config_colors[config_index]
        short_label = format_tile_short(config)

        ln_items = sorted(
            [(k, value) for (cfg, k), value in last_normal_points.items() if cfg == config],
            key=lambda item: item[0],
        )
        if ln_items:
            ax3.plot(
                [k for k, _ in ln_items],
                [value["M"] for _, value in ln_items],
                color=line_color,
                marker="s",
                linewidth=2,
                markersize=7,
                label=f"{short_label} last normal",
            )

        ip_items = sorted(
            [(k, value) for (cfg, k), value in inflection_points.items() if cfg == config],
            key=lambda item: item[0],
        )
        if ip_items:
            ax3.plot(
                [k for k, _ in ip_items],
                [value["M"] for _, value in ip_items],
                color=line_color,
                linestyle="--",
                marker="o",
                linewidth=2,
                markersize=7,
                label=f"{short_label} inflection",
            )

    ax3.set_xlabel("K Value", fontsize=10)
    ax3.set_ylabel("M (Matrix Size)", fontsize=10)
    ax3.set_title("Last Normal vs Inflection Points by Tile Config", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    fig.tight_layout()
    output_file = parsed_results_dir / "ddr_ratio_analysis.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nVisualization saved to: {output_file}")

    subplot_count = max(1, len(tile_configs) * len(k_values))
    cols = 4
    rows = int(np.ceil(subplot_count / cols))
    fig2, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).flatten()

    plot_index = 0
    for config in tile_configs:
        tile_label = format_tile_label(config)
        short_label = format_tile_short(config)
        df_config = df[df["tile_label"] == tile_label]

        for k in k_values:
            if plot_index >= len(axes):
                break

            ax = axes[plot_index]
            plot_index += 1
            df_k = df_config[df_config["K"] == k].sort_values("M")
            if df_k.empty:
                ax.set_visible(False)
                continue

            ax.plot(df_k["M"], df_k["ddr_read_ratio_m"], "o-", linewidth=2, markersize=4, label="kernel_m", color="blue")
            ax.plot(df_k["M"], df_k["ddr_read_ratio_n"], "o-", linewidth=2, markersize=4, label="kernel_n", color="green")
            ax.plot(df_k["M"], df_k["ddr_read_ratio"], "o-", linewidth=2, markersize=4, label="average", color="red")

            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(y=1.05, color="green", linestyle="--", linewidth=1, alpha=0.3)
            ax.axhline(y=1.1, color="red", linestyle="--", linewidth=1, alpha=0.3)

            key = (config, k)
            if key in last_normal_points:
                ln = last_normal_points[key]
                ax.axvline(x=ln["M"], color="green", linestyle=":", linewidth=2, alpha=0.5)
            if key in inflection_points:
                ip = inflection_points[key]
                ax.axvline(x=ip["M"], color="red", linestyle=":", linewidth=2, alpha=0.5)

            ax.set_xlabel("M", fontsize=9)
            ax.set_ylabel("Ratio", fontsize=9)
            ax.set_title(f"{short_label}, K={k}", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

    for ax in axes[plot_index:]:
        ax.set_visible(False)

    fig2.tight_layout()
    output_file2 = parsed_results_dir / "ddr_ratio_detailed.png"
    fig2.savefig(output_file2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Detailed plots saved to: {output_file2}")

    if last_normal_points:
        last_normal_df = pd.DataFrame([
            {
                "tile_m": value["tile_m"],
                "tile_n": value["tile_n"],
                "tile_k": value["tile_k"],
                "tile_label": value["tile_label"],
                "K": k,
                "Last_Normal_M": value["M"],
                "Last_Normal_N": value["N"],
                "Ratio": value["ratio"],
                "Theoretical_Read_Bytes": value["theoretical_bytes"],
                "Theoretical_Read_MB": value["theoretical_mb"],
            }
            for (_, k), value in last_normal_points.items()
        ])
        last_normal_csv = parsed_results_dir / "last_normal_points.csv"
        last_normal_df.to_csv(last_normal_csv, index=False)
        print(f"\nLast normal points saved to: {last_normal_csv}")

        print("\n" + "=" * 100)
        print("LAST NORMAL POINTS (ratio <= 1.05)")
        print("=" * 100)
        print(f"{'tile':<24} {'K':<6} {'M':<6} {'N':<6} {'Theoretical Read (MB)':<25} {'Ratio':<8}")
        print("-" * 100)
        for key in sorted(last_normal_points.keys(), key=lambda item: (config_sort_key(item[0]), item[1])):
            ln = last_normal_points[key]
            print(f"{format_tile_short(key[0]):<24} {ln['K']:<6} {ln['M']:<6} {ln['N']:<6} {ln['theoretical_mb']:<25.2f} {ln['ratio']:<8.3f}")

        print(f"\nAverage theoretical read size at last normal: {last_normal_df['Theoretical_Read_MB'].mean():.2f} MB")
        print(f"Median theoretical read size at last normal: {last_normal_df['Theoretical_Read_MB'].median():.2f} MB")
        print(f"Min theoretical read size: {last_normal_df['Theoretical_Read_MB'].min():.2f} MB")
        print(f"Max theoretical read size: {last_normal_df['Theoretical_Read_MB'].max():.2f} MB")

    if inflection_points:
        inflection_df = pd.DataFrame([
            {
                "tile_m": value["tile_m"],
                "tile_n": value["tile_n"],
                "tile_k": value["tile_k"],
                "tile_label": value["tile_label"],
                "K": k,
                "Inflection_M": value["M"],
                "Inflection_N": value["N"],
                "Ratio_at_inflection": value["ratio"],
                "Theoretical_Read_Bytes": value["theoretical_bytes"],
                "Theoretical_Read_MB": value["theoretical_mb"],
            }
            for (_, k), value in inflection_points.items()
        ])
        inflection_csv = parsed_results_dir / "inflection_points.csv"
        inflection_df.to_csv(inflection_csv, index=False)
        print(f"\nInflection points (ratio > 1.1) saved to: {inflection_csv}")

    if "agg" not in plt.get_backend().lower():
        plt.show()

    return 0


def main():
    PARSED_RESULTS_DIR.mkdir(exist_ok=True)
    log_path, log_file, original_stdout, original_stderr = setup_logging(PARSED_RESULTS_DIR, "analyze_ddr_ratio")
    exit_code = 0

    try:
        print(f"Parsed results directory: {PARSED_RESULTS_DIR}")
        print(f"Log file: {log_path}\n")
        exit_code = analyze_ddr_ratios(PARSED_RESULTS_DIR)
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        teardown_logging(log_file, original_stdout, original_stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
