"""
Pump performance trends: healthy characteristic curves vs measured operating
points, plus a KPI summary table per dataset.

Run:  python -m analysis.performance_report
"""
from __future__ import annotations

import pandas as pd

from analysis.hydraulic_calculations import compute_kpis, performance_curve
from config.pump_config import get_config


def kpi_summary(df: pd.DataFrame, cfg=None, last_seconds: float = 300.0) -> pd.Series:
    """Mean KPIs over the final `last_seconds` of a dataset while running."""
    k = compute_kpis(df, cfg)
    mask = (df["time_s"] > df["time_s"].max() - last_seconds) & (df["speed_rpm"] > 10)
    return k[mask].mean(numeric_only=True)


def plot_performance(datasets: dict[str, pd.DataFrame], path: str, cfg=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cfg or get_config()
    fig, (ax_h, ax_e) = plt.subplots(1, 2, figsize=(14, 5.5))
    for n, ls in ((1450, "-"), (1300, "--")):
        c = performance_curve(cfg, n)
        ax_h.plot(c.flow_m3h, c.head_m, "k" + ls, lw=1.5, label=f"pump curve {n} rpm")
        ax_e.plot(c.flow_m3h, c.efficiency_pct, "k" + ls, lw=1.5, label=f"pump η {n} rpm")
    q = c.flow_m3h.to_numpy()
    h_sys = cfg.system.static_head_m + (cfg.pump.rated_head_m - cfg.system.static_head_m) * (q / cfg.pump.rated_flow_m3h) ** 2
    ax_h.plot(q, h_sys, color="grey", lw=1, label="system curve (valve open)")

    for name, df in datasets.items():
        k = compute_kpis(df, cfg)
        run = (df.speed_rpm > 10).to_numpy()
        ax_h.scatter(df.flow_m3h[run], k.head_m[run], s=2, alpha=0.3, label=name)
        ax_e.scatter(df.flow_m3h[run], k.pump_eff_est_pct[run], s=2, alpha=0.3, label=name)
    ax_h.set(xlabel="Flow Q [m³/h]", ylabel="Head H [m]", title="H–Q: curve vs measured points",
             xlim=(0, 45), ylim=(0, 35))
    ax_e.set(xlabel="Flow Q [m³/h]", ylabel="Pump efficiency [%]",
             title="η–Q: curve vs estimated from measurements", xlim=(0, 45), ylim=(0, 90))
    for ax in (ax_h, ax_e):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, markerscale=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    names = ["healthy_baseline", "healthy_steady", "fault_flow_restriction", "fault_seal_leakage",
             "fault_cavitation", "fault_bearing_degradation"]
    data = {n: pd.read_csv(f"data/raw/{n}.csv") for n in names}
    plot_performance({n: data[n] for n in ("healthy_baseline", "fault_flow_restriction", "fault_cavitation")},
                     "reports/pump_performance.png")
    cols = ["head_m", "hydraulic_power_kw", "pump_eff_est_pct", "wire_to_water_eff_pct", "sec_kwh_m3",
            "bep_ratio", "flow_deficit_m3h", "head_deviation_pct", "npsh_margin"]
    table = pd.DataFrame({n: kpi_summary(d) for n, d in data.items()}).T[cols]
    pd.set_option("display.width", 200)
    print("KPI summary (mean of last 5 min while running):")
    print(table.round(3).to_string())
