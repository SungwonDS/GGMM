"""s14_report.py가 만든 CSV를 LaTeX 표(results/report/tab_*.tex)로 바꾼다."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "results" / "report"


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("*", r"$^{*}$")


def synth_table():
    df = pd.read_csv(OUT / "synth_runs.csv")
    wide = df.pivot_table(index=["scenario", "n", "seed"], columns="method", values=["KL", "K", "IAE"])
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"분포 & $n$ & KL BGM & KL VGGMM & 비율 & 승 & $K$ BGM & $K$ VGGMM \\", r"\midrule"]
    order = list(dict.fromkeys(df.scenario))
    for s in order:
        for n in sorted(df.n.unique()):
            w = wide.loc[(s, n)]
            kb, kv = w[("KL", "BGM")], w[("KL", "VGGMM")]
            ratio = np.median(kb / np.maximum(kv, 1e-5))
            wins = int((kv < kb).sum())
            lines.append(f"{esc(s) if n == sorted(df.n.unique())[0] else ''} & {n} & {kb.mean():.4f} & "
                         f"{kv.mean():.4f} & {('$>$100' if ratio > 100 else f'{ratio:.1f}')} & {wins}/{len(kb)} & "
                         f"{w[('K', 'BGM')].mean():.1f} & {w[('K', 'VGGMM')].mean():.1f} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_synth.tex").write_text("\n".join(lines), encoding="utf-8")
    kb, kv = wide[("KL", "BGM")], wide[("KL", "VGGMM")]
    return {"wins": int((kv < kb).sum()), "total": len(kb),
            "ratio_median": float(np.median(kb / np.maximum(kv, 1e-5)))}


def real_table():
    df = pd.read_csv(OUT / "real_runs.csv")
    g = df.groupby(["column", "method"]).agg(ll=("test_ll", "mean"), K=("K", "mean"),
                                             t=("fit_s", "mean"), n=("n_train", "first")).unstack("method")
    wide = df.pivot_table(index=["column", "seed"], columns="method", values="test_ll")
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"열 & $n_{\mathrm{train}}$ & ll BGM & ll VGGMM & 차이 & 승 & $K$ BGM & $K$ VGGMM & 시간(s) BGM/VGGMM \\",
             r"\midrule"]
    for col in dict.fromkeys(df.column):
        r = g.loc[col]
        w = wide.loc[col]
        lines.append(f"{esc(col)} & {int(r[('n', 'BGM')])} & {r[('ll', 'BGM')]:.3f} & {r[('ll', 'VGGMM')]:.3f} & "
                     f"{r[('ll', 'VGGMM')] - r[('ll', 'BGM')]:+.3f} & {int((w['VGGMM'] > w['BGM']).sum())}/{len(w)} & "
                     f"{r[('K', 'BGM')]:.1f} & {r[('K', 'VGGMM')]:.1f} & "
                     f"{r[('t', 'BGM')]:.1f}/{r[('t', 'VGGMM')]:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_real.tex").write_text("\n".join(lines), encoding="utf-8")
    return {"wins": int((wide["VGGMM"] > wide["BGM"]).sum()), "total": len(wide)}


def kmax_table():
    df = pd.read_csv(OUT / "kmax_runs.csv")
    ks = sorted(df.K_max.unique())
    lines = [r"\begin{tabular}{ll" + "r" * len(ks) + "}", r"\toprule",
             "데이터 & 항목 & " + " & ".join(f"$K_{{\\max}}={k}$" for k in ks) + r" \\", r"\midrule"]
    for d in dict.fromkeys(df.data):
        for method in ("BGM", "VGGMM"):
            s = df[(df.data == d) & (df.method == method)].set_index("K_max")
            lines.append(f"{esc(d) if method == 'BGM' else ''} & {method} 유효 $K$ & " +
                         " & ".join(f"{int(s.K[k])}" for k in ks) + r" \\")
            lines.append(f" & {method} ll & " + " & ".join(f"{s.test_ll[k]:.3f}" for k in ks) + r" \\")
            lines.append(f" & {method} 시간(s) & " + " & ".join(f"{s.fit_s[k]:.1f}" for k in ks) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_kmax.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    for f in (synth_table, real_table, kmax_table):
        try:
            print(f.__name__, f())
        except FileNotFoundError as e:
            print(f.__name__, "skipped:", e.filename)
