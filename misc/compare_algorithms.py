"""Compare one CSV algorithm against every competitor using attack-wise HV.

For each adversarial attack, the script calculates a normalized four-objective
hypervolume for every algorithm. It then compares the algorithm selected with
``--reference-algorithm`` against every other algorithm.

Each output row reports:
  * attacks won by the reference algorithm / total attacks;
  * mean signed HV difference: mean(HV_reference - HV_competitor).

Positive differences favor the reference algorithm. Results are descriptive
comparisons of the supplied architecture sets, not repeated-run significance
tests.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV


DEFAULT_ATTACKS = ["FGSM", "BIM_10", "PGD_7", "PGD_10", "PGD_20", "CW_0.1", "CW_0.01"]

LATEX_NAMES = {
    "r2-emoa-60-40": r"R2-EMOA-RNAS$_{60}$",
    "r2-emoa-75-25": r"R2-EMOA-RNAS$_{75}$",
    "r2-emoa-unif": r"R2-EMOA-RNAS$_{\mathrm{Unif}}$",
    "r2-emoa-one-shot-60-40": r"R2-EMOA-One-shot$_{60}$",
    "nsganet": "NSGA-Net",
    "sms-emoa": "SMS-EMOA",
    "moead": "MOEA/D",
    "moras": "MORAS",
    "nevonas": "NEvoNAS",
    "cars": "CARS",
    "random-search": "Random Search",
}


def read_csv_allowing_markdown_fences(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8-sig").strip()
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return pd.read_csv(io.StringIO("\n".join(lines)))


def validate_csv(df: pd.DataFrame) -> None:
    required = {
        "algorithm", "dataset", "model", "flops", "params",
        "attack", "accuracy", "loss",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if df[list(required)].isnull().any().any():
        raise ValueError("Required CSV columns contain missing values.")
    duplicated = df.duplicated(["algorithm", "model", "attack"], keep=False)
    if duplicated.any():
        example = df.loc[duplicated, ["algorithm", "model", "attack"]].head(10)
        raise ValueError(
            "Duplicated algorithm/model/attack keys were found:\n"
            + example.to_string(index=False)
        )


def select_dataset(df: pd.DataFrame, dataset: str | None) -> tuple[pd.DataFrame, str]:
    available = sorted(df["dataset"].dropna().astype(str).unique())
    if dataset is None:
        if len(available) != 1:
            raise ValueError(f"CSV contains multiple datasets {available}; use --dataset.")
        dataset = available[0]
    if dataset not in available:
        raise ValueError(f"Dataset {dataset!r} not found; available: {available}")
    return df.loc[df["dataset"].astype(str) == dataset].copy(), dataset


def pivot_measure(df: pd.DataFrame, measure: str) -> pd.DataFrame:
    return (
        df.pivot(
            index=["algorithm", "model", "flops", "params"],
            columns="attack",
            values=measure,
        )
        .reset_index()
        .rename_axis(columns=None)
    )


def nondominated_mask(F: np.ndarray, atol: float) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        no_worse = np.all(F <= F[i] + atol, axis=1)
        strictly_better = np.any(F < F[i] - atol, axis=1)
        dominators = no_worse & strictly_better
        dominators[i] = False
        keep[i] = not np.any(dominators)
    return keep


def clean_front(F: np.ndarray, decimals: int, atol: float) -> np.ndarray:
    unique = np.unique(np.round(np.asarray(F, dtype=np.float64), decimals), axis=0)
    return unique[nondominated_mask(unique, atol)]


def build_objectives(group: pd.DataFrame, attack: str, formulation: str) -> np.ndarray:
    if formulation == "accuracy":
        clean = 100.0 - group["clean"].to_numpy(dtype=np.float64)
        adversarial = 100.0 - group[attack].to_numpy(dtype=np.float64)
    elif formulation == "loss":
        clean = group["clean"].to_numpy(dtype=np.float64)
        adversarial = group[attack].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unknown formulation: {formulation}")

    return np.column_stack(
        [
            clean,
            adversarial,
            group["flops"].to_numpy(dtype=np.float64),
            group["params"].to_numpy(dtype=np.float64),
        ]
    )


def calculate_attack_hv(
    models: pd.DataFrame,
    attacks: list[str],
    formulation: str,
    reference_margin: float,
    duplicate_decimals: int,
    dominance_atol: float,
) -> pd.DataFrame:
    algorithms = sorted(models["algorithm"].unique())
    values = pd.DataFrame(index=algorithms, columns=attacks, dtype=float)
    indicator = HV(ref_point=np.full(4, 1.0 + reference_margin))

    for attack in attacks:
        if "clean" not in models.columns or attack not in models.columns:
            raise ValueError(f"Required clean/{attack} columns were not found.")
        complete = models.dropna(subset=["clean", attack, "flops", "params"])
        if complete.empty:
            raise ValueError(f"No complete solutions are available for {attack!r}.")

        pooled = build_objectives(complete, attack, formulation)
        ideal = pooled.min(axis=0)
        ranges = pooled.max(axis=0) - ideal
        if np.any(ranges <= 0):
            bad = np.flatnonzero(ranges <= 0).tolist()
            raise ValueError(f"Zero pooled objective ranges for {attack}: {bad}")

        for algorithm, group in complete.groupby("algorithm", sort=True):
            F = build_objectives(group, attack, formulation)
            F_normalized = (F - ideal) / ranges
            F_nd = clean_front(F_normalized, duplicate_decimals, dominance_atol)
            values.loc[algorithm, attack] = float(indicator(F_nd))

    if values.isna().any().any():
        missing = values.isna().stack()
        raise ValueError(
            "Missing algorithm/attack HV values: "
            + str(missing[missing].index.tolist())
        )
    return values


def compare_reference(
    attack_hv: pd.DataFrame,
    reference_algorithm: str,
    tie_atol: float,
) -> pd.DataFrame:
    if reference_algorithm not in attack_hv.index:
        available = ", ".join(map(str, attack_hv.index))
        raise ValueError(
            f"Reference algorithm {reference_algorithm!r} was not found. "
            f"Available identifiers: {available}"
        )

    reference = attack_hv.loc[reference_algorithm].to_numpy(dtype=np.float64)
    n_attacks = len(attack_hv.columns)
    rows = []

    for competitor in attack_hv.index:
        if competitor == reference_algorithm:
            continue
        differences = reference - attack_hv.loc[competitor].to_numpy(dtype=np.float64)
        wins = int(np.sum(differences > tie_atol))
        losses = int(np.sum(differences < -tie_atol))
        ties = int(n_attacks - wins - losses)
        rows.append(
            {
                "reference_algorithm": reference_algorithm,
                "competitor": competitor,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "number_of_attacks": n_attacks,
                "mean_signed_hv_difference": float(np.mean(differences)),
                "median_signed_hv_difference": float(np.median(differences)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["mean_signed_hv_difference", "wins"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )


def latex_name(name: str) -> str:
    return LATEX_NAMES.get(name, name.replace("_", r"\_"))


def write_latex(
    comparison: pd.DataFrame,
    output: Path,
    dataset: str,
    formulation: str,
    reference_algorithm: str,
    precision: int,
    tie_atol: float,
) -> None:
    objective_text = "classification error" if formulation == "accuracy" else "loss"
    reference_name = latex_name(reference_algorithm)
    n_attacks = int(comparison["number_of_attacks"].iloc[0])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            rf"\caption{{Pairwise comparison of {reference_name} against each "
            rf"competitor using normalized four-objective HV based on {objective_text} "
            rf"for {dataset.upper()}. Wins denote the number of attacks for which the "
            rf"reference algorithm obtains higher HV out of {n_attacks}. The signed "
            r"difference is the attack-wise mean of reference HV minus competitor HV; "
            r"positive values favor the reference algorithm.}"
        ),
        rf"\label{{tab:reference_hv_{formulation}_{dataset}}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Competitor & Wins & Mean signed $\Delta$HV \\",
        r"\midrule",
    ]

    for _, row in comparison.iterrows():
        competitor = latex_name(str(row["competitor"]))
        wins = int(row["wins"])
        losses = int(row["losses"])
        total = int(row["number_of_attacks"])
        difference = float(row["mean_signed_hv_difference"])
        wins_text = f"{wins}/{total}"
        difference_text = f"{difference:+.{precision}f}"

        if wins > losses and difference > tie_atol:
            wins_text = rf"\textbf{{{wins_text}}}"
            difference_text = rf"\textbf{{{difference_text}}}"

        lines.append(
            f"{competitor} & {wins_text} & {difference_text}" + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_precision(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 12:
        raise argparse.ArgumentTypeError("precision must be between 0 and 12")
    return parsed

# python3 compare_algorithms.py --csv ../test-evaluations.csv --dataset cifar10 --reference-algorithm r2-emoa-60-40 --formulation both --precision 3 --output-dir reference_tables/
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one algorithm against all competitors using attack-wise HV."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--reference-algorithm", required=True)
    parser.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS)
    parser.add_argument("--formulation", choices=["accuracy", "loss", "both"], default="both")
    parser.add_argument("--reference-margin", type=float, default=0.1)
    parser.add_argument("--duplicate-decimals", type=int, default=12)
    parser.add_argument("--dominance-atol", type=float, default=0.0)
    parser.add_argument("--tie-atol", type=float, default=1e-12)
    parser.add_argument("--precision", type=validate_precision, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reference_margin <= 0:
        raise ValueError("--reference-margin must be greater than zero.")
    if args.tie_atol < 0 or args.dominance_atol < 0:
        raise ValueError("Tolerance values must be non-negative.")

    df = read_csv_allowing_markdown_fences(args.csv)
    validate_csv(df)
    df, dataset = select_dataset(df, args.dataset)
    available = sorted(df["algorithm"].astype(str).unique())
    if args.reference_algorithm not in available:
        raise ValueError(
            f"Reference algorithm {args.reference_algorithm!r} was not found. "
            f"Available identifiers: {available}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    formulations = (
        ["accuracy", "loss"] if args.formulation == "both" else [args.formulation]
    )
    generated: list[Path] = []

    for formulation in formulations:
        measure = "accuracy" if formulation == "accuracy" else "loss"
        models = pivot_measure(df, measure)
        attack_hv = calculate_attack_hv(
            models=models,
            attacks=args.attacks,
            formulation=formulation,
            reference_margin=args.reference_margin,
            duplicate_decimals=args.duplicate_decimals,
            dominance_atol=args.dominance_atol,
        )
        comparison = compare_reference(
            attack_hv,
            args.reference_algorithm,
            args.tie_atol,
        )

        safe_algorithm = args.reference_algorithm.replace("/", "-")
        prefix = f"reference_hv_{safe_algorithm}_{formulation}"
        hv_path = args.output_dir / f"hv_by_attack_{formulation}.csv"
        csv_path = args.output_dir / f"{prefix}.csv"
        tex_path = args.output_dir / f"{prefix}.tex"

        attack_hv.to_csv(hv_path, index_label="algorithm")
        comparison.to_csv(csv_path, index=False)
        write_latex(
            comparison=comparison,
            output=tex_path,
            dataset=dataset,
            formulation=formulation,
            reference_algorithm=args.reference_algorithm,
            precision=args.precision,
            tie_atol=args.tie_atol,
        )
        generated.extend([hv_path, csv_path, tex_path])

    print("Generated files:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
