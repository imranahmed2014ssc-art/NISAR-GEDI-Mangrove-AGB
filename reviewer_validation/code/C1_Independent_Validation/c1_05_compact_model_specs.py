from pathlib import Path
import ast

ROOT = Path(r"D:\NISAR_Mangrove_Paper")

OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C1_Independent_Validation"
    / "C1_05_COMPACT_MODEL_SPECS.txt"
)

SCRIPTS = {
    "RF":
        ROOT / "05_ML_DL" / "RF" /
        "phase7b_random_forest.py",

    "XGBOOST":
        ROOT / "05_ML_DL" / "XGBoost" /
        "phase7c_xgboost.py",

    "SVR":
        ROOT / "05_ML_DL" / "SVR" /
        "phase7d_svr.py",

    "EXTRA_TREES":
        ROOT / "05_ML_DL" / "ExtraTrees" /
        "phase7e_extra_trees.py",

    "MLP":
        ROOT / "05_ML_DL" / "DNN" /
        "phase7f_mlp_dnn.py",

    "GAM":
        ROOT / "05_ML_DL" / "GAM" /
        "phase7g_gam.py",
}

IMPORTANT_WORDS = [
    "FEATURE",
    "PARAM",
    "SEARCH",
    "N_ITER",
    "N_SEARCH",
    "INNER",
    "OUTER",
    "RANDOM_SEED",
    "N_JOBS",
]


def get_target_names(node):
    names = []

    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return names

    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)

    return names


def source_segment(text, node):
    seg = ast.get_source_segment(text, node)
    return seg if seg is not None else ""


lines_out = []

lines_out.append("=" * 90)
lines_out.append("C1 COMPACT HISTORICAL MODEL SPECIFICATIONS")
lines_out.append("=" * 90)

for label, path in SCRIPTS.items():

    lines_out.append("")
    lines_out.append("=" * 90)
    lines_out.append(label)
    lines_out.append("=" * 90)
    lines_out.append(f"FILE: {path}")
    lines_out.append(f"EXISTS: {path.exists()}")

    if not path.exists():
        continue

    text = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    tree = ast.parse(text)

    # --------------------------------------------------------
    # Relevant top-level constants
    # --------------------------------------------------------

    lines_out.append("")
    lines_out.append("SETTINGS / SEARCH SPACES")
    lines_out.append("-" * 90)

    for node in tree.body:

        if not isinstance(
            node,
            (ast.Assign, ast.AnnAssign)
        ):
            continue

        names = get_target_names(node)

        if not names:
            continue

        keep = False

        for name in names:

            upper = name.upper()

            if any(
                word in upper
                for word in IMPORTANT_WORDS
            ):
                keep = True

        if keep:
            seg = source_segment(
                text,
                node
            )

            lines_out.append(seg)
            lines_out.append("")

    # --------------------------------------------------------
    # Estimator/search constructor calls
    # --------------------------------------------------------

    lines_out.append(
        "MODEL / SEARCH CONSTRUCTORS"
    )
    lines_out.append("-" * 90)

    constructors = [
        "RandomForestRegressor",
        "XGBRegressor",
        "SVR",
        "ExtraTreesRegressor",
        "MLPRegressor",
        "RandomizedSearchCV",
        "GridSearchCV",
        "GroupKFold",
        "StandardScaler",
        "SplineTransformer",
        "Ridge",
    ]

    seen = set()

    for node in ast.walk(tree):

        if not isinstance(node, ast.Call):
            continue

        func_name = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id

        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name not in constructors:
            continue

        seg = source_segment(
            text,
            node
        )

        if not seg:
            continue

        # Keep output compact.
        if len(seg) > 2000:
            seg = seg[:2000] + "\n...[truncated]"

        key = (
            func_name,
            seg
        )

        if key in seen:
            continue

        seen.add(key)

        lines_out.append("")
        lines_out.append(
            f"[{func_name}]"
        )
        lines_out.append(seg)

    lines_out.append("")


OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUT.write_text(
    "\n".join(lines_out),
    encoding="utf-8"
)

print("")
print("=" * 80)
print("C1 COMPACT MODEL SPEC EXTRACTION COMPLETE")
print("=" * 80)

for label, path in SCRIPTS.items():
    print(
        f"{label:12s}: "
        f"{'FOUND' if path.exists() else 'MISSING'}"
    )

print("")
print("Output:")
print(OUT)

print("")
print("No model was fitted.")
print("No previous file was modified.")
print("=" * 80)
