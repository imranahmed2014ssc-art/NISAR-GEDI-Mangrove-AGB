from pathlib import Path
import ast
import re

ROOT = Path(r"D:\NISAR_Mangrove_Paper")

OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C1_Independent_Validation"
    / "C1_04_FULL_HISTORICAL_MODEL_SPEC.txt"
)

SCRIPTS = [
    ROOT / "05_ML_DL" / "RF" /
    "phase7b_random_forest.py",

    ROOT / "05_ML_DL" / "XGBoost" /
    "phase7c_xgboost.py",

    ROOT / "05_ML_DL" / "SVR" /
    "phase7d_svr.py",

    ROOT / "05_ML_DL" / "ExtraTrees" /
    "phase7e_extra_trees.py",

    ROOT / "05_ML_DL" / "DNN" /
    "phase7f_mlp_dnn.py",

    ROOT / "05_ML_DL" / "GAM" /
    "phase7g_gam.py",

    ROOT / "05_ML_DL" /
    "Predictor_Expansion" /
    "Phase8C4_Scripts" /
    "phase8c4_expanded9p_spatial_cv.py",
]

KEY_ASSIGNMENTS = (
    "FEATURE",
    "PARAM",
    "N_ITER",
    "INNER",
    "RANDOM",
    "SEED",
    "SPLIT",
    "SCOR",
    "SEARCH",
)

KEY_FUNCTIONS = (
    "tune",
    "make",
    "group",
    "fit",
    "metric",
)

KEY_LINES = [
    "RandomForestRegressor",
    "XGBRegressor",
    "SVR(",
    "ExtraTreesRegressor",
    "MLPRegressor",
    "LinearRegression",
    "SplineTransformer",
    "Ridge",
    "GAM",
    "GroupKFold",
    "RandomizedSearchCV",
    "ParameterSampler",
    "GridSearchCV",
    "StandardScaler",
]


def assignment_names(node):
    names = []

    targets = []

    if isinstance(node, ast.Assign):
        targets = node.targets

    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]

    for t in targets:
        if isinstance(t, ast.Name):
            names.append(t.id)

    return names


def get_lines(text, node):
    lines = text.splitlines()

    start = max(
        getattr(node, "lineno", 1) - 1,
        0
    )

    end = min(
        getattr(node, "end_lineno", start + 1),
        len(lines)
    )

    return "\n".join(
        f"{i + 1:4d}: {lines[i]}"
        for i in range(start, end)
    )


report = []

report.append("=" * 100)
report.append(
    "C1 HISTORICAL MODEL / TUNING SPECIFICATION AUDIT"
)
report.append("=" * 100)

for path in SCRIPTS:

    report.append("")
    report.append("")
    report.append("=" * 100)
    report.append(f"FILE: {path}")
    report.append("=" * 100)

    if not path.exists():
        report.append("FILE NOT FOUND")
        continue

    text = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    lines = text.splitlines()

    try:
        tree = ast.parse(text)

    except Exception as e:
        report.append(
            f"AST PARSE ERROR: {e}"
        )
        continue

    # --------------------------------------------------------
    # Imports
    # --------------------------------------------------------

    report.append("")
    report.append("IMPORTS")
    report.append("-" * 100)

    for node in tree.body:

        if isinstance(
            node,
            (ast.Import, ast.ImportFrom)
        ):
            report.append(
                get_lines(text, node)
            )

    # --------------------------------------------------------
    # Relevant global constants / parameter spaces
    # --------------------------------------------------------

    report.append("")
    report.append(
        "GLOBAL SETTINGS / PARAMETER SPACES"
    )
    report.append("-" * 100)

    found_assignments = False

    for node in tree.body:

        if not isinstance(
            node,
            (ast.Assign, ast.AnnAssign)
        ):
            continue

        names = assignment_names(node)

        keep = any(
            any(
                token in name.upper()
                for token in KEY_ASSIGNMENTS
            )
            for name in names
        )

        if keep:
            found_assignments = True
            report.append(
                get_lines(text, node)
            )
            report.append("")

    if not found_assignments:
        report.append(
            "No matching global assignments detected."
        )

    # --------------------------------------------------------
    # Relevant tuning / construction functions
    # --------------------------------------------------------

    report.append("")
    report.append(
        "RELEVANT FUNCTIONS"
    )
    report.append("-" * 100)

    found_functions = False

    for node in tree.body:

        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue

        name_lower = node.name.lower()

        if any(
            token in name_lower
            for token in KEY_FUNCTIONS
        ):
            found_functions = True

            report.append("")
            report.append(
                f"FUNCTION: {node.name}"
            )
            report.append(
                get_lines(text, node)
            )
            report.append("")

    if not found_functions:
        report.append(
            "No matching functions detected."
        )

    # --------------------------------------------------------
    # Model-constructor/context lines
    # --------------------------------------------------------

    report.append("")
    report.append(
        "MODEL / CV / SEARCH CONTEXT"
    )
    report.append("-" * 100)

    selected_line_numbers = set()

    for i, line in enumerate(lines):

        if any(
            key.lower() in line.lower()
            for key in KEY_LINES
        ):

            for j in range(
                max(0, i - 5),
                min(len(lines), i + 11)
            ):
                selected_line_numbers.add(j)

    previous = None

    for j in sorted(selected_line_numbers):

        if (
            previous is not None
            and j > previous + 1
        ):
            report.append(
                "    ..."
            )

        report.append(
            f"{j + 1:4d}: {lines[j]}"
        )

        previous = j


OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUT.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("")
print("=" * 80)
print("C1 HISTORICAL MODEL SPEC EXTRACTION COMPLETE")
print("=" * 80)

print("")
print("Scripts inspected:")

for p in SCRIPTS:
    print(
        f"{'FOUND' if p.exists() else 'MISSING'} : {p}"
    )

print("")
print("Output:")
print(OUT)

print("")
print("No models were fitted.")
print("No existing result was modified.")
print("=" * 80)
