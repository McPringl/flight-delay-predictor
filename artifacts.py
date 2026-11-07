"""
Save and load the lookup tables the web app needs.

Why not pickle? Loading a pickle can run any code it contains, so a tampered
file could compromise your computer. Parquet and JSON only hold data.

Files written to the models/ folder:
    app_meta.json          feature list, category lists, overall late rate, best tree count
    encodings.parquet      the historical late-rate lookup tables (one row per key)
    climatology.parquet    typical weather per airport / month / hour
    route_info.parquet     typical distance and flight time per route
"""

import json
from pathlib import Path

import pandas as pd


def save_artifacts(model_dir: Path, *, features, categories, encodings,
                   climatology, route_info, best_iteration) -> None:
    """Write everything the app needs as JSON and parquet (no pickle)."""
    overall = float(encodings["overall"])
    # Each encoding table is a Series (key -> rate). Store them as one long table.
    rows = []
    for name, table in encodings.items():
        if name == "overall":
            continue
        rows.append(pd.DataFrame({"table": name,
                                  "key": table.index.astype(str),
                                  "rate": table.values.astype(float)}))
    pd.concat(rows, ignore_index=True).to_parquet(model_dir / "encodings.parquet", index=False)

    climatology.to_parquet(model_dir / "climatology.parquet", index=False)
    route_info.reset_index().rename(columns={"index": "route"}) \
        .to_parquet(model_dir / "route_info.parquet", index=False)

    meta = {
        "features": list(features),
        "categories": {k: [str(v) for v in vals] for k, vals in categories.items()},
        "overall": overall,
        "best_iteration": int(best_iteration),
    }
    (model_dir / "app_meta.json").write_text(json.dumps(meta, indent=2))


def load_artifacts(model_dir: Path) -> dict:
    """Read the files written by save_artifacts() back into the same structure."""
    meta = json.loads((model_dir / "app_meta.json").read_text())

    encodings = {"overall": meta["overall"]}
    long = pd.read_parquet(model_dir / "encodings.parquet")
    for name, group in long.groupby("table"):
        encodings[name] = pd.Series(group["rate"].to_numpy(), index=group["key"].to_numpy())

    climatology = pd.read_parquet(model_dir / "climatology.parquet")
    route_info = pd.read_parquet(model_dir / "route_info.parquet").set_index("route")

    return {
        "features": meta["features"],
        "categories": meta["categories"],
        "encodings": encodings,
        "climatology": climatology,
        "route_info": route_info,
        "best_iteration": meta["best_iteration"],
    }
