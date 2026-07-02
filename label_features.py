"""
离线打标签脚本：读取 data/features/*.parquet，按固定周期(默认10分钟)
生成事件合约的涨跌标签，输出到 data/labeled/labeled_features.parquet

用法:
    python label_features.py --horizon-min 10
"""
import argparse
import glob
import os

import pandas as pd

FEATURES_DIR = "data/features"
OUTPUT_DIR = "data/labeled"


def load_features() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*.parquet")))
    if not files:
        raise SystemExit(f"No feature files found in {FEATURES_DIR}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def add_labels(df: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    horizon_ms = horizon_min * 60 * 1000
    df["future_ts"] = df["timestamp_ms"] + horizon_ms

    # 用 merge_asof 找到每行 t+horizon 时刻最接近的未来价格
    future = df[["timestamp_ms", "price"]].rename(
        columns={"timestamp_ms": "future_ts", "price": "future_price"}
    )
    df = pd.merge_asof(
        df.sort_values("future_ts"),
        future.sort_values("future_ts"),
        on="future_ts",
        direction="nearest",
        tolerance=60_000,  # 找不到1分钟内的对应点就丢弃(说明还没采集到未来数据)
    ).sort_values("timestamp_ms").reset_index(drop=True)

    df = df.dropna(subset=["future_price"])
    df["future_return"] = (df["future_price"] - df["price"]) / df["price"]
    df["label"] = (df["future_return"] > 0).astype(int)  # 1=涨, 0=跌/平
    return df.drop(columns=["future_ts"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-min", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_features()
    labeled = add_labels(df, args.horizon_min)

    out_path = os.path.join(OUTPUT_DIR, "labeled_features.parquet")
    labeled.to_parquet(out_path, index=False)

    print(f"Loaded {len(df)} feature rows, produced {len(labeled)} labeled rows "
          f"(dropped {len(df) - len(labeled)} without a future point yet).")
    print(f"Label balance: {labeled['label'].mean():.4f} up-ratio")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
