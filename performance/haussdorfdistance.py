
import os, math, json, argparse, glob
import numpy as np
import pandas as pd
import nibabel as nib
from surface_distance import compute_surface_distances

def load_nifti(path):
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    if img.affine is not None and np.linalg.det(img.affine) != 0:
        spacing = nib.affines.voxel_sizes(img.affine)
    else:
        spacing = img.header.get_zooms()[:3]
    return data, spacing
def remap_labels(arr):
    """Convert 1,2,4 → 1 and 3 → 2; leave everything else 0."""
    remapped = np.zeros_like(arr)
    remapped[np.isin(arr, [1, 2, 4])] = 1
    remapped[arr == 3] = 2
    return remapped

def hd95_stats_for_label(manual_arr, auto_arr, spacing, label):
    gt = (manual_arr == label)
    pr = (auto_arr   == label)

    if not gt.any() and not pr.any():
        return dict(hd95=np.nan, mean=np.nan, std=np.nan, se=np.nan, n=0)
    if gt.any() ^ pr.any():  # present in only one mask
        return dict(hd95=math.inf, mean=math.inf, std=0.0, se=0.0, n=0)

    sd = compute_surface_distances(gt, pr, spacing_mm=spacing)
    d = np.concatenate([sd["distances_gt_to_pred"].ravel(),
                        sd["distances_pred_to_gt"].ravel()])
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(hd95=np.nan, mean=np.nan, std=np.nan, se=np.nan, n=0)

    hd95 = float(np.percentile(d, 95))
    mean = float(np.mean(d))
    std  = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    se   = float(std / np.sqrt(d.size)) if d.size > 0 else np.nan
    return dict(hd95=hd95, mean=mean, std=std, se=se, n=int(d.size))

def find_pairs(manual_dir, auto_dir):
    def stem(p):
        base = os.path.basename(p)
        if base.endswith(".nii.gz"):
            return base[:-7]
        elif base.endswith(".nii"):
            return base[:-4]
        return os.path.splitext(base)[0]

    mfiles = glob.glob(os.path.join(manual_dir, "*.nii")) + \
             glob.glob(os.path.join(manual_dir, "*.nii.gz"))
    afiles = glob.glob(os.path.join(auto_dir, "*.nii")) + \
             glob.glob(os.path.join(auto_dir, "*.nii.gz"))

    mmap = {stem(p): p for p in mfiles}
    amap = {stem(p): p for p in afiles}

    common = sorted(set(mmap.keys()) & set(amap.keys()))
    missing_auto = sorted(set(mmap.keys()) - set(amap.keys()))
    missing_manual = sorted(set(amap.keys()) - set(mmap.keys()))
    return [(c, mmap[c], amap[c]) for c in common], missing_manual, missing_auto

def run(manual_dir, auto_dir, labels, out_csv, out_wide_csv=None):
    pairs, missing_manual, missing_auto = find_pairs(manual_dir, auto_dir)
    if not pairs:
        raise SystemExit("No matching case stems between folders.")

    rows = []
    for case_id, mpath, apath in pairs:
        try:
            m_arr, m_sp = load_nifti(mpath)
            a_arr, a_sp = load_nifti(apath)


# Apply label remapping 
            m_arr = remap_labels(m_arr)
            a_arr = remap_labels(a_arr)

            if m_arr.shape != a_arr.shape:
                raise ValueError(f"Shape mismatch for {case_id}: {m_arr.shape} vs {a_arr.shape} "
                                 f"(resample needed).")
            if not np.allclose(m_sp, a_sp, rtol=0, atol=1e-5):
                # Use manual spacing for distances; warn in output
                spacing_note = f"spacing differs: manual {m_sp} auto {a_sp}"
            else:
                spacing_note = ""

            for lab in labels:
                stats = hd95_stats_for_label(m_arr, a_arr, m_sp, lab)
                rows.append({
                    "case": case_id,
                    "label": int(lab),
                    "hd95_mm": stats["hd95"],
                    "mean_mm": stats["mean"],
                    "std_mm": stats["std"],
                    "se_mm": stats["se"],
                    "n_samples": stats["n"],
                    "note": spacing_note
                })
        except Exception as e:
            rows.append({
                "case": case_id, "label": None,
                "hd95_mm": np.nan, "mean_mm": np.nan, "std_mm": np.nan, "se_mm": np.nan,
                "n_samples": 0, "note": f"ERROR: {e}"
            })

    df = pd.DataFrame(rows).sort_values(["case","label"], kind="stable")
    df.to_csv(out_csv, index=False)


    if out_wide_csv:
 
        metrics = ["hd95_mm", "mean_mm", "std_mm", "se_mm"]
        pieces = []
        for m in metrics:
            pivot = df.pivot_table(index="case", columns="label", values=m, aggfunc="first")
            pivot.columns = [f"{m}_label{int(c)}" for c in pivot.columns]
            pieces.append(pivot)
        wdf = pd.concat(pieces, axis=1)
        wdf = wdf.reset_index()
        wdf.to_csv(out_wide_csv, index=False)

    by_label = df.dropna(subset=["label"]).groupby("label")

  
    summary = by_label["hd95_mm"].agg(["count", "mean", "std"]).rename(columns={
        "count": "cases",
        "mean": "mean_hd95_mm",
        "std": "sd_hd95_mm"
    })
    summary["se_hd95_mm"] = summary["sd_hd95_mm"] / np.sqrt(summary["cases"])

    
    summary = summary[["cases", "mean_hd95_mm", "se_hd95_mm"]]

    print("\nPer-label HD95 summary across cases (with SE):")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    
    if missing_manual:
        print(f"\nWarning: {len(missing_manual)} cases only in AUTO folder (no MANUAL): {missing_manual[:5]}{' ...' if len(missing_manual)>5 else ''}")
    if missing_auto:
        print(f"Warning: {len(missing_auto)} cases only in MANUAL folder (no AUTO): {missing_auto[:5]}{' ...' if len(missing_auto)>5 else ''}")
    print(f"\nWrote per-case metrics to: {out_csv}")
    if out_wide_csv:
        print(f"Wrote wide per-case table to: {out_wide_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch HD95/SD/SE for multilabel NIfTI segmentations.")
    ap.add_argument("--manual_dir", required=True, help="Folder with manual .nii/.nii.gz")
    ap.add_argument("--auto_dir", required=True, help="Folder with automatic .nii/.nii.gz")
    ap.add_argument("--labels", nargs="+", type=int, default=[1,2], help="Labels to evaluate")
    ap.add_argument("--out_csv", default="hd95_per_case_long.csv", help="Output long CSV")
    ap.add_argument("--out_wide_csv", default="hd95_per_case_wide.csv", help="Optional wide CSV")
    args = ap.parse_args()
    run(args.manual_dir, args.auto_dir, args.labels, args.out_csv, args.out_wide_csv)
