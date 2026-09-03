"""
Hand-eye calibration SOLVER (offline).

Reads calib_pairs.csv (camera-frame metres + robot-frame mm), computes the
best-fit rigid transform  camera -> robot base  (Kabsch/Umeyama), reports the
per-marker residual so you can spot bad touches, and saves handeye.json.

Usage:  python pde4445-dev/calib_solve.py
"""
import os, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "calib_pairs.csv")
OUT  = os.path.join(HERE, "handeye.json")


def rigid_transform(A, B):
    """Best-fit R,t so that R@A_i + t ~= B_i.  A,B: (N,3), same units.
    Returns R (3,3), t (3,), and a diagnostic scale (should be ~1.0)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    cA, cB = A.mean(0), B.mean(0)
    AA, BB = A - cA, B - cB
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))          # reflection guard
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cB - R @ cA
    scale = float((S * np.array([1, 1, d])).sum() / (AA ** 2).sum())
    return R, t, scale


def main():
    A, B, ids = [], [], []
    with open(CSV) as f:
        for row in csv.DictReader(f):
            if row.get("robot_x_mm", "").strip() == "":
                continue                                     # marker not touched yet
            try:
                A.append([float(row["cam_x_m"]) * 1000.0,     # metres -> mm
                          float(row["cam_y_m"]) * 1000.0,
                          float(row["cam_z_m"]) * 1000.0])
                B.append([float(row["robot_x_mm"]),
                          float(row["robot_y_mm"]),
                          float(row["robot_z_mm"])])
                ids.append(row.get("id", "?"))
            except (KeyError, ValueError):
                continue

    A, B = np.array(A), np.array(B)
    n = len(A)
    if n < 4:
        print(f"Only {n} usable pairs — need >=4 (ideally 8+). Fill more robot_* cells.")
        return

    R, t, scale = rigid_transform(A, B)
    pred = (R @ A.T).T + t
    err = np.linalg.norm(pred - B, axis=1)                    # per-marker mm
    rms = float(np.sqrt((err ** 2).mean()))

    print(f"Pairs used: {n}   |   scale (want ~1.00): {scale:.3f}")
    if abs(scale - 1.0) > 0.05:
        print("  ! scale is off — likely a units mix-up or a few bad points.")
    print("\n  id    residual(mm)")
    for i, e in sorted(zip(ids, err), key=lambda x: -x[1]):
        flag = "   <-- large, consider dropping" if (e > 3 * rms and e > 5) else ""
        print(f"  {str(i):>3}    {e:6.2f}{flag}")
    print(f"\nRMS residual: {rms:.2f} mm   |   max: {err.max():.2f} mm")
    print("Target RMS < 5 mm  ->  " + ("GOOD ✓" if rms < 5 else "HIGH: fix flagged touches and re-run"))

    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    json.dump({
        "note": "p_robot_mm = R @ (p_camera_metres * 1000) + t_mm",
        "R": R.tolist(), "t_mm": t.tolist(), "T_4x4": T.tolist(),
        "rms_mm": rms, "n_pairs": n, "scale": scale
    }, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
