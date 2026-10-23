"""
AMPCA: OSM algorithm (Algorithm 3.1)GPU implementation.
"""

import time
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torch

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
QR_SAVE_ITERS = [1, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000]
SAVE_TOP_N = 1260
PROGRESS_EVERY = 1000
LE_BATCH_SIZE = 4


def make_shared_init(m, k1, p, k2, seed):
    torch.manual_seed(seed)
    U0 = torch.randn((m, max(m, k1)), device=DEVICE)
    U0, _ = torch.linalg.qr(U0)
    U0 = U0[:, :k1].cpu().numpy()

    V0 = torch.randn((p, max(p, k2)), device=DEVICE)
    V0, _ = torch.linalg.qr(V0)
    V0 = V0[:, :k2].cpu().numpy()
    return U0, V0


def init_data_qr(X_list, k1, k2, img_idx=0):
    X = X_list[img_idx % len(X_list)]
    X = torch.tensor(X, device=DEVICE)
    U, _ = torch.linalg.qr(X)
    U = U[:, :k1].cpu().numpy()
    V, _ = torch.linalg.qr(X.T)
    V = V[:, :k2].cpu().numpy()
    return U, V


def build_S_diag(dim, roi_ranges):
    s = torch.ones(dim, device=DEVICE)
    for (lo, hi) in roi_ranges:
        s[max(0, lo - 1):min(dim, hi)] = 0.0
    return s.cpu().numpy()


def scatter_row_gpu(Z_tensor, V_gpu):
    """
    Z_tensor: (n, m, p) 预载在 GPU 上的残差张量
    V_gpu: (p, k2) 投影矩阵
    """
    # Y = Z * V  -> (n, m, k2)
    Y = torch.matmul(Z_tensor, V_gpu)
    # G = sum(Y_i * Y_i^T)
    # 矩阵化写法：将 Y 展平为 (m, n*k2)，然后做一次矩阵乘法
    Y_reshaped = Y.permute(1, 0, 2).reshape(Y.shape[1], -1)
    G = torch.matmul(Y_reshaped, Y_reshaped.T)
    return G


def scatter_col_gpu(Z_tensor, U_gpu):
    """
    Z_tensor: (n, m, p)
    U_gpu: (m, k1)
    """
    # Y = Z^T * U -> (n, p, k1)
    Y = torch.matmul(Z_tensor.transpose(1, 2), U_gpu)
    Y_reshaped = Y.permute(1, 0, 2).reshape(Y.shape[1], -1)
    G = torch.matmul(Y_reshaped, Y_reshaped.T)
    return G


def compute_LE_gpu(Z_stack):
    """Definition: LE=sum_i ||Zi||_2^2
    """
    total = torch.zeros((), dtype=Z_stack.dtype, device=Z_stack.device)
    for start in range(0, len(Z_stack), LE_BATCH_SIZE):
        values = torch.linalg.svdvals(Z_stack[start:start + LE_BATCH_SIZE])
        total += torch.sum(values[:, 0] ** 2)
        if start == 0 or (start // LE_BATCH_SIZE + 1) % 25 == 0:
            print(f"  [LE/GPU] {min(start+LE_BATCH_SIZE,len(Z_stack))}/{len(Z_stack)}", flush=True)
    value = float(total)
    if not math.isfinite(value) or value < 0:
        raise ValueError("LE is not finite/nonnegative")
    return value


def surrogate_coefficients(LE, beta, S1_diag, S2_diag, c_mode, lam, mu):
    """修改处（2026-09-28）：采用 0<lambda,mu<1/(2LE+1) 的充分界。

    beta>=0 时，-beta*S1.T*S1 为负半定，因此
    lambda_max(M_U) <= lambda_max(2*E1t+0.5*Ut*Ut.T) <= LE+0.5。
    V方向同理。这里直接界定最大特征值。
    仅更换自动步长/手动步长校验的上界；更新矩阵中的beta惩罚仍然保留。
    保持原函数参数接口、0.9安全系数及 gamma=C+1/(2*lambda) 不变。
    """
    if beta < 0:
        raise ValueError("The stated sufficient condition assumes beta>=0")
    modes = {"ZERO": 0.0, "LE": LE, "LE_SQUARED": LE ** 2}
    if c_mode not in modes:
        raise ValueError("C_MODE must be ZERO, LE or LE_SQUARED")
    C = modes[c_mode]
    # 【唯一系数修改】旧界分母为 2*LE+1+2*beta*max(S_diag**2)。
    # 新界利用惩罚项的负半定性，去掉分母中的beta项；与迭代t无关。
    upper_U = 1.0 / (2.0 * LE + 1.0)
    upper_V = 1.0 / (2.0 * LE + 1.0)
    lam = 0.9 * upper_U if lam is None else float(lam)
    mu = 0.9 * upper_V if mu is None else float(mu)
    # Do not silently change user-supplied parameters if they violate the paper.
    if not (0 < lam < upper_U and 0 < mu < upper_V):
        raise ValueError(f"Sufficient bounds require 0<lambda<{upper_U:.6e}, "
                         f"0<mu<{upper_V:.6e}; supplied {lam}, {mu}")
    gamma_U, gamma_V = C + 1.0/(2.0*lam), C + 1.0/(2.0*mu)
    if not all(math.isfinite(v) for v in (C,gamma_U,gamma_V)):
        raise ValueError("Nonfinite surrogate coefficient")
    return C, lam, mu, gamma_U, gamma_V, upper_U, upper_V


def qr_positive_diagonal(W):
    """Unique reduced QR for full-column-rank W: diag(R)>0.

    This avoids arbitrary sign jumps in the Frobenius proximal displacement.
    """
    Q, R = torch.linalg.qr(W, mode="reduced")
    signs = torch.where(torch.diagonal(R) < 0, -1.0, 1.0)
    return Q * signs.unsqueeze(0)


def append_iteration_row(path, row, first=False):
    # Close/flush after every complete row: Ctrl+C does not lose earlier iterations.
    with Path(path).open('x' if first else 'a',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(row))
        if first:
            writer.writeheader()
        writer.writerow(row)


def synchronize_iteration_device():
    """【修改处】排除前置异步工作，且确保本轮 CUDA 计算完成后再停表。"""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def mpca_qr(X_list, Psi, beta, S1_diag, S2_diag, k1, k2,
            U0, V0,
            lam=None, mu_prox=None,
            tol=1e-4, max_iter=20000,
            save_iters=None,
            verbose=True, c_mode="ZERO", objective_dir=None):
    if save_iters is None: save_iters = QR_SAVE_ITERS
    if not 1 <= max_iter <= 20000:
        raise ValueError("MAX_ITER_QR must be in [1,20000]; extra iterations are disabled")
    if objective_dir is None:
        raise ValueError("objective_dir is required for iteration logging")
    objective_dir = Path(objective_dir)
    objective_dir.mkdir(parents=True,exist_ok=True)
    objective_csv = objective_dir / 'qr_iteration_history.csv'
    if objective_csv.exists():
        raise FileExistsError(f"Refusing to overwrite previous iteration log: {objective_csv}")


    print("  [GPU] 预载数据并建立 Z_stack (Double Precision)...")
    X_stack = torch.tensor(np.stack(X_list), device=DEVICE, dtype=torch.float64)
    Psi_g = torch.tensor(Psi, device=DEVICE, dtype=torch.float64)
    Z_stack = X_stack - Psi_g  # (n, m, p) 的张量
    del X_stack


    U = torch.tensor(U0, device=DEVICE, dtype=torch.float64)
    V = torch.tensor(V0, device=DEVICE, dtype=torch.float64)
    beta_g = float(beta)
    S1_g = torch.tensor(S1_diag, device=DEVICE, dtype=torch.float64)
    S2_g = torch.tensor(S2_diag, device=DEVICE, dtype=torch.float64)

    m, p = Psi_g.shape
    print("  [GPU] 开始计算LE=sum ||Xi-Psi||_2^2...",flush=True)
    LE = compute_LE_gpu(Z_stack)
    C,lam,mu_prox,gamma_U,gamma_V,upper_U,upper_V = surrogate_coefficients(
        LE,beta,S1_diag,S2_diag,c_mode,lam,mu_prox)
    print(f"  C_MODE={c_mode} LE={LE:.12e} C={C:.12e}",flush=True)
    print(f"  lambda={lam:.12e} mu={mu_prox:.12e} gamma_U={gamma_U:.12e} gamma_V={gamma_V:.12e}",flush=True)
    print(f"  新增项: M_U=P-beta*S1^T*S1+E1t; M_V=Q-beta*S2^T*S2+E2t",flush=True)
    print(f"  逐轮 dist 和计算时间 → {objective_csv}",flush=True)
    metadata = dict(C_MODE=c_mode,C=C,LE=LE,LE_definition='sum of squared matrix spectral norms; no averaging',
        lam=lam,mu=mu_prox,prox_U=1/(2*lam),prox_V=1/(2*mu_prox),gamma_U=gamma_U,gamma_V=gamma_V,
        lambda_strict_upper=upper_U,mu_strict_upper=upper_V,beta=beta,max_iter=max_iter,
        global_lower_bound_certified=(C>=LE**2),
        qr_convention='reduced QR with positive R diagonal',update='simultaneous U,V from old U_t,V_t',
        device=str(DEVICE),dtype=str(Z_stack.dtype),
        lower_bound_does_not_guarantee_improvement_after_QR=True,
        timing_definition='sum of synchronized U/V update, QR and subspace-distance compute blocks',
        timing_excludes=['preprocessing','LE','CSV writes',
                         'progress prints','snapshot transfers',
                         'reconstruction','plots','all image/matrix/result saves'],
        snapshot_policy='cache basis matrices during iterations; reconstruct/save only after the loop',
        elapsed_seconds_column='cumulative iteration compute seconds; NOT total wall-clock seconds')
    (objective_dir/'surrogate_parameters.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

    betaS1 = beta_g * S1_g ** 2
    betaS2 = beta_g * S2_g ** 2

    dist_U_hist, dist_V_hist = [], []
    snapshots = []
    save_set = {i for i in save_iters if 1 <= i <= max_iter}
    diag_u = torch.arange(m,device=DEVICE)
    diag_v = torch.arange(p,device=DEVICE)

    synchronize_iteration_device()
    loop_wall_t0 = time.perf_counter()
    iteration_compute_seconds = 0.0
    converged = False

    # 3. --- 进入 GPU 加速循环 ---
    for t in range(max_iter):
        iter_num = t + 1
        # 上轮日志/快照等异步工作不混入本轮计时。
        synchronize_iteration_device()
        compute_t0 = time.perf_counter()

        # 使用向量化函数计算行/列散射矩阵 (此时只需要 2 个参数)
        E1t = 0.5 * scatter_row_gpu(Z_stack, V)  # 调用你新定义的函数
        E2t = 0.5 * scatter_col_gpu(Z_stack, U)  # 调用你新定义的函数

        P = E1t + 0.5 * (U @ U.T)
        Q_mat = E2t + 0.5 * (V @ V.T)

        M_U_original = P.clone()
        M_U_original[diag_u,diag_u] -= betaS1
        M_U = M_U_original + E1t
        A_U = -M_U / gamma_U
        A_U[diag_u,diag_u] += 1.0

        M_V_original = Q_mat.clone()
        M_V_original[diag_v,diag_v] -= betaS2
        M_V = M_V_original + E2t
        A_V = -M_V / gamma_V
        A_V[diag_v,diag_v] += 1.0

        # GPU 矩阵求解： solve(A, b)
        Ut = torch.linalg.solve(A_U, U)
        Vt = torch.linalg.solve(A_V, V)

        # 正交化
        U_new = qr_positive_diagonal(Ut)[:, :k1]
        V_new = qr_positive_diagonal(Vt)[:, :k2]

        # --- 收敛计算 (subspace distance) ---
        s_u = torch.linalg.svdvals(U_new.T @ U)
        dU = float(torch.sqrt(torch.clamp(1.0 - s_u[-1] ** 2, min=0.0)))
        s_v = torch.linalg.svdvals(V_new.T @ V)
        dV = float(torch.sqrt(torch.clamp(1.0 - s_v[-1] ** 2, min=0.0)))

        # 【修改处】计时到 dist 完成即停止，后续日志/快照均在计时外。
        synchronize_iteration_device()
        iteration_seconds = time.perf_counter() - compute_t0
        iteration_compute_seconds += iteration_seconds
        dist_U_hist.append(dU);
        dist_V_hist.append(dV)

        row = dict(iteration=iter_num, dist_U=dU, dist_V=dV,
                   elapsed_seconds=iteration_compute_seconds,
                   iteration_compute_seconds=iteration_seconds)
        append_iteration_row(objective_csv, row, first=(iter_num == 1))
        if verbose and (iter_num==1 or iter_num%PROGRESS_EVERY==0 or iter_num==max_iter or (dU<tol and dV<tol)):
            print(f"  [QR] {iter_num}/{max_iter} dist_U={dU:.3e} dist_V={dV:.3e} "
                  f"迭代计算时间={iteration_compute_seconds:.4f}s",flush=True)

        U, V = U_new, V_new

        if iter_num in save_set or iter_num == max_iter:
            snapshots.append((iter_num, U.cpu().numpy(), V.cpu().numpy()))

        if dU < tol and dV < tol:
            converged = True
            if not snapshots or snapshots[-1][0] != iter_num:
                snapshots.append((iter_num, U.cpu().numpy(), V.cpu().numpy()))
            break

    loop_wall_elapsed = time.perf_counter() - loop_wall_t0
    elapsed = iteration_compute_seconds
    status = 'subspace_tolerance_reached' if converged else 'max_iterations_reached'
    print(f"  [QR] {status}: {t+1}轮；小位移不等价于原目标的驻点收敛",flush=True)
    (objective_dir/'qr_stop_status.json').write_text(json.dumps(dict(status=status,iterations=t+1,
        elapsed_seconds=elapsed,iteration_compute_seconds=elapsed,
        iteration_wall_seconds=loop_wall_elapsed,
        excluded_loop_overhead_seconds=max(0.0,loop_wall_elapsed-elapsed),
        timing_definition=metadata['timing_definition'],
        dist_U=dist_U_hist[-1],dist_V=dist_V_hist[-1]),indent=2),encoding='utf-8')
    # 最终结果转回 CPU 交给后续绘图逻辑
    U_final, V_final = U.cpu().numpy(), V.cpu().numpy()

    return U_final, V_final, t + 1, dist_U_hist, dist_V_hist, elapsed, snapshots


def reconstruct(X_list, Psi, U, V):
    UUT = U @ U.T; VVT = V @ V.T
    return [UUT @ (Xi - Psi) @ VVT + Psi for Xi in X_list]


def load_images_from_dir(image_dir, target_size=None, to_gray=True):
    img_dir = Path(image_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"目录不存在：{img_dir}")


    paths = sorted([p for p in img_dir.rglob('*')
                    if p.suffix.lower() in SUPPORTED_EXTS])

    if not paths:
        raise RuntimeError(f"未找到图像：{img_dir}")

    print(f"  📸 找到 {len(paths)} 张图像（包含子文件夹）")

    names, arrays = [], []
    for p in paths:
        try:
            img = Image.open(p)
            img = img.convert("L") if to_gray else img.convert("RGB")
            if target_size:
                img = img.resize(target_size, Image.LANCZOS)

            rel_name = "_".join(p.relative_to(img_dir).parts)
            names.append(rel_name)

            arrays.append(np.array(img, dtype=np.float64))
        except Exception as e:
            print(f"   跳过损坏文件 {p.name}: {e}")

    if not target_size:
        shapes = {a.shape for a in arrays}
        if len(shapes) > 1:
            raise RuntimeError(f"图像尺寸不一致 {shapes}，请设置 TARGET_SIZE 或检查数据")

    return names, arrays


def save_image(arr, path):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def _write_single_dist_txt(hist, path, alg, direction, tol):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Algorithm : {alg}\n")
        f.write(f"# Metric    : dist_{direction} = sqrt(1-sigma_min({direction}_t^T {direction}_{{t-1}})^2)"
                f"  [subspace distance]  in [0, 1]\n")
        f.write(f"# tol       : {tol}\n")
        f.write(f"# iter\tdist_{direction}\n")
        for i, v in enumerate(hist):
            f.write(f"{i+1}\t{v:.12e}\n")


def save_all_iter_txts(qr_dU, qr_dV, folder, tol):
    _write_single_dist_txt(qr_dU, folder / "qr_dist_U.txt", "OSM", "U", tol)
    _write_single_dist_txt(qr_dV, folder / "qr_dist_V.txt", "OSM", "V", tol)
    print("  qr_dist_U.txt / qr_dist_V.txt")


def save_summary_txt(path, params,
                     qr_iters, qr_elapsed, qr_dU, qr_dV):

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  AMPCA EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("[PARAMETERS]\n")
        for k, v in params.items():
            f.write(f"  {k:<28} = {v}\n")

        f.write("\n[QR INITIALIZATION]\n")
        f.write(f"  Generated by SEED={params['SEED']} via QR(randn)\n")
        f.write(f"  INIT_MODE = {params['INIT_MODE']}\n")

        f.write("\n[CONVERGENCE CRITERION  —  subspace distance]\n")
        f.write("  dist(S_t, S_{t-1}) = sqrt(1 - sigma_min(U_t^T U_{t-1})^2)  in [0,1]\n")
        f.write("  Converged when dist_U < tol AND dist_V < tol\n\n")

        f.write(f"  {'Metric':<40} {'QR':>14}\n")
        f.write("  " + "-" * 70 + "\n")

        def _row(label, value, fmt="{:.6f}"):
            f.write(f"  {label:<40} {fmt.format(value):>14}\n")

        _row("Iterations",
             qr_iters, "{:d}")
        _row("Iteration compute time (s)", qr_elapsed, "{:.4f}")
        _row("Final dist_U (subspace distance, consecutive)",
             qr_dU[-1], "{:.4e}")
        _row("Final dist_V (subspace distance, consecutive)",
             qr_dV[-1], "{:.4e}")
        f.write("\n[TIMING SCOPE]\n")
        f.write("  Includes only synchronized U/V updates, QR and dist calculations.\n")
        f.write("  Excludes loading, LE, logging, snapshots,\n")
        f.write("  reconstruction, plotting and image/matrix/result saving.\n")


def _ax_style(ax):
    ax.set_facecolor("#f8f9fa")
    for sp in ax.spines.values(): sp.set_color("#555")
    ax.tick_params(colors="#333")
    ax.grid(True, color="#ddd", alpha=0.7)


def plot_convergence_curve(qr_dU, qr_dV, tol, qr_label, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    c_q = "#ff7f0e"
    for ax, (dq, title, yl) in zip(axes, [
        (qr_dU, "Row Subspace  U  —  dist_U", "dist_U"),
        (qr_dV, "Col Subspace  V  —  dist_V", "dist_V"),
    ]):
        _ax_style(ax)
        xq = np.arange(1, len(dq)+1)
        ax.semilogy(xq, dq, color=c_q, lw=2, ls="--", marker="s", ms=3,
                    markevery=max(1, len(dq)//15),
                    label=f"Algorithm 3.1 (OSM) (QR, {qr_label})")
        ax.axhline(tol, color="#888", ls=":", lw=1.5, label=f"tol={tol:.0e}")
        ax.set_xlabel("Iteration t", fontsize=11)
        ax.set_ylabel(f"{yl}  [subspace distance, in [0,1]]", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, facecolor="white", edgecolor="#ccc")
    fig.suptitle(f"Convergence Curve (subspace distance)  —  QR init: {qr_label}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  收敛曲线图 → {path}")


def plot_reconstruction_grid(names, originals, recon_qr,
                             n_show, qr_label, path):
    n   = min(n_show, len(originals))
    fig = plt.figure(figsize=(12, 3.8 * n))
    fig.patch.set_facecolor("#0d0d0d")
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.06, wspace=0.04)
    headers = ["Original", f"QR Recon\n({qr_label})", "QR Residual×3"]

    def _ps(o, r):
        mse = np.mean((o - np.clip(r, 0, 255)) ** 2)
        return 10 * np.log10(255 ** 2 / mse) if mse > 1e-10 else 100.0

    for row in range(n):
        orig = originals[row]
        qr   = np.clip(recon_qr[row],  0, 255)
        data = [orig, qr, (orig - qr) * 3]
        cms  = ["gray"] * 2 + ["RdBu_r"]
        for col, (d, cm) in enumerate(zip(data, cms)):
            ax   = fig.add_subplot(gs[row, col])
            vmin = d.min() if col >= 2 else 0
            vmax = d.max() if col >= 2 else 255
            ax.imshow(d, cmap=cm, vmin=vmin, vmax=vmax,
                      aspect="auto", interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(headers[col], color="white",
                             fontsize=10, fontweight="bold", pad=5)
            if col == 1:
                ax.text(0.02, 0.02, f"PSNR={_ps(orig,qr):.1f}dB",
                        transform=ax.transAxes, color="#f06292", fontsize=8, va="bottom")
            if col == 0:
                lbl = names[row][:16] + ("…" if len(names[row]) > 16 else "")
                ax.set_ylabel(lbl, color="#aaa", fontsize=7,
                              rotation=0, labelpad=58, va="center")
    fig.suptitle(f"AMPCA Reconstruction  (QR init: {qr_label})",
                 color="white", fontsize=14, fontweight="bold", y=1.002)
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  重建对比图 → {path}")


def save_snapshot_images(X_list, Psi, names, snapshots,
                         base_dir, top_n, tag):
    save_n = min(top_n, len(names))
    for iter_num, U_snap, V_snap in snapshots:
        subdir = base_dir / f"iter_{iter_num}"
        subdir.mkdir(exist_ok=True)
        recon = reconstruct(X_list[:save_n], Psi, U_snap, V_snap)
        for i in range(save_n):
            stem = Path(names[i]).stem
            save_image(recon[i], subdir / f"{stem}_{tag}_iter{iter_num}.png")
        print(f"  ✓ {tag.upper()} iter={iter_num:>6}: 前{save_n}张 → {subdir.name}/")


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("本实验要求GPU。CUDA不可用，请检查pytorch_env解释器；不会静默改用CPU。")
    # ╔══════════════════════════════════════════════════════════╗
    #   ===== 参数设置 (所有可调参数集中在此) =====
    # ╚══════════════════════════════════════════════════════════╝

    # ── 路径 ─────────────────────────────────────────────────
    IMAGE_DIR  = r""
    OUTPUT_DIR = r""

    # ── 图像预处理 ────────────────────────────────────────────
    TO_GRAY     = True
    TARGET_SIZE = None

    # ── 投影维数 ──────────────────────────────────────────────
    K1 = 50
    K2 = 50

    # ── ROI 约束 ─────────────────────────────────────────────
    BETA    = 1e12
    ROW_ROI = [(240, 400)]
    COL_ROI = [(190, 320)]

    # ── 步长 ─────────────────────────────────────────────────
    LAMBDA  = None
    MU_PROX = None
    C_MODE  = "ZERO"        #  "ZERO" / "LE" / "LE_SQUARED"
    if C_MODE not in {"ZERO", "LE", "LE_SQUARED"}:
        raise ValueError("Invalid C_MODE")
    OUTPUT_DIR += "_OSM_clean_surrogate_" + C_MODE + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # ── 迭代控制 ─────────────────────────────────────────────
    TOL         = 1e-4
    MAX_ITER_QR  = 20000

    SEED   = 2024
    N_SHOW = 6

    INIT_MODE   = 'random'
    DATA_QR_IDX = 0

    QR_SNAP_ITERS = [1,20,50, 200, 500, 1000, 2000, 3000, 5000, 8000, 10000, 12000, 15000, 18000, 20000]

    # ╚══════════════════════════════════════════════════════════╝

    print("=" * 70)
    print("  AMPCA 医学图像重建  —  QR 单算法 GPU 版（只统计迭代计算时间）")
    print(f"  收敛准则：子空间距离  dist = sqrt(1-σ_min²)  ∈ [0,1]")
    print(f"  初始化策略：{INIT_MODE}")
    print(f"  C_MODE={C_MODE}，单次只跑这一种C；迭代日志仅记录 dist 和计算时间")
    print("  计时仅含 U/V 更新、QR 和 dist；所有重建图统一在迭代结束后保存")
    print(f"  独立输出目录：{OUTPUT_DIR}")
    print("=" * 70)

    print(f"\n[加载图像] {IMAGE_DIR}")
    names, X_list = load_images_from_dir(IMAGE_DIR, TARGET_SIZE, TO_GRAY)
    if X_list[0].ndim == 3:
        X_list = [x.mean(axis=2) for x in X_list]
    m, p  = X_list[0].shape
    n_img = len(X_list)
    Psi   = np.mean(np.stack(X_list), axis=0)
    print(f"  n={n_img},  m×p={m}×{p},  K1={K1}, K2={K2},  β={BETA:.2e}")
    print(f"  ROW_ROI={ROW_ROI},  COL_ROI={COL_ROI}")
    print(f"  TOL={TOL},  MAX_ITER_QR={MAX_ITER_QR}")

    S1_diag = build_S_diag(m, ROW_ROI)
    S2_diag = build_S_diag(p, COL_ROI)

    print(f"\n[生成 QR 初始解]  INIT_MODE={INIT_MODE}  SEED={SEED}")
    if INIT_MODE == 'random':
        U0, V0 = make_shared_init(m, K1, p, K2, SEED)
        qr_label = "random_shared"
        print(f"  U0 shape={U0.shape},  V0 shape={V0.shape}")
        print("  ✓ 保留原随机正交初始化方法与 SEED")

    elif INIT_MODE == 'data_qr':
        U0, V0 = init_data_qr(X_list, K1, K2, DATA_QR_IDX)
        qr_label = f"data_qr_img{DATA_QR_IDX}"
        print(f"  取第 {DATA_QR_IDX} 张图 QR 分解初始化")
        print("  ✓ 保留原数据 QR 初始化方法")

    else:
        raise ValueError(f"未知 INIT_MODE: {INIT_MODE!r}")

    U0_qr, V0_qr = U0, V0

    print(f"\n>>> Algorithm 3.1 (OSM): QR Retraction ...")
    print(f"    快照轮次: {[i for i in QR_SNAP_ITERS if i<=MAX_ITER_QR]}；最多{MAX_ITER_QR}轮，不再追加")
    print(f"    快照缓存 U/V；结束后统一保存前 {SAVE_TOP_N} 张图像（固定索引）")
    objective_dir = Path(OUTPUT_DIR)/"results"
    objective_dir.mkdir(parents=True,exist_ok=True)
    (U2, V2, it2, qr_dU_h, qr_dV_h, elapsed2, snapshots_qr) = mpca_qr(
        X_list, Psi, BETA, S1_diag, S2_diag, K1, K2,
        U0_qr, V0_qr,
        lam=LAMBDA, mu_prox=MU_PROX,
        tol=TOL, max_iter=MAX_ITER_QR,
        save_iters=QR_SNAP_ITERS,
        verbose=True,c_mode=C_MODE,objective_dir=objective_dir)
    print(f"    迭代计算时间（不含日志和保存）={elapsed2:.4f}s")
    print(f"    final dist_U={qr_dU_h[-1]:.2e}  dist_V={qr_dV_h[-1]:.2e}")
    print(f"    实际快照轮次: {[s[0] for s in snapshots_qr]}")

    print("\n[迭代已结束；以下重建、快照、绘图及保存均不计入计算时间]")
    recon_qr  = reconstruct(X_list, Psi, U2, V2)

    out   = Path(OUTPUT_DIR)
    d_qr  = out / "QR_reconstructed"
    d_res = out / "results"
    for d in [d_qr, d_res]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[保存最终重建图像（全部 {n_img} 张）]")
    for i, name in enumerate(names):
        stem = Path(name).stem
        save_image(recon_qr[i],  d_qr  / f"{stem}_qr.png")
    print(f"  QR  → {d_qr}")

    print(f"\n[结束后统一保存 QR 迭代快照（各轮次前{SAVE_TOP_N}张）]")
    save_snapshot_images(X_list, Psi, names, snapshots_qr,
                         d_qr, SAVE_TOP_N, "qr")

    print(f"\n[保存 txt 结果文件] → {d_res}")
    save_all_iter_txts(qr_dU_h, qr_dV_h, d_res, TOL)

    params_log = {
        "IMAGE_DIR": IMAGE_DIR, "N_images": n_img,
        "Image_size": f"{m}x{p}", "K1": K1, "K2": K2,
        "BETA": BETA, "ROW_ROI": ROW_ROI, "COL_ROI": COL_ROI,
        "INIT_MODE": INIT_MODE, "SEED": SEED,
        "TOL": TOL, "MAX_ITER_QR": MAX_ITER_QR,
        "LAMBDA": LAMBDA, "MU_PROX": MU_PROX,
        "C_MODE": C_MODE,
        "ACTUAL_COEFFICIENTS": "results/surrogate_parameters.json",
        "QR_SNAP_ITERS": QR_SNAP_ITERS,
        "SAVE_TOP_N": SAVE_TOP_N,
        "TIMING": "U/V updates + QR + dist only, synchronized GPU wall time",
        "SNAPSHOT_POLICY": "all image reconstruction/saving after iteration",
    }
    save_summary_txt(d_res / "summary.txt", params_log,
                     it2, elapsed2, qr_dU_h, qr_dV_h)
    print("  summary.txt")

    print(f"\n[生成可视化图像]")
    plot_convergence_curve(qr_dU_h, qr_dV_h,
                           TOL, qr_label, d_res / "convergence_curve.png")
    plot_reconstruction_grid(names, X_list, recon_qr,
                             N_SHOW, qr_label,
                             d_res / "reconstruction_comparison.png")

    print(f"\n{'='*70}")
    print(f"  {'指标':<38}  {'QR':>12}")
    print(f"  {'-'*64}")
    print(f"  {'迭代次数':<38}  {it2:>12d}")
    print(f"  {'迭代计算时间 (s，不含日志/保存)':<38}  {elapsed2:>12.4f}")
    print(f"  {'最终 dist_U (子空间距离, 连续)':<38}  "
          f"{qr_dU_h[-1]:>12.2e}")
    print(f"  {'最终 dist_V (子空间距离, 连续)':<38}  "
          f"{qr_dV_h[-1]:>12.2e}")
    print(f"{'='*70}")
    print(f"\n✓ 全部结果 → {out.resolve()}")
    matrix_dir = out / "matrices"
    matrix_dir.mkdir(exist_ok=True)

    # 保存 QR 结果
    np.save(matrix_dir / "U_qr.npy", U2)
    np.save(matrix_dir / "V_qr.npy", V2)
    np.save(matrix_dir / "Psi_train.npy", Psi)

    print(f"✅ 最终投影矩阵 U, V 和训练集均值 Psi 已保存至: {matrix_dir}")


if __name__ == "__main__":
    main()
