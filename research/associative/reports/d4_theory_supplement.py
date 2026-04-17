"""D4 supplementary numerical checks.

Verifies theoretical claims from d4_theory.md:
1. Asymmetry of SOMA's effective weight matrix (Jacobian).
2. Whether the symmetrised energy E_sym decreases during recall.
3. Contraction constant estimation from iteration trajectories.
4. Spectral radius of the Jacobian at fixed points.

Run: python research/associative/reports/d4_theory_supplement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# -- path setup --------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from research.associative.datasets import make_binary_patterns  # noqa: E402
from soma.research.attractor_mode import (  # noqa: E402
    make_attractor_soma,
    recall_pattern,
    store_pattern,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = 50
N_PATTERNS = 10
SEED = 42


# -- helpers -----------------------------------------------------------


def single_step_output(soma, probe: torch.Tensor, modality: str = "text") -> torch.Tensor:
    """Run one forward step and return the output tensor (detached)."""
    result = soma.step(inputs={modality: probe}, eval_mode=True)
    out = result["outputs"].get(modality)
    if out is None:
        return torch.zeros_like(probe)
    return out.detach().clone()


def estimate_jacobian(soma, x: torch.Tensor, eps: float = 1e-3,
                      modality: str = "text") -> torch.Tensor:
    """Estimate the Jacobian of SOMA's input-output map at x via finite differences.

    Uses central differences for better accuracy.
    """
    d = x.shape[0]
    J = torch.zeros(d, d, device=x.device)

    for i in range(d):
        x_plus = x.clone()
        x_plus[i] += eps
        y_plus = single_step_output(soma, x_plus, modality)

        x_minus = x.clone()
        x_minus[i] -= eps
        y_minus = single_step_output(soma, x_minus, modality)

        J[:, i] = (y_plus - y_minus) / (2 * eps)

    return J


# -- main analysis -----------------------------------------------------


def main() -> None:
    torch.manual_seed(SEED)

    # Build SOMA attractor (same config as D2/D3)
    soma = make_attractor_soma(
        pattern_dim=D,
        n_associators=8,
        n_integrators=4,
        hebbian_lr=0.01,
        base_lr=0.001,
        seed=SEED,
        device=DEVICE,
    )

    # Generate and store patterns
    ds = make_binary_patterns(
        n_patterns=N_PATTERNS, dim=D, mask_frac=0.2, device=DEVICE, seed=SEED
    )
    for i in range(N_PATTERNS):
        store_pattern(soma, ds.patterns[i], n_presentations=3)

    print("=" * 70)
    print("D4 SUPPLEMENT: Numerical checks for SOMA attractor theory")
    print(f"Device: {DEVICE}, D={D}, N={N_PATTERNS}")
    print("=" * 70)

    # --- Check 1: Jacobian asymmetry ---
    print("\n## 1. Jacobian Asymmetry at Stored Patterns\n")
    asymmetry_ratios: list[float] = []
    spectral_radii: list[float] = []
    jacobians: list[torch.Tensor] = []

    for i in range(min(5, N_PATTERNS)):
        J = estimate_jacobian(soma, ds.patterns[i])
        jacobians.append(J)
        J_norm = float(torch.norm(J).item())

        if J_norm < 1e-10:
            print(f"Pattern {i}: Jacobian norm ~ 0 (degenerate)")
            asymmetry_ratios.append(0.0)
            spectral_radii.append(0.0)
            continue

        J_asym = (J - J.T) / 2
        ratio = float(torch.norm(J_asym).item()) / J_norm
        asymmetry_ratios.append(ratio)

        # Spectral radius
        try:
            eigvals = torch.linalg.eigvals(J)
            sr = float(eigvals.abs().max().item())
        except RuntimeError:
            sr = float("nan")
        spectral_radii.append(sr)

        print(f"Pattern {i}: ||J||={J_norm:.6f}, asymmetry={ratio:.4f}, "
              f"spectral_radius={sr:.6f}")

    mean_asym = sum(asymmetry_ratios) / max(len(asymmetry_ratios), 1)
    valid_sr = [s for s in spectral_radii if s == s]  # filter NaN
    mean_sr = sum(valid_sr) / max(len(valid_sr), 1) if valid_sr else float("nan")
    print(f"\nMean asymmetry ratio: {mean_asym:.4f}")
    print(f"  (0.0 = perfectly symmetric, 0.707 = maximally asymmetric)")
    print(f"Mean spectral radius: {mean_sr:.6f}")
    print(f"  (< 1.0 = contraction mapping, > 1.0 = potentially divergent)")

    # --- Check 2: Recall trajectory analysis ---
    print("\n## 2. Recall Trajectory: Energy and Cosine During Iteration\n")

    # Use pattern 0 with 20% noise as probe
    probe = ds.probes[0]
    target = ds.patterns[0]

    rr = recall_pattern(soma, probe, max_iters=20, convergence_tol=1e-5,
                        convergence_window=5)
    trajectory = rr.outputs_per_iter

    # Compute symmetrised energy using the Jacobian at the fixed point
    J_fp = estimate_jacobian(soma, trajectory[-1])
    W_sym = (J_fp + J_fp.T) / 2

    print("Iter | ||x||      | ||dx||     | E_sym        | Cos(x, target)")
    print("-----|------------|------------|--------------|---------------")
    e_sym_decreasing = True
    prev_e_sym = None
    for t, x in enumerate(trajectory):
        x_norm = float(x.norm().item())
        dx_norm = float((x - trajectory[t - 1]).norm().item()) if t > 0 else 0.0
        e_sym = float((-0.5 * x @ W_sym @ x).item())
        cos_sim = float(torch.nn.functional.cosine_similarity(
            x.unsqueeze(0), target.unsqueeze(0)).item())
        print(f"  {t:2d} | {x_norm:10.6f} | {dx_norm:10.6f} | {e_sym:12.6f} | {cos_sim:13.6f}")
        if prev_e_sym is not None and e_sym > prev_e_sym + 1e-8:
            e_sym_decreasing = False
        prev_e_sym = e_sym

    print(f"\nE_sym monotonically decreasing: {e_sym_decreasing}")
    print(f"Converged: {rr.converged} (iter {rr.convergence_iter})")

    # --- Check 3: Contraction constant estimation ---
    print("\n## 3. Contraction Constant Estimation\n")

    # Estimate from all patterns
    all_ratios: list[float] = []
    for pi in range(min(5, N_PATTERNS)):
        rr_i = recall_pattern(soma, ds.probes[pi], max_iters=20,
                              convergence_tol=1e-8, convergence_window=3)
        traj_i = rr_i.outputs_per_iter
        for t in range(1, len(traj_i) - 1):
            d_curr = float((traj_i[t + 1] - traj_i[t]).norm().item())
            d_prev = float((traj_i[t] - traj_i[t - 1]).norm().item())
            if d_prev > 1e-10:
                all_ratios.append(d_curr / d_prev)

    if all_ratios:
        print(f"Number of step-pairs measured: {len(all_ratios)}")
        print(f"Contraction ratios (first 10): "
              f"{[f'{r:.4f}' for r in all_ratios[:10]]}")
        mean_c = sum(all_ratios) / len(all_ratios)
        max_c = max(all_ratios)
        min_c = min(all_ratios)
        print(f"Mean contraction ratio: {mean_c:.4f}")
        print(f"Min: {min_c:.4f}, Max: {max_c:.4f}")
        print(f"  (< 1.0 confirms contraction; estimate c ~ {mean_c:.3f})")
        if mean_c < 1.0:
            import math
            iters_to_1e5 = math.log(1e-5 / 1.0) / math.log(mean_c) if mean_c > 0 else float("inf")
            print(f"  Predicted iterations to 1e-5 convergence: {iters_to_1e5:.1f}")
    else:
        print("Insufficient trajectory data for contraction estimation.")

    # --- Check 4: Nearest-pattern accuracy at fixed point ---
    print("\n## 4. Fixed-Point Nearest-Pattern Verification\n")
    correct = 0
    for i in range(N_PATTERNS):
        rr_i = recall_pattern(soma, ds.probes[i], max_iters=50)
        fp = rr_i.final_output
        sims = torch.cosine_similarity(fp.unsqueeze(0), ds.patterns, dim=1)
        nearest = int(sims.argmax().item())
        if nearest == i:
            correct += 1
    print(f"Nearest-pattern accuracy (20% noise): {correct}/{N_PATTERNS} "
          f"= {correct / N_PATTERNS:.1%}")

    # --- Check 5: SVD analysis of Jacobian ---
    print("\n## 5. Singular Value Spectrum of Jacobian\n")
    if jacobians:
        J_mean = torch.stack(jacobians).mean(dim=0)
        try:
            svs = torch.linalg.svdvals(J_mean)
            top_10 = svs[:10].tolist()
            print(f"Top 10 singular values: {[f'{s:.6f}' for s in top_10]}")
            effective_rank = int((svs > 1e-4).sum().item())
            print(f"Effective rank (sv > 1e-4): {effective_rank} / {D}")
            print(f"Condition number: {svs[0] / (svs[-1] + 1e-10):.2f}")
            total_energy = float((svs ** 2).sum().item())
            cumulative = torch.cumsum(svs ** 2, dim=0) / total_energy
            rank_90 = int((cumulative < 0.9).sum().item()) + 1
            rank_99 = int((cumulative < 0.99).sum().item()) + 1
            print(f"Rank for 90% energy: {rank_90}")
            print(f"Rank for 99% energy: {rank_99}")
        except RuntimeError as e:
            print(f"SVD failed: {e}")

    print("\n" + "=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
