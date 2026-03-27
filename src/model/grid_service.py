import numpy as np
from dataclasses import dataclass
from typing import Optional
from .constants import PhysicalConstants

@dataclass
class MeshConfig:
    total_length: float
    n_cells: int
    ignition_zone_length: float = 0.4
    ignition_zone_res: float = 0.1
    # 可选：当给定时，把“点火区（从 cell0 起）”的总物理长度固定为该值，
    # 然后在该物理长度内部再用 ignition_zone_res 控制细网格单元长度/数量。
    ignition_zone_total_length: Optional[float] = None
    # 可选：细网格（等分）起始轴向位置 z（m），默认等于 ignition_zone_length（第一段结束后立即细分）
    ignition_zone_fine_start_z: Optional[float] = None
    # 可选：点火区细网格段“前细后粗”几何级比（=1 表示等分）
    # 例如 1.08 表示每个后续单元长度约为前一单元的 1.08 倍
    ignition_zone_stretch_ratio: float = 1.0
    min_grid_size: float = 0.1

class AdaptiveMeshGenerator:
    """
    Generates a 1D grid with variable resolution.
    Prioritizes resolution in the ignition zone.
    """
    def __init__(self, config: MeshConfig):
        self.cfg = config

    def generate(self):
        """
        Returns:
            dz_list (np.array): Array of cell lengths
            z_positions (np.array): Array of cell center positions
        """
        L = self.cfg.total_length
        N = self.cfg.n_cells
        dz_list = np.zeros(N)
        fixed_len = 0.0
        
        # Explicit Sizes derived from constants or config
        # Use config values, falling back to PhysicalConstants if not provided (though config has defaults)
        
        if N > 2:
            # 0. First Cell (Tuyere / Inlet)
            # Use defined ignition zone length first step or a specific large inlet cell?
            # Existing logic used FIRST_CELL_LENGTH (0.4)
            # Let's align with the config.
            dz_list[0] = self.cfg.ignition_zone_length
            fixed_len += self.cfg.ignition_zone_length
            
            # 1. Ignition/Flame Zone
            # 两种模式：
            # - ignition_zone_total_length=None：复用历史逻辑（fine_end 固定为 min(20,N)）
            # - ignition_zone_total_length!=None：固定点火区物理总长度，再用 ignition_zone_res 决定内部划分
            fine_res = self.cfg.ignition_zone_res
            if self.cfg.ignition_zone_total_length is None:
                fine_end = min(20, N)
                for k in range(1, fine_end):
                    dz_list[k] = fine_res
                    fixed_len += fine_res
            else:
                target_total = float(self.cfg.ignition_zone_total_length)
                target_total = min(target_total, L)  # safety: 不超过总长度
                L0 = float(self.cfg.ignition_zone_length)
                z_fs = self.cfg.ignition_zone_fine_start_z
                if z_fs is None:
                    z_fs = L0
                else:
                    z_fs = float(z_fs)
                    z_fs = max(L0, min(z_fs, target_total))

                # 第一段 [0,L0] 已放置；可选过渡段 [L0, z_fs]；细网格段 [z_fs, target_total] 内等分
                gap_len = max(z_fs - L0, 0.0)
                interior_len = max(target_total - z_fs, 0.0)
                total_budget = max(N - 1, 0)

                def _allocate_two_segments(
                    budget: int,
                    g_len: float,
                    i_len: float,
                    fd: float,
                ) -> tuple[int, int]:
                    """在 budget 个单元内分配过渡段与细网格段的单元数（均 >=1 若对应长度>0）。"""
                    if budget <= 0:
                        return 0, 0
                    if g_len <= 1e-15 and i_len <= 1e-15:
                        return 0, 0
                    if g_len <= 1e-15:
                        ni = max(1, int(round(i_len / fd))) if fd > 0 else 1
                        return 0, min(ni, budget)
                    if i_len <= 1e-15:
                        ng = max(1, int(round(g_len / fd))) if fd > 0 else 1
                        return min(ng, budget), 0
                    ng0 = max(1, int(round(g_len / fd))) if fd > 0 else 1
                    ni0 = max(1, int(round(i_len / fd))) if fd > 0 else 1
                    if ng0 + ni0 <= budget:
                        return ng0, ni0
                    # 按比例压缩到 budget（保持两段都有至少 1 格若物理长度仍为正）
                    w = g_len + i_len
                    ng = max(1, int(round(budget * (g_len / w))))
                    ni = budget - ng
                    if ni < 1:
                        ni = 1
                        ng = min(budget - 1, max(1, ng))
                    return ng, ni

                n_gap, n_int = _allocate_two_segments(total_budget, gap_len, interior_len, fine_res)

                idx = 1
                if n_gap > 0 and gap_len > 1e-12:
                    dz_gap = gap_len / n_gap
                    for _ in range(n_gap):
                        dz_list[idx] = dz_gap
                        fixed_len += dz_gap
                        idx += 1
                if n_int > 0 and interior_len > 1e-12:
                    ratio = float(self.cfg.ignition_zone_stretch_ratio)
                    # ratio<=1 时退化为等分；ratio>1 时采用几何级数实现“前细后粗”
                    if ratio <= 1.0 + 1e-12:
                        dz_int = interior_len / n_int
                        for _ in range(n_int):
                            dz_list[idx] = dz_int
                            fixed_len += dz_int
                            idx += 1
                    else:
                        # S = a * (r^n - 1)/(r - 1)  ->  a = S*(r-1)/(r^n-1)
                        r_pow_n = ratio ** n_int
                        denom = max(r_pow_n - 1.0, 1e-12)
                        dz_first = interior_len * (ratio - 1.0) / denom
                        dz_curr = dz_first
                        for _ in range(n_int):
                            dz_list[idx] = dz_curr
                            fixed_len += dz_curr
                            idx += 1
                            dz_curr *= ratio

                fine_end = idx
                
            # 2. Remaining Length Distributed
            rem_cells = N - fine_end
            if rem_cells > 0:
                # 严格模式（固定点火区总长度）下尽量保持总长度 L 不被破坏
                rem_len = L - fixed_len
                dz_avg = rem_len / rem_cells if rem_cells > 0 else self.cfg.min_grid_size
                if self.cfg.ignition_zone_total_length is None:
                    # 历史逻辑：使用 min_grid_size 做安全下限
                    dz_avg = max(dz_avg, self.cfg.min_grid_size)
                for k in range(fine_end, N):
                    dz_list[k] = dz_avg
        else:
            # Fallback for small N
            dz_list[:] = L / N

        # Compute Z positions (Centers)
        z_positions = np.zeros(N)
        current_z = 0.0
        for k in range(N):
            z_positions[k] = current_z + dz_list[k]/2.0
            current_z += dz_list[k]
            
        return dz_list, z_positions
