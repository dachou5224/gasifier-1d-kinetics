import math

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
    # 网格族（P1.5-A，2026-10-30）：
    #   "legacy"              —— 现行点火区加密逻辑；3 <= N <= 20 时只覆盖入口一小段
    #   "uniform_full_length" —— dz_i = L / N，铺满全长；论文全部新轮次使用这一族
    #   "graded_full_length"  —— P1.9-A（2026-11-05）：全长铺满，燃烧区块 [0, z_b] 等长
    #                            细格 dz0，其余段沿用均匀 N=40 的格长；z_b 与 dz0 由
    #                            graded_zone_length / graded_zone_dz 给出，块内格数
    #                            n_block = round(z_b/dz0)，其余格数 = N - n_block。
    #   "graded_block_tail"   —— P1.10-A（2026-11-05）：块内格数向上取整、余段用固定格长
    #                            平铺、不足一格做尾格。详见 AdaptiveMeshGenerator 内注释。
    #   "wsr_plus_pfr"        —— P1.11-A（2026-11-06）：首格长度恰为 L_wsr（烧嘴近场全混流区），
    #                            其后 [L_wsr, L] 用等长 dz_down 的串联 CSTR（末格吸收余数）。
    mesh_family: str = "legacy"
    graded_zone_length: Optional[float] = None
    graded_zone_dz: Optional[float] = None
    # 仅 "graded_block_tail" 使用：余段（块之后到 L）的目标格长。
    rest_cell_length: Optional[float] = None
    # 仅 "wsr_plus_pfr" 使用：首格物理长度与下游格长。
    wsr_length: Optional[float] = None
    pfr_dz: Optional[float] = None

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
        family = str(getattr(self.cfg, "mesh_family", "legacy") or "legacy").lower()

        if family == "uniform_full_length":
            # 均匀铺满：每个单元 dz = L / N（旧行为保持不变，需显式选择）
            dz_list[:] = L / float(N)
        elif family == "graded_full_length":
            # P1.9-A：燃烧区块细格 + 其余段粗格，两段严格铺满全长。
            zb = self.cfg.graded_zone_length
            dz0 = self.cfg.graded_zone_dz
            if zb is None or dz0 is None:
                raise ValueError(
                    "graded_full_length 需要同时给出 graded_zone_length 与 graded_zone_dz"
                )
            zb = float(zb)
            dz0 = float(dz0)
            if not (0.0 < zb < L) or dz0 <= 0.0:
                raise ValueError(f"graded_full_length 参数越界：z_b={zb}, dz0={dz0}, L={L}")
            n_block = int(round(zb / dz0))
            if n_block < 1 or n_block > N:
                raise ValueError(
                    f"graded_full_length 块内格数越界：n_block={n_block}, N={N}"
                )
            n_rest = N - n_block
            if n_rest < 0:
                raise ValueError("graded_full_length 需要 N >= n_block")
            # 块的实际长度取 n_block*dz0（不是 z_b），这样 Σdz 逐位等于 L；
            # 对本轮声明档位 n_block*dz0 == z_b（整数档位）。
            zb_eff = float(n_block) * dz0
            dz_list[:n_block] = dz0
            if n_rest > 0:
                dz_list[n_block:] = (L - zb_eff) / float(n_rest)
        elif family == "graded_block_tail":
            # P1.10-A（2026-11-05 卡第 3.1 节）：确定性构造，不留实现自由。
            #   1  z_b_target = graded_zone_length            （卡面取 max(L_mix, 0.515)）
            #   2  n_block    = ceil(z_b_target / dz0)        ← 向上取整
            #   3  z_b_eff    = n_block · dz0
            #   4  rest_len   = L − z_b_eff
            #   5  n_rest     = floor(rest_len / rest_dz0)
            #   6  tail       = rest_len − n_rest · rest_dz0  → > 0 时追加 1 格
            #   7  N          = n_block + n_rest + (tail > 0 ? 1 : 0)
            # 块内格长严格为 dz0；余段除末格外严格为 rest_dz0；Σdz = L 在 1e-12
            # 相对容差内成立（n_rest·rest_dz0 的浮点和与 rest_len 可差 ~1e-15，
            # 不要断言"逐位相等"）。调用方传入的 n_cells 必须等于此处导出的 N。
            zb_t = self.cfg.graded_zone_length
            dz0 = self.cfg.graded_zone_dz
            rest_dz = self.cfg.rest_cell_length
            if zb_t is None or dz0 is None or rest_dz is None:
                raise ValueError(
                    "graded_block_tail 需要 graded_zone_length、graded_zone_dz、"
                    "rest_cell_length 三者同时给出"
                )
            zb_t = float(zb_t)
            dz0 = float(dz0)
            rest_dz = float(rest_dz)
            if dz0 <= 0.0 or rest_dz <= 0.0 or zb_t <= 0.0:
                raise ValueError(
                    f"graded_block_tail 参数必须为正：z_b_target={zb_t}, "
                    f"dz0={dz0}, rest_dz0={rest_dz}"
                )
            n_block = int(math.ceil(zb_t / dz0 - 1e-12))
            zb_eff = float(n_block) * dz0
            rest_len = L - zb_eff
            if rest_len < -1e-12 * L:
                raise ValueError(
                    f"graded_block_tail 块长超过总长：z_b_eff={zb_eff}, L={L}"
                )
            n_rest = int(math.floor(max(rest_len, 0.0) / rest_dz + 1e-12))
            tail = max(rest_len, 0.0) - float(n_rest) * rest_dz
            has_tail = bool(tail > 1e-12)
            n_derived = n_block + n_rest + (1 if has_tail else 0)
            if int(N) != n_derived:
                raise ValueError(
                    "graded_block_tail 的 N_cells 与构造不一致："
                    f"调用方 N_cells={N}, 构造导出 N={n_derived} "
                    f"(n_block={n_block}, n_rest={n_rest}, tail={tail:.6e})"
                )
            dz_list[:n_block] = dz0
            if n_rest > 0:
                dz_list[n_block:n_block + n_rest] = rest_dz
            if has_tail:
                dz_list[n_block + n_rest] = tail
        elif family == "wsr_plus_pfr":
            # P1.11-A（2026-11-06 卡第 1 节）：烧嘴近场全混流区 + 下游串联格。
            #   1  首格 dz[0] = L_wsr（体积 = A·L_wsr，即一个全混流区）
            #   2  RL = L − L_wsr
            #   3  n_full = floor(RL / dz_down + 1e-12)  ← 完整的 dz_down 格数
            #   4  rem = RL − n_full·dz_down；rem > 1e-12 时追加末格 = rem
            #   5  N = 1 + n_full + (rem > 1e-12 ? 1 : 0)
            # 首格长度**不随网格加密改变**，这是卡面的核心要求：全混流区尺寸是物理参数。
            # 下游格是串联 CSTR（tanks-in-series），加密使其逼近平推流。
            wsr = self.cfg.wsr_length
            dz_down = self.cfg.pfr_dz
            if wsr is None or dz_down is None:
                raise ValueError("wsr_plus_pfr 需要同时给出 wsr_length 与 pfr_dz")
            wsr = float(wsr)
            dz_down = float(dz_down)
            if not (0.0 < wsr < L) or dz_down <= 0.0:
                raise ValueError(
                    f"wsr_plus_pfr 参数越界：L_wsr={wsr}, dz_down={dz_down}, L={L}"
                )
            RL = L - wsr
            n_full = int(math.floor(RL / dz_down + 1e-12))
            rem = RL - float(n_full) * dz_down
            has_rem = bool(rem > 1e-12)
            n_derived = 1 + n_full + (1 if has_rem else 0)
            if int(N) != n_derived:
                raise ValueError(
                    "wsr_plus_pfr 的 N_cells 与构造不一致："
                    f"调用方 N_cells={N}, 构造导出 N={n_derived} "
                    f"(L_wsr={wsr}, dz_down={dz_down}, n_full={n_full}, rem={rem:.6e})"
                )
            dz_list[0] = wsr
            if n_full > 0:
                dz_list[1:1 + n_full] = dz_down
            if has_rem:
                dz_list[1 + n_full] = rem
        elif N > 2:
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
