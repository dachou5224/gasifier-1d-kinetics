"""
主线 8 组分与 JAX JIT 9 组分（含 COS）单点映射真源。

``MAINLINE_SPECIES_NAMES`` 为唯一主线顺序真源；``material.SPECIES_NAMES`` 由其列表化。

JAX 状态向量前 9 维顺序为：
[O2, CH4, CO, CO2, H2O, H2, N2, H2S, COS]
"""
from __future__ import annotations

import numpy as np

# 与 StateVector.gas_moles / 求解器主线一致（唯一真源）
MAINLINE_SPECIES_NAMES: tuple[str, ...] = (
    "O2", "CH4", "CO", "CO2", "H2S", "H2", "N2", "H2O",
)

# 与 jax_residuals / jax_solver 中 x[:9] 一致
JAX_9_SPECIES_NAMES: tuple[str, ...] = (
    "O2", "CH4", "CO", "CO2", "H2O", "H2", "N2", "H2S", "COS"
)

# 主线索引 i -> JAX 9 维索引（COS 无主线对应，由调用方置零）
_MAINLINE_I_TO_JAX: tuple[int, ...] = (0, 1, 2, 3, 7, 5, 6, 4)


def mainline_gas8_to_jax9(gas8: np.ndarray) -> np.ndarray:
    """主线 8 维摩尔流量 → JAX 9 维（COS=0）。"""
    g = np.zeros(9, dtype=np.asarray(gas8).dtype)
    g8 = np.asarray(gas8).reshape(8)
    for i in range(8):
        g[_MAINLINE_I_TO_JAX[i]] = g8[i]
    return g


def jax9_to_mainline_gas8(gas9: np.ndarray) -> np.ndarray:
    """JAX 9 维 → 主线 8 维（丢弃 COS 或按需在外部校验）。"""
    out = np.zeros(8, dtype=np.asarray(gas9).dtype)
    g9 = np.asarray(gas9).reshape(9)
    for i in range(8):
        out[i] = g9[_MAINLINE_I_TO_JAX[i]]
    return out


def state12_mainline_inlet_to_jax(inlet_gas8: np.ndarray, Ws: float, Xc: float, T: float) -> np.ndarray:
    """构造 JAX 用的 12 维入口向量 [9 气 + Ws + Xc + T]。"""
    g9 = mainline_gas8_to_jax9(inlet_gas8)
    return np.concatenate([g9, np.array([Ws, Xc, T], dtype=np.asarray(inlet_gas8).dtype)])


def state12_jax_to_mainline_gas8(row12: np.ndarray) -> np.ndarray:
    """12 维 JAX 状态中的前 9 维气相 → 主线 8 维。"""
    return jax9_to_mainline_gas8(row12[:9])
