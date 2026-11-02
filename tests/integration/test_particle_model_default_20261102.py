"""颗粒尺寸模型缺省值的回归测试（2026-11-02）。

背景：缺省值原为 shrinking_legacy，仅靠 `paper_op_conds` 逐个覆盖为
fortran_constant。任何绕过它的代码会静默拿到旧模型。现将缺省改为
fortran_constant，并显式锁定历史复现使用的旧模型，产物带标记。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_REPO_ROOT / "src"))


def test_default_particle_model_is_fortran_constant():
    """缺省必须是 fortran_constant，与 Wen 1979 / legacy Fortran 一致。"""
    from model.input_contract import PARTICLE_MODEL_DEFAULT, resolve_particle_model

    assert PARTICLE_MODEL_DEFAULT == "fortran_constant"
    assert resolve_particle_model({}) == "fortran_constant"
    assert resolve_particle_model({"ParticleSizeModel": "fortran_constant"}) == "fortran_constant"
    assert resolve_particle_model({"ParticleSizeModel": "shrinking_legacy"}) == "shrinking_legacy"


def test_unknown_particle_model_raises_instead_of_falling_back():
    """非法模型名必须报错。

    静默退回缺省会让整轮结果用错物理却毫无异常——比直接崩溃更危险。
    """
    from model.input_contract import resolve_particle_model

    with pytest.raises(ValueError, match="ParticleSizeModel must be one of"):
        resolve_particle_model({"ParticleSizeModel": "shrinkng_legacy"})  # 拼写错误
    with pytest.raises(ValueError):
        resolve_particle_model({"ParticleSizeModel": ""})


def test_cell_uses_the_shared_default():
    """Cell 与 jax 路径必须共用同一处解析，不能再各自写默认值。"""
    import inspect

    from model.cell import Cell

    src = inspect.getsource(Cell._calc_physics_props)
    assert "resolve_particle_model" in src
    assert "'shrinking_legacy'" not in src, "cell must not hard-code a default any more"


def test_cell_and_jax_paths_agree_on_the_default():
    """同一 op_conds 下，NumPy 与 JAX 两条路径必须选中同一模型。"""
    from model.input_contract import uses_fortran_default_model

    assert uses_fortran_default_model({}) is True
    assert uses_fortran_default_model({"ParticleSizeModel": "fortran_constant"}) is True
    assert uses_fortran_default_model({"ParticleSizeModel": "shrinking_legacy"}) is False


def test_legacy_mesh_helper_pins_the_old_model_and_marks_it():
    """历史复现必须显式锁定旧模型，并能在产物里被识别。

    否则缺省值一改，历史轮次的复现就会静默换模型，而产物看不出区别。
    """
    from model.revision_evidence_common import legacy_mesh_op_conds, particle_model_marker

    op = legacy_mesh_op_conds({})
    assert op["AllowTruncatedMesh"] is True
    assert op["ParticleSizeModel"] == "shrinking_legacy"

    marker = particle_model_marker(op)
    assert marker == {"particle_size_model": "shrinking_legacy",
                      "particle_model_legacy_model": True}

    # 论文口径不带 legacy 标记
    assert particle_model_marker({}) == {"particle_size_model": "fortran_constant",
                                         "particle_model_legacy_model": False}


def test_legacy_helper_respects_an_explicit_model():
    """调用方显式指定模型时，helper 不得覆盖。"""
    from model.revision_evidence_common import legacy_mesh_op_conds, particle_model_marker

    op = legacy_mesh_op_conds({"ParticleSizeModel": "fortran_constant"})
    assert op["ParticleSizeModel"] == "fortran_constant"
    assert particle_model_marker(op)["particle_model_legacy_model"] is False


def test_historical_diagnostics_declare_the_marker_columns():
    """两个产出逐用例诊断的模块必须把标记写进列定义与行。"""
    from model.cell0_residual_diagnostic import CELL0_RESIDUAL_COLUMNS
    from model.model_case_diagnostic_baseline import REQUIRED_DIAGNOSTIC_COLUMNS

    for columns in (CELL0_RESIDUAL_COLUMNS, REQUIRED_DIAGNOSTIC_COLUMNS):
        assert "particle_size_model" in columns
        assert "particle_model_legacy_model" in columns
