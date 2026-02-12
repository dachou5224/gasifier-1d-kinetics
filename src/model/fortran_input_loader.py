import os
import re
from typing import Dict, Any, Tuple


def _dulong_hhv_mj_kg(C: float, H: float, O: float, S: float) -> float:
    """
    Rough HHV (MJ/kg) from ultimate analysis (wt%). Dulong-style.
    和 validation_loader 中保持一致，避免重复魔数。
    """
    c, h, o, s = C / 100.0, H / 100.0, O / 100.0, S / 100.0
    return 33.5 * c + 144.0 * h - 18.0 * o + 10.0 * s


def _parse_fortran_namelist(path: str) -> Dict[str, Dict[str, Any]]:
    """
    解析 reference_fortran/input_副本.txt 中的 &readdata ... &end 块。

    返回:
      name -> {字段: 值}
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 找出每个 &readdata ... &end 块
    blocks = re.findall(r"&readdata(.*?)&end", text, flags=re.S | re.I)
    cases: Dict[str, Dict[str, Any]] = {}

    for block in blocks:
        params: Dict[str, Any] = {}
        # 去掉换行和续行符
        cleaned = block.replace("\n", " ").replace("&", " ")
        # 按逗号切分 key=value
        for token in cleaned.split(","):
            token = token.strip()
            if not token:
                continue
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            key = k.strip().lower()
            val = v.strip()
            # 字符串: 形如 'texaco i-1'
            if val.startswith("'") and val.endswith("'"):
                val_parsed = val.strip("'").strip()
            else:
                try:
                    val_parsed = float(val)
                except ValueError:
                    # 无法解析成 float 的，直接原样保留
                    val_parsed = val
            params[key] = val_parsed

        name = str(params.get("name", "unknown")).strip()
        cases[name] = params

    return cases


def load_fortran_cases(
    path: str,
    default_pressure_atm: float = 24.0,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    从 Fortran 的 input_副本.txt 读取工况，转成 (COAL_DATABASE, VALIDATION_CASES) 风格。

    假设:
      - yc,yoxy,yh,ys,yn,yash: as-received（不含水？）质量分数 + xmois 为水分。
      - foxy,fsteam: 以 tfcoal 为基的质量比 (g O2 / g 煤, g H2O / g 煤)。
      - tfcoal: 近似认为是 t/h，映射为 FeedRate_kg_h = tfcoal * 1000.
      - 压力 pt ≈ 24 atm（在 Source1_副本.for 的 data 里给出）。

    验证工况默认水煤浆浓度 60%；若工况名含 "dry"/"干粉"/"pulverized" 则按干粉气化用 100。

    返回:
      coal_db:  name_Coal -> {Cd, Hd, Od, Nd, Sd, Ad, Mt, Vd, FCd, HHV_d}
      cases:    name -> {inputs: {...}, expected: {}}  (无显式文献期望组分，暂留空)
    """
    raw_cases = _parse_fortran_namelist(path)
    coal_db: Dict[str, Dict[str, Any]] = {}
    cases: Dict[str, Dict[str, Any]] = {}

    P_Pa = default_pressure_atm * 101325.0

    for name, p in raw_cases.items():
        yc = float(p.get("yc", 0.0))
        yoxy = float(p.get("yoxy", 0.0))
        yh = float(p.get("yh", 0.0))
        ys = float(p.get("ys", 0.0))
        yn = float(p.get("yn", 0.0))
        yash = float(p.get("yash", 0.0))
        xmois = float(p.get("xmois", 0.0))

        sum_dry = yc + yoxy + yh + ys + yn + yash
        if sum_dry <= 0.0:
            # 退化情况，跳过
            continue

        # 干基（含灰）质量百分数
        Cd = yc / sum_dry * 100.0
        Od = yoxy / sum_dry * 100.0
        Hd = yh / sum_dry * 100.0
        Sd = ys / sum_dry * 100.0
        Nd = yn / sum_dry * 100.0
        Ad = yash / sum_dry * 100.0

        # 水分：按 as-received (dry+moist) 的质量百分数近似
        Mt = xmois / (sum_dry + xmois + 1e-12) * 100.0

        # 挥发/固定碳：无法从 Fortran 直接获得，这里简单假设
        # 保持与 validation_loader 一致：Vd 在 25–45% 范围，FCd 为余量
        Vd = 35.0
        FCd = max(0.0, 100.0 - Ad - Vd)

        coal_key = f"{name}_Coal"
        HHV_d_MJ_kg = _dulong_hhv_mj_kg(Cd, Hd, Od, Sd)

        coal_db[coal_key] = {
            "Cd": Cd,
            "Hd": Hd,
            "Od": Od,
            "Nd": Nd,
            "Sd": Sd,
            "Ad": Ad,
            "Vd": Vd,
            "FCd": FCd,
            "Mt": Mt,
            "HHV_d": HHV_d_MJ_kg * 1000.0,  # kJ/kg
        }

        # 质量比（g/g）近似为 O2/coal, steam/coal 质量比
        foxy = float(p.get("foxy", 0.0))
        fsteam = float(p.get("fsteam", 0.0))
        tfcoal = float(p.get("tfcoal", 100.0))  # 近似 t/h

        feed_rate_kg_h = tfcoal * 1000.0

        # 入口温度：Fortran 有 ta(煤)、tsteam、toxy，这里简单用 ta 近似整体 T_in
        ta = float(p.get("ta", 500.0))  # K

        # Fortran 的 fsteam 已包含煤浆水（作为蒸汽），无需额外液态水蒸发
        # 因此 Fortran 工况统一设为 SlurryConcentration=100（干粉），水通过 steam_flow 以蒸汽形式进入
        slurry_pct = 100.0

        inputs = {
            "coal": coal_key,
            "FeedRate": feed_rate_kg_h,
            "Ratio_OC": foxy,
            "Ratio_SC": fsteam,
            "P": P_Pa,
            "TIN": ta,
            "HeatLossPercent": 2.0,
            "SlurryConcentration": slurry_pct,
        }

        cases[name] = {
            "inputs": inputs,
            "expected": {},  # 暂无文献出口组分，这里先留空
        }

    return coal_db, cases


def get_fortran_input_path() -> str:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "reference_fortran", "input_副本.txt")

