# gasifier-physics-audit/scripts/balance_checker.py
import numpy as np

def check_atom_balance(inlet_flows, outlet_flows, species_atoms):
    """
    核算元素质量守恒。
    :param inlet_flows: 输入摩尔流率字典，如 {'CH4': 10, 'H2O': 5}
    :param outlet_flows: 输出摩尔流率字典
    :param species_atoms: 组分原子数定义，如 {'CH4': {'C':1, 'H':4}, ...}
    """
    atoms = ['C', 'H', 'O', 'N']
    balance_report = {}
    
    for a in atoms:
        sum_in = sum(inlet_flows.get(s, 0) * species_atoms[s].get(a, 0) for s in inlet_flows)
        sum_out = sum(outlet_flows.get(s, 0) * species_atoms[s].get(a, 0) for s in outlet_flows)
        
        # 计算相对误差
        error = abs(sum_in - sum_out) / sum_in if sum_in > 0 else 0
        balance_report[a] = {"in": sum_in, "out": sum_out, "error_percent": error * 100}
    
    return balance_report

# 示例调用逻辑...