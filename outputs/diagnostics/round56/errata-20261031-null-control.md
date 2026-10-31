# Round 56 null control —— 更正说明（2026-10-31）

**本文件不修改 `huayi_round56_result.md`。** 那份文件保留当时运行的实际输出，包括现已确认有缺陷的 null objective。这里记录缺陷、修复与修复后的数值，供引用者对照。

## 缺陷

null 控制用 `syngas_flow_mean` 做循环平移（cyclic shift）来打乱被选特征。问题在于该特征同时是 `dry_syngas_flow` 这个 target 的真值来源：

- `_prediction_rows_for_alpha` 里 `dry_syngas_flow` 的真值取自 `row["syngas_flow_mean"]`
- 同一 target 的预测取自 `row["syngas_dry_volumetric_flow_value"]`——这是该行自身输入的携带值，**不经过打乱**

于是打乱后：真值被替换成邻行的流量，预测仍是本行原值，残差里混入了与该扰动完全无关的流量误差。

`huayi_round56_result.md` 的 "Known limitations" 已记有「dry syngas flow prediction remains carried metadata target under WGS comparators」，当时未意识到它与 null 控制的特征打乱会相互作用。

## 数值影响

从当次运行的预测缓存 `_null_feature_shuffle_train.csv` 重算，不重跑求解器（预测本身正确，错的只是记录下来的真值）：

| 量 | 原记录 | 更正后 |
|---|---:|---:|
| null objective（5 target 均值） | 1.9611542806500364 | 1.9566848374 |
| selected model train objective | 1.956680603180225 | 1.956680603180225（未受影响） |
| margin = null − selected | 4.49e-03 | **4.23e-06** |
| `dry_syngas_flow` 分量（归一化 RMSE） | 0.022347 | 0.0 |

`dry_syngas_flow` 修正后残差恒为 0——该 target 的预测就是该行输入，本来就是退化目标，不该携带信息。

## 对结论的影响

`control_explains_improvement` 在两个版本下都是 `False`，**判决未变**。但：

- 原 margin 的 99.9% 由上述缺陷产生
- 更正后的 margin 相对 objective 仅 2.1e-6，处于数值噪声量级

因此该 null 控制**不具备判别力**。`huayi_round56_result.md` 中的 `control_explains=False` 应读作「不可判别」，**不能**作为「控制组无法解释改进」的正面证据。

`round56` 的 `Decision: WGS_COMPARATOR_CONDITIONAL` 主要依据 held-out delta 与 worsened-target 计数，不依赖这一条；但若后续论证用到 null 控制，须引用本文件而非原 result 文件。

## 代码修复

`scripts/generate_huayi_round56_wgs_comparators.py`：

- `_prediction_rows_for_alpha` 新增 `target_rows` 参数，真值一律取自未扰动行
- null 控制分别构造 `original_train` 与 `shuffled_train`，后者只作为模型输入
- 缓存增加 `cache_fingerprint`（方法 + 逐窗 alpha + 真值来源），不符即整体作废

回归测试：`tests/integration/test_review_fixes_20261031.py::test_null_shuffle_keeps_ground_truth_from_unshuffled_rows`。

## 复现

```bash
python -m pytest tests/integration/test_review_fixes_20261031.py -q
```

修复后的缓存 `_null_feature_shuffle_train.csv` 中 `dry_syngas_flow` 的 `target_value` 已按原始行回填，`residual` 相应归零；该缓存为中间产物，不随 `round-*` 目录入库。
