# Fortran 壁面热损的符号表（P1.8-A，2026-11-04）

来源：`reference_fortran/Source1_副本.for`（行号为去 CR 后的行号）。
`&readdata` namelist 只有 `name, ta, tsteam, toxy, yoxy, yh, yc, ys, yn, yash, xmois, foxy, fsteam, tfcoal`
（第 40–41 行），**下表没有任何一个符号出现在 namelist 里**。

| 符号 | 定义 / 单位 | 首次赋值行 | 是否被重新赋值 | Python 对应量 |
|---|---|---|---|---|
| `dt` | 反应器直径，**cm**。`data` 语句给 152.4 cm（= 1.524 m），与 `at = 18241.5 cm²` 满足 π/4·dt² = at，确认是直径 | 32 | **否**（`data` 初值，全程序未再赋值） | `FORTRAN_DT_CM`（常量，不取 Python 几何） |
| `height` | 反应器/积分高度，**cm**。`data` 给 330 cm（= 3.3 m）；第 787–788 行 `!change 325 into height-5` / `dhs = hs-(height-5.)` 说明积分到 `height−5` = 325 cm 为止，所以它是**总高度**而非有效长度 | 32 | **否** | `FORTRAN_HEIGHT_CM`（常量） |
| `at` | 反应器截面积，**cm²**（18241.5） | 32 | 否 | 无用（Python 用几何 D） |
| `pt` | 压力，**atm**（24.0） | 32 | 否 | `op_conds['P']` |
| `nc` | 一个气室内的子步数，无量纲。`nc = 3`（第 145 行注释：`nc is the number of step sizes in each gas compartment`） | 146 | **否** | 不单独出现；`nc·delh` → `dz` |
| `delh` | **一个子步的轴向长度，cm**。`data` 给 25，但 25 只用于**第一个气室**；第 852–858 行在每个新气室开头重设为 `0.5` 或 `1.0`（`hs.le.40 → 0.5`；`poxy.gt.0.01 → 0.5`；否则 `1.0`），并在迭代不收敛时被 `delh = delh/2`（第 367、523 行）折半 | 96 | **是**（852–858、367、523） | `dz`（Python 一个单元 = 一个气室） |
| `hs` | 颗粒已走过的轴向高度，**cm**；初值 `hsin`（第 221 行），每个子步 `hs = hs + delh`（第 277、763 行） | 221 | **是**（每子步累进；每气室由 `hsin` 重置） | 单元出口 `z + dz/2` |
| `fcoal` | 入炉煤流量，**g/s，dmmf 基**。`fcoal = tfcoal*(1.-yash)*(1.-xmois)`（第 66 行），`tfcoal` 来自 namelist | 66 | **是**（由 namelist 的 `tfcoal/yash/xmois` 算出） | 干煤流量（`W_dry`，g/s） |
| `rheat1` | **本气室挥发分项放热**，cal/g dmmf。由 `pyroly` 写入：`wl*(xh2*ht(1)/2 + xco*ht(2)/28 + xch4*ht(3)/16 + xtar*ht(4)/78)/100 − cop*ht(2)/28 − ch4p*ht(3)/16 − h2p*ht(1)/2`（第 1087–1088 行），其中 `ht(n, tg)` 是组分在**气相温度**下的焓（`heatup` 第 1131 行起，cal/mol） | 248（每气室清零） | **是**：每个气室开头清零（248、865），由第 314 行调用 `pyroly` 赋值 | 无直接对应；挥发分能量走 `PyrolysisSource.energy_src_W` |
| `rheat2` | 气相燃烧项，cal/g dmmf。`rheat2 += rhta+rhtb+rhtc-rhtd`（第 556 行）与 `+qmref*dch46/16`（第 587、743 行），`rhta/b/c/d` 由 `q1/qrh/ht` 与碳消耗量 `cwl/cwl2/cwl3/cop` 组合 | 249 | **是**（每气室清零 417/611，多处累加） | 由 `CO_Ox/H2_Ox/CH4_Ox/MSR` 的反应焓体现 |
| `rheat3` | WGS 项，cal/g dmmf。`rheat3 += qwg*wgl*(1-fwgs)`（第 588、744 行），`fwgs = 1.0`（第 100 行）是 WGS 放热被固相吸收的份额 | 250 | **是** | 由 `WGS/RWGS` 的反应焓体现 |
| `rhts` | 被固相吸收的热，cal/g dmmf：`rhts += cwl*qrh − cwl2*qcsm/12 − cwl3*qcbco2/12 − cwl5*qcbhy/12`（第 558 行）与 `−qwg*wgl*fwgs`（第 745 行） | 251 | **是**（每气室清零 417/611） | C+O2/C+H2O/C+CO2/C+H2 的反应焓（含固相） |
| `rheat` | **净放热**，cal/g dmmf：`rheat = rheat1+rheat2+rheat3 − rhts`（第 772 行） | 772 | **是**（每气室清零 869） | `reaction_heat_release_w()` 的返回值 ÷ 煤流量 |
| `hl` | 燃烧区传给壁面的放热份额，**%**。`data hl/30./`（第 30 行，注释 `combustion heat percent transfered to the wall`） | 30 | **否** | `FORTRAN_HL_PERCENT = 30` |
| `u0` | 气-壁总换热系数，**cal/(s·cm²·K)**，`u0 = 3.4e-3`（第 214 行） | 214 | **否** | `FORTRAN_U0_CAL_PER_S_CM2_K` |
| `tw` | 壁温，**K**：`tw = 2100. − 600.*hs/height`（第 348、777 行）。第 216 行的 `tw = 1700.` 只是**初始化占位**，两条 hloss 分支都会覆盖它 | 216（占位） | **是**（348/777 覆盖） | `gasification_zone_heat_loss_w()` 内的 `tw_K` |
| `tg` | 气室气相温度，**K**（待解未知量；由 Wegstein 迭代收敛，第 218–219、800 行） | 218 | **是**（迭代更新） | 单元 `current.T` |
| `poxyin` | 气室**入口** O2 分压，atm（第 141、147 行算出，每气室由 `poxyin = poxy` 更新，第 882 行）。`poxyin > 0.05` 是燃烧区判据 | 147 | **是**（每气室 882） | 入口 pO2（`Cell._inlet_pO2_is_combustion`） |
| `deltim` | 子步时长，s，由 `vsolid` 回填（`time = delh/us`，第 987 行） | 261 | **是**（每子步） | 不进入本轮的 hloss 式子 |

## 两条 hloss 分支的被乘数不同（必须写明）

- 第 341–350 行（`400` 之后的快路径）：燃烧区 `hloss = fcoal*abs(rheat1)*hl/100`，用 **`rheat1`**；
- 第 772–781 行（主循环 `515→737`）：燃烧区 `hloss = fcoal*abs(rheat)*hl/100`，用 **`rheat`**（净放热）。

本卡第 1.1 节写的是 `fcoal·|rheat|·hl/100`，对应**主循环**那一条，本轮按此转写。

## `nc·delh` 与 `hs` 的映射

`delh` 是**子步**长度，一个气室走 `nc` 个子步，因此 `nc·delh` 就是**一个气室的轴向长度**。
Fortran 里这个长度是变的（第一个气室 25×3 = 75 cm，其后 0.5×3 或 1.0×3 = 1.5 或 3 cm），
Python 的一个单元就是一个气室，故 `nc·delh → dz`、`hs → z + dz/2`（气室出口）。

## 符号不确定性检查

- `dt`/`height` 的单位由 `at = π/4·dt²` 与第 787–788 行的 `height-5` 交叉确认，取 cm，**确定**。
- `delh` 的语义由 `hs = hs+delh`（轴向累进）与 `time = delh/us`（第 987 行，`us` 为 cm/s）交叉确认，
  是轴向长度（cm），**确定**。
- `rheat` 的组成由第 772 行显式给出，**确定**。
- 唯一不确定项：Fortran 的 `rheat1` 把挥发分的**显热当量**（`ht(n,tg)`）当作热量计入，
  而 Python 的挥发分能量走 `PyrolysisSource.energy_src_W`，不重复计入反应热。这一项
  **在 Python 侧没有对应量**，本轮按「净化学放热」口径转写并在 `interpretation.md` 说明量级。
