# HR-0 高重频接口审计与账本

## 0. 审计身份与边界

- audit date: 2026-08-26
- audit branch: `main`
- audit git SHA: `6ffbf91d97f24b46d9aff6966f2e55ca5e81a46b`
- audit remote state: 本次审计开始时本地 `main` 与 `origin/main` 一致（ahead/behind = `0/0`）
- frozen single-pulse physics SHA: `e11d13f103c484953c0f733aa9b410bff385b2b5`

两类 SHA 用途不同：`audit git SHA` 标识本文实际审计的当前实现；`e11d13f...` 仅是历史冻结的单脉冲物理基准。本文不重新定义、替换或移动冻结基准。

本次工作只做静态源码追踪、论文接口审阅和量纲审计。没有修改 `KHz_filament` 传播核心、production config、Raman、电离或 BK-NEE 实现；没有进行正式高重频传播；没有提交 HPC/Slurm 作业。

参考模型：`references/Isaacs 等 - 2022 - Modeling the propagation of a high-average-power train of ultrashort laser pulses.pdf`，重点为论文页 22313-22315 的 Eq. (25)-(33)。

## 1. 结论摘要

1. 当前 `Npulses > 1` 不是“多个独立 source pulse 依次穿过持续演化的介质”，而是同一个数组变量 `E` 的传播输出被直接作为下一轮传播输入；薄透镜和可选的线性预推进也只在 pulse loop 之前执行一次。
2. 跨 pulse 保存的慢变量只有一个二维 `dn_gas[Ny, Nx]`。光场 `E[Nt, Ny, Nx]` 也跨 pulse 保存，但这是当前实现中不符合 pulse-train source 语义的状态继承。`Q2D` 只在单轮 loop 内存在，随后用于更新 `dn_gas`；`last_diag` 只保留最后一发诊断。
3. `E_out(N) -> E_in(N+1)`：**存在，HR0-P0-1 = CONFIRMED**。
4. 当前没有 `dn[z, y, x]`、screen index 或等价 thermal-screen 状态；同一个二维 `dn_gas` 在所有 optical z step 重复使用：**HR0-P0-2 = CONFIRMED**。
5. `heat_Q_per_z` 返回 `J/m^3`，`Qacc` 再沿 z 积分为 `J/m^2`；代码与说明却将 `gamma_heat` 声明为 `Δn/(J/m^3)`，二者相乘不可能得到无量纲 `Δn`：**HR0-P0-3 = CONFIRMED**。
6. 电离沉积与 IB 沉积进入返回给 runner 的 `Qacc`。Raman 则按模式和 `absorption_model` 分流：部分 legacy `poynting` 路径进入 `Qacc`，当前常见 `conv_deriv`/`closed_form` 路径只形成诊断而不进入慢热源，完整 Eq. (27) 路径进入独立的 `Qacc_raman` 但 runner 忽略它：**HR0-P0-4 = PARTIALLY_CONFIRMED**（“不完整”已确认，但存在一个窄路径会进入慢热源）。
7. Isaacs Eq. (27) 对应的单发光学传播能力已经存在；Eq. (25)-(26) 只有一个不完整且语义错误的 pulse loop/二维扩散近似；Eq. (31) 只有未闭合的 `gamma_heat` 接口；Eq. (32) 只实现了二维纯扩散；Eq. (33) 未实现。
8. HR-1 应首先修改 runner 的 pulse/source 接口：保留不可变 source field，每发复制/重建 fresh pulse，只继承明确的 `MediumState`，并保留逐 pulse 诊断。HR-1 不应顺带改变单脉冲传播公式。
9. HR-2 需要 thermal screens，因为 Eq. (31) 的输入是局部 `-∂F_L/∂z`（`J/m^3`），而 Eq. (32)-(33) 在沿 z 放置的一系列二维 screens 上演化。当前对 z 全积分的 `Qacc[J/m^2]` 已经丢失沉积位置，无法表示不同 z 处不同的热透镜。
10. HR-3 应在每个 screen 上用物理闭合的体能量密度更新折射率：正的局部沉积 `q_dep[J/m^3]` 乘以 `-(n0-1)/(rho0*Cv*T0) [m^3/J]`，得到无量纲 `δn`，再按 `1/f_rep` 演化扩散/输运。不能把全程面能量 `J/m^2` 直接乘体积耦合系数。

## 2. Isaacs Eq. (25)-(33) 的接口语义

| 方程 | 作用 | 输入 | 输出/状态 | 空间维度 | 跨 pulse |
|---|---|---|---|---|---|
| Eq. (25)-(26) | 导热主导时，多脉冲轴上热折射率的解析累积/饱和关系 | 单发 `δn1`、脉间隔 `τs`、导热时间 `τD`、pulse index `N` | `δnN` | 解析式为轴上标量；来源是横向二维热扰动 | 是 |
| Eq. (27) | 每一发的 3D NEE 光学传播 | fresh `A(r,tau)`、当前 thermal `δn(r)`、非线性极化/等离子体项 | 本发传播后的光场与沿 z 沉积 | 光场为 `tau,x,y`，并沿 z 推进；热扰动由 longitudinal screens 提供 | 光场否；介质是 |
| Eq. (31) | 将单发局部 fluence loss 转换为瞬时等压热折射率 imprint | pulse 前 `δn-`、局部 `∂F_L/∂z [J/m^3]`、`n0,rho0,Cv,T0` | pulse 后 `δn+` | 每个 longitudinal screen 上的二维 `x,y` 场 | 是 |
| Eq. (32) | 脉间折射率的扩散与平流 | `δn(x,y)`、二维速度 `v(x,y)`、热扩散率 `chi` | 演化后的 `δn(x,y)` | 每个 z screen 独立的二维横向平面 | 是 |
| Eq. (33) | 速度场的黏性、平流、浮力演化 | `v(x,y)`、`δn(x,y)`、运动黏度 `nu`、重力 `g` | 演化后的二维速度场 | 每个 z screen 独立的二维横向平面 | 是 |

论文在 Eq. (31)-(33) 后明确说明：这些方程在沿传播路径放置的一系列 screens 上求解，screen 间距要能够解析 longitudinal energy-deposition curve。因此，论文中的“2D fluid equations”不是一个对整个 z 区间共用的单独二维平面，而是多个 z 位置各自拥有二维横向介质状态。

## 3. 当前多脉冲数据流

### 3.1 源码实际流程

```text
build_transverse_input_field()
    -> E_source_like [Nt, Ny, Nx]
    -> 可选 thin lens（一次）
    -> 可选 linear pre-advance（一次）
    -> E

dn_gas = zeros([Ny, Nx])
delta_t_pulse = 1 / f_rep

for pulse i in range(Npulses):
    E, Q2D, diag = propagate_one_pulse(E, dn_gas=dn_gas, ...)
       |                                  |
       |                                  +-- 同一个 dn_gas 用于本发所有 z step
       +-- 返回本发 z=z_max 的输出场
       +-- Q2D 实际是 Qacc: 对 t、z 均积分后的 [Ny, Nx], J/m^2
       +-- diag 含本发沿 z 诊断及独立 Qacc_raman

    dn_gas = diffuse_dn_gas(dn_gas, Q2D, D_gas, 1/f_rep, ...)
    last_diag = diag

next iteration:
    previous E_out 直接成为下一发 E_in
    updated dn_gas 被继承
```

直接证据：

- 初始光场只在 loop 前生成一次：`runner.py:220`；lens 位于 `runner.py:222-229`；可选预推进位于 `runner.py:275-300`。
- pulse loop 在 `runner.py:307-324`；`E` 同时出现在 `propagate_one_pulse` 的输入位置 `runner.py:309-310` 和返回赋值位置 `runner.py:309`，中间没有 source field 重建或复制。
- `dn_gas` 只初始化为 `zeros((Ny,Nx))`：`runner.py:302`；每发后覆盖为 `diffuse_dn_gas(...)` 返回值：`runner.py:321`。
- `propagate_one_pulse` 在每个 z step 调用 `apply_nonlinear(..., dn_gas=dn_gas)`：`propagate.py:812`；`nonlinear.py:40-43` 将二维 `dn_gas` 广播到时间轴，没有 z 索引。
- `Qacc` 明确初始化为 `[Ny,Nx]` 且注释单位为 `J/m^2`：`propagate.py:474-476`；最终作为第二返回值返回：`propagate.py:1326-1327`。
- runner 将第二返回值命名为 `Q2D`，但没有保存它到输出；只保存最终 `dn_gas` 和最后一发 `last_diag`：`runner.py:309,324,343-355`。

### 3.2 三类对象的生命周期

| 对象/变量 | 当前 shape | 当前物理含义 | 实际单位 | 随 optical z 演化 | 跨 pulse 保存 | 审计结论 |
|---|---:|---|---|---|---|---|
| `E` | `[Nt,Ny,Nx]` complex | 光场包络/电场，初始由时域 Gaussian 与横向 profile 构造 | `V/m`（输入幅值诊断明确为 `V_m`） | 是，函数内原地/重绑定推进 | **是** | 不应作为 pulse-train 的跨发状态；应由 immutable source 为每发产生 fresh copy |
| `dn_gas` | `[Ny,Nx]` real | 慢时间气体折射率扰动 | 无量纲 `Δn` | 否；本发所有 z step 使用同一张图 | 是 | 只有单一二维 thermal plane，缺少 longitudinal screens |
| 瞬时 `Q` | `[Nt,Ny,Nx]` | 电离功率密度 + IB 功率密度 | `W/m^3` | 每个 z step 重算 | 否 | 内部临时量 |
| `Qslice` | `[Ny,Nx]` | 本 z step 对脉内时间积分后的沉积能量密度 | `J/m^3` | 每个 z step 产生 | 否 | 与 Eq. (31) 所需局部体沉积在量纲上相容，但没有作为 screen 状态返回 |
| `Qacc` | `[Ny,Nx]` | `Qslice*dz` 沿全传播区间累积后的面能量 | `J/m^2` | 在函数内跨 z 累积 | 仅以 runner 局部 `Q2D` 传给一次慢时更新 | 已丢失纵向沉积位置 |
| `Qacc_raman` | `[Ny,Nx]` | full Eq. (27) 实际局部 fluence loss 沿 z step 累加形成的横向面能量图 | `J/m^2` | 在函数内跨 z 累积 | 否；只在 `diag` 中返回 | runner 不把它加入慢时热源 |
| `E_dep_z` | `[Nz_record]` | 每个记录 z step 的电离 + IB 总沉积能量 | `J/step` | 是，诊断数组 | 只保留最后一发 | 只有 z 标量历史，没有横向 screen 分布 |
| `E_dep_rot_z` | `[Nz_record]` | 每个记录 z step 的 Raman 总沉积能量 | `J/step` | 是，诊断数组 | 只保留最后一发 | 有诊断不等于进入慢热源 |

## 4. 四个 P0 最终结论

### HR0-P0-1

**verdict: CONFIRMED**

**evidence:**

- `E` 在 `runner.py:220` 生成一次；lens 和 window pre-advance 也只在 loop 前执行（`runner.py:222-300`）。
- `runner.py:309-310` 以当前 `E` 调用 `propagate_one_pulse`，并把返回的 output field 再赋回同名 `E`。
- `runner.py:307-324` 的 loop 内没有 `build_transverse_input_field`、`copy(source_E)` 或等价重置逻辑。

因此当前软件语义是：

```text
E_out(pulse N, z=z_max) -> E_in(loop N+1, local z=0)
```

而不是：

```text
fresh source pulse + medium state left by previous pulses
```

**impact:** `Npulses > 1` 当前表示“把同一传播后光场重复推进 N 次，并同时更新一个二维热场”，不是物理 pulse train。第二轮起的输入面、波前、能量、时空形状和传播距离语义均已改变。

**recommended fix phase:** HR-1。引入不可变 source-field 责任边界；每发在同一输入面重新复制/生成光场，只有 medium state 和明确的慢变量跨发保存。

### HR0-P0-2

**verdict: CONFIRMED**

**evidence:**

- `dn_gas = zeros((grid.Ny, grid.Nx))`：`runner.py:302`。
- `propagate_one_pulse` 只接收一个 `dn_gas` 参数，没有 z/screen 轴：`propagate.py:208-225`。
- 每个 z step 都把同一对象传给 `apply_nonlinear`：`propagate.py:812`。
- `nonlinear.py:40-43` 直接将 `[Ny,Nx]` 加到 `[Nt,Ny,Nx]` 相位，靠时间轴广播；没有 z 查找、插值或 screen 选择。
- `Qslice[J/m^3]` 在每个 z step 产生，但 `Qacc += Qslice*dz`（`propagate.py:876-884`）最终压缩成 `[Ny,Nx] J/m^2`。
- 全仓对运行代码的搜索没有发现 `dn[z,y,x]`、thermal-screen 容器或等价 screen index。

**impact:** 当前实现丢失热沉积的纵向位置，并把一个横向热透镜施加到整个传播区间。它不能表达 filament onset、峰值沉积区和尾部在不同 z 处形成不同热屏，也不能复现论文“screen spacing resolves deposition curve”的接口。

**recommended fix phase:** HR-2。保留 z-resolved deposition 并建立 screen coordinates/weights；每个 optical z step 只读取对应 screen（或经明确插值获得的局部 `dn`）。

### HR0-P0-3

**verdict: CONFIRMED**

**evidence and dimensional chain:**

| 表达式 | 量纲 |
|---|---|
| `ion_source_raw = sum_j U_j * d(rho_j)/dt` | `J * m^-3 s^-1 = W/m^3` |
| `alpha_ib * I` | `m^-1 * W/m^2 = W/m^3` |
| `Q = ion_source_raw + alpha_ib*I` | `W/m^3` |
| `Qslice = sum_t(Q)*dt` | `J/m^3` |
| `Qacc += Qslice*dz` | `J/m^2` |
| declared `gamma_heat` | `Δn per J/m^3 = m^3/J` |
| current `gamma_heat * Qacc` | `(m^3/J)*(J/m^2) = m`，不是无量纲 |

代码证据为 `heat.py:25-37`、`propagate.py:474-476,876-884,1326-1327`、`heat.py:60-83`、`config.py:148-152` 和 `Config_explain.md:173-175`。

**A/B 判断:**

- 情况 A（物理定义有问题）：由当前仓库声明的接口直接支持。代码注释和配置文档都把 `gamma_heat` 定义为体能量密度到 `Δn` 的耦合，而实际传入的是沿 z 积分后的面能量。
- 情况 B（代码一直把 `gamma_heat` 当 `Δn/(J/m^2)`，只有文档错）：没有足够仓库证据支持。若强行把 `gamma_heat` 重新解释为 `m^2/J`，乘法可量纲闭合，但这将不再是 Isaacs Eq. (31) 的局部体沉积转换，且默认数值 `-1e-23` 没有找到相应的面积耦合物理来源。

Isaacs Eq. (31) 的系数为 `-(n0-1)/(rho0*Cv*T0)`，单位是 `m^3/J`；它乘的是局部 `∂F_L/∂z [J/m^3]`，即与当前 `Qslice` 同类，而不是全 z 积分后的 `Qacc[J/m^2]`。

**impact:** 当前 `dn_gas` 的幅值既没有量纲闭合，也无法与 Eq. (31) 做可解释的数值比较。默认 `gamma_heat` 应视为未物理标定的接口值，不能作为高重频科学结论依据。

**recommended fix phase:** HR-3，但依赖 HR-2 先提供每个 screen 的局部体沉积。HR-3 应明确空气热力学常数、符号、screen 厚度/采样方式与时间更新顺序。

### HR0-P0-4

**verdict: PARTIALLY_CONFIRMED**

“Raman 沉积没有完整进入慢时间热源”已经确认；之所以不是无条件 `CONFIRMED`，是因为 legacy 路径在配置字符串恰为 `absorption_model='poynting'` 时会把 `Q_rot_vol` 加入 `Qacc`。其他常见或完整 Eq. (27) 路径没有完成相同闭合。

| Raman 模式/路径 | Raman 能量损失有计算 | 有诊断 | 进入 `Qacc` | 进入 `Qacc_raman` | 最终进入 runner 慢时热源 |
|---|---|---|---|---|---|
| Raman disabled | 否 | 零/关闭状态 | 否 | 否 | 否 |
| `legacy_split` / `historical_fr_mixture`, `conv_deriv` | 是：产生 `Q_rot_vol`、`E_dep_rot_step` | 是：`E_dep_rot_z` | **否**：最终 gate 检查原始字符串是否等于 `poynting` | 否 | **否** |
| `legacy_split` / `historical_fr_mixture`, `poynting` alias | 是；内部仍映射到 conv-deriv 计算 | 是 | **是**：`Q_rot_vol*dz` | 否 | **是** |
| `legacy_split`, `closed_form` / `alpha_local` | 是：产生 `E_dep_rot_step` | 是 | 否：没有构造 `Q_rot_vol` | 否 | **否** |
| `split_energy_closed` | 配置耦合受限；legacy absorption/conv-deriv 被 normalize 拒绝 | 可有 raw/关闭诊断 | 未形成一致慢热源接口 | 否 | **否/接口不清** |
| `full_isaacs_eq27`, feedback ON | 是：field actual energy loss | 是：actual/target loss 与 `E_dep_rot_z` | 否 | **是**：actual local fluence loss | **否**：runner 不读取 `diag['Qacc_raman']` |
| `full_isaacs_eq27_complete`, feedback ON | 是：combined operator 中的 rotational actual loss | 是 | 否 | **是** | **否** |
| 两种 full mode, feedback OFF | target 可计算，actual loss 为 0 | 是 | 否 | 零 | 否 |

关键源码：

- ionization + IB 总是通过 `heat_Q_per_z` 进入 `Qslice`，再进入 `Qacc`：`propagate.py:641-695,876-884`。
- legacy conv-deriv 的 `Q_rot_vol` 为 `J/m^3`：`propagate.py:724-755`；但加入 `Qacc` 的条件使用原始 `absorption_model == 'poynting'`：`propagate.py:882-884`。因此配置写 `conv_deriv` 时，即使执行相同物理计算，也不会加入慢热源。
- closed-form 只生成总沉积能量诊断，没有横向 `Q_rot_vol`：`propagate.py:767-793`。
- full Eq. (27) 把 actual local fluence loss 累加到独立 `Qacc_raman`：`propagate.py:814-825`；`Qacc_raman` 只进入 `diag`：`propagate.py:1250`。
- 函数返回给 runner 的只有 `Qacc`：`propagate.py:1326-1327`；runner 只用该第二返回值更新 `dn_gas`：`runner.py:309-321`。
- `E_dep_total_z = E_dep_z + E_dep_rot_z` 的诊断一致性检查（`diagnostics.py:538-542`）只证明账面求和一致，不证明 Raman 已进入慢热源。

**impact:** 不同 Raman mode 或仅改变同义配置字符串，可能得到不同的慢时间热源组成；full Eq. (27) 路径尤其会出现“能量闭合诊断存在，但下一发完全看不到该 Raman 热沉积”的接口断裂。

**recommended fix phase:** HR-2 定义统一、分量化的 z-resolved deposition contract；HR-3 决定哪些已沉积能量按何时间尺度转化为热，并让 runner 只消费统一的 total thermal source，而不是从 diagnostics 反向拼接。

## 5. Isaacs ↔ 当前代码接口账本

| 功能 | Isaacs 模型 | 当前代码 | 状态 | 主要问题 | 后续阶段 |
|---|---|---|---|---|---|
| 单发光学传播 | Eq. (27) | `propagate_one_pulse`；含 legacy split 与 opt-in full Eq. (27) 变体 | `implemented` | 当前 solver 不等同于论文 PyCAP 的逐项复刻，但单发传播责任已存在 | 保持单脉冲基准；HR-1 不改公式 |
| fresh pulse injection | 每发独立输入脉冲 | `runner.py` 只生成一次 `E` | `not_implemented` | output field 直接成为下一发 input | HR-1 |
| 慢时介质跨 pulse 保存 | 每个 screen 的 `δn` 与 fluid state | 单一 `dn_gas[Ny,Nx]` | `partially_implemented` | 只有折射率；无 screen、velocity、time metadata | HR-1 定义状态边界；HR-2 扩维 |
| 多脉冲热累积 | Eq. (25)-(26) 解析递推/饱和 | pulse loop + `diffuse_dn_gas` | `partially_implemented` | source pulse 语义错误；只有单 screen；新沉积更新顺序不对应“沉积后经历一个脉间隔” | HR-1/HR-3 |
| 单发能量沉积 | Eq. (31) 前置的 local fluence loss | `Qslice`, `Qacc`, `E_dep_*_z` | `partially_implemented` | 分量存在但返回合同分裂；横向与纵向信息不能同时保留 | HR-2 |
| longitudinal heat profile | z-dependent screens | 无；`Qacc` 对 z 积分 | `not_implemented` | 丢失沉积位置 | HR-2 |
| deposition -> `δn` | Eq. (31) | `gamma_heat * Q2D` | `interface_only` | 输入是 `J/m^2`，声明系数要求 `J/m^3`；常数无物理 provenance | HR-3 |
| conduction | Eq. (32) 右端 `chi laplacian(delta n)` | `diffuse_dn_gas` 的二维 FFT damping | `partially_implemented` | 单 screen；更新顺序；边界条件为 FFT 周期边界但未形成高重频物理合同 | HR-3 |
| advection | Eq. (32) 左端 `v dot grad(delta n)` | 无 | `not_implemented` | 无速度场 | later |
| velocity / buoyancy | Eq. (33) | 无 | `not_implemented` | 无 `v`、黏性、重力耦合 | later |
| pulse diagnostics | 逐 pulse 的传播与沉积演化 | 只有 `last_diag` 写出 | `partially_implemented` | 无 pulse axis，无法审计累积过程 | HR-1 |

## 6. 其他重要 P1

### HR0-P1-1：脉间“扩散/注入”算子顺序与物理时间线不一致

当前 `diffuse_dn_gas` 实现：

```text
dn_new = diffuse(old_dn, delta_t_pulse) + gamma_heat * current_Q
```

runner 在本发传播后立即调用它，再让下一发读取 `dn_new`。因此本发新沉积没有经历 `1/f_rep` 的扩散，而旧 `dn` 经历了扩散。Isaacs 的时间线是 pulse 先产生 `δn+`，随后在脉间隔内由 Eq. (32)-(33) 演化，下一发读取演化后的状态。若仍采用 operator split，物理顺序应明确为“注入后演化”，或采用经验证的更高阶拆分；不能继续靠函数名掩盖顺序。

建议：HR-3 在 thermal-screen 和 Eq. (31) 闭合后统一确定更新顺序并增加单 screen 解析扩散测试。不要在 HR-1 修改这个物理算子。

### HR0-P1-2：没有 pulse-resolved 输出合同和 `Npulses>1` 回归测试

runner 每轮覆盖 `last_diag`，最终只保存最后一发的 z 诊断和最终 `dn_gas`；`Q2D` 本身不保存。现有配置和搜索到的测试均使用 `Npulses=1`，没有测试 fresh pulse、介质继承、逐 pulse 热源或 pulse axis shape。

建议：HR-1 在不改变单脉冲传播的前提下增加最小 monkeypatch/CPU 测试，至少断言每发 input field 等于 source copy、medium state 逐发传递、诊断具有 pulse index，且生产默认仍为 `Npulses=1`。

## 7. 后续接口草案（仅职责、shape、单位）

```text
Immutable SourceField [Nt, Ny, Nx], V/m
          |
          +-- fresh copy for pulse 1 --> optical propagation --+
          +-- fresh copy for pulse 2 --> optical propagation --+--> PulseDeposition
          +-- ...                                               |    [Ns, Ny, Nx], J/m^3
                                                               |
MediumState before pulse [Ns, Ny, Nx] -------------------------+
          ^
          |
post-pulse Eq.31 injection -> interpulse Eq.32/33 evolution over 1/f_rep
          |
MediumState for next pulse
```

### 7.1 最小概念对象

| 概念 | 职责 | 最小 shape | 单位/约束 | 跨 pulse |
|---|---|---:|---|---|
| `SourceField` | 描述固定输入面的单发 source；为每个 pulse 产生独立 field | `[Nt,Ny,Nx]` | complex `V/m`；应视为 immutable | 模板是；工作副本否 |
| `MediumState` | 保存下一发会遇到的慢介质 | `dn[Ns,Ny,Nx]`；未来可有 `velocity[Ns,2,Ny,Nx]` | `dn` 无量纲；`velocity` m/s；含 screen z 和当前慢时间 | 是 |
| `PulseDeposition` | 本发在每个 screen 的局部沉积，按通道分量化 | 每分量 `[Ns,Ny,Nx]` | `J/m^3`；至少 ionization、IB、Raman、total | 作为本发事件输入 medium update；可归档 |
| `ThermalScreens` | 定义 `z_screen[Ns]`、采样/聚合权重、光学 z 到 screen 的映射 | `z_screen[Ns]` | m；必须能解析 deposition curve | 是 |
| `PulseDiagnostics` | 保存每发光学与慢时结果，不覆盖前一发 | 标量/数组带前导 pulse 轴 `[Np,...]` | 每字段显式单位；区分 diagnostic loss 与 thermal source | 是/输出 |

### 7.2 推荐阶段边界

- **HR-1：source/runner contract。** 只解决 fresh pulse、medium state 显式传递、pulse-resolved diagnostics 和最小回归测试。保持 `propagate_one_pulse` 的单发数值行为和 production config 不变。
- **HR-2：z-resolved deposition + thermal screens。** 让传播器按约定 screen 输出 `J/m^3` 分量沉积，并在 optical z step 读取局部 screen `dn`。这一阶段建立数据结构与守恒/聚合检查，不先猜测完整流体模型。
- **HR-3：物理闭合的 heat -> δn 与 interpulse evolution。** 实现/校准 Eq. (31) 系数、空气常数、符号、注入-扩散顺序和单 screen 解析测试；随后才决定是否加入 advection/velocity/buoyancy。

## 8. HR-0 验收边界

- 修改生产代码：**no**
- 修改 production config：**no**
- 正式高重频传播：**0**
- HPC/Slurm 提交：**0**
- 本文给出的结论属于接口、量纲和数据流审计；不宣称当前高重频模型已复现 Isaacs 2022，也不改变冻结单脉冲物理基准。
