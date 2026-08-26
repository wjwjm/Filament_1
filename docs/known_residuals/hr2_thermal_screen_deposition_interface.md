# HR-2 Thermal Screen Deposition Interface：正式设计基线与待收敛问题

## 状态与冻结规则

- 记录日期：2026-08-26
- 阶段：HR-2（高重频 thermal-screen deposition 接口）
- 正式冻结：问题 1、2、3、6、7
- 尚未收敛：问题 4、5
- 关联文档：`docs/architecture/hr0_high_repetition_interface_ledger.md`、`docs/architecture/hr1_pulse_train_orchestration.md`
- 参考资料：Isaacs et al. 2022；`references/曾庆伟 - 2022 - 飞秒强激光在不同大气环境中传输成丝及其热沉积过程研究.pdf`

本文中的“固定/冻结”是指：下述结论作为 HR-2 当前实现和文档的正式设计决策；除非后续验证明确失败，否则不再反复改变架构。问题 4、5 的候选方案和物理分析单独列出，不得因其写入本文而视为已冻结。

HR-2 的总体目标是建立：

$$
\text{single-pulse optical propagation}
\rightarrow
\text{mechanism-resolved local deposition}
\rightarrow
\text{longitudinal thermal screens}.
$$

HR-2 只建立 longitudinal schedule、deposition representation、数据接口和守恒闭合。真正的 Eq. (31) 热折射率更新、Eq. (32) 脉间扩散/输运和 Eq. (33) 速度场/浮力属于 HR-3 或更后阶段。

---

# A. HR-2 正式设计基线（已冻结）

## 1. Thermal screen 的物理与数值语义

HR-2 采用 longitudinal 2D thermal screens 表示沿传播方向分布的慢介质位置。

第 $k$ 个 thermal state screen 对应固定 longitudinal 坐标：

$$
z_k,
$$

并在横向平面上承载未来的介质状态：

$$
S_k(x,y).
$$

HR-2 阶段主要建立 screen 坐标、沉积接口和数据结构，不在本阶段定义或推进完整的 $S_k$ 热/流体状态。

为使局域能量沉积具有有限体积意义，第 $k$ 个 screen 同时关联一个 longitudinal deposition interval：

$$
[z_k,z_{k+1}],
$$

或等价 control volume，宽度为：

$$
\Delta z_k=z_{k+1}-z_k.
$$

必须明确区分：

$$
\boxed{\text{thermal state screen at }z_k}
$$

与

$$
\boxed{\text{finite deposition interval }[z_k,z_{k+1}]}.
$$

二者通过同一 longitudinal framework 相关联，但不是同一个物理或软件对象。screen 是介质状态的空间锚点；interval/control volume 是沉积量积分和守恒闭合的空间单元。

## 2. Optical、deposition、thermal 共用 longitudinal framework

HR-2 正式采用方案 A：

$$
\boxed{
Z_{\rm optical}
=
Z_{\rm deposition}
=
Z_{\rm thermal}
}.
$$

定义一套显式、固定的 longitudinal schedule：

$$
Z=\{z_0,z_1,\ldots,z_K\},
$$

以及：

$$
\Delta z_k=z_{k+1}-z_k.
$$

每个 interval $[z_k,z_{k+1}]$ 同时对应：

- 一次 optical propagation step；
- 一个 deposition interval；
- 一个 thermal longitudinal interval/control volume。

整个 pulse train 的所有 pulse 必须使用完全相同的 schedule：

$$
Z^{(1)}=Z^{(2)}=\cdots=Z^{(N)}.
$$

因此上一发在 $z_k$ 的慢介质状态可以无歧义地传递给下一发同一位置：

$$
S_k^{(N)}\rightarrow S_k^{(N+1)}.
$$

HR-2 不实现真正的 state-dependent adaptive optical solver。

### 方案 B：仅作为后续备选

保留未来方案 B：

$$
\boxed{
\text{fixed thermal anchors}
+
\text{adaptive optical substeps}
},
$$

其中：

$$
\Delta z_{\rm optical}\leq\Delta z_{\rm thermal}.
$$

方案 B 可在 HR-5 或其他后续阶段用于减少 thermal-state 数量和计算成本。HR-2 不实现：

- accept/reject；
- retry；
- local error estimator；
- dynamic grow/shrink；
- pulse-dependent z-grid；
- optical/thermal longitudinal remapping。

## 3. Longitudinal schedule 与 spacing 原则

当前 production 中所谓 `auto_substep` 尚不构成真正的自适应步进；实际 schedule 主要由：

$$
dz_{\rm base}+dz_{\rm focus}+dz_{\rm final}
$$

组成。因此 HR-2 不把现有 base/focus 两档步长直接视为最终 thermal grid。

HR-2 必须建立显式的 longitudinal coordinate truth：

$$
\boxed{
z_{\rm edges}=\{z_0,z_1,\ldots,z_K\}
},
$$

而不是只依靠：

$$
z\leftarrow z+\Delta z
$$

的浮点累计来隐式恢复坐标。

正式 schedule 为 fixed nonuniform grid。它允许在成丝、强电离和强能量沉积区域加密，在平缓区域使用较粗 interval。

### 3.1 主要物理判据

schedule 必须充分解析 longitudinal deposition curve：

$$
\frac{dE_{\rm dep}}{dz}.
$$

候选 schedule 可优先利用已有单脉冲结果中的：

- `z_axis`；
- `dz_used_z`；
- `E_dep_z`；
- `E_dep_rot_z`；
- `I_max_z`；
- `rho_max_z`；
- nonlinear phase diagnostics。

已有结果用于离线识别沉积区域和生成 candidate grid，不要求一开始重新运行完整传播：

$$
\boxed{\text{historical results}\rightarrow\text{candidate grid}}.
$$

最终 spacing 仍必须通过数量有限、边界明确的 convergence calculations 确认：

$$
\boxed{\text{new calculations}\rightarrow\text{final confirmation}}.
$$

这意味着“fixed schedule”是 pulse-train 运行时固定，而不是未经收敛验证永久冻结某一组具体数值节点。

## 6. Deposition conservation / closure contract

HR-2 必须建立显式的逐 interval、逐 pulse 和全局能量账本。

对 deposition channel $c$，interval $k$ 的局域表示记为：

$$
q_{k,c}(x,y).
$$

在候选问题 4 的体能量密度表示下，该 interval 对应的沉积能量为：

$$
E_{k,c}^{\rm screen}
=
\iint q_{k,c}(x,y)\,dx\,dy\,\Delta z_k.
$$

即使问题 4 的最终字段命名或主存储量尚未正式冻结，以下三级 closure 层次已经冻结。

### Level 1：step / interval closure

要求：

$$
E_{k,c}^{\rm screen}
\approx
E_{k,c}^{\rm existing\ diagnostic}.
$$

每个 interval、每个 mechanism 必须能与现有权威 step diagnostic 对应，不能只在整发结束后比较一个总数。

### Level 2：per-pulse channel closure

定义：

$$
E_c^{\rm pulse}=\sum_k E_{k,c}.
$$

要求它与该机制现有的 per-pulse cumulative diagnostic 一致。

### Level 3：global field-energy bookkeeping

进一步检查：

$$
E_{\rm field,in}-E_{\rm field,out}
$$

与所有 dissipative energy-transfer channels 总和之间的闭合。

必须保持：

$$
\boxed{
\text{optical field loss}
\neq
\text{thermalized energy}
}.
$$

HR-2 只建立：

$$
\text{field-energy bookkeeping}
\leftrightarrow
\text{deposition bookkeeping}.
$$

HR-2 不假定所有 deposition 都立即、完全转化为平动热。具体 numerical tolerance 在实施阶段结合浮点精度、现有诊断定义和历史误差分析后确定，但三级闭合结构本身不再改变。

## 7. Transverse optical grid 与 thermal grid 的架构边界

HR-2 第一版采用同一 transverse grid：

$$
x_{\rm thermal}=x_{\rm optical},
$$

$$
y_{\rm thermal}=y_{\rm optical},
$$

即：

$$
(N_x,N_y,L_x,L_y)_{\rm thermal}
=
(N_x,N_y,L_x,L_y)_{\rm optical}.
$$

这一决定用于：

- 避免 transverse interpolation；
- 避免额外守恒误差；
- 简化 deposition 到 thermal-screen 的接口；
- 便于 HR-2 closure 验证。

但软件架构不得永久写死同网格关系。数据结构必须能够在未来表达：

$$
(N_x,N_y,L_x,L_y)_{\rm thermal}
\neq
(N_x,N_y,L_x,L_y)_{\rm optical}.
$$

HR-5 可进一步实现：

$$
\text{high-resolution / smaller optical grid}
+
\text{coarser / larger thermal grid}.
$$

HR-2 不实现 transverse remapping，但接口中不得通过隐式全局 axes 或硬编码 shape 排除未来 remapping。

---

## 已冻结的 longitudinal 数据流

在 interval $[z_k,z_{k+1}]$ 上，输入光场：

$$
E_k(x,y,t)
$$

经过一次固定步长 $\Delta z_k$ 的 optical propagation，得到：

$$
E_{k+1}(x,y,t),
$$

同时产生该 interval、该机制的局域 deposition representation：

$$
q_{k,c}(x,y).
$$

总体关系为：

$$
\boxed{
E_k
\rightarrow
\text{optical step }\Delta z_k
\rightarrow
q_{k,c}(x,y)
\rightarrow
\text{deposition interval / future }S_k
}.
$$

其中：

$$
\boxed{z_k,\Delta z_k}
$$

在整个 pulse train 中保持固定。

---

# B. 尚未收敛的问题

## 4. Authoritative local deposition quantity

### 当前状态

**尚未正式冻结，但已有首选候选定义。**

对 longitudinal interval：

$$
[z_k,z_{k+1}],
\qquad
\Delta z_k=z_{k+1}-z_k,
$$

若机制 $c$ 使介质获得的局域 fluence 为：

$$
\Delta F_{k,c}(x,y)
\quad[\mathrm{J/m^2}],
$$

当前首选 authoritative local deposition representation 为 interval-average volumetric deposition：

$$
\boxed{
q_{k,c}(x,y)
=
\frac{\Delta F_{k,c}(x,y)}{\Delta z_k}
\quad[\mathrm{J/m^3}]
}.
$$

因此：

$$
q_{k,c}\Delta z_k=\Delta F_{k,c}.
$$

### 4.1 选择该定义的理由

Isaacs Eq. (31) 需要的局域量本质为：

$$
-\frac{\partial F_L}{\partial z}.
$$

在有限 interval 上最自然的离散表示是：

$$
-\frac{\Delta F_L}{\Delta z},
$$

其单位为 $\mathrm{J/m^3}$。该定义也能正确比较 nonuniform grid 上不同 $\Delta z_k$ 的局域沉积强度；不能直接以不同厚度 interval 的 $\Delta F_k$ 大小代表相同意义的局域强度。

### 4.2 建议同时保留的三个账本层次

即使 authoritative quantity 最终冻结为 $q_{k,c}$，逻辑上仍建议同时保留：

$$
q_{k,c}(x,y)
\quad[\mathrm{J/m^3}],
$$

$$
\Delta F_{k,c}(x,y)
=q_{k,c}\Delta z_k
\quad[\mathrm{J/m^2}],
$$

以及：

$$
E_{k,c}
=
\iint\Delta F_{k,c}(x,y)\,dx\,dy
\quad[\mathrm{J}].
$$

即：

$$
\boxed{
q_{k,c}
\rightarrow
\Delta F_{k,c}
\rightarrow
E_{k,c}
}.
$$

三者分别服务于局域体沉积表示、横向局域守恒和 step-global diagnostic。

### 4.3 冻结前仍需确认

- `q` 表示 interval average、screen sample，还是明确命名的 deposition-bin average；
- 数组的 authoritative 存储字段和派生字段，避免三份冗余数据产生不一致；
- 与现有 `Qslice`、full-operator local fluence loss、`E_dep_*_z` 的逐机制映射；
- 符号约定：存储正的 medium energy gain，还是带符号的 optical fluence derivative；
- interval index 与 state-screen index 的边界/末端约定。

只有这些接口细节和问题 5 的 channel 语义共同确认后，问题 4 才转为正式冻结。

## 5. Deposition channels 与 thermalization 边界

### 当前状态

**尚未收敛，是 HR-2 当前最主要的物理接口问题。**

当前候选 mechanism-resolved deposition vector 为：

$$
\boxed{
\mathbf q_{\rm dep}
=
\{q_{\rm ion},q_{\rm IB},q_{\rm Raman}\}
}.
$$

候选 optical-to-medium 总能量转移账本为：

$$
q_{\rm dep,total}
=
q_{\rm ion}
+
q_{\rm IB}
+
q_{\rm Raman}.
$$

这一定义若被接受，只代表 optical field 到 medium 的能量转移，不等于 HR-3 的 thermal source。

### 5.1 Ionization deposition

当前 ionization source 本质为：

$$
P_{\rm ion}
=
\sum_j U_j\frac{\partial\rho_j}{\partial t},
$$

单位为：

$$
\mathrm{J}\times\mathrm{m^{-3}s^{-1}}
=
\mathrm{W/m^3}.
$$

时间积分得到：

$$
q_{\rm ion}=\int P_{\rm ion}\,dt
\quad[\mathrm{J/m^3}].
$$

从 optical energy bookkeeping 看，$q_{\rm ion}$ 是清晰的 transferred/deposited energy channel：光场为产生自由电子支付 ionization-potential energy。

但：

$$
\boxed{q_{\rm ion}\text{ may be deposition}}
$$

不等于：

$$
\boxed{q_{\rm heat,ion}=q_{\rm ion}}.
$$

其转化为平动热可能经过电子动能弛豫、复合、激发和分子过程；thermalization timing/efficiency 留给 HR-3。

### 5.2 Inverse Bremsstrahlung deposition

IB 功率密度为：

$$
P_{\rm IB}=\alpha_{\rm IB}I,
$$

时间积分得到：

$$
q_{\rm IB}
=
\int\alpha_{\rm IB}I\,dt
\quad[\mathrm{J/m^3}].
$$

IB 能量通过自由电子碰撞吸收进入介质，相较 ionization-potential energy 通常更接近直接热化。但 HR-2 仍只将其称为 IB deposition，不提前规定 thermalization efficiency。

### 5.3 Raman deposition

完整 Isaacs Eq. (27) operator 当前可给出 actual local field fluence loss：

$$
\Delta F_{{\rm Raman},k}(x,y).
$$

因此数学上可以定义：

$$
q_{{\rm Raman},k}
=
\frac{\Delta F_{{\rm Raman},k}}{\Delta z_k}.
$$

数值表示本身不是主要困难。尚未冻结的物理问题是：

$$
\boxed{
\text{rotational excitation energy}
\stackrel{?}{=}
\text{thermal energy available before the next pulse}
}.
$$

它取决于 rotational relaxation、collisional transfer、repetition period，以及 Isaacs 对 Raman energy deposition/heat source 的具体处理。

### 5.4 HR-2 不定义 `q_total_heat`

HR-2 应保留 mechanism-resolved deposition，而不直接创建或固定：

$$
q_{\rm total\ heat}.
$$

未来 HR-3 才定义 thermalization model：

$$
q_{\rm thermal}
=
\mathcal T
(q_{\rm ion},q_{\rm IB},q_{\rm Raman}),
$$

一般形式可写为：

$$
q_{\rm thermal}
=
\eta_{\rm ion}q_{\rm ion}
+
\eta_{\rm IB}q_{\rm IB}
+
\eta_{\rm Raman}q_{\rm Raman}.
$$

HR-2 不决定 $\eta_c$，也不得把现有 `Qacc_raman` 简单加入慢时间热源来替代这一物理判断。

### 5.5 问题 5 的核心待决项

需要回到 Isaacs 原文和曾庆伟学位论文逐项确认：

> Isaacs thermal Eq. (31) 中 $-\partial F_L/\partial z$ 对应当前代码哪些 channel 的和？尤其 rotational Raman loss 是否应与 ionization、IB 同等且立即地进入下一发之前的 thermal source？

在该问题有来源支持的答案之前，`q_dep,total` 只能作为 field-to-medium deposition bookkeeping；不得宣称它就是 `q_thermal`。

---

# C. 当前设计决策矩阵

| 问题 | 设计状态 | 当前结论 | 允许改变的条件 |
|---|---|---|---|
| 1. Screen 语义 | **frozen** | state screen 与 finite deposition interval 分离但共用坐标框架 | 后续验证证明该分离无法闭合或无法表达介质状态 |
| 2. Longitudinal framework | **frozen** | HR-2 采用 fixed shared optical/deposition/thermal schedule（方案 A） | 明确验证失败；方案 B 仅留作后续 |
| 3. Schedule/spacing | **frozen** | 显式 `z_edges`、fixed nonuniform grid；历史结果生成 candidate，新计算确认 | 收敛测试否定该 schedule 构造原则 |
| 4. Authoritative quantity | **open / preferred candidate** | 首选 $q_{k,c}=\Delta F_{k,c}/\Delta z_k$，并派生 $\Delta F$、$E$ | 完成字段、符号、索引和 channel 映射后冻结 |
| 5. Channel semantics | **open** | 保留 ionization、IB、Raman 分量；deposition 不等于 thermalization | 需 Isaacs/曾庆伟来源与当前算子证据 |
| 6. Closure | **frozen** | interval、per-pulse channel、global field-energy 三级闭合 | 仅 tolerance 数值可在实施后确定 |
| 7. Transverse grid | **frozen** | HR-2 同网格，但数据结构不得永久写死同网格 | 后续 HR-5 可实现 remapping，不改变 HR-2 接口边界 |

## 四个根本问题

HR-2 最终仍需完整回答：

$$
\boxed{\text{沉积在哪里？}}
$$

$$
\boxed{\text{沉积了多少？}}
$$

$$
\boxed{\text{由什么机制沉积？}}
$$

$$
\boxed{\text{怎样无损地交给下一阶段的慢介质模型？}}
$$

其中问题 1–3、6、7 已固定“在哪里、如何离散、如何闭合、网格边界”；问题 4、5 继续收敛“权威局域量”和“机制/热化语义”。在问题 4、5 正式冻结前，不进入 HR-2 production implementation。
