# HR-2 Thermal Screen Deposition Interface：正式设计基线

## 状态与冻结规则

- 记录日期：2026-08-26
- 阶段：HR-2（高重频 thermal-screen deposition 接口）
- 正式冻结：问题 1–7
- 关联文档：`docs/architecture/hr0_high_repetition_interface_ledger.md`、`docs/architecture/hr1_pulse_train_orchestration.md`
- 参考资料：Isaacs et al. 2022；`references/曾庆伟 - 2022 - 飞秒强激光在不同大气环境中传输成丝及其热沉积过程研究.pdf`

本文中的“固定/冻结”是指：下述结论作为 HR-2 当前实现和文档的正式设计决策；除非后续验证明确失败，否则不再反复改变架构。具体 schedule 节点、数值容差和 HR-3 thermalization 参数仍需验证，但不改变 HR-2 的接口语义。

HR-2 的总体目标是建立：

$$
\text{single-pulse optical propagation}
\rightarrow
\text{mechanism-resolved local deposition}
\rightarrow
\text{longitudinal thermal screens}.
$$

$$
\boxed{
\text{HR-2}
=
\text{fixed longitudinal screen framework}
+
\text{mechanism-resolved local deposition}
+
\text{energy closure}
}.
$$

HR-2 不实现 Eq. (31) heat-to-$\Delta n_{\rm gas}$、Eq. (32) diffusion 或 Eq. (33) advection/buoyancy。这些属于 HR-3/HR-4。此边界禁止在 HR-2 实现时顺带把旧 `heat.py` 扩展成完整热模型。

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

按照问题 4 已冻结的体能量密度表示，该 interval 对应的沉积能量为：

$$
E_{k,c}^{\rm screen}
=
\iint q_{k,c}(x,y)\,dx\,dy\,\Delta z_k.
$$

以下三级 closure 层次正式冻结。

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

与所有已知 dissipative energy-transfer channels 总和之间的闭合。

但 $E_{\rm field,in}-E_{\rm field,out}$ 只能作为 closure diagnostic，绝不能直接作为 heat source，因为它还可能包含：

- domain escape；
- diffraction/outflow；
- numerical error；
- 未来的 scattering；
- 其他非局域热化的 field-energy change。

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

# B. 已冻结的 deposition 定义与机制边界

## 4. Authoritative local deposition quantity

### 冻结结论

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

authoritative local deposition representation 正式冻结为 interval-average volumetric deposition：

$$
\boxed{
q_{k,c}(x,y)
=
\frac{\Delta F_{k,c}(x,y)}{\Delta z_k}
\quad[\mathrm{J/m^3}]
}.
$$

符号约定正式冻结为：

$$
\boxed{q_{k,c}(x,y)\geq 0}
$$

表示 medium energy gain，而不是保存带负号的 optical fluence derivative。因此：

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

### 4.2 三个账本层次与 canonical representation

逻辑上保留：

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

代码中不得维护三套相互独立的 authoritative arrays。$q_{k,c}$ 是 canonical representation；$\Delta F_{k,c}$ 和 $E_{k,c}$ 必须由 $q_{k,c}$、$\Delta z_k$、$dx$、$dy$ 推导，避免账本漂移。

### 4.3 索引合同

- $K+1$ 个 longitudinal edges/state anchors：$z_0,\ldots,z_K$；
- $K$ 个 deposition intervals：$[z_k,z_{k+1}]$，$k=0,\ldots,K-1$；
- $q_{k,c}$ 是 interval average，不是 state-screen point sample；
- 最末端 $z_K$ 是 state anchor/边界，不对应额外 deposition interval。

现有 `Qslice`、full-operator actual local fluence loss 和 `E_dep_*_z` 到该合同的逐机制映射属于 HR-2 实现任务，但不得改变上述 shape、单位、符号和索引语义。

## 5. Deposition channels 与 thermalization 边界

### 冻结结论

mechanism-resolved deposition vector 正式冻结为：

$$
\boxed{
\mathbf q_{\rm dep}
=
\{q_{\rm ion},q_{\rm IB},q_{\rm Raman}\}
}.
$$

optical-to-medium 总能量转移账本正式冻结为：

$$
q_{\rm dep,total}
=
q_{\rm ion}
+
q_{\rm IB}
+
q_{\rm Raman}.
$$

这里的 deposition 严格表示 optical field 到 medium 的吸收性能量转移，不等于 HR-3 的 thermal source：

$$
\boxed{q_{\rm dep}\neq q_{\rm thermal}}.
$$

### 5.1 Ionization deposition

Ionization deposition 必须使用正的 photoionization creation rate：

$$
P_{\rm ion}
=
\sum_j U_j W_j(I)(N_{j,0}-\rho_j),
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

不得把一般意义的净变化率：

$$
U_j\frac{d\rho_j}{dt}
$$

永久定义为 ionization deposition。未来加入 recombination 或 attachment 后，净 $d\rho_j/dt$ 可出现负项；这不能解释为激光从介质中重新取回电离能。HR-2 的 canonical ionization deposition 只取正的光致生成项。

当前 production 配置中 $\beta_{\rm rec}=0$，因此这一语义修正不改变当前冻结单脉冲基准数值，但 HR-2 实现必须建立独立、明确的 positive-creation source，不能继续依赖包含复合项的 `drho_dt_u_sum` 名义。

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

当前三份 production config 的 effective IB 均为零：`sigma_ib=0`；`nu_ei_const` 在 `khz_config*.json` 中为 `0`，在 `config_ref.json` 中为 `null` 并退回零 `sigma_ib`。因此当前 40–120 fs production 路径满足：

$$
q_{\rm IB}^{\rm current}=0.
$$

HR-2 仍保留 $q_{\rm IB}$ channel，以便未来非零碰撞吸收配置不需要改变 deposition schema。

### 5.3 Raman deposition

Raman deposition 的 authoritative source 正式冻结为完整 Isaacs Eq. (27) operator 实际造成的 local field-fluence loss：

$$
\Delta F_{{\rm Raman},k}(x,y).
$$

因此：

$$
q_{{\rm Raman},k}
=
\frac{\Delta F_{{\rm Raman},k}}{\Delta z_k}.
$$

历史 `Q_rot_vol` / `w_R` 不作为 HR-2 authoritative Raman deposition source，因为它们不是完整算子实际局域 field-fluence loss，且既有审计已确认它们不能替代严格的 Eq. (10)/full-operator energy-transfer closure。

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

### 5.5 HR-3 thermalization boundary

对当前 Isaacs benchmark，HR-3 第一版可采用待独立验证的 slow-time closure：

$$
q_{\rm thermal}=q_{\rm ion}+q_{\rm Raman},
$$

理由是相关 microscopic relaxation time 相对 kHz pulse separation 很短，且当前 $q_{\rm IB}=0$。但这属于 HR-3 thermalization assumption，不是 HR-2 deposition 定义；HR-3 必须记录来源、适用 repetition-rate 范围和验证结果。

---

# C. 当前设计决策矩阵

| 问题 | 设计状态 | 当前结论 | 允许改变的条件 |
|---|---|---|---|
| 1. Screen 语义 | **frozen** | state screen 与 finite deposition interval 分离但共用坐标框架 | 后续验证证明该分离无法闭合或无法表达介质状态 |
| 2. Longitudinal framework | **frozen** | HR-2 采用 fixed shared optical/deposition/thermal schedule（方案 A） | 明确验证失败；方案 B 仅留作后续 |
| 3. Schedule/spacing | **frozen** | 显式 `z_edges`、fixed nonuniform grid；历史结果生成 candidate，新计算确认 | 收敛测试否定该 schedule 构造原则 |
| 4. Authoritative quantity | **frozen** | $q_{k,c}=\Delta F_{k,c}/\Delta z_k\geq0$ 为 canonical interval-average `J/m^3`；$\Delta F$、$E$ 为派生账本 | 后续验证证明该表示无法闭合或无法对接 Eq. (31) |
| 5. Channel semantics | **frozen** | $q_{\rm dep}=q_{\rm ion}+q_{\rm IB}+q_{\rm Raman}$；positive creation、IB absorption、full-operator actual Raman loss；deposition 不等于 thermalization | 新物理机制加入时扩展 channel，不重定义既有语义 |
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

七个问题现已全部形成 HR-2 正式设计基线：

- 问题 1–3 固定“screen/interval 是什么、纵向网格怎样统一、schedule 如何生成和验证”；
- 问题 4–5 固定“权威局域量、符号、索引、mechanism channels 以及 deposition/thermalization 边界”；
- 问题 6 固定三级 energy closure；
- 问题 7 固定 HR-2 同横向网格但接口不永久写死同网格。

下一步可以进入有边界的 HR-2 software design/implementation，但不得越界实现 Eq. (31)–(33) 或直接改造 slow-time heat source。
