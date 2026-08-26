# HR-2 Thermal Screen Deposition Interface：待冻结问题清单

## 状态

- 记录日期：2026-08-26
- 阶段：HR-2（高重频热扩散链路的前置接口定义）
- 状态：待讨论 / 待冻结（7 个问题按下方顺序逻辑推进）
- 关联文档：`docs/architecture/hr0_high_repetition_interface_ledger.md`、`docs/architecture/hr1_pulse_train_orchestration.md`

## 核心目标

HR-2 的目标不是"开始做热扩散"，而是先把下面这条接口完全定义清楚：

$$
\text{single-pulse optical propagation}
\rightarrow
q_{\rm dep}(x,y,z)
\rightarrow
\text{longitudinal thermal screens}
$$

HR-2 只讨论并冻结以下 7 个问题，顺序按该逻辑推进。

---

## 七个待冻结问题

### 1. Thermal screen 的数据语义

- 一张 screen 代表什么：论文意义上应理解为位于 $z_k$ 的二维介质状态，而不是直接定义成有限厚度 cell。
- 每张 screen 至少需要承载哪些量。
- HR-2 阶段哪些只是 deposition 数据，哪些属于未来 HR-3 才出现的 medium state。
- 需要明确区分：

$$
\text{screen position } z_k
$$

与

$$
\text{deposition interval/bin}
$$

这两个概念，不能默认它们完全相同。

### 2. Longitudinal screen grid 如何布置

- $z_k$ 的覆盖范围应该从哪里到哪里。
- 使用均匀 spacing，还是允许非均匀 spacing。
- screen spacing 的物理判据是什么。
- Isaacs 只要求"充分解析 longitudinal energy-deposition curve"，因此需要进一步确定自己的收敛标准。
- 尤其要决定是否允许成丝/强沉积区域加密、远离该区域变稀。

### 3. Optical z-step 与 thermal screen 之间如何耦合

这是 HR-2 最关键的数值接口。

- optical propagation 有自己的 z stepping，甚至可以是 adaptive。
- thermal screen 是另一套 z 网格。
- 需要明确：

$$
z_{\rm optical}
\rightarrow
z_{\rm screen}
$$

的映射规则。

- 如果一个 optical step 跨过一个 screen，如何处理。
- 是在 screen 位置强制截步、插值、累计到 bin 后投影，还是采用其他方式。
- 这里必须做到与 optical solver 解耦，不能要求"一 optical step = 一 thermal screen"。

### 4. 局域 deposition 的物理量到底定义成什么

HR-2 应明确唯一基础量。倾向冻结为：

$$
q_{\rm dep}(x,y,z)
\quad [\mathrm{J/m^3}]
$$

即单发脉冲在当前位置向介质沉积的体能量密度。

同时需要区分：

$$
\text{optical power loss } [\mathrm{W/m^3}]
$$

脉内积分后成为：

$$
\text{local deposition } [\mathrm{J/m^3}]
$$

再沿 z 积分才成为：

$$
Q_{\rm acc}(x,y) [\mathrm{J/m^2}]
$$

HR-2 必须确定哪一个量作为以后 HR-3 的正式输入。

### 5. 不同沉积机制如何表示

需要逐项审计并决定至少是否保留：

$$
q_{\rm ion}, \quad q_{\rm IB}, \quad q_{\rm Raman}
$$

以及：

$$
q_{\rm dep,total}.
$$

这里要解决的不是"这些能量最后是不是都变成热"，而是：

- 当前代码中哪些机制已经有可靠的局域 $\mathrm{J/m^3}$ 表示；
- 哪些目前只有能量诊断；
- 哪些虽然造成 optical loss，但还没有进入 slow-medium source；
- 是否需要在 HR-2 保留 mechanism-resolved deposition，而不是过早只存一个 total。

应尽量保留分量，因为：

$$
\text{deposited energy}
\neq
\text{thermalized energy}
$$

这一层物理判断应留给 HR-3。

### 6. Deposition conservation / closure 如何定义

HR-2 必须有非常明确的验收物理关系。

例如从 thermal representation 重新积分：

$$
E_{\rm dep}
=
\int dz \int dxdy\, q_{\rm dep}(x,y,z)
$$

离散形式则根据 screen/deposition-bin 定义写成相应求和。

然后要与现有诊断逐机制核对：

$$
E_{\rm ion}^{\rm reconstructed} \approx E_{\rm ion}^{\rm existing}
$$

$$
E_{\rm IB}^{\rm reconstructed} \approx E_{\rm IB}^{\rm existing}
$$

$$
E_{\rm Raman}^{\rm reconstructed} \approx E_{\rm Raman}^{\rm existing}.
$$

这里需要提前确定：

- 比较的是哪一套"权威诊断"；
- 用绝对误差还是相对误差；
- 允许多大容差；
- adaptive step / projection 是否影响闭合定义。

### 7. Transverse grid 的架构边界

这个问题 HR-2 必须讨论，但不一定在 HR-2 真正做降采样。

要确定的是：

$$
(x,y)_{\rm optical}
$$

与

$$
(x,y)_{\rm thermal}
$$

是否在软件架构上强制相同。

建议明确：

$$
\boxed{
\text{HR-2 第一版可以同网格运行，但接口不能写死同网格}
}
$$

这样以后 HR-5 才能发展：

$$
\text{high-resolution optical grid}
+
\text{larger / lower-resolution thermal grid}
$$

而不需要重构整个 HR 数据流。

---

## 四个根本问题（压缩版）

如果进一步压缩，HR-2 实际上就是要回答四个根本问题：

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

---

## 建议讨论顺序

1. **先讨论 1–3**：screen 及 longitudinal 离散；
2. **再讨论 4–5**：deposition 的物理语义；
3. **最后讨论 6–7**：闭合和网格架构。

这样 HR-2 的物理定义会先稳定下来，再进入任何代码设计。