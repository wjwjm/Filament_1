> **历史交接快照 — SUPERSEDED / NOT CURRENT PLAN（2026-09-16）。**
> 以下正文保留原审查前状态与被撤回HR4C-authoritative方案，均不得作为当前路线/授权。
> 当前 E5-0 = CLOSED / STREAMING_PREPARATION_CONTRACT_ACCEPTED；E5-1 = NOT STARTED。
> implementation_authorized=false；formal_execution_authorized=false；resource_execution_gate=NOT_RELEASED；ENTRY_BLOCKED_BY_S5_FINAL。
> 当前正式依据：[E5-0 closeout](E5_0_CLOSEOUT_20260916.md)。

# HR-4E-5：Formal E5 新对话上下文

**更新时间：** 2026-09-16
**用途：** 后续 Codex 对话承接；优先理解研究阶段、当前门槛、已作设计决定与授权边界。
**工作分支：** `codex/hr4e5s-s5-lifecycle`
**最近已推送文档提交：** `cbfe2e3`（其前置设计提交：`c92118f`）
**Formal E5 entry gate：** `ENTRY_BLOCKED_BY_S5_FINAL`

## 0. 一句话状态

HR-4E-5 处在“**S5 最终资格收尾等待 + Formal E5 设计已收束、待人工审查**”的位置：

```text
S5-FINAL terminal evidence
      └── 尚是 Formal E5 实施/提交的硬门槛

E5-0 authority/interface/resource design
      └── 已完成并 READY_FOR_MANUAL_REVIEW

E5-1 legal N=3 full-z engineering canary
      └── 已完成设计，未获实施或运行授权

E5-2 full-z/endurance/formal production validation
      └── 未开始
```

本轮只完成只读审计、设计和文档/审查包；**没有**修改生产物理、运行时、配置、LUT、源数组或历史结果，**没有**创建算例、提交 Slurm/GPU 作业，也**没有**干预 S5。

---

## 1. 整个研究问题的阶段构思（重点）

| 层级 | 阶段目的 | 当前结论/进度 | 后续进入条件 |
| --- | --- | --- | --- |
| 历史基础 | HR-1 至 HR-4D 建立并冻结光学、HR-2/HR-3、HR-4 慢介质及 restart-safe 生命周期基础。 | 已有冻结基线与历史资格证据；仅作继承背景，不在当前重新验证或改写。 | 不自动重跑或改基线。 |
| HR-4E 前置资格 | E3/E4、S3/S4R 等以 48-screen 资格窗口验证执行器、屏障、恢复和拓扑。 | 资格结果只能说明其记录范围；不可升级为正式多脉冲全 z 输入。1+4 为已测试首选，1+2 为已验证低资源回退；1+6 为 `NOT TESTED / RESOURCE_UNAVAILABLE`。 | 仅供后续工程路径和资源判断参考。 |
| S5-FINAL 收尾 | 完成集成 worker-loss/recovery 资格的终态、精确比对与 provenance closeout。 | **当前 Formal E5 硬门槛。** 最后一次只读快照中作业 `244700` 为 `PENDING (Priority)`；该快照不是终态结果，后续必须重新核验 scheduler/controller/evidence。 | S5-FINAL terminal PASS、`sacct` 终态、精确/provenance 审计和 F01--F06 跨 SHA 继承表。 |
| **E5-0（当前）** | 选定唯一慢介质权威；定义跨脉冲、PRE_0、restart、资源与最小 glue；形成网页人工审查包。 | **`E5_0_AUTHORITY_INTERFACE_READY_FOR_MANUAL_REVIEW`。** 设计已收束；实施仍未授权。 | 人工审查接受设计；S5 完成后，再分别授权实施和资源。 |
| E5-1 | 合法的“小”多脉冲工程闭环、精确比较和 restart canary。 | 设计 verdict：`E5_1_SMALL_CASE_REQUIRES_FULL_Z`；候选为 `Npulses=3`，未运行。 | S5 PASS + 单独授权的最小 glue + PRE_0 哈希 + 资源/保留策略 + 测试/预检。 |
| E5-2 | 全 z、耐久/跨 allocation restart、正式规模验证及科学结论。 | `NOT_STARTED`。 | E5-1 工程闭环通过，且另有正式科学问题、脉冲数、资源、诊断保留和验收标准授权。 |

### 历史完成阶段如何使用

HR-1--HR-4D 与 E3/E4/S3/S4R 的完成/资格结果是**非重点背景**：它们提供冻结算子、输入 provenance、已验证的局部执行/恢复证据和拓扑参考。它们不是 Formal E5 已完成的证明，也不允许把 48-screen 记录、单脉冲配置、历史耗时或资格队列参数直接作为正式全 z 多脉冲科学定义。

---

## 2. 已作出的 E5-0 设计决定（后续不得重新引入歧义）

### 2.1 唯一权威

Formal E5 的唯一 canonical slow-medium authority 为：

```text
HR4DPulseController + HR4CThreeFieldStore
```

- `HR4CThreeFieldStore`：`AUTHORITATIVE_STATE_STORE` + `TRANSACTION_STAGING`；其 manifest-selected 三场 slot 是唯一 canonical PRE/POST。
- `HR4DPulseController`：脉冲生命周期、phase/pulse-index/provenance 元数据；generation 与 HR4C manifest 绑定。
- `HR4DPulseTransaction`：光学/HR-3 到 canonical POST 的事务接口。
- 未来 Streaming：只能是 `STREAMING_EXECUTION_ADAPTER`，可有可恢复 per-screen staging、claims、barrier/telemetry；不得拥有独立 pulse index、独立 authority promotion 或正式 `authoritative_generation.json`。
- 现有 `StreamingLifecycle CURRENT/POST/NEXT` 是 S3/S4R/S5 的资格实现和历史证据，**不是** Formal E5 的第二权威。

### 2.2 固定的跨脉冲契约

```text
canonical PRE_p (HR4C)
  -> fresh immutable E_source.copy()
  -> propagate_one_pulse reads PRE.delta_n once per interval
  -> authoritative HR-3 POST:
       delta_n_post = delta_n_pre + increment
       vx_post = vx_pre
       vy_post = vy_pre
  -> atomic canonical POST_p commit
  -> non-authoritative hydro NEXT staging + full barrier
  -> coordinator fills HR4C staging
  -> atomic canonical PRE_(p+1) commit
  -> generation and pulse index advance exactly once
```

最后一脉冲停在 `POST_final`，不做 final interpulse。光学输出绝不作为下一脉冲光学输入；慢介质仅经 canonical PRE 继承。

真实 restart 入口是：

```text
HR4DPulseController(..., resume=True)
  -> HR4CThreeFieldStore.open_existing(...)
```

不存在 `HR4DPulseController.open`。

### 2.3 PRE_0 三场候选契约

`PRE_0` 必须同时绑定 `{delta_n, vx, vy}`，不能只引用一个 `delta_n` 文件。

| 场 | 候选来源/规则 | 证据与待完成项 |
| --- | --- | --- |
| `delta_n` | E1B 冻结 HR-3B 源数组 | raw SHA-256 `70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467`；array SHA-256 `5990da24bec80937bf3be9985777b3797be9adffedb776dd2f2e840b118d8ad9`。 |
| `vx` | 现有 HR4C batch initializer 的确定性 float64 零初始化 | 规则已有 HR4C/S3 资格证据；未来 preflight 仍须 materialize 并记录持久化文件 hash。 |
| `vy` | 同 `vx` | 同上。 |

共同候选 layout 为历史全 z `[15000,351,301]`、float64、`dx=dy=1e-5 m`、generation 0、phase PRE、pulse index 0；z-edge、config/source manifest 与三场实际持久化 hash 必须在未来非覆盖 preflight 中一起绑定。该规则是候选，不是已创建的 Formal E5 输入。

### 2.4 E5-1 合法小算例

- Verdict：`E5_1_SMALL_CASE_REQUIRES_FULL_Z`。
- “small” 仅指工程脉冲数：`Npulses=3`；它产生两次真正的 `POST -> PRE_next -> optical` rollover，并检查最终 `POST_final`。
- 当前没有可合法缩短的连续 z 域：冻结光学输入起点与完整慢介质覆盖均对应 full-z；旧 48-screen 是峰附近抽选资格记录，不是状态域、也不覆盖连续传播区间。
- 比较设计：serial HR4D-authority reference 对比同输入、同 `PRE_0`、同 full-z、同 N=3、同算子/精度且仅加入非权威 adapter 的 candidate。逐脉冲精确比较 source identity、PRE/POST/next-PRE 三场 hash、HR-3 deposition ledger、barrier、generation/pulse counters；物理趋势幅值不是工程 PASS 门槛。

### 2.5 资源结论（仅模型，不是 allocation 批准）

对历史 illustrative geometry `K=15000, Ny=351, Nx=301, float64`：

- 一组三场 generation：`G = 35.422 GiB`。
- 已选 canonical HR4C 两 slot 稳态：`2G = 70.845 GiB`。
- 若为 restart barrier 保留完整非权威 NEXT staging，一次 hydro transaction 条件性为 `3G = 106.267 GiB`。
- 上述数字不是 campaign peak、quota estimate 或 retained-data upper bound；checkpoints、历史、诊断、光场、worker/transfer buffer、原子临时文件、container/filesystem overhead 和保留策略均另计。
- S3 initializer 的 `selected = np.asarray(source[indices])` 加 `zero = np.zeros_like(selected)` 在 full volume 会额外 materialize 两个 host fields（23.615 GiB raw，外加 mapped source），故 Formal E5 **不得**采用该初始化方式。
- 一份 fp64 complex optical field 为约 0.605 GiB；immutable source + working copy 至少约 1.209 GiB GPU payload，尚未含 kernels/diagnostics/transfers。

已知 site-level 缺口：quota、purge policy、numeric QoS limit、GPU model/VRAM/CUDA、observed MaxRSS/VRAM/I/O、正式 retention/checkpoint policy。历史 48-screen 耗时不得线性外推为 Formal E5 full-z runtime。

---

## 3. 当前应做什么、绝不能做什么

### 现在可做

1. 对 E5-0 审查包作人工/web 审查，确认唯一权威、PRE_0、E5-1 和资源逻辑是否接受。
2. 等待并在 S5 状态变化后，以最新 scheduler/controller、`sacct` 和持久化审计证据完成 S5 closeout；不要从旧 `PENDING` 快照推断当前状态。
3. 保持设计与实施/运行授权分离；可做后续只读证据整理，但不可借此绕过 gate。

### 现在禁止

- 不实现 formal orchestration glue，不添加正式 entry/adapter，不改变 `propagate_one_pulse`、HR-2、HR-3、HR-4、ionization、Raman、精度、LUT、frozen configs 或 source arrays。
- 不创建/覆盖 run root、checkpoint、PRE_0 实际数组或历史结果。
- 不提交、取消、hold、requeue、重启或轮询干预 S5-FINAL；尤其不以 E5-0 设计作为 S5 之外的替代执行路径。
- 不将 48-screen、`Npulses=1`、queue/block 数、1+6 或历史耗时自动提升为 Formal E5 参数/资源承诺。

---

## 4. 下一次对话的最小起点

先阅读以下文件，再根据用户新的明确授权行动：

1. `docs/physics_decisions/hr4e5/e5_0/formal_hr4e5_e5_0_authority_interface_design_20260916.md`
2. `docs/physics_decisions/hr4e5/e5_0/formal_hr4e5_e5_0_authority_contract.json`
3. `docs/physics_decisions/hr4e5/e5_0/formal_hr4e5_e5_0_freeze_candidate.json`
4. `docs/physics_decisions/hr4e5/e5_0/formal_hr4e5_e5_0_resource_budget.csv`
5. `docs/physics_decisions/hr4e5/e5_1/formal_hr4e5_e5_1_small_case_candidate_20260916.md`
6. `docs/physics_decisions/hr4e5_hpc_audit_pack_20260916.md`
7. `docs/review_bundles/HR4E5_E5_0_AUTHORITY_INTERFACE_WEB_REVIEW_20260916.zip`

若用户只要求“检查 S5 是否收尾”，先做 live read-only scheduler + controller + `sacct` + persisted-artifact 核验；终态、队列状态或生成文件本身都不能代替科学/工程 closeout。若用户要求“开始 E5-1”，应先指出：还缺 S5 PASS、实施任务授权、PRE_0 三场 materialized hashes、site resources/retention contract，以及强制的 N=3 exact/restart test 计划。

---

## 5. 本轮审查包与提交

- 设计提交：`c92118f` — `docs(hr4e5): select E5-0 authority and cross-pulse contract`
- 审查包提交：`cbfe2e3` — `docs(hr4e5): add authority-interface web review bundle`
- 审查 ZIP：`docs/review_bundles/HR4E5_E5_0_AUTHORITY_INTERFACE_WEB_REVIEW_20260916.zip`
- ZIP SHA-256：`83EFA73AFD1805EF48151F1F6FF50549DE95BBD9B501E4B47FBFC8E8C8FA17AE`

该 ZIP 已复开校验：16 个条目、15 条内部 SHA-256、全部 bundled JSON/CSV 可解析，并含 HR4D、HR4C、Streaming、S3、runner 和 `propagate.py` 的带 blob/行号/用途源码摘录。
