# HR-1 Pulse-train orchestration

## 基线与边界

- HR-1 start SHA: `82a2f02449399d6673dcfc4d54c8cfe608b6395a`
- frozen single-pulse physics SHA: `e11d13f103c484953c0f733aa9b410bff385b2b5`
- design basis: `docs/architecture/hr0_high_repetition_interface_ledger.md`

HR-1 只修复 runner 的 pulse/source ownership，并增加轻量逐 pulse 诊断。未修改热沉积公式、`gamma_heat`、扩散/注入顺序、二维 `dn_gas` 结构、Raman thermal routing、电离、BK-NEE 或 production config。

## 旧行为

旧 pulse loop 将传播返回值重新赋给 `E`，下一轮又把该 `E` 作为输入：

```text
E_out(pulse N) -> E_in(pulse N+1)
```

因此 `Npulses > 1` 会重复传播上一发的 output field，而不是让独立 source pulses 穿过持续演化的介质。

## 新行为

source plane 定义在所有 pulse-independent 光学预处理完成之后：

```text
build_transverse_input_field
    -> thin lens
    -> optional focus-window linear pre-advance
    -> E_source
```

`E_source` 不直接传给 `propagate_one_pulse`。每轮创建独立工作副本：

```text
E_pulse = E_source.copy()
E_out, Q2D, diag = propagate_one_pulse(E_pulse, dn_gas=dn_gas, ...)
```

本发 `E_out` 仅用于本发最终输出/诊断，不再成为下一发的 optical input。

## 状态所有权

| 状态 | 生命周期 | HR-1 行为 |
|---|---|---|
| `E_source[Nt,Ny,Nx]` | 整个 pulse loop | 只读 source；每发复制 |
| `E_pulse`, `E_out` | 单个 pulse | pulse-local；不跨发继承 |
| `dn_gas[Ny,Nx]` | 跨 pulse | 保持原有递推和更新调用 |
| `Q2D`, propagation intermediates | 单个 pulse | 保持现有行为 |
| `last_diag` | 最后一发 | 保留原输出合同 |

## Pulse-level diagnostics

NPZ 新增三个字段，不删除或重命名已有字段：

| 字段 | shape | 含义 |
|---|---:|---|
| `pulse_index` | `[Npulses]` | 1-based pulse index |
| `pulse_dn_gas_min` | `[Npulses]` | 每发完成现有 thermal update 后的 `dn_gas` 最小值 |
| `pulse_dn_gas_max` | `[Npulses]` | 每发完成现有 thermal update 后的 `dn_gas` 最大值 |

`return_results=True` 同步新增 `pulse_summary`，原有 `E_final`、`I_final`、`diagnostics` 和 `axes` 保持不变。

## Tests

`Filament_python/tests/test_runner_multipulse_orchestration.py` 包含两个独立测试：

1. multipulse contract：fake propagation 故意原地破坏每发工作场并返回明显不同的 output；测试确认三发输入值始终等于 source、工作数组不 alias、传播调用次数为三、介质输入按 `M0 -> M1 -> M2` 继承，且 pulse summary 为三条。
2. `Npulses=1` equivalence：tiny CPU runner 与同一 source/medium 的直接 `propagate_one_pulse` 比较，覆盖 `E_final`、`I_final`、z、rho 和 deposition/energy diagnostics。

修改前后还使用同一 tiny CPU 配置生成选择性单脉冲基线；两次 NPZ SHA256 均为：

```text
40D56BC16ECB540048701E18C3A4E0BF0CB9C15E577DDDB1BFC527AECD66AABD
```

该结果证明本次 runner 组织调整对该 tiny 单脉冲路径为 bitwise-equivalent；它不构成高重频物理有效性结论。

## Remaining HR-2 / HR-3 blockers

- HR-2：仍只有 `dn_gas[Ny,Nx]`，没有 longitudinal thermal screens 或 z-resolved transverse deposition。
- HR-3：`Qacc[J/m^2]` 与当前 `gamma_heat` 体积耦合定义仍未量纲闭合。
- HR-3：现有 diffusion/injection 顺序保持未改。
- HR-2/HR-3：Raman 的 diagnostic deposition、`Qacc_raman` 与 runner thermal source 仍未统一。
- Eq. (32) advection 与 Eq. (33) velocity/buoyancy 仍未实现。

HPC/Slurm jobs submitted: `0`。
