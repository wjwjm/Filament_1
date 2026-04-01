# Filament_1

本仓库用于 **kHz 高重频激光在空气中的成丝（filamentation）数值仿真**。程序核心覆盖：

- 线性传播（`uppe` / `paraxial` / `bk_nee`）；
- 非线性效应（Kerr、自陡峭、拉曼、电离、等离子体相位与吸收）；
- 多脉冲慢时演化（热扩散与折射率通道 `dn_gas` 累积）。

适合两类场景：
1. 新手快速上手；
2. 间隔较久后快速恢复“程序运行逻辑 + 参数位置感”。

---

## 1. 代码结构总览（先知道去哪看）

- 运行入口：`Filament_python/KHz_filament/cli.py`
- 配置读取与派生：`Filament_python/KHz_filament/confio.py`
- 配置数据结构（默认值来源）：`Filament_python/KHz_filament/config.py`
- 单脉冲传播主循环（Strang 分裂）：`Filament_python/KHz_filament/propagate.py`
- 非线性算子：`Filament_python/KHz_filament/nonlinear.py`
- 诊断与输出：`Filament_python/KHz_filament/diagnostics.py`
- 示例配置（建议新手先从这里改）：
  - `Filament_python/config_ref.json`（轻量）
  - `Filament_python/khz_config.json`（较大网格）

---

## 2. 一次仿真的完整执行流程（从命令到输出）

### Step A：命令行启动

```bash
python -m Filament_python.KHz_filament.cli Filament_python/config_ref.json
```

`cli.py` 会调用 `run_from_file()`，随后进入 `run_demo()`。

### Step B：加载配置 + 自动补齐

`confio.load_all()` 会读取 JSON/YAML/TOML，然后做两类关键补齐：

1. **自动补时间窗**：若 `grid.Twin` 缺省，按 `Twin = 8 * tau_fwhm`。
2. **自动反推电场峰值 `E0_peak`**（当 `E0_peak == 0` 时）：
   - 给了 `beam.energy_J` → 用能量反推；
   - 给了 `beam.P0_peak` → 用峰值功率反推；

并且 `energy_J`、`P0_peak` 彼此互斥，冲突会直接报错。

### Step C：建网格、造入射场、可选加透镜相位

`run_demo()` 里：

1. 用 `make_axes()` 建立 `x/y/t/Ω/k⊥` 轴；
2. 用 `gaussian_beam_xy * gaussian_pulse_t` 生成 `E(x,y,t)`；
3. 若有 `beam.focal_length`，会在频域加薄透镜相位（UPPE 走按频率折射率的 achromatic 版本）。

### Step D：单脉冲传播（核心）

`propagate_one_pulse()` 按 **Strang splitting** 执行：

1. 线性半步；
2. 非线性整步（Kerr/自陡峭/电离/等离子体/拉曼 + 吸收）；
3. 再线性半步。

并在 z 方向累积诊断量（`U_z`、`I_max_z`、`rho_onaxis_max_z`、`w_mom_z`、`fwhm_*` 等）。

### Step E：多脉冲慢时迭代

若 `run.Npulses > 1`，外层循环会在每个脉冲后：

- 根据沉积热源 `Q2D` 调 `diffuse_dn_gas()` 更新 `dn_gas`；
- 下一个脉冲继续在更新后的背景介质上传播。

### Step F：保存输出

最终将坐标 + 诊断数组写入 `.npz`（默认 `khzfil_out.npz`），用于后处理或与实验趋势对照。

---

## 3. 参数设置重点（新手最常改、最容易互相耦合）

> 建议按“从几何到物理、从稳定到精细”的顺序改参数。

### 3.1 `grid`（采样与算力成本）

- `Nx, Ny, Nt`：空间/时间采样点数，直接决定内存与速度；
- `Lx, Ly`：横向窗口，太小会裁边，太大则浪费分辨率；
- `Twin`：时间窗，要覆盖主脉冲及非线性展宽尾部。

建议：先小网格 smoke test（例如 `128×128×96`），再放大。

### 3.2 `beam`（入射条件）

关键字段：

- `lam0`, `n0`, `w0`, `tau_fwhm`, `focal_length`, `n2_air`；
- 三选一给幅值入口：`E0_peak` **或** `energy_J` **或** `P0_peak`。

推荐新手优先用 `energy_J`（物理直观），让程序自动反推 `E0_peak`。

### 3.3 `propagation`（数值稳定性中枢）

- `linear_model`: `uppe` / `paraxial` / `bk_nee`。
- 步长主控：`z_max`, `dz`, `auto_substep`, `dz_min`, `grow_factor`, `shrink_factor`。
- 稳定阈值：
  - `max_linear_phase`
  - `max_alpha_dz`
  - `max_kerr_phase`
  - `imax_growth_limit`
- 焦区细化：`focus_window_step`, `focus_center_m`, `focus_halfwidth_m`, `dz_focus`。
- 围焦窗裁剪：`limit_focus_window`, `window_halfwidth_m`（大幅省时，适合调参）。
- 内存相关：`full_linear_factorize`（UPPE 下显著影响显存峰值）。

经验：先保证稳定（不过冲、不爆振），再追求更细步长和更大网格。

### 3.4 `ionization`（最容易影响电子密度量级）

核心是 `species` 列表，每个组分定义 `name/rate/fraction` 与模型参数。

常用 `rate`：

- `ppt_talebpour_i_full`（N2/O2 常用）
- `popruzhenko_atom_i_full`（原子/离子场景）
- `ppt_talebpour_i_legacy`（回归对照）
- `mpa_fact` 或 `off`

其它关键开关：

- `time_mode`: `full` / `qs_peak` / `qs_mean` / `qs_mean_esq`
- `integrator`: `rk4` / `euler`（主要对 `full` 有意义）
- `cycle_avg_samples`
- `beta_rec`, `sigma_ib`, `nu_ei_const`
- 限幅：`I_cap`, `W_cap`

建议：做模型对照时固定 `time_mode=full` + `rk4`，一次只改一项。

### 3.5 `raman`（延迟响应与吸收）

- `enabled`, `f_R`, `model`, `T2`, `T_R`; 
- 吸收相关：`absorption`, `absorption_model`；
- `method` 与 `chunk_pixels` 影响计算方式与显存占用。

### 3.6 `heat` + `run`（慢时累积）

- `run.Npulses`：脉冲数；
- `heat.f_rep`：重频（决定脉冲间隔）；
- `heat.D_gas`, `heat.gamma_heat`：热扩散和折射率耦合。

若只看单脉冲物理，先用 `Npulses=1`。

---

## 4. 推荐的上手路径（10 分钟版本）

1. 安装依赖：
   ```bash
   pip install -r Filament_python/requirements.txt
   ```
2. 先跑轻量配置：
   ```bash
   python -m Filament_python.KHz_filament.cli Filament_python/config_ref.json
   ```
3. 检查输出中的关键曲线是否“形态合理”：`U_z`、`I_max_z`、`rho_onaxis_max_z`、`w_mom_z`。
4. 只改 1~2 个参数重跑（优先 `beam.energy_J`、`propagation.dz`、`ionization.time_mode`）。
5. 再切到 `khz_config.json` 做更高分辨率测试。

---

## 5. 最小质量检查（建议每次改动后执行）

```bash
python -m compileall Filament_python/KHz_filament
pytest -q Filament_python/tests/test_sanity.py
```

如需快速运行自检（非长时大作业），可附加：

```bash
python Filament_python/tests/minimal_run.py
```

---

## 6. 补充文档

- 更详细中文参数手册：`Filament_python/README.md`
- 参数说明草稿：`Filament_python/KHz_filament/Config_explain.md`


---

## 7. 目录结构（2026-04 更新）

```text
Filament_1/
├─ README.md
├─ AGENTS.md
├─ Filament_python/
│  ├─ README.md
│  ├─ KHz_filament/
│  │  ├─ README.md
│  │  ├─ ionization/
│  │  │  └─ README.md
│  │  └─ ...
│  ├─ tools/
│  │  └─ README.md
│  ├─ tests/
│  │  └─ README.md
│  └─ matlab/
│     └─ README.md
└─ references/
   ├─ README.md
   └─ papers/
      ├─ README.md
      ├─ talebpour1999.pdf
      └─ popruzhenko2008.pdf
```

说明：
- 参考文献 PDF 已统一移动到 `references/papers/`。
- 每个主要子目录均提供 `README.md` 用于说明职责与入口文件。
