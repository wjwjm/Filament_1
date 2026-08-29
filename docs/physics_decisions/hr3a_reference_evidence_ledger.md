# HR-3A thermalization reference evidence ledger

**Recorded:** 2026-08-29

**Branch:** `HR-3`

**Start main SHA:** `7bd2aa6b3bcaf27c66336f4e9f98fffd26ad445c`
**Gate status:** **BLOCKED_AT_REFERENCE_GATE**

## Scope and stop condition

HR-3A may define only the authoritative transition

\[
q_{\rm deposition}\;\longrightarrow\;q_{\rm thermal}.
\]

It must not implement post-acoustic/isobaric state construction, temperature,
density, refractive-index screens, interpulse diffusion, or any HR-2E
convergence work.  This ledger records a blocked reference gate.  It does not
freeze a thermalization contract, modify production code or configuration, or
authorize a new propagation calculation.

## Required two-repository audit

The task requires two identified GitHub repositories whose stored literature
jointly supports the HR-3A contract.  The following locations were inspected
before any production implementation:

| Location | Result |
|---|---|
| `git remote -v` | One repository only: `origin`, `https://github.com/wjwjm/Filament_1.git`. |
| `AGENTS.md` | Identifies the local HR boundaries and the two core PDFs, but no second GitHub repository. |
| `docs/`, `Filament_python/`, `results/`, `README.md`, `.gitmodules` | HR-0/HR-1/HR-2 records cite the local PDFs and `origin`; no identifiable second repository or its HR-3 references. |
| Local sibling/workspace search | The only Git repository at the `D:\` workspace level is `D:\Filament_1`; inspected local Codex workspace records did not identify a second HR reference repository. |

Therefore the required second repository is **not identified**.  It would be
incorrect to substitute a public web result, SeaRay, or any other repository.

## Confirmed material in the identified repository

| Repository | Source file | Location in source | Supported statement | Direct HR-3A implication |
|---|---|---|---|---|
| `wjwjm/Filament_1` | `references/Isaacs 等 - 2022 - Modeling the propagation of a high-average-power train of ultrashort laser pulses.pdf` | PDF p. 2, Sec. 2; article p. 22307 | With scattering excepted, the listed loss mechanisms deposit heat; electron energy is transferred collisionally on tens-of-ps scales, recombination/attachment processes heat on ns scales, and rotational/vibrational excitation becomes heat on a molecular-collision time scale. | Supports distinguishing fs deposition from later microscopic thermalization; it does not by itself satisfy the two-repository requirement. |
| `wjwjm/Filament_1` | same Isaacs PDF | PDF p. 4, Sec. 2.2; article p. 22309 | The rotational-Raman fluence loss is identified as energy density injected into rotational excitation. | Supports using only the HR-2 authoritative Raman-deposition channel if a later gate adopts complete eventual thermalization. |
| `wjwjm/Filament_1` | same Isaacs PDF | PDF p. 10, Sec. 5; article p. 22315 | The modeled pulse deposits energy that is assumed to heat air on a sufficiently short time scale; scattering and linear molecular absorption are omitted, and the pulse is too short for inverse-Bremsstrahlung heating, leaving rotational and ionization absorption. | Supports the candidate 40--120 fs benchmark channel set `ion + Raman` with inactive IB, but cannot alone freeze `q_th,c=q_dep,c` as the requested dual-source contract. |
| `wjwjm/Filament_1` | `references/曾庆伟 - 2022 - 飞秒强激光在不同大气环境中传输成丝及其热沉积过程研究.pdf` | Repository file confirmed; not a second repository | This PDF is locally archived in the same identified repository. | It cannot supply the missing second-repository provenance by itself. |

## Consequence for the candidate contract

The following candidate remains **unadopted**:

\[
q_{\rm th,ion}=q_{\rm ion},\qquad
q_{\rm th,Raman}=q_{\rm Raman},\qquad
q_{\rm th,IB}=q_{\rm IB}.
\]

The missing second-repository evidence prevents dual-source confirmation of:

1. complete eventual thermalization for each authoritative channel;
2. the absence of a required branching efficiency or radiative/chemical loss;
3. the exact applicability range of the complete-thermalization approximation;
4. the proposed zero-IB treatment as a frozen HR-3A model contract.

No `eta_ion`, `eta_Raman`, or `eta_IB` parameter has been introduced.  No
legacy `Qacc`, `gamma_heat`, field-loss diagnostic, recombination-derived
source, or Raman operator diagnostic has been reinterpreted as heat.

## Required resolution

Provide or identify the intended second GitHub repository and the HR-3A
reference file(s) within it.  Then this ledger can be completed with source
locations from both repositories before deciding whether to implement the
thermal ledger.

## Preserved upstream status

- HR-2 core deposition interface: **CLOSED**.
- HR-2E longitudinal convergence debt: **DEFERRED**.
- Production longitudinal schedule: **NOT FROZEN**.
- New HPC jobs / Slurm jobs under this branch: **0**.
