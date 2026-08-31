# HR-3A thermalization reference evidence ledger

**Recorded:** 2026-08-29

**Source branch:** `HR-3` at `7d74370ab56049591b69f01ceede3e14a0e0ecec`,
merged to `main` by `654fb0236b9c119ab7d89524c08cf0b84fe9181e`

**Start main SHA:** `7bd2aa6b3bcaf27c66336f4e9f98fffd26ad445c`

**Reference gate:** **PASS**

## Scope

This ledger records the two core references stored in the single GitHub
repository `wjwjm/Filament_1`. The task wording means two reference works in
that repository, not two separate repositories.

HR-3A defines only the authoritative transition

\[
q_{\rm deposition}\;\longrightarrow\;q_{\rm thermal}.
\]

It stops at complete microscopic thermalization. It does not imply
instantaneous fs translational heating, and it does not implement post-acoustic
or isobaric state construction, temperature, density, refractive-index
screens, interpulse diffusion, or HR-2E convergence work.

## Reference evidence

| Source file | Page / section / equation | Evidence | HR-3A consequence |
|---|---|---|---|
| `references/Isaacs 等 - 2022 - Modeling the propagation of a high-average-power train of ultrashort laser pulses.pdf` | PDF p. 2, Sec. 2; article p. 22307 | Apart from scattering, the listed loss mechanisms deposit heat. Electron energy transfer is collisional on tens-of-ps scales; recombination and attachment heat on a ns scale; rotational and vibrational excitation becomes heat on a molecular-collision time scale. | fs optical deposition and subsequent microscopic thermalization are distinct stages. |
| same Isaacs PDF | PDF p. 4, Sec. 2.2; article p. 22309 | Rotational-Raman fluence loss is identified as energy density injected into rotational excitation. | The only Raman input is HR-2's authoritative positive rotational medium-energy gain, not signed field loss. |
| same Isaacs PDF | PDF p. 6--7, Sec. 3; article pp. 22311--22312, Eqs. (21)--(23) | Heating first raises temperature without a density perturbation; acoustic and then isobaric evolution follow on longer time scales. | HR-3A ends before acoustic/isobaric conversion; those are HR-3B work. |
| same Isaacs PDF | PDF p. 10, Sec. 5; article p. 22315 | The model assumes all pulse energy loss heats air on a sufficiently short time scale. Scattering and linear molecular absorption are omitted, and the pulse is too short for inverse-Bremsstrahlung heating, leaving rotational and ionization absorption. | Supports complete microscopic thermalization for the current 40--120 fs Isaacs-compatible channel set, with IB inactive. |
| `references/曾庆伟 - 2022 - 飞秒强激光在不同大气环境中传输成丝及其热沉积过程研究.pdf` | PDF p. 26, Sec. 1.2.2.1 | Optical energy initially enters free-electron kinetic/potential and molecular rotational degrees of freedom, then evolves through microscopic processes toward the medium's thermodynamic state. | Supports a separate microscopic-thermalization layer rather than identifying optical loss with immediate macroscopic state change. |
| same Zeng PDF | PDF p. 27, Sec. 1.2.2.2 | Molecular thermalization is stated to occur on a ns scale, before the longer gas-dynamical response. | Confirms the HR-3A / HR-3B temporal boundary. |
| same Zeng PDF | PDF p. 43, Sec. 2.3.2, Eq. (2-15) | The temperature estimate explicitly assumes that all deposited energy converts to thermal energy. | Supports unit-efficiency complete thermalization with no free channel efficiency parameter. |
| same Zeng PDF | PDF p. 44, Sec. 2.3.2, Eqs. (2-19)--(2-22) | The deposition-to-temperature model uses unit-length deposited energy; its nonlinear loss density includes inverse Bremsstrahlung and multiphoton ionization. | Retain the IB thermal-source channel generically, while preserving exact zero when HR-2 reports it inactive. |

## Frozen HR-3A approximation

Both references support a complete eventual-heating **model approximation** for
the deposited energy supplied to the thermal model and introduce no
mechanism-specific thermalization efficiency. They do not establish a measured
channel-specific branching law. HR-3A therefore freezes the following
reference-compatible model contract:

\[
q_{\rm th,ion}=q_{\rm ion},\qquad
q_{\rm th,Raman}=q_{\rm Raman},\qquad
q_{\rm th,IB}=q_{\rm IB},
\]

with

\[
q_{\rm thermal}=q_{\rm th,ion}+q_{\rm th,Raman}+q_{\rm th,IB}.
\]

This is a data-contract identity under the complete microscopic-thermalization
approximation. It is not a claim that translational temperature changes during
the fs propagation step or that each microscopic pathway has independently
measured unit efficiency. No `eta_ion`, `eta_Raman`, or `eta_IB` parameter is
introduced.

## Channel and anti-double-counting rules

- Inputs are only the HR-2 authoritative `q_ion`, `q_IB`, and `q_Raman`
  interval maps, their schedule, geometry, units, and authority metadata.
- Ionization is not recomputed from net electron-density change, recombination,
  or attachment.
- Raman is not reconstructed from signed actual field loss, `Q_rot_vol`,
  `w_R`, `E_dep_rot_z`, `Qacc_raman`, or any legacy diagnostic.
- IB remains an explicit channel. For the current benchmark it is inactive and
  must remain exactly zero.
- Field-energy loss is closure evidence only; it is never a thermal source.
- Scattering and linear molecular absorption are not added because HR-2 has
  not authenticated either channel as deposition.

## Reference differences and resolution

Isaacs explicitly excludes scattering and linear molecular absorption in the
benchmark. Zeng studies broader atmospheric cases, including particle
scattering, but its thermal-response equations still assume complete conversion
of the deposited energy supplied to them. This is not a conflict for HR-3A:
only HR-2's existing authoritative channels are consumed, and no new channel is
created from field loss or scattering.

## Preserved upstream status

- HR-2 core deposition interface: **CLOSED**.
- HR-2E longitudinal convergence debt: **DEFERRED**.
- Production longitudinal schedule: **NOT FROZEN**.
- New HPC jobs / Slurm jobs under this branch: **0**.
