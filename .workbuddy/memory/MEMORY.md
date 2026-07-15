# MEMORY.md - 长期记忆

## 超算账号分工
- **scvi806@NC-N50R5**（宁夏超算）：用于申请 **GPU** 任务
- **t0s000727@BSCC-T**（北京超算）：用于申请 **CPU** 任务
- **Filament_1 项目默认使用 GPU 超算**（即 scvi806@NC-N50R5）

## 仿真结果下载规范
- 下载超算仿真结果时，**必须**触发 `filament-download` skill
- 不要手动 SCP 下载，必须走 skill 规范流程（超算端重命名 → SCP 到 OneDrive 目录）
- 下载目标：`C:\Users\wangj\OneDrive\博士\学业\A_超快成丝\仿真保存数据\`（.mat）和 `运行out文件\`（.out）
- 触发词：下载仿真结果、下载超算结果、download filament results 等
