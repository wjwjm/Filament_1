# S4-6I实施边界

用户于2026-11-02批准本轮完整计划和既有多次CPU提交授权。研究基线54d8c3f2c101a5268a197f475e68c75e04333562，科学子树821fec5bdacd832b0347e1f2e84285cb54172e57，分支codex/cpu-residual-pointwise-20261102。

所有角色都使用INTENSITY_P24、EQ10_P48、PHASE_ABSORB_P24及L1。第一阶段只有OFF/COST计时差异，尚不实现新数学候选。成本测量有效后才按冻结排名选择一个单位；随后原尺寸真实输入局部REF/P1/P24对照，合格后与联合基线进行原尺寸14窗比较。不会追加Eq10单项诊断。

新增私有cpu_s4_6i_residual_parallel及默认关闭的6E/H compile/attach接口。原科学文件、公开配置、六元组接口、FFT/IIR/归约/闭合/Qacc语义不改。所有新增AST和helper受逆变换、源码/闭包/实际后端绑定，禁止全局monkeypatch。历史执行SHA和证据不改写。

全尺寸[384,351,301]、complex128、冻结入口/PRE及空1.5GiB线性缓存。W1微网格仅工程检查。当前代码/本地测试/调度状态和原尺寸exact分别登记；DEFAULT=I0、PRODUCTION_QUALIFIED=false。
