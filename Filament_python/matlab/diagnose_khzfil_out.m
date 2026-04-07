function summary = diagnose_khzfil_out(matFile, cfgFile, figSelect, zShiftCm)
%DIAGNOSE_KHZFIL_OUT 诊断 khzfil_out.mat 中光丝关键量随 z 的变化。
% 用法：
%   summary = diagnose_khzfil_out('khzfil_out.mat');
%   summary = diagnose_khzfil_out('matlab_output/khzfil_out.mat');
%   summary = diagnose_khzfil_out('khzfil_out.mat', 'khz_config.json');
%   summary = diagnose_khzfil_out('khzfil_out.mat', 'khz_config.json', {'plasma'});
%   summary = diagnose_khzfil_out('khzfil_out.mat', '', {'plasma'}, -20);
%
% 输出：
%   summary: 结构体，包含焦点位置、峰值强度、峰值等离子体密度、能量漂移等。
%
% 图像选择（figSelect，可选）：
%   - 空/未提供：输出全部图像（默认行为）。
%   - 字符串或 cellstr：仅输出指定图像，如 'plasma' 或 {'plasma','energy'}。
%   - 支持别名：'all' | 'intensity' | 'plasma' | 'beam' | 'energy' | 'fwhm' | 'rho_tz'
% 坐标平移（zShiftCm，可选）：
%   - 直接指定绘图坐标平移量（单位 cm）。
%   - 例如 zShiftCm=-20：整体向左平移 20 cm；zShiftCm=+20：整体向右平移 20 cm。
%
% 图像：
%   Figure 1: I_max_z / I_onaxis_max_z / I_center_t0_z
%   Figure 2: rho_onaxis_max_z / rho_max_z
%   Figure 3: w_mom_z 及其最小值（焦点估计）
%   Figure 4: U_z（能量守恒监测）
%   Figure 5: fwhm_plasma_z / fwhm_fluence_z
%   Figure 6: rho_onaxis_t_z (z-t 热图，若数据存在)

if nargin < 1 || strlength(string(matFile)) == 0
    matFile = 'khzfil_out.mat';
end
if nargin < 2
    cfgFile = '';
end
if nargin < 3
    figSelect = [];
end
if nargin < 4 || isempty(zShiftCm)
    zShiftCm = 0;
end
validateattributes(zShiftCm, {'numeric'}, {'scalar','finite'}, mfilename, 'zShiftCm', 4);

S = load(matFile);
assert(isfield(S, 'z_axis'), '缺少 z_axis，无法做 z 向诊断。');
z = colvec(S.z_axis);
z_m = z;
[z_plot_m, z_plot_cm, z_label, z_shift_meta] = build_plot_axis(z_m, zShiftCm);
fig_flags = parse_figure_selection(figSelect);

summary = struct();
summary.file = matFile;
summary.Nz = numel(z);
summary.z_plot_label = z_label;
summary.z_plot_shift_applied = z_shift_meta.applied;
summary.z_manual_shift_cm = z_shift_meta.shift_cm;

% 关键 summary 指标始终计算，不依赖图像选择
if isfield(S, 'I_max_z')
    Imax = sanitize_positive(colvec(S.I_max_z));
    [summary.I_max_peak, iIp] = max(Imax, [], 'omitnan');
    if ~isempty(iIp) && ~isnan(summary.I_max_peak)
        summary.z_Imax_peak_m = z_m(iIp);
    end
end
if isfield(S, 'rho_onaxis_max_z')
    rhoOn = sanitize_positive(colvec(S.rho_onaxis_max_z));
    [summary.rho_onaxis_peak, irp] = max(rhoOn, [], 'omitnan');
    if ~isempty(irp) && ~isnan(summary.rho_onaxis_peak)
        summary.z_rho_onaxis_peak_m = z_m(irp);
    end
end
if isfield(S, 'w_mom_z')
    w = colvec(S.w_mom_z);
    [wmin, iw] = min(w, [], 'omitnan');
    summary.w_mom_min_m = wmin;
    summary.z_focus_est_m = z_m(iw);
end
if isfield(S, 'U_z')
    U = colvec(S.U_z);
    U0 = U(find(~isnan(U), 1, 'first'));
    if ~isempty(U0)
        summary.U0_J = U0;
        summary.U_end_J = U(find(~isnan(U), 1, 'last'));
        summary.U_drift_pct = summary.U_end_J / U0 * 100 - 100;
    end
end

% --------- Figure 1: 强度诊断 ---------
if fig_flags.intensity
    figure('Name', 'Filament diagnostics: intensity vs z', 'Color', 'w');
    tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

    nexttile;
    hold on; grid on; box on;
    if isfield(S, 'I_max_z')
        Imax = sanitize_positive(colvec(S.I_max_z));
        semilogy(z_plot_cm, Imax, 'LineWidth', 1.8, 'DisplayName', 'I\_max\_z');
        if isfield(summary, 'z_Imax_peak_m')
            xline(to_plot_cm(summary.z_Imax_peak_m, z_shift_meta), '--', 'I_{max} peak', 'LabelVerticalAlignment','middle');
        end
    end
    if isfield(S, 'I_onaxis_max_z')
        semilogy(z_plot_cm, sanitize_positive(colvec(S.I_onaxis_max_z)), 'LineWidth', 1.6, 'DisplayName', 'I\_onaxis\_max\_z');
    end
    if isfield(S, 'I_center_t0_z')
        semilogy(z_plot_cm, sanitize_positive(colvec(S.I_center_t0_z)), 'LineWidth', 1.2, 'DisplayName', 'I\_center\_t0\_z');
    end
    xlabel(z_label); ylabel('Intensity (W/m^2)');
    
    title('强度相关诊断（对数坐标）'); legend('Location','best');

    nexttile;
    hold on; grid on; box on;
    if isfield(S, 'I_max_z')
        Imaxn = colvec(S.I_max_z) / max(colvec(S.I_max_z), [], 'omitnan');
        plot(z_plot_cm, Imaxn, 'LineWidth', 1.8, 'DisplayName', 'I\_max\_z / max');
    end
    if isfield(S, 'I_peak_q99_z')
        q99n = colvec(S.I_peak_q99_z) / max(colvec(S.I_peak_q99_z), [], 'omitnan');
        plot(z_plot_cm, q99n, 'LineWidth', 1.4, 'DisplayName', 'I\_peak\_q99\_z / max');
    end
    xlabel(z_label); ylabel('Normalized');
    title('峰值与稳峰（归一化）'); legend('Location','best');
end

% --------- Figure 2: 等离子体密度 ---------
if fig_flags.plasma && (isfield(S, 'rho_onaxis_max_z') || isfield(S, 'rho_max_z'))
    figure('Name', 'Filament diagnostics: plasma density vs z', 'Color', 'w');
    hold on; grid on; box on;
    if isfield(S, 'rho_onaxis_max_z')
        semilogy(z_plot_cm, sanitize_positive(colvec(S.rho_onaxis_max_z)), 'LineWidth', 1.8, 'DisplayName', '\rho\_onaxis\_max\_z');
    end
    if isfield(S, 'rho_max_z')
        semilogy(z_plot_cm, sanitize_positive(colvec(S.rho_max_z)), 'LineWidth', 1.4, 'DisplayName', '\rho\_max\_z');
    end
    if isfield(S, 'rho_peak_q99_z')
        semilogy(z_plot_cm, sanitize_positive(colvec(S.rho_peak_q99_z)), '--', 'LineWidth', 1.2, 'DisplayName', '\rho\_peak\_q99\_z');
    end
    % yline(1e25, ':k', '1e25 m^{-3} (air neutral density scale)', 'LabelVerticalAlignment','bottom');
    xlabel(z_label); ylabel('Electron density (m^{-3})');
    xlim([-20,20]);
    title('等离子体密度诊断（对数坐标）');
    legend('Location','best');
end

% --------- Figure 3: 光斑二阶矩半径 ---------
if fig_flags.beam && isfield(S, 'w_mom_z')
    w = colvec(S.w_mom_z);
    [~, iw] = min(w, [], 'omitnan');
    wmin = summary.w_mom_min_m;

    figure('Name', 'Filament diagnostics: beam radius vs z', 'Color', 'w');
    plot(z_plot_cm, w*1e3, 'LineWidth', 1.8); grid on; box on;
    hold on;
    plot(z_plot_cm(iw), wmin*1e3, 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'w_{mom} min');
    xline(z_plot_cm(iw), '--r', sprintf('focus z = %.2f cm', z_plot_cm(iw)));
    xlabel(z_label); ylabel('w_{mom} (mm)');
    title('二阶矩光斑半径（焦点估计）');
end

% --------- Figure 4: 能量守恒 ---------
if fig_flags.energy && isfield(S, 'U_z')
    if ~isfield(summary, 'U0_J')
        warning('U_z 全为 NaN，跳过能量图绘制。');
    else
        U = colvec(S.U_z);
        dU = (U - summary.U0_J) / summary.U0_J * 100;

        figure('Name', 'Filament diagnostics: pulse energy vs z', 'Color', 'w');
        yyaxis left
        plot(z_plot_cm, U, 'LineWidth', 1.8);
        ylabel('U(z) (J)');
        yyaxis right
        plot(z_plot_cm, dU, '--', 'LineWidth', 1.3);
        ylabel('\DeltaU/U_0 (%)');
        grid on; box on;
        xlabel(z_label);
        title('脉冲能量与相对漂移');
    end
end

% --------- Figure 5: FWHM 诊断 ---------
if fig_flags.fwhm && (isfield(S, 'fwhm_plasma_z') || isfield(S, 'fwhm_fluence_z'))
    figure('Name', 'Filament diagnostics: FWHM vs z', 'Color', 'w');
    hold on; grid on; box on;
    if isfield(S, 'fwhm_plasma_z')
        plot(z_plot_cm, colvec(S.fwhm_plasma_z)*1e6, 'LineWidth', 1.8, 'DisplayName', 'FWHM plasma');
    end
    if isfield(S, 'fwhm_fluence_z')
        plot(z_plot_cm, colvec(S.fwhm_fluence_z)*1e6, 'LineWidth', 1.6, 'DisplayName', 'FWHM fluence');
    end
    xlabel(z_label); ylabel('FWHM diameter (\mum)');
    title('通道横向尺度（FWHM）'); legend('Location','best');
end

% --------- Figure 6: on-axis rho(t,z) ---------
if fig_flags.rho_tz && isfield(S, 'rho_onaxis_t_z')
    figure('Name', 'Filament diagnostics: rho on-axis (z-t map)', 'Color', 'w');
    rhozt = S.rho_onaxis_t_z;
    if isfield(S, 't_axis')
        t = colvec(S.t_axis) * 1e15;
        imagesc(z_plot_cm, t, log10(max(rhozt.', 1))); axis xy;
        ylabel('t (fs)');
    else
        imagesc(z_plot_cm, 1:size(rhozt,2), log10(max(rhozt.', 1))); axis xy;
        ylabel('time index');
    end
    xlabel(z_label);
    title('log_{10}(\rho_{on-axis}(t,z))');
    cb = colorbar; cb.Label.String = 'log_{10}(m^{-3})';
    colormap(turbo);
end

% --------- 文本报告 ---------
fprintf('\n========== Filament quick summary ==========' );
fprintf('\nFile: %s\n', matFile);
if isfield(summary, 'z_focus_est_m')
    fprintf('Focus estimate from w_mom min: z = %.4g m (%.3f cm), w_min = %.4g m\n', ...
        summary.z_focus_est_m, summary.z_focus_est_m*100, summary.w_mom_min_m);
end
if z_shift_meta.applied
    fprintf('Plot z-axis shifted by %+g cm (manual)\n', z_shift_meta.shift_cm);
end
if isfield(summary, 'I_max_peak')
    fprintf('I_max peak: %.4e W/m^2 @ z = %.4g m\n', summary.I_max_peak, summary.z_Imax_peak_m);
end

if isfield(summary, 'rho_onaxis_peak')
    fprintf('rho_onaxis peak: %.4e m^-3 @ z = %.4g m\n', summary.rho_onaxis_peak, summary.z_rho_onaxis_peak_m);
end
if isfield(summary, 'U_drift_pct')
    fprintf('Energy drift (end vs start): %.3f %%\n', summary.U_drift_pct);
end

warns = sanity_checks(S, z_m);
if ~isempty(warns)
    fprintf('\n[Sanity warnings]\n');
    for k = 1:numel(warns)
        fprintf(' - %s\n', warns{k});
    end
end
fprintf('============================================\n\n');

end

function flags = parse_figure_selection(figSelect)
flags = struct('intensity', true, 'plasma', true, 'beam', true, ...
    'energy', true, 'fwhm', true, 'rho_tz', true);

if nargin < 1 || isempty(figSelect)
    return;
end

if ischar(figSelect) || isstring(figSelect)
    keys = string(figSelect);
elseif iscell(figSelect)
    keys = string(figSelect);
else
    error('figSelect 必须是字符串、string 数组或 cellstr。');
end

keys = lower(strtrim(keys(:)));
keys(keys == "") = [];
if isempty(keys)
    return;
end

if any(keys == "all")
    return;
end

flags.intensity = false;
flags.plasma = false;
flags.beam = false;
flags.energy = false;
flags.fwhm = false;
flags.rho_tz = false;

for i = 1:numel(keys)
    k = keys(i);
    switch k
        case {"intensity", "i", "figure1", "fig1"}
            flags.intensity = true;
        case {"plasma", "rho", "density", "figure2", "fig2"}
            flags.plasma = true;
        case {"beam", "w_mom", "radius", "figure3", "fig3"}
            flags.beam = true;
        case {"energy", "u", "figure4", "fig4"}
            flags.energy = true;
        case {"fwhm", "width", "figure5", "fig5"}
            flags.fwhm = true;
        case {"rho_tz", "rho-onaxis-t", "figure6", "fig6"}
            flags.rho_tz = true;
        otherwise
            warning('Unknown figSelect key ignored: %s', k);
    end
end
end

function x = colvec(x)
    x = x(:);
end

function y = sanitize_positive(y)
    y = y(:);
    y(y <= 0) = NaN;
end

function [z_plot_m, z_plot_cm, z_label, meta] = build_plot_axis(z_m, z_shift_cm)
z_plot_m = z_m + z_shift_cm / 100;
z_plot_cm = z_m * 100 + z_shift_cm;
z_label = 'z (cm)';
meta = struct('applied', abs(z_shift_cm) > 0, 'shift_cm', z_shift_cm);
if meta.applied
    z_label = sprintf('z (cm), shifted %+g cm', z_shift_cm);
end
end

function x_cm = to_plot_cm(x_m, meta)
x_cm = x_m * 100;
if isstruct(meta) && isfield(meta, 'shift_cm')
    x_cm = x_cm + meta.shift_cm;
end
end

function warns = sanity_checks(S, z)
% 基于项目 README 的“合理性包络”给出轻量报警。
warns = {};

if isfield(S, 'U_z')
    U = colvec(S.U_z);
    U = U(~isnan(U));
    if numel(U) >= 2
        rel = (U(end) - U(1)) / U(1);
        if rel > 0.10
            warns{end+1} = sprintf('U_z 相对起点增长 %.1f%% (>10%%)，建议检查步长/增益项设置。', rel*100); %#ok<AGROW>
        end
    end
end

if isfield(S, 'I_max_z')
    I = colvec(S.I_max_z);
    I = I(~isnan(I) & I>0);
    if numel(I) >= 3
        jump = I(2:end) ./ I(1:end-1);
        if any(jump > 10)
            warns{end+1} = 'I_max_z 出现相邻步 >10x 跳变，建议排查 dz、裁剪和边界设置。'; %#ok<AGROW>
        end
    end
end

if isfield(S, 'rho_onaxis_max_z')
    rho = colvec(S.rho_onaxis_max_z);
    rho_valid = rho(~isnan(rho));
    if ~isempty(rho_valid) && any(rho_valid > 1e25)
        warns{end+1} = 'rho_onaxis_max_z 超过 ~1e25 m^-3，可能偏离空气中性粒子密度量级。'; %#ok<AGROW>
    end
end

if isfield(S, 'w_mom_z')
    w = colvec(S.w_mom_z);
    if any(diff(sign(diff(w(~isnan(w))))) ~= 0)
        % 非严格报警：有振荡趋势时提示用户关注
        osc = sum(abs(diff(sign(diff(w(~isnan(w)))))) > 0);
        if osc > max(5, 0.1*numel(w))
            warns{end+1} = 'w_mom_z 变化振荡偏多，建议减小 dz 或检查边界反射。'; %#ok<AGROW>
        end
    end
end

if isfield(S, 'fwhm_plasma_z')
    fp = colvec(S.fwhm_plasma_z);
    if any(fp <= 0 | isnan(fp))
        warns{end+1} = 'fwhm_plasma_z 存在 <=0 或 NaN，建议检查阈值与诊断计算。'; %#ok<AGROW>
    end
end
if isfield(S, 'fwhm_fluence_z')
    ff = colvec(S.fwhm_fluence_z);
    if any(ff <= 0 | isnan(ff))
        warns{end+1} = 'fwhm_fluence_z 存在 <=0 或 NaN，建议检查阈值与诊断计算。'; %#ok<AGROW>
    end
end

if nargin >= 2 && ~isempty(z)
    if any(diff(z) <= 0)
        warns{end+1} = 'z_axis 非严格递增，可能导致曲线解释错误。'; %#ok<AGROW>
    end
end
end
