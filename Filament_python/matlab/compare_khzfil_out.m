function out = compare_khzfil_out(matFiles, fieldName, labels, zShiftCm, opts)
%COMPARE_KHZFIL_OUT 对比多个 khzfil_out .mat 文件中的同一诊断量。
% 用法：
%   out = compare_khzfil_out({'a.mat','b.mat'}, 'I_max_z');
%   out = compare_khzfil_out(["a.mat","b.mat"], 'w_mom_z', {'run A','run B'});
%   out = compare_khzfil_out(files, 'rho_onaxis_max_z', [], -20, struct('normalize', true));
%
% 输入：
%   matFiles   : 多个 .mat 文件（cellstr / string array / char）
%   fieldName  : 要对比的字段名（字符串）
%   labels     : 图例标签（可选；默认使用文件名）
%   zShiftCm   : z 轴平移（cm，可选；标量或与文件数等长向量；默认 0）
%   opts       : 选项结构体（可选）
%       .yscale          = 'linear' | 'log'（默认按字段自动选择）
%       .normalize       = true | false（默认 false）
%       .title           = 自定义标题
%       .showPeakMarkers = true | false（默认 true）
%
% 输出：
%   out: 结构体，包含绘图配置与每个数据集峰值/峰值位置等摘要信息。

if nargin < 1 || isempty(matFiles)
    error('matFiles 不能为空，至少需要 2 个 .mat 文件。');
end
if nargin < 2 || strlength(string(fieldName)) == 0
    error('fieldName 不能为空。');
end
if nargin < 3
    labels = [];
end
if nargin < 4 || isempty(zShiftCm)
    zShiftCm = 0;
end
if nargin < 5 || isempty(opts)
    opts = struct();
end

files = normalize_file_list(matFiles);
N = numel(files);
if N < 2
    error('compare_khzfil_out 需要至少 2 个文件进行对比。');
end

fieldName = char(string(fieldName));
labels = normalize_labels(labels, files);
zShiftCm = normalize_shift(zShiftCm, N);
opts = normalize_opts(opts, fieldName);

figure('Name', sprintf('khzfil compare: %s', fieldName), 'Color', 'w');
hold on; grid on; box on;
set(gca, 'YScale', opts.yscale);

out = struct();
out.fieldName = fieldName;
out.opts = opts;
out.datasets = repmat(struct( ...
    'file', '', ...
    'label', '', ...
    'z_shift_cm', 0, ...
    'unit_scale', 1, ...
    'unit_name', '', ...
    'z_peak_m', NaN, ...
    'z_peak_cm_plot', NaN, ...
    'peak_value_raw', NaN, ...
    'peak_value_plot', NaN, ...
    'max_value_raw', NaN, ...
    'max_value_plot', NaN), N, 1);

for i = 1:N
    Si = load(files{i});
    if ~isfield(Si, 'z_axis')
        error('文件 "%s" 缺少 z_axis。', files{i});
    end
    if ~isfield(Si, fieldName)
        error('文件 "%s" 缺少字段 "%s"。', files{i}, fieldName);
    end

    z = colvec(Si.z_axis);
    y_raw = colvec(Si.(fieldName));

    if numel(z) ~= numel(y_raw)
        error('文件 "%s" 中 z_axis 与 %s 长度不一致。', files{i}, fieldName);
    end

    [y_plot, y_unit_name, unit_scale] = convert_field_units(fieldName, y_raw);

    if strcmp(opts.yscale, 'log')
        y_plot(y_plot <= 0) = NaN;
        y_raw(y_raw <= 0) = NaN;
    end

    if opts.normalize
        denom = max(y_plot, [], 'omitnan');
        if isempty(denom) || isnan(denom) || denom == 0
            error('文件 "%s" 的字段 "%s" 无法归一化（最大值无效）。', files{i}, fieldName);
        end
        y_draw = y_plot ./ denom;
    else
        y_draw = y_plot;
    end

    z_plot_cm = z * 100 + zShiftCm(i);
    plot(z_plot_cm, y_draw, 'LineWidth', 1.6, 'DisplayName', labels{i});

    [peak_val_raw, idx_peak] = max(y_raw, [], 'omitnan');
    if isempty(idx_peak) || isnan(peak_val_raw)
        z_peak_m = NaN;
        z_peak_cm_plot = NaN;
        peak_plot = NaN;
    else
        z_peak_m = z(idx_peak);
        z_peak_cm_plot = z_peak_m * 100 + zShiftCm(i);
        if opts.normalize
            peak_plot = max(y_draw, [], 'omitnan');
        else
            peak_plot = y_plot(idx_peak);
        end
        if opts.showPeakMarkers && ~isnan(peak_plot)
            plot(z_peak_cm_plot, peak_plot, 'o', 'HandleVisibility', 'off');
        end
    end

    out.datasets(i).file = files{i};
    out.datasets(i).label = labels{i};
    out.datasets(i).z_shift_cm = zShiftCm(i);
    out.datasets(i).unit_scale = unit_scale;
    out.datasets(i).unit_name = y_unit_name;
    out.datasets(i).z_peak_m = z_peak_m;
    out.datasets(i).z_peak_cm_plot = z_peak_cm_plot;
    out.datasets(i).peak_value_raw = peak_val_raw;
    out.datasets(i).peak_value_plot = peak_plot;
    out.datasets(i).max_value_raw = max(y_raw, [], 'omitnan');
    out.datasets(i).max_value_plot = max(y_plot, [], 'omitnan');
end

xlabel('z (cm)');
ylabel(compose_ylabel(fieldName, opts.normalize, out.datasets(1).unit_name));
if strlength(string(opts.title)) > 0
    title(opts.title);
else
    title(compose_title(fieldName, opts.normalize));
end
legend('Location', 'best');

end

function files = normalize_file_list(matFiles)
if ischar(matFiles)
    files = {matFiles};
elseif isstring(matFiles)
    files = cellstr(matFiles(:));
elseif iscell(matFiles)
    files = cellfun(@char, matFiles(:), 'UniformOutput', false);
else
    error('matFiles 必须是 char/string/cellstr。');
end

files = files(:);
for k = 1:numel(files)
    if strlength(string(files{k})) == 0
        error('matFiles 中包含空文件名。');
    end
end
end

function labelsOut = normalize_labels(labels, files)
N = numel(files);
if nargin < 1 || isempty(labels)
    labelsOut = cell(N,1);
    for k = 1:N
        [~, name, ext] = fileparts(files{k});
        labelsOut{k} = [name, ext];
    end
    return;
end

if ischar(labels)
    labelsOut = {labels};
elseif isstring(labels)
    labelsOut = cellstr(labels(:));
elseif iscell(labels)
    labelsOut = cellfun(@char, labels(:), 'UniformOutput', false);
else
    error('labels 必须是 char/string/cellstr。');
end

if numel(labelsOut) ~= N
    error('labels 数量必须与 matFiles 一致。');
end
end

function zShift = normalize_shift(zShiftCm, N)
validateattributes(zShiftCm, {'numeric'}, {'vector','finite'}, mfilename, 'zShiftCm');
if isscalar(zShiftCm)
    zShift = repmat(double(zShiftCm), N, 1);
else
    if numel(zShiftCm) ~= N
        error('zShiftCm 必须是标量或与 matFiles 等长。');
    end
    zShift = double(zShiftCm(:));
end
end

function opts = normalize_opts(opts, fieldName)
if ~isstruct(opts)
    error('opts 必须是结构体。');
end

if ~isfield(opts, 'normalize') || isempty(opts.normalize)
    opts.normalize = false;
end
if ~isfield(opts, 'showPeakMarkers') || isempty(opts.showPeakMarkers)
    opts.showPeakMarkers = true;
end
if ~isfield(opts, 'title')
    opts.title = '';
end

if ~isfield(opts, 'yscale') || isempty(opts.yscale)
    opts.yscale = default_yscale(fieldName);
else
    opts.yscale = lower(char(string(opts.yscale)));
end

if ~ismember(opts.yscale, {'linear', 'log'})
    error('opts.yscale 必须是 ''linear'' 或 ''log''。');
end
opts.normalize = logical(opts.normalize);
opts.showPeakMarkers = logical(opts.showPeakMarkers);
end

function yscale = default_yscale(fieldName)
logFields = {'I_max_z','I_onaxis_max_z','I_center_t0_z', ...
             'rho_onaxis_max_z','rho_max_z','rho_peak_q99_z'};
if any(strcmp(fieldName, logFields))
    yscale = 'log';
else
    yscale = 'linear';
end
end

function [y, unitName, unitScale] = convert_field_units(fieldName, yRaw)
y = yRaw;
unitScale = 1;

if strcmp(fieldName, 'w_mom_z')
    y = yRaw * 1e3;
    unitName = 'mm';
    unitScale = 1e3;
elseif startsWith(fieldName, 'fwhm_')
    y = yRaw * 1e6;
    unitName = '\mum';
    unitScale = 1e6;
elseif contains(fieldName, 'I_')
    unitName = 'W/m^2';
elseif contains(fieldName, 'rho')
    unitName = 'm^{-3}';
elseif strcmp(fieldName, 'U_z')
    unitName = 'J';
else
    unitName = 'a.u.';
end
end

function s = compose_title(fieldName, normalize)
if normalize
    s = sprintf('%s (normalized comparison)', fieldName);
else
    s = sprintf('%s (multi-file comparison)', fieldName);
end
end

function ylab = compose_ylabel(fieldName, normalize, unitName)
if normalize
    ylab = sprintf('%s / max(%s)', fieldName, fieldName);
else
    ylab = sprintf('%s (%s)', fieldName, unitName);
end
end

function x = colvec(x)
x = x(:);
end
