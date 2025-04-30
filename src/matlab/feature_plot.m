% Pfad zur .mat-Datei
matfile_path = 'C:\Users\anton\Documents\Studium\Bachelorarbeit\GPR_Daten_mat\radargrams.mat';

% .mat-Datei laden
load(matfile_path, 'radargrams');

% Radargramm-Daten aus der gewünschten Zelle holen
data = radargrams{9};

[nt, nx] = size(data);
dt = 1e-9; % Abtastintervall

% Speicher für Attribute anlegen
amplitude_envelope = zeros(nt, nx);
instantaneous_phase = zeros(nt, nx);
instantaneous_frequency = zeros(nt, nx);

for ix = 1:nx
    trace = data(:, ix);
    analytic_signal = hilbert(trace);
    amplitude_envelope(:, ix) = abs(analytic_signal);
    instantaneous_phase(:, ix) = angle(analytic_signal);
    instantaneous_frequency(:, ix) = [0; diff(unwrap(angle(analytic_signal))) / dt] / (2*pi);
end

% Parameter für Umgebung (env_size = 0 → 1x1, env_size = 1 → 3x3 usw.)
env_size = 7;
window_size = 2 * env_size + 1;

% Pixelbezogener Absolutgradient (Sobel-Filter)
[Gx, Gy] = gradient(double(data));
abs_gradient = sqrt(Gx.^2 + Gy.^2);

% Umgebungsbezogene Features über Moving Window (mit 'same' für zentriertes Fenster)
mean_env    = movmean(movmean(data, window_size, 1, 'Endpoints','shrink'), window_size, 2, 'Endpoints','shrink');
median_env  = medfilt2(data, [window_size, window_size], 'symmetric');
std_env     = stdfilt(data, true(window_size)); % lokale Standardabweichung

% Umgebungs-Entropie (lokale Texturmaßzahl)
entropy_env = entropyfilt(data, true(window_size));

% Umgebungsbezogene Features mit ordfilt2
num_elements = window_size^2;
max_env = ordfilt2(data, num_elements, true(window_size));
min_env = ordfilt2(data, 1, true(window_size));
p75 = ordfilt2(data, round(0.75 * num_elements), true(window_size));
p25 = ordfilt2(data, round(0.25 * num_elements), true(window_size));

% Range und IQR Berechnung
range_env = max_env - min_env;
iqr_env = p75 - p25;

% Schiefe Berechnung nach Formel: a3 = Σ((xᵢ - x̄)/s)³/n
kernel = ones(window_size) / window_size^2;  % Normalisierter Kernel für Moving Average
mean_local = conv2(data, kernel, 'same');    % Lokaler Mittelwert x̄
diff_local = data - mean_local;              % (xᵢ - x̄)
var_local = conv2(diff_local.^2, kernel, 'same'); % Lokale Varianz
std_local = sqrt(var_local);                 % Lokale Standardabweichung s
norm_diff = diff_local ./ std_local;         % (xᵢ - x̄)/s
skewness_env = conv2(norm_diff.^3, kernel, 'same'); % Σ((xᵢ - x̄)/s)³/n

% Kurtosis (ähnliches Prinzip wie Schiefe)
kurtosis_env = conv2(norm_diff.^4, kernel, 'same') - 3; % -3 für Excess Kurtosis

% Neue Features berechnen
% Mean instantaneous frequency über Umgebung
mean_inst_freq = movmean(movmean(instantaneous_frequency, window_size, 1, 'Endpoints','shrink'), window_size, 2, 'Endpoints','shrink');

% Gradient der instantaneous phase
[Gx_phase, Gy_phase] = gradient(instantaneous_phase);
abs_gradient_phase = sqrt(Gx_phase.^2 + Gy_phase.^2);

% Mean der Wölbung über Umgebung
mean_kurtosis = movmean(movmean(kurtosis_env, window_size, 1, 'Endpoints','shrink'), window_size, 2, 'Endpoints','shrink');

% Gradient der Schiefe
[Gx_skew, Gy_skew] = gradient(skewness_env);
abs_gradient_skew = sqrt(Gx_skew.^2 + Gy_skew.^2);


% Gemeinsame Farbschranken für bestimmte Plots
cmin = min(data(:));
cmax = max(data(:));

% Erste Figur mit 4 Features
figure('Name', 'Features 1-4');

subplot(2,2,1);
imagesc(data);
colormap(gray);
title('Radargramm (Amplitude)');
xlabel('Spur'); ylabel('Zeit (Samples)'); colorbar;
caxis([cmin cmax]);

subplot(2,2,2);
imagesc(abs_gradient);
colormap(parula);
title('Absoluter Gradient');
xlabel('Spur'); ylabel('Zeit'); colorbar;

subplot(2,2,3);
imagesc(mean_env);
colormap(jet);
title('Mittelwert (Umgebung)');
xlabel('Spur'); ylabel('Zeit'); colorbar;
caxis([cmin cmax]);

subplot(2,2,4);
imagesc(median_env);
colormap(jet);
title('Median (Umgebung)');
xlabel('Spur'); ylabel('Zeit'); colorbar;
caxis([cmin cmax]);

sgtitle(sprintf('Umgebungsgröße: %d x %d - Teil 1', window_size, window_size));

% Zweite Figur mit 4 Features
figure('Name', 'Features 5-8');

subplot(2,2,1);
imagesc(std_env);
colormap(hot);
title('Standardabweichung');
xlabel('Spur'); ylabel('Zeit'); colorbar;

subplot(2,2,2);
imagesc(entropy_env);
colormap(turbo);
title('Entropie (Umgebung)');
xlabel('Spur'); ylabel('Zeit'); colorbar;

subplot(2,2,3);
imagesc(max_env);
colormap(jet);
title('Maximum (Umgebung)');
xlabel('Spur'); ylabel('Zeit'); colorbar;
caxis([cmin cmax]);

subplot(2,2,4);
imagesc(min_env);
colormap(jet);
title('Minimum (Umgebung)');
xlabel('Spur'); ylabel('Zeit'); colorbar;
caxis([cmin cmax]);

sgtitle(sprintf('Umgebungsgröße: %d x %d - Teil 2', window_size, window_size));

% Dritte Figur mit 4 zusätzlichen Features
figure('Name', 'Features 9-12');

subplot(2,2,1);
imagesc(range_env);
colormap(jet);
title('Range (Max-Min)');
xlabel('Spur'); ylabel('Zeit'); colorbar;

subplot(2,2,2);
imagesc(iqr_env);
colormap(jet);
title('Interquartilsabstand');
xlabel('Spur'); ylabel('Zeit'); colorbar;

subplot(2,2,3);
imagesc(skewness_env);
colormap(jet);
title('Schiefe');
xlabel('Spur'); ylabel('Zeit'); colorbar;

subplot(2,2,4);
imagesc(kurtosis_env);
colormap(jet);
title('Wölbung');
xlabel('Spur'); ylabel('Zeit'); colorbar;

sgtitle(sprintf('Umgebungsgröße: %d x %d - Teil 3', window_size, window_size));

% Vierte Figur mit instantaneous phase und frequency
figure('Name', 'Features 13-16');

subplot(2,2,1);
imagesc(instantaneous_phase);
colormap(hsv);
title('Instantaneous Phase');
xlabel('Spur'); ylabel('Zeit'); colorbar;

subplot(2,2,2);
imagesc(instantaneous_frequency);
colormap(jet);
title('Instantaneous Frequency');
xlabel('Spur'); ylabel('Zeit'); colorbar;

subplot(2,2,3);
imagesc(mean_inst_freq);
colormap(jet);
title('Mean Instantaneous Frequency');
xlabel('Spur'); ylabel('Zeit'); colorbar;

subplot(2,2,4);
imagesc(amplitude_envelope);
colormap(jet);
title('Amplitude Envelope');
xlabel('Spur'); ylabel('Zeit'); colorbar;

sgtitle(sprintf('Umgebungsgröße: %d x %d - Teil 4', window_size, window_size));

% Fünfte Figur mit neuen Features
figure('Name', 'Features 17-20');

subplot(2,2,1);
imagesc(abs_gradient_phase);
colormap(jet);
title('Abs Gradient Phase');
xlabel('Spur'); ylabel('Zeit'); colorbar;

subplot(2,2,2);
imagesc(mean_kurtosis);
colormap(jet);
title('Mean Wölbung');
xlabel('Spur'); ylabel('Zeit'); colorbar;

subplot(2,2,3);
imagesc(abs_gradient_skew);
colormap(jet);
title('Abs Gradient Schiefe');
xlabel('Spur'); ylabel('Zeit'); colorbar;

sgtitle(sprintf('Umgebungsgröße: %d x %d - Teil 5', window_size, window_size));
