% This script initializes a neural network model, trains it with different hyperparameter settings,
% and records activation changes over multiple runs and epochs.
% It also visualizes the results for each layer in response to different frequencies.

%% Define Hyperparameters
format compact, clear, clc, close all
% rng('default'); rng(0);  % Uncomment for reproducibility

nRuns = 3; % Number of training runs
% Layer names and number of units
namesLayers_all = {'MGv', 'MGm', 'Auditory Cortex', 'Amygdala'};
nLayers = length(namesLayers_all);
nUnits_MGv = 8;  % Number of units in the MGv layer
nUnits_MGm = 3;  % Number of units in the MGm layer
nUnits_AC = 8;   % Number of units in the Auditory Cortex
nUnits_Amy = 3;  % Number of units in the Amygdala

freq_train = 4;  % Frequency used for training
x_thr = 0;  % Activation threshold
x_sat = 1;  % Activation saturation
flag_US = 0; %1=US is on, 0=off

% Hyperparameter configurations (epochs, nFreq, learning rate, lateral inhibition)
hyperP_all = {
    [1e3, 16, 0.1, 0.2], ...  % Standard
    [1e3, 9, 0.1, 0.2], ...   % Reduce nFreq to 9
    [1e3, 31, 0.1, 0.2], ...  % Increase nFreq to 31
    [1e3, 16, 0.1, 0], ...    % No lateral inhibition (mu = 0)
    [1e3, 16, 0.1, 1], ...    % Strongest lateral inhibition (mu = 1)
    [1e3, 16, 0.1, 0.5], ...  % Moderate lateral inhibition (mu = 0.5)
    [1e3, 16, 0.05, 0.2], ... % Reduced learning rate
    [1e3, 16, 0.5, 0.2] ...   % Increased learning rate
    };
nHyperP = length(hyperP_all);

% Define color scheme for plotting
colors = [
    46,  134, 193;  % Sky Blue
    231, 76,  60;   % Soft Red
    39,  174, 96;   % Emerald Green
    241, 196, 15;   % Mustard Yellow
    155, 89,  182;  % Amethyst Purple
    230, 126, 34;   % Tangerine Orange
    26,  188, 156;  % Teal
    192, 57,  43;   % Crimson
    ] / 255;  % Normalize to MATLAB's [0,1] range

%% Loop through each hyperparameter setting
for iHyperP = 1%1:nHyperP
    hyperP = hyperP_all{iHyperP};
    nEpochs = hyperP(1);  % Number of training epochs
    nUnits_CS = hyperP(2); % Number of frequency components (CS)
    nFreqs = nUnits_CS - 1; % Number of unique frequencies
    learning_rate = hyperP(3); % Learning rate
    mu = hyperP(4);  % Lateral inhibition factor

    % Create folder to save results
    nameFolder = sprintf('Fig/[nEpoch%d][nFreq%d][LR%.1f][mu%.1f]', nEpochs, nFreqs, learning_rate, mu);
    if isempty(dir(nameFolder)), mkdir(nameFolder), end

    fprintf('\n===================== Hyperparameter Set %d Initialized =====================\n\n', iHyperP)

    %% Initialize Activation History Storage
    % These arrays store activation values for each layer across runs, epochs, and frequencies
    a_MGv_allEpoch = nan(nRuns, nEpochs, nFreqs, nUnits_MGv);
    a_MGm_allEpoch = nan(nRuns, nEpochs, nFreqs, nUnits_MGm);
    a_AC_allEpoch = nan(nRuns, nEpochs, nFreqs, nUnits_AC);
    a_Amy_allEpoch = nan(nRuns, nEpochs, nFreqs, nUnits_Amy);

    % Initialize arrays for forward pass activations (before training)
    a_MGv_allFreq_pre = nan(nRuns, nFreqs, nUnits_MGv);
    a_MGm_allFreq_pre = nan(nRuns, nFreqs, nUnits_MGm);
    a_AC_allFreq_pre = nan(nRuns, nFreqs, nUnits_AC);
    a_Amy_allFreq_pre = nan(nRuns, nFreqs, nUnits_Amy);

    % Unconditioned stimulus (US) inputs (disabled during development phase)
    US_inputs = 0;
    w_US_MGm = 0;
    w_US_Amy = 0;
    str_cond = 'develop';

    d = 10; % Scaling factor for initial weight values
    clc, fprintf('\nStarting Training...\n')

    %% Loop Through Runs
    for iRun = 1:nRuns
        if nRuns~=1 && ~mod(iRun, round(nRuns/5)), fprintf('  Run #%d/%d...\n', iRun, nRuns), end

        % Initiate and normalize random weight matrix (0-1)
        w_CS_MGv = fxn_normUnit(rand(nUnits_CS, nUnits_MGv)/d);
        w_CS_MGm = fxn_normUnit(rand(nUnits_CS, nUnits_MGm)/d);
        w_MGv_AC = fxn_normUnit(rand(nUnits_MGv, nUnits_AC)/d);
        w_MGm_AC = fxn_normUnit(rand(nUnits_MGm, nUnits_AC)/d);
        w_AC_Amy = fxn_normUnit(rand(nUnits_AC, nUnits_Amy)/d); % Subcortical pathway
        w_MGm_Amy = fxn_normUnit(rand(nUnits_MGm, nUnits_Amy)/d);

        % Loop through each epoch
        for iEpoch = 1:nEpochs

            % Shuffle the sequence of frequencies fed into the model
            freq_all = randperm(nFreqs);

            % Loop through each input frequency to examine activations
            for freq = freq_all

                % Define CS input at a frequency
                CS_inputs = zeros(1, nUnits_CS);
                CS_inputs(freq:freq+1) = 1;

                % ------------------------------------------
                % Calculate activation of each unit
                % MGv, from CS
                a_MGv = fxn_lateral_inhibition(CS_inputs * w_CS_MGv, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
                % MGm
                a_MGm = fxn_lateral_inhibition(CS_inputs * w_CS_MGm + US_inputs*w_US_MGm, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
                % Audi cortex
                a_AC = fxn_lateral_inhibition(a_MGv * w_MGv_AC + a_MGm*w_MGm_AC, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
                % Amygdala
                a_Amy = fxn_lateral_inhibition(a_MGm*w_MGm_Amy + a_AC * w_AC_Amy + US_inputs*w_US_Amy, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
                % -----------------------------------------

                % ------------------------------------------
                % Update weights by applying Stent-Hebb learning
                % weight_norm = fxn_SHupdate(a_s, a_r, weight, learning_rate)
                w_CS_MGv = fxn_SHupdate(CS_inputs, a_MGv, w_CS_MGv, learning_rate);
                w_CS_MGm = fxn_SHupdate(CS_inputs, a_MGm, w_CS_MGm, learning_rate);
                w_MGv_AC = fxn_SHupdate(a_MGv, a_AC, w_MGv_AC, learning_rate);
                w_MGm_AC = fxn_SHupdate(a_MGm, a_AC, w_MGm_AC, learning_rate);
                w_AC_Amy = fxn_SHupdate(a_AC, a_Amy, w_AC_Amy, learning_rate);
                w_MGm_Amy = fxn_SHupdate(a_MGm, a_Amy, w_MGm_Amy, learning_rate);
                % ------------------------------------------

                % Store averaged activation of all layers for all freq and all epochs
                a_MGv_allEpoch(iRun, iEpoch, freq, :) = a_MGv;
                a_MGm_allEpoch(iRun, iEpoch, freq, :) = a_MGm;
                a_AC_allEpoch(iRun, iEpoch, freq, :) = a_AC;
                a_Amy_allEpoch(iRun, iEpoch, freq, :) = a_Amy;

                %         fprintf('  Freq %d/%d\n', freq, nFreq)
            end % freq

            if nRuns==1 && ~mod(iEpoch, round(nEpochs/5)), fprintf('Epoch #%d/%d\n', iEpoch, nEpochs), end

        end % iEpoch

        % fprintf('\n===================== [%s] Training DONE =====================\n\n', flag_SXPA)

        %% Run the forward pass using the trained weight
        for freq = 1:nFreqs

            % Define CS input at a frequency
            CS_inputs = zeros(1, nUnits_CS);
            CS_inputs(freq:freq+1) = 1;

            % MGv, from CS
            a_MGv = fxn_lateral_inhibition(CS_inputs * w_CS_MGv, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
            % MGm
            a_MGm = fxn_lateral_inhibition(CS_inputs * w_CS_MGm + US_inputs*w_US_MGm, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
            % Audi cortex
            a_AC = fxn_lateral_inhibition(a_MGv * w_MGv_AC + a_MGm*w_MGm_AC, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
            % Amygdala
            a_Amy = fxn_lateral_inhibition(a_AC * w_AC_Amy + a_MGm*w_MGm_Amy + US_inputs*w_US_Amy, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
            % ------------------------------------------
            % ------------------------------------------
            % Store summed activation of all layers for all freq and all epochs
            a_MGv_allFreq_pre(iRun, freq, :) = a_MGv;
            a_MGm_allFreq_pre(iRun, freq, :) = a_MGm;
            a_AC_allFreq_pre(iRun, freq, :) = a_AC;
            a_Amy_allFreq_pre(iRun, freq, :) = a_Amy;

        end % freq

        % Plot activations of forward pass as a fxn of freq (RF) per Run
        a_allFreq_pre = {a_MGv_allFreq_pre, a_MGm_allFreq_pre, a_AC_allFreq_pre, a_Amy_allFreq_pre};

        figure('Position', [0 0 1.2e3, 1.5e3])
        % color_perEpoch = 'k';
        for iLayer = 1:nLayers
            subplot(2,2, iLayer), hold on

            nUnits = size(a_allFreq_pre{iLayer}, 3);
            str_legend = cell(nUnits, 1);
            for iUnit = 1:nUnits

                % Calculate the mean and sd across Runs
                a = squeeze(a_allFreq_pre{iLayer}(iRun, :, iUnit));

                % plot ave and sem
                plot(1:nFreqs, a, 'color', colors(iUnit, :), 'linewidth', 2)

                % Indicate the BF
                [~, iFreq_max] = max(a);
                % xline(iFreq_max, '-', 'color', color_, 'handlevisibility', 'off', 'linewidth', 1.5);
                str_legend{iUnit} = sprintf('U#%d (BF=%d)', iUnit, iFreq_max);
            end

            % title(sprintf('%s (BF=%.1f vs. %.1f)', namesLayers_all{iLayer}, iFreq_max, BF_fromPaper(iLayer)))
            title(namesLayers_all{iLayer}, 'fontsize', 20)
            xlabel('Frequency (a.u.)', 'fontsize', 18)
            ylabel(sprintf('Activation'), 'fontsize', 18)
            xticks(1:nFreqs), xlim([0, nFreqs+2])
            legend(str_legend, 'Location', 'southeast', 'fontsize', 12)
            % ylim([min(a_ave_all(:, :, iLayer),[], 'all'), max(a_ave_all(:, :, iLayer),[], 'all')])
        end % iLayer
        % set(findall(gcf, '-property', 'fontsize'), 'fontsize', 15);

        sgtitle(sprintf('%s\nActivation per layer [Run#%d]', nameFolder, iRun), 'fontsize', 20)

        %% Train with pairing US
        if flag_US
            US_inputs = 1;
            w_US_MGm = .4;
            w_US_Amy = .4;
            str_cond = 'condition';

            for iEpoch = 1:nEpochs

                % Loop through each input frequency to examine activations
                for freq = freq_train; % freq_all

                    % Define CS input at a frequency
                    CS_inputs = zeros(1, nUnits_CS);
                    CS_inputs(freq:freq+1) = 1;

                    % ------------------------------------------
                    % Calculate activation of each unit
                    % MGv, from CS
                    a_MGv = fxn_lateral_inhibition(CS_inputs * w_CS_MGv, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
                    % MGm
                    a_MGm = fxn_lateral_inhibition(CS_inputs * w_CS_MGm + US_inputs*w_US_MGm, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
                    % Audi cortex
                    a_AC = fxn_lateral_inhibition(a_MGv * w_MGv_AC + a_MGm*w_MGm_AC, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
                    % Amygdala
                    a_Amy = fxn_lateral_inhibition(a_MGm*w_MGm_Amy + a_AC * w_AC_Amy + US_inputs*w_US_Amy, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
                    % -----------------------------------------

                    % ------------------------------------------
                    % Update weights by applying Stent-Hebb learning
                    % weight_norm = fxn_SHupdate(a_s, a_r, weight, learning_rate)
                    w_CS_MGv = fxn_SHupdate(CS_inputs, a_MGv, w_CS_MGv, learning_rate);
                    w_CS_MGm = fxn_SHupdate(CS_inputs, a_MGm, w_CS_MGm, learning_rate);
                    w_MGv_AC = fxn_SHupdate(a_MGv, a_AC, w_MGv_AC, learning_rate);
                    w_MGm_AC = fxn_SHupdate(a_MGm, a_AC, w_MGm_AC, learning_rate);
                    w_AC_Amy = fxn_SHupdate(a_AC, a_Amy, w_AC_Amy, learning_rate);
                    w_MGm_Amy = fxn_SHupdate(a_MGm, a_Amy, w_MGm_Amy, learning_rate);
                    % ------------------------------------------

                    % Store averaged activation of all layers for all freq and all epochs
                    %                 a_MGv_allEpoch(iRun, iEpoch, freq, :) = a_MGv;
                    %                 a_MGm_allEpoch(iRun, iEpoch, freq, :) = a_MGm;
                    %                 a_AC_allEpoch(iRun, iEpoch, freq, :) = a_AC;
                    %                 a_Amy_allEpoch(iRun, iEpoch, freq, :) = a_Amy;

                    %         fprintf('  Freq %d/%d\n', freq, nFreq)
                end % freq

                if nRuns==1 && ~mod(iEpoch, round(nEpochs/5)), fprintf('Epoch #%d/%d\n', iEpoch, nEpochs), end

            end % iEpoch
        end
        % fprintf('\n===================== [%s] Training DONE =====================\n\n', flag_SXPA)

        %% Run the forward pass using the trained weight
        for freq = 1:nFreqs

            % Define CS input at a frequency
            CS_inputs = zeros(1, nUnits_CS);
            CS_inputs(freq:freq+1) = 1;

            % MGv, from CS
            a_MGv = fxn_lateral_inhibition(CS_inputs * w_CS_MGv, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
            % MGm
            a_MGm = fxn_lateral_inhibition(CS_inputs * w_CS_MGm + US_inputs*w_US_MGm, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
            % Audi cortex
            a_AC = fxn_lateral_inhibition(a_MGv * w_MGv_AC + a_MGm*w_MGm_AC, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
            % Amygdala
            a_Amy = fxn_lateral_inhibition(a_AC * w_AC_Amy + a_MGm*w_MGm_Amy + US_inputs*w_US_Amy, mu, x_thr, x_sat); % fxn_getAct(CS_inputs, w_CS_MGv, mu, x_thr, x_sat);
            % ------------------------------------------
            % ------------------------------------------
            % Store summed activation of all layers for all freq and all epochs
            a_MGv_allFreq_post(iRun, freq, :) = a_MGv;
            a_MGm_allFreq_post(iRun, freq, :) = a_MGm;
            a_AC_allFreq_post(iRun, freq, :) = a_AC;
            a_Amy_allFreq_post(iRun, freq, :) = a_Amy;

        end % freq


        a_allFreq_post = {a_MGv_allFreq_post, a_MGm_allFreq_post, a_AC_allFreq_post, a_Amy_allFreq_post};

        %% Plot activations of forward pass as a fxn of freq (RF) per Run
        figure('Position', [0 0 1.2e3, 1.5e3])
        % color_perEpoch = 'k';
        for iLayer = 1:nLayers
            subplot(2,2, iLayer), hold on

            nUnits = size(a_allFreq_pre{iLayer}, 3);
            str_legend = cell(nUnits, 1);
            for iUnit = 1:nUnits

                % Calculate the mean and sd across Runs
                a_pre = squeeze(a_allFreq_pre{iLayer}(iRun, :, iUnit));
                a_post = squeeze(a_allFreq_post{iLayer}(iRun, :, iUnit));

                % plot ave and sem
                plot(1:nFreqs, a_pre, '-o', 'color', colors(iUnit, :), 'linewidth', 2)
                plot(1:nFreqs, a_post, '--s', 'color', colors(iUnit, :), 'linewidth', 2)

                % Indicate the BF
                [~, iFreq_max] = max(a_pre);
                % xline(iFreq_max, '-', 'color', color_, 'handlevisibility', 'off', 'linewidth', 1.5);
                str_legend{iUnit} = sprintf('U#%d (BF=%d)', iUnit, iFreq_max);
            end

            % title(sprintf('%s (BF=%.1f vs. %.1f)', namesLayers_all{iLayer}, iFreq_max, BF_fromPaper(iLayer)))
            title(namesLayers_all{iLayer}, 'fontsize', 20)
            xlabel('Frequency (a.u.)', 'fontsize', 18)
            ylabel(sprintf('Activation'), 'fontsize', 18)
            xticks(1:nFreqs), xlim([0, nFreqs+2])
            legend(str_legend, 'Location', 'southeast', 'fontsize', 12)
            % ylim([min(a_ave_all(:, :, iLayer),[], 'all'), max(a_ave_all(:, :, iLayer),[], 'all')])
        end % iLayer
        % set(findall(gcf, '-property', 'fontsize'), 'fontsize', 15);

        sgtitle(sprintf('%s\nActivation per layer [Run#%d]', nameFolder, iRun), 'fontsize', 20)

        %%%%%%%%%
        if iRun<=5, saveas(gcf, sprintf('%s/SX_Run#%d.jpg', nameFolder, iRun)), end, close all

    end % iRun
    fprintf('\n===================== [%s] Forward pass DONE =====================\n\n', flag_SXPA)
end % iHyperP

%% Helper funcitions

function activation = fxn_ramp(input, x_thr, x_sat)
% Ramp function (thresholding & saturation). [Eq 1]
activation = min(max(0, input - x_thr), x_sat);
end

function w_norm = fxn_normUnit(w)
assert(sum(sum(w, 1)==0)==0, '!!! ALERT: zeros appear in denominator !!!')
w_norm = w./sum(w, 1);% Normalize across input dimension
end

function a_r = fxn_lateral_inhibition(input_r, mu, x_thr, x_sat)
% input_r: input to the recieving unit
% a_r: activation of the receiving unit
[input_max, imax] = max(input_r);
a_max = fxn_ramp(input_max, x_thr, x_sat); % Eq 3a
a_r = fxn_ramp(input_r - mu * a_max, x_thr, x_sat); % Eq 3b
a_r(imax) = a_max;
end

function weight_norm = fxn_SHupdate(act_send, act_recv, weight, learning_rate)
% Implements Hebbian learning with conditional weight updates:

% Create a mask for neurons where activation is above average
active_mask = act_send > mean(act_send);

% Calculating delta [Eq 4a]
weight_delta = learning_rate * ((active_mask .* act_send)' * act_recv);  % fire together, wire together

% Update weight
weight_updated = weight + weight_delta;

% Normalize weights across incoming connections per neuron [Eq 4b]
% so that the sum of the weights of all incoming connections to a unit was kept constant
% weight_norm = weight_updated./sum(weight_updated, 2);% Normalize across input dimension
weight_norm = fxn_normUnit(weight_updated);

end
