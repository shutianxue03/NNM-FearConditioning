
%% Illustration of the ramp function
input = linspace(-1, 2, 1e2);
activation = fxn_ramp(input, x_thr, x_sat);
figure, plot(input, activation, 'k', 'LineWidth', 5)
xlabel('Input (x)'), xticks([-3]), xticklabels([-3]), ylim([-1, 2])
ylabel('Activation'), yticks([0]), yticklabels([0]), ylim([-.25, 1.25])

yline(0, 'k--', 'LineWidth', 2)
set(findall(gcf, '-property', 'fontsize'), 'fontsize', 30);
text(-.2, .4, 'x_{thr}', 'fontsize', 40)
text(1, .6, 'x_{sat}', 'fontsize', 40)
