function main()
  clear all; close all;

  load a.mat;

  x = [1, 5]

  [em, ec] = process(e(:, 2))
  [hm, hc] = process(h(:, 2))
  [h5m, h5c] = process(h5(:, 2))
  [sm, sc] = process(s(:, 2))
  [s5m, s5c] = process(s5(:, 2))
  [rm, rc] = process(r(:, 2))

  %errorbar(x, [em, em], [ec, ec], '*-');
  %hold on;
  %errorbar(x, [rm, rm], [rc, rc], 'o-');
  %errorbar(x, [hm, h5m], [hc, h5c], '+-');

  plot(x, [em, em], '*-');
  hold on;
  plot(x, [hm, h5m], '+-');
  plot(x, [sm, s5m], '+--');
  plot(x, [rm, rm], 'o-');
  xlim([0, 6]);

  legend('E-JQTP', 'Occupancy-based Query Selection', 'Active Sampling', 'Random Query');
  xlabel('m in Algorithm 1');
  ylabel('Q-Value');
  
  figure;
  [em, ec] = process(e(:, 3))
  [hm, hc] = process(h(:, 3))
  [h5m, h5c] = process(h5(:, 3))
  [sm, sc] = process(s(:, 3))
  [s5m, s5c] = process(s5(:, 3))
  [rm, rc] = process(r(:, 3))

  plot(x, [em, em], '*-');
  hold on;
  plot(x, [hm, h5m], '+-');
  plot(x, [sm, s5m], '+--');
  plot(x, [rm, rm], 'o-');
  xlim([0, 6]);

  legend('E-JQTP', 'Occupancy-based Query Selection', 'Active Sampling', 'Random Query');
  xlabel('m in Algorithm 1');
  ylabel('Computation Time');
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end

