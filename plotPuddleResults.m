function main()
  clear all; close all;

  load a.mat;

  x = [1, 2, 3, 5, 7, 9, 50, 100];
  len = size(x, 2);

  [em, ec] = process(e(:, 2))
  [rm, rc] = process(r(:, 2))
  hm = []; sm = [];
  for i = 1:8
    hm(end + 1) = mean(h(i:8:80, 2))
    sm(end + 1) = mean(s(i:8:80, 2))
  end

  %errorbar(x, [em, em], [ec, ec], '*-');
  %hold on;
  %errorbar(x, [rm, rm], [rc, rc], 'o-');
  %errorbar(x, [hm, h5m], [hc, h5c], '+-');

  plot(x, ones(1, len) * em, '*-');
  hold on;
  plot(x, hm, '+-');
  plot(x, sm, '+--');
  plot(x, ones(1, len) * rm, 'o-');

  legend('Exhaustive', 'Occupancy-based', 'Active Sampling', 'Random Query');
  xlabel('m in Algorithm 1');
  ylabel('Q-Value');
  
  figure;
  [em, ec] = process(e(:, 3))
  [rm, rc] = process(r(:, 3))
  hm = []; sm = [];
  for i = 1:8
    hm(end + 1) = mean(h(i:8:80, 3))
    sm(end + 1) = mean(s(i:8:80, 3))
  end

  plot(x, ones(1, len) * em, '*-');
  hold on;
  plot(x, hm, '+-');
  plot(x, sm, '+--');
  plot(x, ones(1, len) * rm, 'o-');

  legend('Exhaustive', 'Occupancy-based', 'Active Sampling', 'Random Query');
  xlabel('m in Algorithm 1');
  ylabel('Computation Time');
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end

