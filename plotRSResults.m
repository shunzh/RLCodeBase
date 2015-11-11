function main()
  clear all; close all;

  responseTimes = [0, 10, 20];
  filename = 'rs_out';

  [jm, jc] = process(filename, [2, 5, 8])
  [am, ac] = process(filename, [11, 14, 17])
  [rm, rc] = process(filename, [20, 23, 26])
  [nm, nc] = process(filename, [29, 29, 29])

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');
  errorbar(responseTimes, rm, rc, '+-');
  errorbar(responseTimes, nm, nc, '--');

  legend('E-JQTP', 'AQTP', 'Random Query', 'No Query');
  xlabel('Response Time');
  ylabel('Q-Value');
  axis([-1, 25, 4, 6]);

  
  figure;

  [jm, jc] = process(filename, [3, 6, 9])
  [am, ac] = process(filename, [12, 15, 18])
  [rm, rc] = process(filename, [21, 24, 27])
  [nm, nc] = process(filename, [30, 30, 30])

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');
  errorbar(responseTimes, rm, rc, '+-');
  errorbar(responseTimes, nm, nc, '--');

  legend('E-JQTP', 'AQTP', 'Random Query', 'No Query');
  xlabel('Response Time');
  ylabel('Computation Time');
  axis([-1, 25, 0, 15]);
end

function [m, ci] = process(filename, lines)
  data = [];
  for i=0:399
    raw = load([filename, '.', num2str(i)]);
    data = [data, raw(lines)];
  end

  n = 400;
  m = mean(data, 2);
  ci = 1.96 * std(data, 0, 2) / sqrt(n);
end
