function main()
  system('scp $ut:/u/menie482/workspace/CMP/results.tar.gz .');
  system('tar xvzf results.tar.gz');

  clear all; close all;

  responseTimes = [0, 10, 20];
  filename = 'rs_out';

  [jm, jc] = process(filename, [1, 4, 7])
  [am, ac] = process(filename, [10, 13, 16])
  [tm, tc] = process(filename, [19, 22, 25])
  [rm, rc] = process(filename, [28, 31, 34])
  [nm, nc] = process(filename, [37, 37, 37])

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');
  errorbar(responseTimes, tm, tc, 'x-');
  errorbar(responseTimes, rm, rc, '+-');
  errorbar(responseTimes, nm, nc, '--');

  legend('E-JQTP', 'AQTP', 'Random Query, Optimal TP', 'Optimal Query, Prior TP', 'No Query');
  xlabel('Response Time');
  ylabel('Accumulated Return');
  xlim([-1, 25]); 
 

  figure;

  [jm, jc] = process(filename, [2, 5, 8])
  [am, ac] = process(filename, [11, 14, 17])
  [tm, tc] = process(filename, [20, 23, 26])
  [rm, rc] = process(filename, [29, 32, 35])
  [nm, nc] = process(filename, [38, 38, 38])

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');
  errorbar(responseTimes, tm, tc, 'x-');
  errorbar(responseTimes, rm, rc, '+-');
  errorbar(responseTimes, nm, nc, '--');

  legend('E-JQTP', 'AQTP', 'Random Query, Optimal TP', 'Optimal Query, Prior TP', 'No Query');
  xlabel('Response Time');
  ylabel('Q-Value');
  xlim([-1, 25]); 
  
  figure;

  [jm, jc] = process(filename, [3, 6, 9])
  [am, ac] = process(filename, [12, 15, 18])
  [tm, tc] = process(filename, [21, 24, 27])
  [rm, rc] = process(filename, [30, 33, 36])
  [nm, nc] = process(filename, [39, 39, 39])

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');
  errorbar(responseTimes, tm, tc, 'x-');
  errorbar(responseTimes, rm, rc, '+-');
  errorbar(responseTimes, nm, nc, '--');

  legend('E-JQTP', 'AQTP', 'Random Query, Optimal TP', 'Optimal Query, Prior TP', 'No Query');
  xlabel('Response Time');
  ylabel('Computation Time (sec)');
  xlim([-1, 25]); 


  figure;

  jd = getData(filename, [5]);
  ad = getData(filename, [14]);
  hist(jd - ad);
  xlabel('Difference in Q value between E-JQTP and AQTP');
  ylabel('Frequency');
end

function data = getData(filename, lines)
  data = [];
  for i=0:399
    try
      raw = load([filename, '.', num2str(i)]);
      data = [data, raw(lines)];
    catch
      disp(['Unable to load ', num2str(i)]);
    end
  end
end

function [m, ci] = process(filename, lines)
  data = getData(filename, lines);
  n = size(data, 2);
  m = mean(data, 2);
  ci = 1.96 * std(data, 0, 2) / sqrt(n);
end

