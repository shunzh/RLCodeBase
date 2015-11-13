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
  [wm, wc] = process(filename, [37, 40, 43])
  [nm, nc] = process(filename, [46, 46, 46])

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');
  errorbar(responseTimes, tm, tc, 'x-');
  errorbar(responseTimes, rm, rc, '+-');
  errorbar(responseTimes, nm, nc, '--');

  legend('E-JQTP', 'AQTP', 'Optimal Query, Prior TP', 'Random Query, Optimal TP', 'No Query');
  xlabel('Response Time');
  ylabel('Accumulated Return');
  xlim([-1, 25]); 
 

  figure;

  [jm, jc] = process(filename, [2, 5, 8])
  [am, ac] = process(filename, [11, 14, 17])
  [tm, tc] = process(filename, [20, 23, 26])
  [rm, rc] = process(filename, [29, 32, 35])
  [wm, wc] = process(filename, [38, 41, 44])
  [nm, nc] = process(filename, [47, 47, 47])

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');
  errorbar(responseTimes, tm, tc, 'x-');
  errorbar(responseTimes, rm, rc, '+-');
  errorbar(responseTimes, nm, nc, '--');

  legend('E-JQTP', 'AQTP', 'Optimal Query, Prior TP', 'Random Query, Optimal TP', 'No Query');
  xlabel('Response Time');
  ylabel('Q-Value');
  xlim([-1, 25]); 
  
  figure;

  [jm, jc] = process(filename, [3, 6, 9])
  [am, ac] = process(filename, [12, 15, 18])
  [tm, tc] = process(filename, [21, 24, 27])
  [rm, rc] = process(filename, [30, 33, 36])
  [wm, wc] = process(filename, [39, 42, 45])
  [nm, nc] = process(filename, [48, 48, 48])

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');
  errorbar(responseTimes, tm, tc, 'x-');
  errorbar(responseTimes, rm, rc, '+-');
  errorbar(responseTimes, nm, nc, '--');

  legend('E-JQTP', 'AQTP', 'Optimal Query, Prior TP', 'Random Query, Optimal TP', 'No Query');
  xlabel('Response Time');
  ylabel('Computation Time (sec)');
  xlim([-1, 25]); 


  figure;

  jd = getData(filename, [5]);
  ad = getData(filename, [14]);
  hist(jd - ad);
  disp('supports');
  find(jd - ad)
  xlabel('Difference in Q value between E-JQTP and AQTP');
  ylabel('Frequency');
end

function data = getData(filename, lines)
  data = [];
  for i=0:999
    try
      raw = load([filename, '.', num2str(i)]);
      if size(raw, 1) == 39
        data = [data, raw(lines)];
      end
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

