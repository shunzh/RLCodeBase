function main()
  system('scp $ut:/u/menie482/workspace/CMP/results.tar.gz .');
  system('tar xvzf results.tar.gz');

  clear all; close all;

  responseTimes = [0, 5, 10, 15, 20];
  filename = 'rs_out';

  retMat = 1:3:200
  jMat = retMat(1:5)
  aMat = retMat(6:10)
  anfMat = retMat(11:15)
  pMat = retMat(16:20)
  rMat = retMat(21:25)
  nMat = retMat([26 26 26 26 26])

  [jm, jc] = process(filename, jMat + 1)
  [am, ac] = process(filename, aMat + 1)
  [anm, anc] = process(filename, anfMat + 1)
  [pm, pc] = process(filename, pMat + 1)
  [rm, rc] = process(filename, rMat + 1)
  [nm, nc] = process(filename, nMat + 1)

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');
  errorbar(responseTimes, anm, anc, 'o--');
  errorbar(responseTimes, pm, pc, 'x-');
  errorbar(responseTimes, rm, rc, '+-');
  errorbar(responseTimes, nm, nc, '--');

  legend('E-JQTP', 'AQTP', 'AQTP No Filtering', 'Prior TP, BR Query', 'Random Query, BR TP', 'No Query');
  xlabel('Response Time');
  ylabel('Q-Value');
  xlim([-1, 25]); 
  
  figure;

  [jm, jc] = process(filename, jMat + 2)
  [am, ac] = process(filename, aMat + 2)
  [anm, anc] = process(filename, anfMat + 2)
  [pm, pc] = process(filename, pMat + 2)
  [rm, rc] = process(filename, rMat + 2)
  [nm, nc] = process(filename, nMat + 2)

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');
  errorbar(responseTimes, anm, anc, 'o--');
  errorbar(responseTimes, pm, pc, 'x-');
  errorbar(responseTimes, rm, rc, '+-');
  errorbar(responseTimes, nm, nc, '--');

  legend('E-JQTP', 'AQTP', 'AQTP No Filtering', 'Prior TP, BR Query', 'Random Query, BR TP', 'No Query');
  xlabel('Response Time');
  ylabel('Computation Time (sec.)');
  xlim([-1, 25]); 


  figure;

  keyboard
  jd = getData(filename, [jMat(3)] + 1);
  ad = getData(filename, [aMat(3)] + 1);
  hist(jd - ad);
  disp('supports');
  find(jd - ad)
  xlabel('Difference in Q value between E-JQTP and AQTP');
  ylabel('Frequency');
end

function data = getData(filename, lines)
  data = [];
  for i=0:499
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

