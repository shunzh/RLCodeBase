function main()
  system('scp $ut:/u/menie482/workspace/CMP/results.tar.gz .');
  system('tar xvzf results.tar.gz');

  clear all; close all;

  responseTimes = [0, 5, 10, 15, 20];
  filename = 'rsCorner_out';

  retMat = 1:3:200;
  jMat = retMat(1:5);
  aMat = retMat(6:10);
  anfMat = retMat(11:15);
  pMat = retMat(16:20);
  rMat = retMat(21:25);
  nMat = retMat([26 26 26 26 26]);

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

  %legend('E-JQTP', 'AQTP-QF', 'AQTP', 'Prior TP, BR Query', 'Random Query, BR TP', 'No Query');
  xlabel('Response Time');
  ylabel('Q-Value');
  xlim([-1, 25]); 
  set(gcf, 'PaperUnits', 'inches');
  set(gcf, 'PaperPosition', [2.5 2.5 4 3]);
  print('-depsc2', '-painters', 'rsCornerQ.eps');
  
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

  %legend('E-JQTP', 'AQTP-QF', 'AQTP-NQF', 'Prior TP, BR Query', 'Random Query, BR TP', 'No Query');
  xlabel('Response Time');
  ylabel('Computation Time (sec.)');
  xlim([-1, 25]); 
  set(gcf, 'PaperUnits', 'inches');
  set(gcf, 'PaperPosition', [2.5 2.5 4 3]);
  print('-depsc2', '-painters', 'rsCornerTime.eps');

  % figure;

  % jd = getData(filename, [jMat(3)] + 1);
  % ad = getData(filename, [aMat(3)] + 1);
  % histogram(jd - ad, 'FaceColor', [.8, .8, .8], 'BinWidth', 0.01);
  % hold on
  % xlim([0, 0.1]);
  % ylim([0, 200]);
  % plot(0, sum(jd - ad == 0), '*k');
  % xlabel('Difference in Q value between E-JQTP and AQTP-QF');
  % ylabel('Frequency');
end

function data = getData(filename, lines)
  data = [];
  for i=0:199
    try
      raw = load([filename, '.', num2str(i)]);
      if size(raw, 1) == 78
        data = [data, raw(lines)];
      else
        disp(['Invalid datum #', num2str(i)]);
      end
    catch
      disp(['Unable to load ', num2str(i)]);
    end
  end
end

function [m, ci] = process(filename, lines)
  data = getData(filename, lines);
  n = size(data, 2);
  m = []; ci = [];
  for i=1:size(data,1)
    dataRow = data(i, :);
    filter = dataRow < 60; % condor may suspend my jobs
    dataRow = dataRow(filter);

    m(end + 1) = mean(dataRow);
    ci(end + 1) = 1.96 * std(dataRow) / sqrt(n);
  end
end

