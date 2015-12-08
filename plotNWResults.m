function main()
  system('scp $ut:/u/menie482/workspace/CMP/results.tar.gz .');
  system('tar xvzf results.tar.gz');

  clear all; close all;

  responseTimes = [6, 10, 20];
  filename = 'nw_out';

  retMat = 1:3:200;
  jMat = retMat(1:3);
  aMat = retMat(4:6);
  anfMat = retMat(7:9);

  [jm, jc] = process(filename, jMat + 1)
  [am, ac] = process(filename, aMat + 1)
  [anm, anc] = process(filename, anfMat + 1)

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');
  errorbar(responseTimes, anm, anc, 'o--');

  legend('E-JQTP', 'AQTP-QF', 'AQTP-NQF');
  xlabel('Response Time');
  ylabel('Q-Value');
  xlim([-1, 25]); 
  
  figure;

  [jm, jc] = process(filename, jMat + 2)
  [am, ac] = process(filename, aMat + 2)
  [anm, anc] = process(filename, anfMat + 2)

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');
  errorbar(responseTimes, anm, anc, 'o--');

  legend('E-JQTP', 'AQTP-QF', 'AQTP-NQF');
  xlabel('Response Time');
  ylabel('Computation Time (sec.)');
  xlim([-1, 25]); 

  figure;

  jd = getData(filename, [jMat(3)] + 1);
  ad = getData(filename, [aMat(3)] + 1);
  histogram(jd - ad, 'FaceColor', [.8, .8, .8]);
  hold on
  plot(0, sum(jd - ad == 0), '*k');
  xlabel('Difference in Q value between E-JQTP and AQTP-QF');
  ylabel('Frequency');
end

function data = getData(filename, lines)
  data = [];
  for i=0:199
    try
      raw = load([filename, '.', num2str(i)]);
      if size(raw, 1) == 27
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
  m = []; ci = [];
  for i=1:size(data,1)
    dataRow = data(i, :);

    m(end + 1) = mean(dataRow);
    ci(end + 1) = 1.96 * std(dataRow) / sqrt(n);
  end
end

