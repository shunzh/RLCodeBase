function main()
  system('scp $ut:/u/menie482/workspace/CMP/results.tar.gz .');
  system('tar xvzf results.tar.gz');

  clear all; close all;

  responseTimes = [5, 10, 15];
  filename = 'rsP_out';

  retMat = 1:3:200;
  jMat = retMat(1:3);
  aMat = retMat(4:6);

  [jm, jc] = process(filename, jMat + 1)
  [am, ac] = process(filename, aMat + 1)

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');

  legend('E-JQTP', 'AQTP');
  xlabel('Response Time');
  ylabel('Q-Value');
  xlim([-1, 25]); 
  

  figure;

  [jm, jc] = process(filename, jMat + 2)
  [am, ac] = process(filename, aMat + 2)

  errorbar(responseTimes, jm, jc, '*-');
  hold on;
  errorbar(responseTimes, am, ac, 'o-');

  legend('E-JQTP', 'AQTP');
  xlabel('Response Time');
  ylabel('Computation Time (sec.)');
  xlim([-1, 25]); 

  figure;

  jd = getData(filename, [jMat(3)] + 1);
  ad = getData(filename, [aMat(3)] + 1);
  hist(jd - ad);
  disp('supports');
  hold on
  plot(0, sum(jd - ad == 0), '*r');
  xlabel('Difference in Q value between E-JQTP and AQTP');
  ylabel('Frequency');
end

function data = getData(filename, lines)
  data = [];
  for i=0:499
    try
      raw = load([filename, '.', num2str(i)]);
      if size(raw, 1) == 18
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
    %filter = dataRow < 60; % condor may suspend my jobs
    %dataRow = dataRow(filter);

    m(end + 1) = mean(dataRow);
    ci(end + 1) = 1.96 * std(dataRow) / sqrt(n);
  end
end

