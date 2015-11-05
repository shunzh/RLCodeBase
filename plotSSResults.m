function main()
  clear all; close all;

  [am, aci] = process('ssAqtp_out', 1);
  [a2m, a2ci] = process('ssAqtp2_out', 1);
  errorbar([60, 120], [am, a2m], [aci, a2ci], '*-');
  hold on
  [jm, jci] = process('ssJqtp_out', 1);
  [j2m, j2ci] = process('ssJqtp2_out', 1);
  errorbar([60, 120], [jm, j2m], [jci, j2ci], '+--');

  legend('AQTP-QE', 'E-JQTP');
  xlabel('Number of States');
  ylabel('Accumulated Return');


  figure;

  [am, aci] = process('ssAqtp_out', 2);
  [a2m, a2ci] = process('ssAqtp2_out', 2);
  errorbar([60, 120], [am, a2m], [aci, a2ci], '*-');
  hold on
  [jm, jci] = process('ssJqtp_out', 2);
  [j2m, j2ci] = process('ssJqtp2_out', 2);
  errorbar([60, 120], [jm, j2m], [jci, j2ci], '+--');

  legend('AQTP-QE', 'E-JQTP');
  xlabel('Number of States');
  ylabel('Optimal Q-Value');


  figure;

  [am, aci] = process('ssAqtp_out', 3);
  [a2m, a2ci] = process('ssAqtp2_out', 3);
  errorbar([60, 120], [am, a2m], [aci, a2ci], '*-');
  hold on
  [jm, jci] = process('ssJqtp_out', 3);
  [j2m, j2ci] = process('ssJqtp2_out', 3);
  errorbar([60, 120], [jm, j2m], [jci, j2ci], '+--');

  legend('AQTP-QE', 'E-JQTP', 'Location', 'northwest');
  xlabel('Number of States');
  ylabel('Computation Time (sec.)');
end

function [m, ci] = process(filename, lineId)
  ret = [];
  for i=0:399
    [status results] = system(['tail -n 3 ', filename, '.', num2str(i)]);
    results = str2num(results);
    if status == 0 && size(results, 1) == 3
      ret(end + 1) = results(lineId);
    end
  end

  n = size(ret, 2)
  m = mean(ret);
  ci = 1.96 * std(ret) / sqrt(n);
end
