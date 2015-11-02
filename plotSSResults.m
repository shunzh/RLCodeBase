function main()
  clear all; close all;

  [am, aci] = process('ssAqtp_out', 1);
  [a2m, a2ci] = process('ssAqtp2_out', 1);
  errorbar([100, 400], [am, a2m], [aci, a2ci], '*-');
  hold on
  [jm, jci] = process('ssJqtp_out', 1);
  [j2m, j2ci] = process('ssJqtp2_out', 1);
  errorbar([100, 400], [jm, j2m], [jci, j2ci], '+--');

  legend('AQTP', 'JQTP');
  xlabel('Number of States');
  ylabel('Accumulated Return');
  axis([0, 500, 0.2, 0.8]);


  figure;

  [am, aci] = process('ssAqtp_out', 3);
  [a2m, a2ci] = process('ssAqtp2_out', 3);
  errorbar([100, 400], [am, a2m], [aci, a2ci], '*-');
  hold on
  [jm, jci] = process('ssJqtp_out', 3);
  [j2m, j2ci] = process('ssJqtp2_out', 3);
  errorbar([100, 400], [jm, j2m], [jci, j2ci], '+--');

  legend('AQTP', 'JQTP');
  xlabel('Number of States');
  ylabel('Computation Time (sec.)');
  axis([0, 500, 0, 250]);
end

function [m, ci] = process(filename, lineId)
  ret = [];
  for i=0:799
    [status results] = system(['tail -n 3 ', filename, '.', num2str(i)]);
    ret(end + 1) = results(lineId);
  end

  n = size(ret, 2)
  m = mean(ret);
  ci = 1.96 * std(ret) / sqrt(n);
end
