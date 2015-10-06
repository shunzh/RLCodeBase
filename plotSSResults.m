function main()
  clear all; close all;

  plot([100, 400], [0.478, 0.133], '*-');
  hold on
  plot([100, 400], [0.502, 0.150], '+--');

  legend('AQTP', 'JQTP');
  xlabel('Number of States');
  ylabel('Accumulated Return');
  axis([0, 500, 0, 0.6]);


  figure;

  plot([100, 400], [2.674, 44.043], '*-');
  hold on
  plot([100, 400], [10.463, 68.599], '+--');

  legend('AQTP', 'JQTP');
  xlabel('Number of States');
  ylabel('Computation Time (sec.)');
  axis([0, 500, 0, 100]);
end
