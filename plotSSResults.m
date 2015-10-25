function main()
  clear all; close all;

  errorbar([100, 400], [0.598, 0.204], [0.058, 0.049], '*-');
  hold on
  errorbar([100, 400], [0.647, 0.215], [0.062, 0.044], '+--');

  legend('AQTP', 'JQTP');
  xlabel('Number of States');
  ylabel('Accumulated Return');
  axis([0, 500, 0, 0.8]);


  figure;

  errorbar([100, 400], [3.838, 45.927], [0.269, 1.306], '*-');
  hold on
  errorbar([100, 400], [11.517, 155.710], [0.135, 1.485], '+--');

  legend('AQTP', 'JQTP');
  xlabel('Number of States');
  ylabel('Computation Time (sec.)');
  axis([0, 500, 0, 250]);
end
