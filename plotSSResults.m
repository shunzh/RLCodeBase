function main()
  clear all; close all;

  errorbar([100, 400], [0.674, 0.271], [0.058, 0.047], '*-');
  hold on
  errorbar([100, 400], [0.715, 0.296], [0.059, 0.049], 'o-');
  errorbar([100, 400], [0.727, 0.300], [0.120, 0.046], '+--');

  legend('AQTP', 'AQTP-RS', 'JQTP');
  xlabel('Number of States');
  ylabel('Accumulated Return');
  axis([0, 500, 0.2, 0.8]);


  figure;

  errorbar([100, 400], [3.507, 45.360], [0.165, 1.341], '*-');
  hold on
  errorbar([100, 400], [7.528, 90.511], [0.234, 2.618], 'o-');
  errorbar([100, 400], [10.607, 157.824], [0.141, 2.046], '+--');

  legend('AQTP', 'AQTP-RS', 'JQTP');
  xlabel('Number of States');
  ylabel('Computation Time (sec.)');
  axis([0, 500, 0, 250]);
end
