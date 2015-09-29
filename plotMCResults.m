function main()
  clear all; close all;
  t = 0:0.1:0.6;

  plot(t, 2.276 * ones(1,7), '-square');
  hold on;
  plot([0.05, 0.1, 0.2, 0.3, 0.5], [2.289, 2.286, 2.092, 1.949, 1.498],'--+');
  plot(t, 2.254 * ones(1,7), '-d');
  plot(t, 2.158 * ones(1,7), '-*');
  plot(t, 1.817 * ones(1,7), '-o');

  axis([0, 0.6, 1, 3]);
  legend('JQTP', 'JQTP with clustering','AQTP','Optimal Query for Prior Belief Policy', 'Prior Belief Policy')
  xlabel('Cluster Radius');
  ylabel('Accumulated Return');
end
