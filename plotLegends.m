function main()
  C = [0, 0.4470, 0.7410;
       0.8500, 0.3250, 0.0980;
       0.9290, 0.6940, 0.1250;
       0.4940, 0.1840, 0.5560;
       0.4660, 0.6740, 0.1880;
       0.3010, 0.7450, 0.9330];

  Y = zeros(2, 3)';
  
  h = bar(Y);
  for iBarSeries = 1:size(Y,2)
    set(h(iBarSeries), 'FaceColor', C(iBarSeries, :));
  end

  axis off
  %legend('Greedy q^*_\Pi', 'Sampling N=10', 'Sampling N=20', 'Sampling N=50');
  legend('Opt q^*_\Pi', 'Greedy q^*_\Pi', 'Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  %legend('Greedy q^*_\Pi', 'Query Projection');

  %set(gcf,'PaperUnits','inches','PaperPosition',[0 0 3 1.5])
  print('-dpng', ['aprilLegend.png'], '-r100');
end
