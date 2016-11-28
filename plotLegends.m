function main()
  C = repmat((1:5) / 6, 3, 1)'
  %C = repmat((1:6) / 7, 3, 1)';
  Y = ones(5, 3)';
  
  h = bar(Y);
  for iBarSeries = 1:size(Y,2)
    set(h(iBarSeries), 'FaceColor', C(iBarSeries, :), 'EdgeColor', 'none');
  end

  axis off
  legend('Greedy q^*_\Pi', 'Sampling N=10', 'Sampling N=20', 'Sampling N=50');
  %legend('Opt q^*_\Pi', 'Greedy q^*_\Pi', 'Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  %legend('Greedy q^*_\Pi', 'Query Projection', 'Belief Change', 'Disagreement', 'Random Query');

  %set(gcf,'PaperUnits','inches','PaperPosition',[0 0 3 1.5])
  %print('-deps', ['aprilLegend.eps'], '-r100');
end
