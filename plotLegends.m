function main()
  C = repmat((1:3) / 4, 3, 1)'
  Y = ones(3, 3)';
  
  h = bar(Y);
  for iBarSeries = 1:size(Y,2)
    set(h(iBarSeries), 'FaceColor', C(iBarSeries, :), 'EdgeColor', 'none');
  end

  axis off
  legend('Greedy q^*_\Pi', 'Feature Based', 'Random Query');
  %legend('Opt q^*_\Pi', 'Greedy q^*_\Pi', 'Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  %legend('Greedy q^*_\Pi', 'Query Projection', 'Belief Change', 'Disagreement', 'Random Query');

  %set(gcf,'PaperUnits','inches','PaperPosition',[0 0 3 1.5])
  print('-deps', ['rsLegend.eps'], '-r100');
end
