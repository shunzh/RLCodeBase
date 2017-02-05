function main()
  C = repmat((1:3) / 4, 3, 1)'
  Y = ones(3, 3)';
  
  h = bar(Y);
  for iBarSeries = 1:size(Y,2)
    set(h(iBarSeries), 'FaceColor', C(iBarSeries, :), 'EdgeColor', 'none');
  end

  axis off
  legend('MILP', 'Feature Based', 'Random Query');

  print('-deps', ['rsLegend.eps'], '-r100');
end
