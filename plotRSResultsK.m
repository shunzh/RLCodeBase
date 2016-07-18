function main()
  agentNames = {'OPT-POLICY', 'MILP-QI-POLICY', 'MILP-POLICY'};

  qv = zeros(size(agentNames, 2), 3);
  time = zeros(size(agentNames, 2), 3);
  for agentId = 1 : size(agentNames, 2)
    filename = strcat(agentNames(agentId), '3_5.out');
    data = load(char(filename));
    filename = strcat(agentNames(agentId), '3_5_1.out');
    data = [data; load(char(filename))];
    qv(agentId, 2) = mean(data(:, 1));
    time(agentId, 2) = mean(data(:, 2));

    for k = 3:4
      filename = strcat(agentNames(agentId), '3_5_', num2str(k), '.out');
      data = load(char(filename));
      qv(agentId, k) = mean(data(:, 1));
      time(agentId, k) = mean(data(:, 2));
    end
  end

  markers = {'*-', 'x-', 'x--'};
  for agentId = 1 : size(agentNames, 2)
    datum = qv(agentId, 2:4)
    plot(2:4, datum, markers{agentId});
    hold on;
  end
  legend('Optimal Policy Query', 'Policy Query w/ QI', 'Policy Query');
  xlabel('k');
  ylabel('Q-Value');

  figure;
  for agentId = 1 : size(agentNames, 2)
    datum = time(agentId, 2:4)
    plot(2:4, datum, markers{agentId});
    hold on;
  end
  legend('Optimal Policy Query', 'Policy Query w/ QI', 'Policy Query');
  xlabel('k');
  ylabel('Computation Time (sec.)');
end

