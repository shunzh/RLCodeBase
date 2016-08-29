function main()
  agentNames = {'OPT-POLICY', 'OPT-POLICY-ACT', 'JQTP', 'MILP-QI-POLICY', 'MILP-POLICY', 'MILP-QI', 'MILP', 'AS', 'RQ'};
  numOfRocks = [1, 3, 5];
  rewardNums = [3, 5, 7];

  qv = zeros(size(agentNames, 2), size(numOfRocks, 2), size(rewardNums, 2));
  time = zeros(size(agentNames, 2), size(numOfRocks, 2), size(rewardNums, 2));
  dataSet = cell(size(agentNames, 2), size(numOfRocks, 2), size(rewardNums, 2));
  for agentId = 1 : size(agentNames, 2)
    for numOfRock = numOfRocks
      filename = strcat(agentNames(agentId), num2str(numOfRock), '_', num2str(5), '.out');
      data = load(char(filename));
      % some data are in files with postfix _1 due to an implementation bug
      filename = strcat(agentNames(agentId), num2str(numOfRock), '_', num2str(5), '_1.out');
      if exist(char(filename), 'file')
        data = [data; load(char(filename))];
      end
      qv(agentId, numOfRock, 5) = mean(data(:, 1));
      time(agentId, numOfRock, 5) = mean(data(:, 2));

      dataSet{agentId, numOfRock, 5} = data(:, 1);
    end
  end

  keyboard

  markers = {'*-', '*--', '+-', 'x-', 'x--', 's-', 's--', '^-', 'd-'};
  for agentId = 1 : size(agentNames, 2)
    agentNames{agentId}
    datum = qv(agentId, numOfRocks, 5)
    plot(numOfRocks, datum, markers{agentId});
    hold on;
  end
  legend('Optimal Policy Query', 'Action Query from OP', 'Optimal Action Query', 'Policy Query w/ QI', 'Policy Query', 'QP w/ QI', 'QP', 'Active Sampling', 'Random Query');
  xlabel('Number of Rocks');
  ylabel('Q-Value');

  figure;
  for agentId = 1 : size(agentNames, 2)
    datum = time(agentId, numOfRocks, 5)
    plot(numOfRocks, datum, markers{agentId});
    hold on;
  end
  legend('Optimal Policy Query', 'Action Query from OP', 'Optimal Action Query', 'Policy Query w/ QI', 'Policy Query', 'QP w/ QI', 'QP', 'Active Sampling', 'Random Query');
  xlabel('Number of Rocks');
  ylabel('Computation Time (sec.)');

  figure;
  histogram(dataSet{1, 3, 5} - dataSet{4, 3, 5}, 5, 'FaceColor', [.8, .8, .8]);
  hold on
  plot(0, sum(dataSet{1, 3, 5} - dataSet{4, 3, 5} == 0), '*r');
  xlabel('Optimal Policy Query - Policy Query w/ QI')
  ylabel('Frequency')

  figure;
  histogram(dataSet{3, 3, 5} - dataSet{2, 3, 5}, 5, 'FaceColor', [.8, .8, .8]);
  hold on
  plot(0, sum(dataSet{3, 3, 5} - dataSet{2, 3, 5} == 0), '*r');
  xlabel('Optimal Action Query - Action Query from OP')
  ylabel('Frequency')
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end

