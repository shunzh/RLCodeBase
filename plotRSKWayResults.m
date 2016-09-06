function main()
  agentNames = {'JQTP', 'MILP-QI-POLICY', 'MILP-POLICY', 'MILP-QI', 'MILP', 'AS', 'RQ'};
  numOfRocks = [2, 3, 4, 5];
  rewardNums = [5];

  qv = zeros(size(agentNames, 2), size(numOfRocks, 2), size(rewardNums, 2));
  time = zeros(size(agentNames, 2), size(numOfRocks, 2), size(rewardNums, 2));
  dataSet = cell(size(agentNames, 2), size(numOfRocks, 2), size(rewardNums, 2));
  for agentId = 1 : size(agentNames, 2)
    for numOfRock = numOfRocks
      filename = strcat(agentNames(agentId), num2str(numOfRock), '_', num2str(5), '.out');
      data = load(char(filename));
      qv(agentId, numOfRock, 5) = mean(data(:, 1));
      time(agentId, numOfRock, 5) = mean(data(:, 2));

      dataSet{agentId, numOfRock, 5} = data(:, 1);
    end
  end

  markers = {'+-', 'x-', 'x--', 's-', 's--', '^-', 'd-'};
  for agentId = 1 : size(agentNames, 2)
    agentNames{agentId}
    datum = qv(agentId, numOfRocks, 5)
    plot(numOfRocks, datum, markers{agentId});
    hold on;
  end
  legend('Optimal Action Query', 'Policy Query w/ QI', 'Policy Query', 'QP w/ QI', 'QP', 'Active Sampling', 'Random Query');
  xlabel('Connectivity in Transition');
  ylabel('Q-Value');

  figure;
  for agentId = 1 : size(agentNames, 2)
    datum = time(agentId, numOfRocks, 5)
    plot(numOfRocks, datum, markers{agentId});
    hold on;
  end
  legend('Optimal Action Query', 'Policy Query w/ QI', 'Policy Query', 'QP w/ QI', 'QP', 'Active Sampling', 'Random Query');
  xlabel('Connectivity in Transition');
  ylabel('Computation Time (sec.)');
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end

