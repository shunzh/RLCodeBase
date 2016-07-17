function main()
  agentNames = {'OPT-POLICY', 'JQTP', 'MILP-QI-POLICY', 'MILP-POLICY', 'MILP-QI', 'MILP', 'AS', 'RQ'};
  numOfRocks = [1, 3, 5];
  rewardNums = [3, 5, 7];

  qv = zeros(size(agentNames, 2), size(numOfRocks, 2), size(rewardNums, 2));
  time = zeros(size(agentNames, 2), size(numOfRocks, 2), size(rewardNums, 2));
  for agentId = 1 : size(agentNames, 2)
    for numOfRock = numOfRocks
      filename = strcat(agentNames(agentId), num2str(numOfRock), '_', num2str(5), '.out');
      data = load(char(filename));
      qv(agentId, numOfRock, 5) = mean(data(:, 1));
      time(agentId, numOfRock, 5) = mean(data(:, 2));
    end
  end
  for agentId = 1 : size(agentNames, 2)
    for rewardNum = rewardNums
      filename = strcat(agentNames(agentId), num2str(3), '_', num2str(rewardNum), '.out');
      data = load(char(filename));
      qv(agentId, 3, rewardNum) = mean(data(:, 1));
      time(agentId, 3, rewardNum) = mean(data(:, 2));
    end
  end

  markers = {'*-', '+-', 'x-', 'x--', 's-', 's--', '^-', 'd-'}
  for agentId = 1 : size(agentNames, 2)
    datum = qv(agentId, numOfRocks, 5)
    plot(numOfRocks, datum, markers{agentId});
    hold on;
  end
  legend('Optimal Policy Query', 'Optimal Action Query', 'Policy Query w/ QI', 'Policy Query', 'QP w/ QI', 'QP', 'Active Sampling', 'Random Query');
  xlabel('Number of Rocks');
  ylabel('Q-Value');
 
  figure;
  for agentId = 1 : size(agentNames, 2)
    datum = qv(agentId, 3, rewardNums)
    plot(rewardNums, datum(:), markers{agentId});
    hold on;
  end
  legend('Optimal Policy Query', 'Optimal Action Query', 'Policy Query w/ QI', 'Policy Query', 'QP w/ QI', 'QP', 'Active Sampling', 'Random Query');
  xlabel('Number of Reward Candidates');
  ylabel('Q-Value');
 
  %ylabel('Computation Time (sec.)');
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end

