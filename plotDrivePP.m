function main()
  agents = {'MILP-POLICY', 'MILP-SIMILAR', 'SIMILAR-VARIATION', 'SIMILAR-DISAGREE', 'SIMILAR-RANDOM'};

  % driving
  rewardCandNums = [5];
  numOfQueries = [1];
  numOfResponses = [2, 3, 4];
  trajLens = [3];

  % default values of variables
  rewardCand_ = 5;
  numOfQuery_ = 1;
  trajLen_ = 3;

  dataM = cell(size(agents, 2), max(rewardCandNums), max(numOfQueries), max(numOfResponses), max(trajLens));
  for agentId = 1 : size(agents, 2)
    for numOfResponse = numOfResponses
      filename = strcat(agents(agentId), num2str(rewardCand_), '_', num2str(numOfQuery_), '_', num2str(numOfResponse), '_', num2str(trajLen_), '.out');
      data = load(char(filename));
      [m, ci] = process(data(:, 1));
      dataM{agentId, rewardCand_, numOfQuery_, numOfResponse, trajLen_} = m;
      dataCI{agentId, rewardCand_, numOfQuery_, numOfResponse, trajLen_} = ci;
    end
  end

  markers = {'o--', '*-', '+-', 'x-', 's-'};

  for agentId = 1 : size(agents, 2)
    %errorbar(numOfResponses, cell2mat(dataM(agentId, rewardCand_, numOfQuery_, numOfResponses, trajLen_)),...
    %                         cell2mat(dataCI(agentId, rewardCand_, numOfQuery_, numOfResponses, trajLen_)), markers{agentId});
    errorbar(numOfResponses, cell2mat(dataM(agentId, rewardCand_, numOfQuery_, numOfResponses, trajLen_)),...
                             zeros(1, size(numOfResponses, 2)), markers{agentId});
    hold on
  end
  legend('Greedy q^*_\Pi', 'Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  xlabel('Number of Responses');
  ylabel('Q-Value');
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
