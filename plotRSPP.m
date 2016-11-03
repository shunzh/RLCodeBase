function main()
  agents = {'MILP-SIMILAR', 'MILP-SIMILAR-VARIATION', 'MILP-SIMILAR-DISAGREE', 'MILP-SIMILAR-RANDOM'};

  rewardCandNums = [5, 10];
  numOfQueries = [1, 2];
  numOfResponses = [2, 3];
  trajLens = [2, 3, 4];

  % default values of variables
  rewardCand_ = 5;
  numOfQuery_ = 1;
  numOfResponse_ = 2;
  trajLen_ = 3;

  dataM = cell(size(agents, 2), max(rewardCandNums), max(numOfQueries), max(numOfResponses), max(trajLens));
  for agentId = 1 : size(agents, 2)
    for rewardCand = rewardCandNums
      filename = strcat(agents(agentId), num2str(rewardCand), '_', num2str(numOfQuery_), '_', num2str(numOfResponse_), '_', num2str(trajLen_), '.out');
      data = load(char(filename));
      [m, ci] = process(data(:, 1));
      dataM{agentId, rewardCand, numOfQuery_, numOfResponse_, trajLen_} = m;
      dataCI{agentId, rewardCand, numOfQuery_, numOfResponse_, trajLen_} = ci;
    end

    for numOfQuery = numOfQueries
      filename = strcat(agents(agentId), num2str(rewardCand_), '_', num2str(numOfQuery), '_', num2str(numOfResponse_), '_', num2str(trajLen_), '.out');
      data = load(char(filename));
      [m, ci] = process(data(:, 1));
      dataM{agentId, rewardCand_, numOfQuery, numOfResponse_, trajLen_} = m;
      dataCI{agentId, rewardCand_, numOfQuery, numOfResponse_, trajLen_} = ci;
    end

    for numOfResponse = numOfResponses
      filename = strcat(agents(agentId), num2str(rewardCand_), '_', num2str(numOfQuery_), '_', num2str(numOfResponse), '_', num2str(trajLen_), '.out');
      data = load(char(filename));
      [m, ci] = process(data(:, 1));
      dataM{agentId, rewardCand_, numOfQuery_, numOfResponse, trajLen_} = m;
      dataCI{agentId, rewardCand_, numOfQuery_, numOfResponse, trajLen_} = ci;
    end

    for trajLen = trajLens
      filename = strcat(agents(agentId), num2str(rewardCand_), '_', num2str(numOfQuery_), '_', num2str(numOfResponse_), '_', num2str(trajLen), '.out');
      data = load(char(filename));
      [m, ci] = process(data(:, 1));
      dataM{agentId, rewardCand_, numOfQuery_, numOfResponse_, trajLen} = m;
      dataCI{agentId, rewardCand_, numOfQuery_, numOfResponse_, trajLen} = ci;
    end
  end

  markers = {'*-', '+-', 'x-', 's-'};

  for agentId = 1 : size(agents, 2)
    errorbar(rewardCandNums, cell2mat(dataM(agentId, rewardCandNums, numOfQuery_, numOfResponse_, trajLen_)),...
                             cell2mat(dataCI(agentId, rewardCandNums, numOfQuery_, numOfResponse_, trajLen_)), markers{agentId});
    hold on
  end
  legend('Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  xlabel('Number of Reward Candidates');
  ylabel('Q-Value');

  figure;
  for agentId = 1 : size(agents, 2)
    errorbar(numOfQueries, cell2mat(dataM(agentId, rewardCand_, numOfQueries, numOfResponse_, trajLen_)),...
                           cell2mat(dataCI(agentId, rewardCand_, numOfQueries, numOfResponse_, trajLen_)), markers{agentId});
    hold on
  end
  legend('Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  xlabel('Number of Queries');
  ylabel('Q-Value');

  figure;
  for agentId = 1 : size(agents, 2)
    errorbar(numOfResponses, cell2mat(dataM(agentId, rewardCand_, numOfQuery_, numOfResponses, trajLen_)),...
                             cell2mat(dataCI(agentId, rewardCand_, numOfQuery_, numOfResponses, trajLen_)), markers{agentId});
    hold on
  end
  legend('Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  xlabel('Number of Responses');
  ylabel('Q-Value');

  figure;
  for agentId = 1 : size(agents, 2)
    errorbar(trajLens, cell2mat(dataM(agentId, rewardCand_, numOfQuery_, numOfResponse_, trajLens)),...
                       cell2mat(dataCI(agentId, rewardCand_, numOfQuery_, numOfResponse_, trajLens)), markers{agentId});
    hold on
  end
  legend('Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  xlabel('Trajectory Lengths');
  ylabel('Q-Value');
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
