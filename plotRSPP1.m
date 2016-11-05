function main()
  %agents = {'MILP-SIMILAR', 'MILP-SIMILAR-VARIATION', 'MILP-SIMILAR-DISAGREE', 'MILP-SIMILAR-RANDOM'};
  agents = {'MILP-SIMILAR', 'SIMILAR-VARIATION', 'SIMILAR-DISAGREE', 'SIMILAR-RANDOM'};

  % driving
  rewardCandNums = [5];
  numOfQueries = [1];
  numOfResponses = [2, 3];
  rewardVars = [0, 0.5, 1.0];
  rewardVarIds = 1:3;

  % default values of variables
  rewardCand_ = 5;
  numOfQuery_ = 1;
  numOfResponse_ = 2;

  dataM = cell(size(agents, 2), max(rewardCandNums), max(numOfQueries), max(numOfResponses), max(rewardVarIds));
  for agentId = 1 : size(agents, 2)
    for rewardVarId = rewardVarIds
      filename = strcat(agents(agentId), num2str(rewardCand_), '_', num2str(numOfQuery_), '_', num2str(numOfResponse_), '_', num2str(rewardVars(rewardVarId)), '.out');
      data = load(char(filename));
      [m, ci] = process(data(:, 1));
      dataM{agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVarId} = m;
      dataCI{agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVarId} = ci;
    end
  end

  markers = {'*-', '+-', 'x-', 's-'};

  for agentId = 1 : size(agents, 2)
    errorbar(rewardVars, cell2mat(dataM(agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVarIds)),...
                         cell2mat(dataCI(agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVarIds)), markers{agentId});
    hold on
  end
  legend('Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  xlabel('Number of Responses');
  ylabel('Q-Value');

  %figure;
  %numOfQuery_ = 2;
  %for agentId = 1 : size(agents, 2)
  %  errorbar(numOfResponses, cell2mat(dataM(agentId, rewardCand_, numOfQuery_, numOfResponses, trajLen_)),...
  %                           cell2mat(dataCI(agentId, rewardCand_, numOfQuery_, numOfResponses, trajLen_)), markers{agentId});
  %  hold on
  %end
  %legend('Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  %xlabel('Number of Responses');
  %ylabel('Q-Value');
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
