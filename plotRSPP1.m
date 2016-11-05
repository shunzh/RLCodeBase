function main()
  %agents = {'MILP-SIMILAR', 'MILP-SIMILAR-VARIATION', 'MILP-SIMILAR-DISAGREE', 'MILP-SIMILAR-RANDOM'};
  agents = {'MILP-SIMILAR', 'SIMILAR-VARIATION', 'SIMILAR-DISAGREE', 'SIMILAR-RANDOM'};

  % driving
  rewardCandNums = [5];
  numOfQueries = [1];
  numOfResponses = [2, 3];
  rewardVars = [1, 2, 3];

  % default values of variables
  rewardCand_ = 5;
  numOfQuery_ = 1;

  dataM = cell(size(agents, 2), max(rewardCandNums), max(numOfQueries), max(numOfResponses), max(rewardVars));
  for agentId = 1 : size(agents, 2)
    for numOfResponse = numOfResponses
      for rewardVar = rewardVars
        filename = strcat(agents(agentId), num2str(rewardCand_), '_', num2str(numOfQuery_), '_', num2str(numOfResponse), '_', num2str(rewardVar-1), '.out');
        data = load(char(filename));
        [m, ci] = process(data(:, 1));
        dataM{agentId, rewardCand_, numOfQuery_, numOfResponse, rewardVar} = m;
        dataCI{agentId, rewardCand_, numOfQuery_, numOfResponse, rewardVar} = ci;
      end
    end
  end

  markers = {'*-', '+-', 'x-', 's-'};

  for numOfResponse = numOfResponses
    figure;
    for agentId = 1 : size(agents, 2)
      errorbar(rewardVars, cell2mat(dataM(agentId, rewardCand_, numOfQuery_, numOfResponse, rewardVars)),...
                           cell2mat(dataCI(agentId, rewardCand_, numOfQuery_, numOfResponse, rewardVars)), markers{agentId});
      hold on
    end
  end
  legend('Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  xlabel('Variance in Reward Candidates');
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
