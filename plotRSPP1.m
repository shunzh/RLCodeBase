function main()
  %agents = {'MILP-SIMILAR', 'MILP-SIMILAR-VARIATION', 'MILP-SIMILAR-DISAGREE', 'MILP-SIMILAR-RANDOM'};
  agents = {'MILP-POLICY', 'MILP-SIMILAR', 'SIMILAR-VARIATION', 'SIMILAR-DISAGREE', 'SIMILAR-RANDOM'};

  % driving
  rewardCandNums = [10];
  numOfQueries = [1];
  numOfResponses = [4];
  rewardVars = [1, 2, 3];

  % default values of variables
  rewardCand_ = 5;
  numOfQuery_ = 1;
  numOfResponse_ = 2;

  agentName = 'MILP-POLICY';
  optData = cell(size(rewardVars));
  for rewardVar = rewardVars
    filename = strcat(agentName, num2str(rewardCand_), '_', num2str(numOfQuery_), '_', num2str(numOfResponse_), '_', num2str(rewardVar), '.out');
    optData{rewardVar} = load(char(filename));
  end

  dataM = cell(size(agents, 2), max(rewardCandNums), max(numOfQueries), max(numOfResponses), max(rewardVars));
  for agentId = 1 : size(agents, 2)
    for rewardVar = rewardVars
      filename = strcat(agents(agentId), num2str(rewardCand_), '_', num2str(numOfQuery_), '_', num2str(numOfResponse_), '_', num2str(rewardVar), '.out');
      data = load(char(filename));
      %[m, ci] = process(data(:, 1) - optData{rewardVar}(:, 1));
      [m, ci] = process(data(:, 1));
      dataM{agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVar} = m;
      dataCI{agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVar} = ci;
    end
  end

  markers = {'*--', '*-', '+-', 'x-', 's-'};

  for numOfResponse = numOfResponses
    figure;
    for agentId = 1 : size(agents, 2)
      %errorbar(rewardVars, cell2mat(dataM(agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVars)),...
      %                     cell2mat(dataCI(agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVars)), markers{agentId});

      % code that does not plot confidence intervals
      errorbar(rewardVars, cell2mat(dataM(agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVars)),...
                           zeros(1, size(rewardVars, 2)), markers{agentId});
      hold on
    end
  end
  legend('Greedy q^*_\Pi', 'Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  xlabel('Reward Settings');
  %ylabel('Q-Value - Q-Value of Greedy Policy Query');
  ylabel('Q-Value');
  set(gca, 'Xtick', rewardVars, 'XtickLabel', {'#1', '#2', '#3'});
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
