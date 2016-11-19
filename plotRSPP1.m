function main()
  %agents = {'MILP-SIMILAR', 'MILP-SIMILAR-VARIATION', 'MILP-SIMILAR-DISAGREE', 'MILP-SIMILAR-RANDOM'};
  agents = {'OPT-POLICY', 'MILP-POLICY', 'MILP-SIMILAR', 'SIMILAR-VARIATION', 'SIMILAR-DISAGREE', 'SIMILAR-RANDOM'};

  % driving
  agentIds = 1 : size(agents, 2);
  rewardCandNums = [10];
  numOfQueries = [1];
  numOfResponses = [4];
  rewardVars = [1, 2, 3];

  % default values of variables
  numOfQuery_ = 1;

  rewardCand_ = 5;
  numOfResponse_ = 3;

  %agentName = 'MILP-POLICY';
  %optData = cell(size(rewardVars));
  %for rewardVar = rewardVars
  %  filename = strcat(agentName, num2str(rewardCand_), '_', num2str(numOfQuery_), '_', num2str(numOfResponse_), '_', num2str(rewardVar), '.out');
  %  optData{rewardVar} = load(char(filename));
  %end

  dataM = cell(max(agentIds), max(rewardCandNums), max(numOfQueries), max(numOfResponses), max(rewardVars));
  dataTM = cell(max(agentIds), max(rewardCandNums), max(numOfQueries), max(numOfResponses), max(rewardVars));
  for agentId = 1 : size(agents, 2)
    for rewardVar = rewardVars
      filename = strcat(agents(agentId), num2str(rewardCand_), '_', num2str(numOfQuery_), '_', num2str(numOfResponse_), '_', num2str(rewardVar), '.out');
      data = load(char(filename));
      [m, ci] = process(data(:, 1));
      dataM{agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVar} = m;
      %dataCI{agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVar} = ci;

      [tm, tci] = process(data(:, 2));
      dataTM{agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVar} = tm;
    end
  end

  markers = {'o--', '*-', '+-', 'x-', 's-'};

  %for agentId = 1 : size(agents, 2)
    %errorbar(rewardVars, cell2mat(dataM(agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVars)),...
    %                     cell2mat(dataCI(agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVars)), markers{agentId});

    % code that does not plot confidence intervals
    %errorbar(rewardVars, cell2mat(dataM(agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVars)),...
    %                     zeros(1, size(rewardVars, 2)), markers{agentId});
  %end

  d = squeeze(cell2mat(dataM(agentIds, rewardCand_, numOfQuery_, numOfResponse_, rewardVars)))'
  h = figure;
  b = bar(d);
  colormap(gray); 

  legend('Greedy q^*_\Pi', 'Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  xlabel('Reward Settings');
  %ylabel('Q-Value - Q-Value of Greedy Policy Query');
  ylabel('EPU');
  set(gca, 'Xtick', rewardVars, 'XtickLabel', {'#1', '#2', '#3'});

  d = squeeze(cell2mat(dataTM(agentIds, rewardCand_, numOfQuery_, numOfResponse_, rewardVars)))'
  h = figure;
  b = bar(d);
  colormap(gray); 

  legend('Greedy q^*_\Pi', 'Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  xlabel('Reward Settings');
  %ylabel('Q-Value - Q-Value of Greedy Policy Query');
  ylabel('Computation Time (sec.)');
  set(gca, 'Xtick', rewardVars, 'XtickLabel', {'#1', '#2', '#3'});
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
