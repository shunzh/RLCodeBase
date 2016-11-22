function main()
  agents = {'MILP-POLICY', 'MILP-SIMILAR', 'SIMILAR-VARIATION', 'SIMILAR-DISAGREE', 'SIMILAR-RANDOM'};

  % driving
  agentIds = 1 : size(agents, 2);
  rewardCandNums = [5];
  numOfQueries = [1];
  numOfResponses = [2, 3, 4];
  trajLens = [3];

  % default values of variables
  rewardCand_ = 5;
  numOfQuery_ = 1;
  trajLen_ = 4;

  for agentId = 1 : size(agents, 2)
    for numOfResponse = numOfResponses
      filename = strcat(agents(agentId), num2str(rewardCand_), '_', num2str(numOfQuery_), '_', num2str(numOfResponse), '_', num2str(trajLen_), '.out');
      data = load(char(filename));
      [m, ci] = process(data(:, 1));
      dataM{agentId, rewardCand_, numOfQuery_, numOfResponse, trajLen_} = m;
      dataCI{agentId, rewardCand_, numOfQuery_, numOfResponse, trajLen_} = ci;
    end
  end

  colors = repmat((1:5) / 6, 3, 1)';
  colors = permute(colors, [3 1 2]);

  %for agentId = 1 : size(agents, 2)
    %errorbar(numOfResponses, cell2mat(dataM(agentId, rewardCand_, numOfQuery_, numOfResponses, trajLen_)),...
    %                         cell2mat(dataCI(agentId, rewardCand_, numOfQuery_, numOfResponses, trajLen_)), markers{agentId});
    %errorbar(numOfResponses, cell2mat(dataM(agentId, rewardCand_, numOfQuery_, numOfResponses, trajLen_)),...
    %                         zeros(1, size(numOfResponses, 2)), markers{agentId});
    %hold on
  %end
  d = squeeze(cell2mat(dataM(agentIds, rewardCand_, numOfQuery_, numOfResponses, trajLen_)))'
  c = squeeze(cell2mat(dataCI(agentIds, rewardCand_, numOfQuery_, numOfResponses, trajLen_)))'
  figure;
  ylim([0; Inf]);
  b = superbar(d, 'E', c, 'BarFaceColor', colors);

  %legend('Greedy q^*_\Pi', 'Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  xlabel('Number of Responses');
  ylabel('EVOI');
  set(gcf,'PaperUnits','inches','PaperPosition',[0 0 4 3])
  print('-deps', ['drive.eps'], '-r100');
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
