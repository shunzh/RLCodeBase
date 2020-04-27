function main()
  agents = {'MILP-POLICY', 'MILP-SIMILAR', 'SIMILAR-VARIATION', 'SIMILAR-DISAGREE', 'SIMILAR-RANDOM'};

  colors = [0.8500, 0.3250, 0.0980;
            0.9290, 0.6940, 0.1250;
            0.4940, 0.1840, 0.5560;
            0.4660, 0.6740, 0.1880;
            0.3010, 0.7450, 0.9330];

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

      [tm, tci] = process(data(:, 2));
      dataTM{agentId, rewardCand_, numOfQuery_, numOfResponse, trajLen_} = tm;
      dataTCI{agentId, rewardCand_, numOfQuery_, numOfResponse, trajLen_} = ci;
    end
  end

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
  set(gca, 'Xtick', 1:3, 'XtickLabel', {'2', '3', '4'});
  set(gcf,'PaperUnits','inches','PaperPosition',[0 0 4 3])
  print('-dpng', ['drive.png'], '-r100');

  d = squeeze(cell2mat(dataTM(agentIds, rewardCand_, numOfQuery_, numOfResponses, trajLen_)))'
  c = squeeze(cell2mat(dataTCI(agentIds, rewardCand_, numOfQuery_, numOfResponses, trajLen_)))'
  figure;
  ylim([0; Inf]);
  b = superbar(d, 'E', c, 'BarFaceColor', colors);

  %legend('Greedy q^*_\Pi', 'Query Projection', 'Belief Change', 'Disagreement', 'Random Query');
  xlabel('Number of Responses');
  ylabel('Time (sec.)');
  set(gca, 'Xtick', 1:3, 'XtickLabel', {'2', '3', '4'});
  set(gcf,'PaperUnits','inches','PaperPosition',[0 0 4 3])
  print('-dpng', ['driveT.png'], '-r100');
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
