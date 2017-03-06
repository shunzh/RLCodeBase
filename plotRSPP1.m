function main()
  %agents = {'OPT-POLICY', 'MILP-POLICY', 'MILP-SIMILAR', 'SIMILAR-VARIATION', 'SIMILAR-DISAGREE', 'SIMILAR-RANDOM'};
  %agents = {'MILP-POLICY', 'MILP-SIMILAR', 'SIMILAR-VARIATION', 'SIMILAR-DISAGREE', 'SIMILAR-RANDOM'};
  agents = {'MILP-SIMILAR', 'SIMILAR-VARIATION', 'SIMILAR-DISAGREE', 'SIMILAR-RANDOM'};

  % driving
  agentIds = 1 : size(agents, 2);
  rewardVars = [1, 2, 3];

  % default values of variables
  numOfQuery_ = 1;
  rewardCand_ = 5;
  numOfResponse_ = 2;

  for agentId = 1 : size(agents, 2)
    for rewardVar = rewardVars
      filename = strcat(agents(agentId), num2str(rewardCand_), '_', num2str(numOfQuery_), '_', num2str(numOfResponse_), '_', num2str(rewardVar), '.out');
      data = load(char(filename));
      [m, ci] = process(data(:, 1));
      dataM{agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVar} = m;
      dataCI{agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVar} = ci;

      [tm, tci] = process(data(:, 2));
      dataTM{agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVar} = tm;
      dataTCI{agentId, rewardCand_, numOfQuery_, numOfResponse_, rewardVar} = ci;
    end
  end

  colors = repmat((1:6) / 7, 3, 1)';
  colors = permute(colors, [3 1 2]);

  d = squeeze(cell2mat(dataM(agentIds, rewardCand_, numOfQuery_, numOfResponse_, rewardVars)))';
  c = squeeze(cell2mat(dataCI(agentIds, rewardCand_, numOfQuery_, numOfResponse_, rewardVars)))';
  %d = [[0.05; 0.05; 0.05], d];
  %c = [[0; 0; 0], c];
  d
  c

  figure;
  ylim([0; Inf]);
  b = superbar(d, 'E', c, 'BarFaceColor', colors);

  set(gca, 'Xtick', rewardVars, 'XtickLabel', {'#1', '#2', '#3'});
  set(gcf,'PaperUnits','inches','PaperPosition',[0 0 4 3])
  print('-deps', ['rsn', num2str(rewardCand_), 'k', num2str(numOfResponse_), '.eps'], '-r100');


  d = squeeze(cell2mat(dataTM(agentIds, rewardCand_, numOfQuery_, numOfResponse_, rewardVars)))';
  c = squeeze(cell2mat(dataTCI(agentIds, rewardCand_, numOfQuery_, numOfResponse_, rewardVars)))';
  %d = [[0.05; 0.05; 0.05], d];
  %c = [[0; 0; 0], c];
  d
  c
  figure;
  ylim([0; Inf]);
  b = superbar(d, 'E', c, 'BarFaceColor', colors);

  set(gca, 'Xtick', rewardVars, 'XtickLabel', {'#1', '#2', '#3'});
  set(gcf,'PaperUnits','inches','PaperPosition',[0 0 4 3])
  print('-deps', ['rsn', num2str(rewardCand_), 'k', num2str(numOfResponse_), 't.eps'], '-r100');
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
