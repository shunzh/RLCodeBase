function main()
  agents = {'MILP-POLICY', 'FEAT-GREEDY', 'FEAT-RANDOM'};

  % driving
  agentIds = 1 : size(agents, 2);
  rewardCandNums = [10, 50, 100];

  % default values of variables
  numOfResponse_ = 3;

  for agentId = 1 : size(agents, 2)
    for rewardCandNum = rewardCandNums
      if agentId == 1 && rewardCandNum == 100
        dataM{agentId, rewardCandNum} = 0;
        dataCI{agentId, rewardCandNum} = 0;
        dataTM{agentId, rewardCandNum} = 0;
        dataTCI{agentId, rewardCandNum} = 0;
      else
        filename = strcat(agents(agentId), num2str(rewardCandNum), '_', num2str(numOfResponse_), '.out');
        data = load(char(filename));
        [m, ci] = process(data(:, 1));
        dataM{agentId, rewardCandNum} = m;
        dataCI{agentId, rewardCandNum} = ci;

        [tm, tci] = process(data(:, 2));
        dataTM{agentId, rewardCandNum} = tm;
        dataTCI{agentId, rewardCandNum} = tci;
      end
    end
  end

  colors = repmat((1:3) / 4, 3, 1)';
  colors
  colors = permute(colors, [3 1 2]);
  colors

  d = squeeze(cell2mat(dataM(agentIds, rewardCandNums)))';
  c = squeeze(cell2mat(dataCI(agentIds, rewardCandNums)))';
  d
  c

  ylim([0; Inf]);
  b = superbar(d, 'E', c, 'BarFaceColor', colors);

  xlabel('Numbers of Reward Candidates')
  ylabel('EVOI');

  set(gca, 'Xtick', 1:3, 'XtickLabel', {'10', '50', '100'});
  set(gcf,'PaperUnits','inches','PaperPosition',[0 0 4 3])
  print('-deps', ['rs.eps'], '-r100');


  d = squeeze(cell2mat(dataTM(agentIds, rewardCandNums)))';
  c = squeeze(cell2mat(dataTCI(agentIds, rewardCandNums)))';
  d
  c
  figure;
  ylim([0; Inf]);
  b = superbar(d, 'E', c, 'BarFaceColor', colors);

  xlabel('Numbers of Reward Candidates')
  ylabel('Computation Time (sec.)');

  set(gca, 'Xtick', 1:3, 'XtickLabel', {'10', '50', '100'});
  set(gcf,'PaperUnits','inches','PaperPosition',[0 0 4 3])
  print('-deps', ['rst.eps'], '-r100');
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
