function main()
  agents = {'MILP-POLICY', 'SAMPLE-POLICY', 'RAND-POLICY'};
  agentNames = {'Policy Grad.', 'Sampling', 'Random'};
  %agents = {'MILP-SIMILAR', 'SIMILAR-VARIATION', 'SIMILAR-DISAGREE'};
  %agentNames = {'Query Proj.', 'Belief Change', 'Disagree'};

  ms = [];
  tms = [];
  cis = [];
  tcis = [];
  for agent = agents
    filename = strcat(agent, '.out');
    d = load(char(filename));
    m = mean(d(:, 1));
    tm = mean(d(:, 2));
    ci = 1.96 * std(d(:, 1)) / sqrt(size(d, 1));
    tci = 1.96 * std(d(:, 2)) / sqrt(size(d, 1));

    ms = [ms, m];
    cis = [cis, ci];

    tms = [tms, tm];
    tcis = [tcis, tci];
  end

  errorbar(1:size(agents, 2), ms, cis, '+')
  ylabel('EVOI');

  set(gca, 'Xtick', 1:size(agents, 2), 'XtickLabel', agents);

  set(gcf,'PaperUnits','inches','PaperPosition',[0 0 5 3])
  ylim([0, 2.5]);
  print('-deps', [char(agents{1}), '.eps'], '-r100');

  %figure;
  %errorbar(1:size(agents, 2), tms, tcis, '+')
  %ylabel('Computation Time (sec.)');

  %set(gca, 'Xtick', 1:size(agents, 2), 'XtickLabel', agents);

  %set(gcf,'PaperUnits','inches','PaperPosition',[0 0 5 3])
  %print('-deps', [char(agents{1}), 't.eps'], '-r100');
end

