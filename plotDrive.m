function main()
  load MILP-SIMILAR2.out
  load MILP-SIMILAR-NAIVE2.out
  load MILP-SIMILAR3.out
  load MILP-SIMILAR-NAIVE3.out
  load MILP-SIMILAR4.out
  load MILP-SIMILAR-NAIVE4.out
  load MILP-SIMILAR5.out
  load MILP-SIMILAR-NAIVE5.out

  sim2 = MILP_SIMILAR2(:, 1);
  sim_naive2 = MILP_SIMILAR_NAIVE2(:, 1);
  sim3 = MILP_SIMILAR3(:, 1);
  sim_naive3 = MILP_SIMILAR_NAIVE3(:, 1);
  sim4 = MILP_SIMILAR4(:, 1);
  sim_naive4 = MILP_SIMILAR_NAIVE4(:, 1);
  sim5 = MILP_SIMILAR5(:, 1);
  sim_naive5 = MILP_SIMILAR_NAIVE5(:, 1);

  [m12, ci1] = process(sim2)
  [m13, ci1] = process(sim3)
  [m14, ci1] = process(sim4)
  [m15, ci1] = process(sim5)
  [m22, ci2] = process(sim_naive2)
  [m23, ci2] = process(sim_naive3)
  [m24, ci2] = process(sim_naive4)
  [m25, ci2] = process(sim_naive5)

  plot([2, 3, 4, 5], [m12, m13, m14, m15]);
  hold on
  plot([2, 3, 4, 5], [m22, m23, m24, m25]);

  legend('sim', 'naive');

  %errorbar([0, 1], [m1, m2], [ci1, ci2], '+')
  %set(gca, 'Xtick', [0, 1], 'XtickLabel', {'Our Heuristic', 'Welson et al. 2012'})
  %ylabel('Q Value')
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
