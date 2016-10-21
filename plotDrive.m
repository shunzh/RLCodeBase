function main()
  load OPT-POLICY3.out
  load MILP-SIMILAR1.out
  load MILP-SIMILAR-NAIVE1.out
  load MILP-SIMILAR2.out
  load MILP-SIMILAR-NAIVE2.out
  load MILP-SIMILAR3.out
  load MILP-SIMILAR-NAIVE3.out
  load MILP-SIMILAR4.out
  load MILP-SIMILAR-NAIVE4.out
  load MILP-SIMILAR5.out
  load MILP-SIMILAR-NAIVE5.out

  opt = OPT_POLICY3(:, 1);

  sim2 = MILP_SIMILAR2(:, 1);
  sim_naive2 = MILP_SIMILAR_NAIVE2(:, 1);
  sim3 = MILP_SIMILAR3(:, 1);
  sim_naive3 = MILP_SIMILAR_NAIVE3(:, 1);
  sim4 = MILP_SIMILAR4(:, 1);
  sim_naive4 = MILP_SIMILAR_NAIVE4(:, 1);
  sim5 = MILP_SIMILAR5(:, 1);
  sim_naive5 = MILP_SIMILAR_NAIVE5(:, 1);

  [m12, ci12] = process(sim2 - opt)
  [m13, ci13] = process(sim3 - opt)
  [m14, ci14] = process(sim4 - opt)
  [m15, ci15] = process(sim5 - opt)
  [m22, ci22] = process(sim_naive2 - opt)
  [m23, ci23] = process(sim_naive3 - opt)
  [m24, ci24] = process(sim_naive4 - opt)
  [m25, ci25] = process(sim_naive5 - opt)

  errorbar(2:5, [m12, m13, m14, m15], [ci12, ci13, ci14, ci15], '*-');
  hold on
  errorbar(2:5, [m22, m23, m24, m25], [ci22, ci23, ci24, ci25], '+--');
  legend('Our Heuristic', 'Welson et al. 2012');
  xlabel('Length of Trajectory');
  ylabel('Q Value - Q Value of Optimal Query');
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
