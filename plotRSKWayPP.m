function main()
  load MILP-SIMILAR10_5.out
  load MILP-SIMILAR-VARIATION10_5.out
  load MILP-SIMILAR-DISAGREE10_5.out
  load MILP-SIMILAR-RANDOM10_5.out

  sim = MILP_SIMILAR10_5(:, 1);
  sim_var = MILP_SIMILAR_VARIATION10_5(:, 1);
  sim_dis = MILP_SIMILAR_DISAGREE10_5(:, 1);
  sim_rand = MILP_SIMILAR_RANDOM10_5(:, 1);

  [m1, ci1] = process(sim)
  [m2, ci2] = process(sim_var)
  [m2, ci2] = process(sim_dis)
  [m2, ci2] = process(sim_rand)
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
