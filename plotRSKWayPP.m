function main()
  %load OPT-POLICY.out
  load MILP-SIMILAR.out
  load MILP-SIMILAR-NAIVE.out

  sim = MILP_SIMILAR(:, 1);
  sim_naive = MILP_SIMILAR_NAIVE(:, 1);

  [m, ci] = process(sim)
  [m, ci] = process(sim_naive)
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
