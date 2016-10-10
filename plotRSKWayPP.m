function main()
  load MILP-SIMILAR.out
  load MILP-SIMILAR-NAIVE.out

  sim = MILP_SIMILAR(:, 1);
  sim_naive = MILP_SIMILAR_NAIVE(:, 1);

  [m1, ci1] = process(sim)
  [m2, ci2] = process(sim_naive)
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
