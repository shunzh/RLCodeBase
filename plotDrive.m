function main()
  load MILP-SIMILAR.out
  load MILP-SIMILAR-NAIVE.out

  sim = MILP_SIMILAR(1:30, 1);
  sim_naive = MILP_SIMILAR_NAIVE(1:30, 1);

  [m1, ci1] = process(sim)
  [m2, ci2] = process(sim_naive)

  errorbar([0, 1], [m1, m2], [ci1, ci2], '+')
  set(gca, 'Xtick', [0, 1], 'XtickLabel', {'Our Heuristic', 'Welson et al. 2012'})
  ylabel('Q Value')
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
