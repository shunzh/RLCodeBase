function main()
  load MILP-SIMILAR3.out
  load MILP-SIMILAR3_2.out
  sim3 = MILP_SIMILAR3(:, 1);
  sim32 = MILP_SIMILAR3_2(:, 1);

  load MILP-SIMILAR-VARIATION3.out
  load MILP-SIMILAR-VARIATION3_2.out
  sim_naive3 = MILP_SIMILAR_VARIATION3(:, 1);
  sim_naive32 = MILP_SIMILAR_VARIATION3_2(:, 1);

  load MILP-SIMILAR-DISAGREE3.out
  load MILP-SIMILAR-DISAGREE3_2.out
  sim_disagree3 = MILP_SIMILAR_DISAGREE3(:, 1);
  sim_disagree32 = MILP_SIMILAR_DISAGREE3_2(:, 1);

  load MILP-SIMILAR-RANDOM3.out
  load MILP-SIMILAR-RANDOM3_2.out
  sim_rand3 = MILP_SIMILAR_RANDOM3(:, 1);
  sim_rand32 = MILP_SIMILAR_RANDOM3_2(:, 1);

  [ms3, cis3] = process(sim3)
  [mn3, cin3] = process(sim_naive3)
  [md3, cid3] = process(sim_disagree3)
  [mr3, cir3] = process(sim_rand3)

  [ms32, cis32] = process(sim32)
  [mn32, cin32] = process(sim_naive32)
  [md32, cid32] = process(sim_disagree32)
  [mr32, cir32] = process(sim_rand32)

  errorbar(1:2, [ms3, ms32], [cis3, cis32], '*-');
  hold on
  errorbar(1:2, [mn3, mn32], [cin3, cin32], '+-');
  errorbar(1:2, [md3, md32], [cid3, cid32], 'd-');
  errorbar(1:2, [mr3, mr32], [cir3, cir32], 'o--');
  legend('Query Projection', 'Belief Change', 'Disagree', 'Random Query');
  xlabel('Number of Queries');
  ylabel('Q Value');
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
