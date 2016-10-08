function main()
  load OPT-POLICY.out
  load MILP-DEMO-BATCH.out
  load MILP-DEMO.out

  demo = MILP_DEMO(:, 1) - OPT_POLICY(:, 1);
  demo_batch = MILP_DEMO_BATCH(:, 1) - OPT_POLICY(:, 1);

  [m, ci] = process(demo)
  [m, ci] = process(demo_batch)
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end
