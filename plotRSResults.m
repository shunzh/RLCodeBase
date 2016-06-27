function main()
  load RQ.out;
  load JQTP.out;
  load AS.out;
  load MILP.out;

  rq = mean(RQ)
  e = mean(JQTP)
  as = mean(AS)
  milp = mean(MILP)
end

function [m, ci] = process(data)
  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);
end

