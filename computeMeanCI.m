function [m, ci] = mci(filename)
  data = load(filename);

  n = size(data, 1);
  m = mean(data);
  ci = 1.96 * std(data) / sqrt(n);

  fprintf('%.3f \\pm %.3f', m, ci)
end
