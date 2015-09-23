load("results");

disp("number of data")
size(results, 1)

disp("results")
sprintf('%.3f', mean(results))
sprintf('%.3f', 1.96 * std(results) / size(results, 1))

load("beliefs");

disp("beliefs")
sprintf('%.3f', mean(beliefs))
