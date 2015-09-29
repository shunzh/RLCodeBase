arg_list = argv();
filename = ["results", arg_list{1}];

results = load(filename);

disp("number of data")
size(results, 1)

disp("results")
sprintf('%.3f', mean(results))
sprintf('%.3f', 1.96 * std(results) / size(results, 1))
