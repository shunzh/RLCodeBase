load('results');
load('beliefs');
load('time');

disp('number of data')
size(results, 1)

disp('results')
sprintf('%.3f', mean(results))
sprintf('%.3f', 1.96 * std(results) / sqrt(size(results, 1)))

disp('time')
sprintf('%.3f', mean(time))
sprintf('%.3f', 1.96 * std(time) / sqrt(size(time, 1)))
