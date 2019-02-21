data = [fscanf(fopen('opt.out', 'r'), '%d'), fscanf(fopen('iisAndRelpi.out', 'r'), '%d'), fscanf(fopen('iisOnly.out', 'r'), '%d'), fscanf(fopen('relpiOnly.out', 'r'), '%d'), fscanf(fopen('maxProb.out', 'r'), '%d'), fscanf(fopen('piHeu.out', 'r'), '%d'), fscanf(fopen('random.out', 'r'), '%d')];

[mean(data);
std(data) / sqrt(size(data, 1))]
