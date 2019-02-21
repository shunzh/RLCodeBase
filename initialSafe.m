best = fscanf(fopen('opt.out', 'r'), '%d');
data = [best, fscanf(fopen('iisAndRelpi.out', 'r'), '%d') - best, fscanf(fopen('iisOnly.out', 'r'), '%d') - best, fscanf(fopen('relpiOnly.out', 'r'), '%d') - best, fscanf(fopen('maxProb.out', 'r'), '%d') - best, fscanf(fopen('piHeu.out', 'r'), '%d') - best, fscanf(fopen('random.out', 'r'), '%d') - best];

[mean(data);
std(data) / sqrt(size(data, 1))]'
