function main()
  system('scp $ut:/u/menie482/workspace/CMP/results.tar.gz .');
  system('tar xvzf results.tar.gz');

  clear all; close all;

  filename = 'rs_out';

  m = {};
  for j = 1:5
    m{j} = [];
  end

  for i=0:499
    try
      raw = load([filename, '.', num2str(i)]);
    catch
      disp(['Unable to load ', num2str(i)]);
    end

    j = 1;
    r = 1;
    while j <= 5
      while raw(r) ~= -1
        m{j}(end + 1) = raw(r);
        r = r + 1;
      end
      r = r + 1;
      j = j + 1;
    end
  end

  for j = 1:5
    j, mean(m{j}), std(m{j})
  end
end

