function main()
  system('scp $ut:/u/menie482/workspace/CMP/results.tar.gz .');
  system('tar xvzf results.tar.gz');

  clear all; close all;

  filename = 'rsStates_out';

  m = {};
  for j = 1:5
    m{j} = zeros(21, 21);
  end

  for i=0:299
    try
      raw = load([filename, '.', num2str(i)]);
    catch
      disp(['Unable to load ', num2str(i)]);
    end

    j = 1;
    r = 1;
    while j <= 5
      if r > size(raw, 1)
        break
      end
      while raw(r,:) ~= [-1, -1]
        if r > size(raw, 1)
          break
        end
        m{j}(raw(r,1)+1, raw(r,2)+1) = m{j}(raw(r,1)+1, raw(r,2)+1) + 1;
        r = r + 1;
      end
      r = r + 1;
      j = j + 1;
      if r > size(raw, 1)
        break
      end
    end
  end

  for j = 1:5
     subplot(1,5,j);
     imagesc(log(m{j}' + 1));
     colormap bone;
  end

  %imagesc(log(m{3}' + 1));
  %colormap bone;
end

