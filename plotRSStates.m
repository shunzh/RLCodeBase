function main()
  system('scp $ut:/u/menie482/workspace/CMP/results.tar.gz .');
  system('tar xvzf results.tar.gz');

  clear all; close all;

  filename = 'rs_out';

  m0 = zeros(20, 20);
  m10 = zeros(20, 20);
  m20 = zeros(20, 20);

  for i=0:499
    try
      raw = load([filename, '.', num2str(i)]);
    catch
      disp(['Unable to load ', num2str(i)]);
    end

    r = 1;
    while raw(r,:) ~= [-1, -1]
      m0(raw(r,1)+1, raw(r,2)+1) = m0(raw(r,1)+1, raw(r,2)+1) + 1;
      r = r + 1;
    end
    r = r + 1;
    while raw(r,:) ~= [-1, -1]
      m10(raw(r,1)+1, raw(r,2)+1) = m10(raw(r,1)+1, raw(r,2)+1) + 1;
      r = r + 1;
    end
    r = r + 1;
    while raw(r,:) ~= [-1, -1]
      m20(raw(r,1)+1, raw(r,2)+1) = m20(raw(r,1)+1, raw(r,2)+1) + 1;
      r = r + 1;
    end
  end

  subplot(1,3,1);
  imagesc(log(m0' + 1));
  colormap bone;

  subplot(1,3,2);
  imagesc(log(m10' + 1));
  colormap bone;

  subplot(1,3,3);
  imagesc(log(m20' + 1));
  colormap bone;
end

