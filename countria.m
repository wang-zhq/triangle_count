% function binfile_read

fid = fopen('soc-LiveJournal1.bin','rb');

A = fread(fid, inf, 'uint');

fclose(fid);

n = length(A);
S = A(1:2:n);
T = A(2:2:n);
EMX = sparse(S+1, T+1, ones(n/2,1));
% dim = max(A) + 1;

EMX = triu(min(EMX + EMX',1),1);
TRMX = EMX*EMX;
TRV = TRMX(EMX>0);
disp(sum(TRV))

% soc-LiveJournal1.bin    312369026
% s24.kron.edgelist   