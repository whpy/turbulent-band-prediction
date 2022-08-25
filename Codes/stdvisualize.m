clear;
clc;

load("sparsedata678_2lowspeed.mat");
load("localcoordinate.mat");
load("subA.mat")
load("subPHI.mat")

figure(114)
reverse = zeros(size(grupred1000));
a = size(grupred1000);
for i =1:a(2)
    reverse(:,i) = (pv(2,i)-pv(1,i))*grupred1000(:,i)+pv(1,i);
end
for i =1:1000
figure(114)
contourf(xtmp,ztmp,reshape(meanfieldsparsedata6782+reverse(i,1:30)*subPHI(:,1:30)',63,130),50,'linestyle','none')
colormap(jet);
daspect([1 1 1]);
caxis([-0.35 0.35]);
xlabel(i)
end