function [precision,tpr,fpr]=c_pcs(tresult,ytest)
idx=find(tresult==0);
tp0=length(find(ytest(idx,:)==0));
fp0=length(find(ytest(idx,:)~=0));
pcs0=tp0/length(idx);
idx=find(tresult==1);
tp1=length(find(ytest(idx,:)==1));
fp1=length(find(ytest(idx,:)~=1));
pcs1=tp1/length(idx);
idx=find(tresult==2);
tp2=length(find(ytest(idx,:)==2));
fp2=length(find(ytest(idx,:)~=2));
tpr=(tp0+tp1+tp2)/(length(find(tresult==0))+length(find(tresult==1))+length(find(tresult==2)));
fpr=(fp0+fp1+fp2)/(length(find(tresult==0))+length(find(tresult==1))+length(find(tresult==2)));
pcs2=tp2/length(idx);
precision=(pcs0+pcs1+pcs2)/3;