function recall=c_rec(tresult,ytest)
idx=find(tresult==0);
tp0=length(find(ytest(idx)==0));
rec0=tp0/length(find(ytest==0));
idx=find(tresult==1);
tp1=length(find(ytest(idx)==1));
rec1=tp1/length(find(ytest==1));
idx=find(tresult==2);
tp2=length(find(ytest(idx)==2));
rec2=tp2/length(find(ytest==2));
recall=(rec0+rec1+rec2)/3;

