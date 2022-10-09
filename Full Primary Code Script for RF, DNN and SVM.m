%% read training data
data1=xlsread('S1.xls');
data1=[data1(:,28:end),data1(:,2)];
data2=xlsread('S2.xlsx');
data2=[data2(:,end-35:end),ones(length(data2(:,1)),1)];
idx=find(data1(:,end)==2);
data3=data1(idx,:);
data1(idx,:)=[];
idx=find(data1(:,end)==1);
data2=[data2;data1(idx,:)];
data1(idx,:)=[];
alldata=[data1;data2;data3];
%% data preprocessing
idx=find(alldata(:,1)<45); 
alldata(idx,:)=[];
idx=find(alldata(:,1)>53);
alldata(idx,:)=[];
L0=length(find(alldata(:,end)==0));%No. of samples labelled with '0' after cleaning
L1=length(find(alldata(:,end)==1));%No. of samples labelled with '1' after cleaning
L2=length(find(alldata(:,end)==2));%No. of samples labelled with '2' after cleaning
data1=alldata(1:L0,:);
data2=alldata(L0+1:L1+L0,:);
data3=alldata(L1+L0+1:L2+L1+L0,:);
xmin=[];xmax=[];
for i=1:36
    xmin=[xmin,min(alldata(:,i))];
    xmax=[xmax,max(alldata(:,i))];
end
%% randomly split the data into 5 fold training & testing dataset
k=5;
traind=cell(k,1);test=cell(k,1);
for i=1:k
    idx1=randperm(L0);
    idx2=randperm(L1);
    idx3=randperm(L2);
    traind{i,1}=[data1(idx1(1:4990),:);data2(idx2(1:845),:);data3(idx3(1:2663),:)];
    test{i,1}=[data1(idx1(4991:end),:);data2(idx2(845:end),:);data3(idx3(2664:end),:)];
end
%% DNN with 5 fold Cross-Validation
mdl=cell(k,1);precision_ann=[];
for i=1:k
    xtrain=(traind{i,1}(:,1:end-1)-xmin)./(xmax-xmin);%normalization
    xtest=(test{i,1}(:,1:end-1)-xmin)./(xmax-xmin);%normalization
    result=[ones(4990,1),zeros(4990,2);zeros(845,1),ones(845,1),zeros(845,1);zeros(2663,2),ones(2663,1)];
    net=patternnet([50,100,50,25]);
    net=train(net,xtrain',result');% DNN training
    mdl{i,1}=net;
    result=net(xtest');
    result=vec2ind(result);
    result=result'-1;
    acc=length(find((result-test{i,1}(:,end))==0))/length(result);
    precision_ann=[precision_ann;acc];
end
optimum_ann=mdl{find(precision_ann==max(precision_ann)),1};
%% 5-fold cross-validation of SVM with different kernel function
P_SVM=[];%Precision of SVM with different kernel in each fold validation
for i=1:k
    xtrain=(traind{i,1}(:,1:end-1)-xmin)./(xmax-xmin);%normalization
    xtest=(test{i,1}(:,1:end-1)-xmin)./(xmax-xmin);%normalization
    t=templateSVM('KernelFunction','linear');
    svm=fitcecoc(xtrain,traind{i,1}(:,end),'Learners',t);%SVM training
    tresult=predict(svm,xtest);
    p1=length(find((tresult-test{i,1}(:,end))==0))/length(tresult);
    t=templateSVM('KernelFunction','polynomial');
    svm=fitcecoc(xtrain,traind{i,1}(:,end),'Learners',t);%SVM training
    tresult=predict(svm,xtest);
    p2=length(find((tresult-test{i,1}(:,end))==0))/length(tresult);
    t=templateSVM('KernelFunction','gaussian');
    svm=fitcecoc(xtrain,traind{i,1}(:,end),'Learners',t);%SVM training
    tresult=predict(svm,xtest);
    p3=length(find((tresult-test{i,1}(:,end))==0))/length(tresult);
    P_SVM=[P_SVM;p1,p2,p3];
end
%% 5-fold average accuracy of RF with different hyperparameters combination
numT=[50,100,150,200,250,300];numV=[10,15,20,25,30,36];accs=[];
for i=1:6
    for j=1:6
        prc=0;
        for m=1:5
            rf=TreeBagger(numT(i),traind{m,1}(:,1:end-1),traind{m,1}(:,end),'NumPredictorsToSample',numV(j));
            tresult=predict(rf,test{m,1}(:,1:end-1));
            tresult=cell2mat(tresult);
            tresult=str2num(tresult);
            prc=prc+length(find((tresult-test{m,1}(:,end))==0))/length(tresult);
        end
        accs=[accs;i,j,prc/5];
    end
end
%% draw a 3-D plot of grid search result of RF
[X,Y]=meshgrid(numT,numV);
[a,b]=size(X);
mesh(X,Y,reshape(accs(:,3),a,b));
xlabel('No.of Trees');ylabel('No.of Variables Selected');zlabel('Precision');
%% Show the average precision of optimum RF and its optimum hyperparameters
disp('average precision of RF using optimum hyperparameters');
disp(min(accs(:,3)));
idx=find(accs(:,3)==min(accs(:,3)));
disp('optimum no. of tree and no. of variable');
disp([numT(accs(idx,1)),numV(accs(idx,2))]);