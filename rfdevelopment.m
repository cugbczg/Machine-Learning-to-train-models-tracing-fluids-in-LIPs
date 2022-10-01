%% READ ME
% �ȶ������� ���ܴ���
% ���һ����һ���ֵ��ܣ���Ϊ��һ���ֺ���
% ������������ͼ����һ��ͼ������������RF��hyperparameters�������������model development
% �ڶ���ͼ��ROC curve ���Ҳ�̫�ử�������ֱ�������۵Ľ����������㣩
% �������ֻ��RF��development��SVM��ANN������һ���ļ��Ĵ�����Ǹ����ң���û�и�����Ҫ�Ļ�������
% RF���������õ�TreeBaggerʵ�ֵģ���grid search���Σ����ĸ�����ÿ�η���ѡ�������������
% �һ�����һ��jaya-RF��ģ�ͣ�������ԵĻ���һ�£��Ǹ�Ч��Ӧ�ò������Ǻ���
% SVM��ֱ�ӶԱ������ֺ˺��������ս����polynomial��ã�
% ����û��hyperparameters optimization��ֱ���õ�default�趨�����еĻ����ý��ͣ���Ϊ�����RF���
% SVM��ANN�Ľ���ڹ������
% ׼ȷ�ʽ�Precision_SVM��precision_ann��F1score��F1score_SVM��F1score_ANN��˫�����ɲ鿴����
%% 5-fold average accuracy of RF with different hyperparameters
% �ò�ͬ���������ѵ��RF
% ��5-fold CV��ƽ�����Ⱥ���RF�ڲ�ͬ����������µı���
numT=[1,50,100,150,200,250,300];numV=[1,10,15,20,25,30,36];accs=[];
for i=1:7
    for j=1:7
        prc=0;
        for m=1:5
            rf=TreeBagger(numT(i),traind{m,1}(:,1:end-1),traind{m,1}(:,end),'NumPredictorsToSample',numV(j));
            tresult=predict(rf,test{m,1}(:,1:end-1));
            tresult=cell2mat(tresult);
            tresult=str2num(tresult);
            prc=prc+length(find((tresult-test{m,1}(:,end))==0))/length(tresult);
        end
        accs=[i,j,prc/5];
    end
end
%% draw a 3-D plot of grid research result of RF
% �������ͼ��������������Ч�����ÿ�������д�����
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
optT=numT(accs(idx,1));
optV=numV(accs(idx,2));
%% 5-fold of optimum RF
precisions=[];F1scores=[];TPRRF=[];FPRRF=[];RFs=cell(5,1);
% TPRRF is the true positive rate of RF 
% FPRRF is the false positive rate
for i=1:5
    rf=TreeBagger(optT,traind{i,1}(:,1:end-1),traind{i,1}(:,end),'NumPredictorsToSample',optV);
    RFs{i,1}=rf;
    tresult=predict(rf,test{i,1}(:,1:end-1));
    tresult=cell2mat(tresult);
    tresult=str2num(tresult);
    prc=length(find((tresult-test{i,1}(:,end))==0))/length(tresult);
    precisions=[precisions;prc];
    [pc,tp,fp]=c_pcs(tresult,test{i,1}(:,end));
    rec=c_rec(tresult,test{i,1}(:,end));
    f1s=2*pc*rec/(rec+pc);
    F1scores=[F1scores;f1s];
    TPRRF=[TPRRF,tp];
    FPRRF=[FPRRF,fp];
end
disp('5-fold precisions and F1-Scores');
disp([100*precisions,F1scores]);
%% Predict Unlabelled Data Using optimum RF
% ֮ǰ�Ҹ��Ľ���õ���200������30�����������ѵ����
% Ҳ�����������õģ�������grid search��5-fold CV֮���õ����ģ����Ԥ��һ��
% classifyresult�Ƿ�������scores�Ǹ��ʣ��ڹ������򿪸��Ƶ�Excel������У�
optidx=find(precisions==max(precisions));
if length(optidx)>1
    optidx=optidx(find(F1scores(optidx)==max(F1scores(optidx))));
    if length(optidx)>1
        optidx=optidx(1);
    end
end
optimumRF=RFs{optidx,1};% Find the best RF in 5-fold CV as final model
[classifyresult,scores]=predict(optimumRF,ulbdata);% Predict the unlabelled data using final RF
%% Draw ROC curves of ANN SVM and RF
[FPRANN,IDX]=sort(FPRANN);
TPRANN=TPRANN(IDX);
[FPRSVM,IDX]=sort(FPRSVM);
TPRSVM=TPRSVM(IDX);
figure (1)
hold on
plot([0,FPRRF,1],[0,TPRRF,1],'-r','linewidth',1);
hold on
plot([0,FPRSVM',1],[0,TPRSVM',1],'-g','linewidth',1);
hold on
plot([0,FPRANN',1],[0,TPRANN',1],'-b','linewidth',1);
legend('RF','SVM','DNN');
xlabel('False Positive Rate');
ylabel('True Positive Rate');