%% READ ME
% 先读工作区 再跑代码
% 最好一部分一部分的跑，因为第一部分很慢
% 最后会生成两个图，第一个图可以用来解释RF的hyperparameters，放在文章里的model development
% 第二个图是ROC curve （我不太会画这个，我直接用五折的结果画了五个点）
% 这个主体只有RF的development，SVM和ANN在另外一个文件的代码里，那个很乱，我没有附，需要的话告诉我
% RF在这里我用的TreeBagger实现的，用grid search调参（树的个数和每次分裂选择的特征个数）
% 我还给了一个jaya-RF的模型，如果可以的话跑一下，那个效果应该不错，但是很慢
% SVM是直接对比了三种核函数（最终结果是polynomial最好）
% 但并没有hyperparameters optimization（直接用的default设定），有的话不好解释，因为结果跟RF差不多
% SVM和ANN的结果在工作区里：
% 准确率叫Precision_SVM和precision_ann，F1score叫F1score_SVM和F1score_ANN，双击即可查看！！
%% 5-fold average accuracy of RF with different hyperparameters
% 用不同超参数组合训练RF
% 用5-fold CV的平均精度衡量RF在不同超参数组合下的表现
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
% 保存这个图，放在文章里，如果效果不好看出来，写个表格
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
% 之前我给的结果用的是200棵树，30变量参与分裂训练的
% 也许这个不是最好的，可以用grid search和5-fold CV之后获得的最佳模型再预测一遍
% classifyresult是分类结果，scores是概率（在工作区打开复制到Excel表里就行）
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