function []=main()
    neuronNum=3;
    k=1;
    type=1; 
    count=0;
    
    abNormal=load('ungradable.mat');
    normal=load('gradable.mat');
    
    Abnormal=abNormal.Feature;
%     a=Abnormal<.002;
%     Abnormal(a)=[];
    Normal=normal.Feature;
    size_abnormal=size(Abnormal,2);
    size_normal=size(Normal,2);
    num=size_normal+size_abnormal;
    index=zeros(num,1);
    while(1)
        while(1)
            r = round(1 + (num-1).*rand(1,1));
            if(index(r)==0)
                index(r)=1;
                count=count+1;
                break;
            end
        end
        if(count==size_abnormal)
            break;
        end
    end
    
    [eval,sen,spe]=svm_test(logical(index),Abnormal,Normal,k,type);
    % ---------------- test svm with different paremeters -----------------
    k=[5,8,10,12,15];
    type=[1,2,3,4];
    for i=1 : length(k)
        for j=1 : length(type)
            [eval(i,j),sen(i,j),spe(i,j)]=svm_test(logical(index),Abnormal,Normal,k(i),type(j));
        end
    end
    R.eval=eval;
    R.sen=sen;
    R.spe=spe;
    R.k=k;
    R.type=type;
    
    save('svm_result','R')

end

function [eval,sen,spe]=NN_test(index,ab_class,norm_class,neuronNum,k,type)

    trainIn(index)=ab_class;
    trainIn(~index)=norm_class;
    trainIn=trainIn';
  
    trainOUT=[index,~index];
    count=round(size(trainIn,1)/k);
    temp=1; 
    eval=0;
    sen=0;c1=0;
    spe=0;c2=0;
    for i=1 : k
        index=zeros(size(trainIn,1),1);
        ind=temp:temp+count-1;
        if(ind(end)>size(index,1))
            ind=temp:size(index,1);
        end
        temp=temp+count;
        index(ind)=1;
        In=find(~index);
        testI=trainIn(ind,:);
        testO=trainOUT(ind,:);
        trainI=trainIn(In,:);
        trainO=trainOUT(In,:);
        [net,tr]=Train(trainI,trainO,neuronNum,type);
        y=round(sim(net,testI'));
        I1=(y==1);
        I=I1.*testO';
        eval=eval+sum(I(:))/size(I,2);
        
        [sensitivity,specificity]=evaluation(testO(:,1),I1(1,:)');
        if(~isnan(sensitivity))
            sen=sen+sensitivity;
            c1=c1+1;
        end
        if(~isnan(specificity))
            spe=spe+specificity;
            c2=c2+1;
        end
    end

    sen=sen/c1*100;
    spe=spe/c2*100;
    eval=(eval/k)*100; 
end


function [net,tr]=Train(trainIn,trainOUT,neuronNum,tt)
%      TFi - Transfer function of ith layer. Default is 'tansig' for
%              hidden layers, and 'purelin' for output layer.
%        BTF - Backprop network training function, default = 'trainlm'.
%        BLF - Backprop weight/bias learning function, default = 'learngdm'.
%        PF  - Performance function, default = 'mse'.
%        IPF - Row cell array of input processing functions.
%              Default is {'fixunknowns','remconstantrows','mapminmax'}.
%        OPF - Row cell array of output processing functions.
%              Default is {'remconstantrows','mapminmax'}.
%        DDF - Data division function, default = 'dividerand';
    if(nargin<4)
        TF='tansig';
        BTF='trainlm';
        BLF='learngdm';
        PF='mse';
        IPF={'fixunknowns','remconstantrows','mapminmax'};
        OPF={'remconstantrows','mapminmax'};
        DDF='dividerand';
    end
    T={'trainlm','trainbr','trainbfg','trainrp','trainscg','traincgb','traincgf','traincgp','trainoss','traingdx','traingdm','traingd'};
    net=newff(trainIn',trainOUT',neuronNum,{},T{tt});
    [net,tr] = train(net,trainIn',trainOUT');
end
 

function [eval,sen,spe]=svm_test(index,ab_class,norm_class,k,type)
   
    trainIn(index)=ab_class;
    trainIn(~index)=norm_class;
    trainOUT=index;
    trainIn=trainIn';
    count=round(size(trainIn,1)/k);
    temp=1; 
    eval=0;
    sen=0;
    spe=0;
    c1=0;
    c2=0;
    kernel={'linear','quadratic','polynomial','rbf','mlp'};
    
    for i=1 : k
        disp(i)
        index=zeros(size(trainIn,1),1);
        if(temp+count-1>size(trainIn,1))
            ind=temp:size(trainIn,1);
        else
            ind=temp:temp+count-1;
        end
        temp=temp+count;
        index(ind)=1;
        In=find(~index);
        testI=trainIn(ind,:);
        testO=trainOUT(ind,:);
        trainI=trainIn(In,:);
        trainO=trainOUT(In,:);
        svm_struct = svmtrain(trainI,trainO,'Kernel_Function',kernel{type},'showplot',true);
        classes = svmclassify(svm_struct,testI,'showplot',false);
        I=~xor(classes,testO);
        eval=eval+sum(I(:))/size(I,1);
        [sensitivity,specificity]=evaluation(testO,classes);
        if(~isnan(sensitivity))
            sen=sen+sensitivity;
            c1=c1+1;
        end
        if(~isnan(specificity))
            spe=spe+specificity;
            c2=c2+1;
        end
    end
    sen=sen/c1*100;
    spe=spe/c2*100;
    eval=(eval/k)*100;
end


function [sensitivity,specificity]=evaluation(label,result)
    In=find(label);
    res=result(In);
    sensitivity=sum(res)/length(res);
    In=find(~label);
    res=~result(In);
    specificity=sum(res)/length(res);
end