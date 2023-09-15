function []=main()
path='..\dataset\';
out_path='..\results';
ImgType='png';
Imgs = dir([path '/' ['*.',ImgType]]);
s=512;
name=[];
for i=1 : length(Imgs)
    image=imread([path,Imgs(i).name]);
    [mask,image]=Cropping(image,3);
    image=image(:,:,2);
    image=double(imresize(image,[s,s]));
    tic
    feature(i,:,:)=QualityAssessment(image);
    toc
    name{i}=Imgs(i).name;
    save([out_path,'curvelet_method1_khatam'],'feature','name');
    disp(i)
end
end
%% -----------------------------------------------------
function [feature]=QualityAssessment(image)
CC = fdct_wrapping(image,1,2,4);
for j=1 : length(CC)
    coef=CC{2};
    C=[];
    for i=1 : length(coef)
        temp=coef{i};
        cc=temp(:);
        C=[C;cc];
    end
    [a,b]=hist(C,-25:1:25);
    a=a./sum(a);
    a([1,end])=[];
    feature(j,:)=[var(a),var(C),skewness(C),kurtosis(C)];
end
end