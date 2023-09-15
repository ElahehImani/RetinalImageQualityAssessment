function []=test()
    % a test function to assess the quality image
    clc
    clear all
    Path='E:\quality\';
    thresh=2;
    ImgType='png';
    Imgs = dir([Path '/' ['*.',ImgType]]);
    for i=1 : length(Imgs)
        image=imread([Path,Imgs(i).name]);
        quality(i,:)=QualityAssessment(image,thresh);
%         class(i)=Classify(quality(i,[1,3]));
        disp(i)
    end
    save('all','quality');
    % quality=0 -> good quality, quality=1 -> poor quality
end