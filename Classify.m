function [class]=Classify(feature)
%   [classes]=Classify(feature) determine the quality of the image
%   feature: feature vector of the image
%   class: quality of the image
%   class=0 -> good quality image, class=1 -> poor quality image

    data=load('train');
    svm_struct=data.svm_struct;
    class = svmclassify(svm_struct,feature,'showplot',false);
end