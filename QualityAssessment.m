function [feature]=QualityAssessment(Image,thresh)  
    % Quality Assessment Using Curvelete Transform
    % Image: input image
    % thresh: a threshold value for cropping the image
    % class: quality of the image
    % class=0 -> good quality, class=1 -> poor quality
    
        s=512;
        tic
        % crop the retinal image
        [mask,Image]=Cropping(Image,thresh);
        image=Image(:,:,2);
        image=double(imresize(image,[s,s]));
        % curvelet transform
        CC = fdct_wrapping(image,1,2,4);
        % use the coefficients of the secound subband
        coef=CC{2};
        C=[];
        % put the coefficents in a vector
        for i=1 : length(coef)
            temp=coef{i};      
            cc=temp(:);
            C=[C;cc];
        end
        % compute the histogram of the coefficents
        [a,b]=hist(C,-100:1:100);  
        a=a./sum(a);
        % compute the variance of coefficients frequency for each of the
        % images
%         feature=var(a);
        feature=[var(a),var(C),kurtosis(C),skewness(C)];
        toc
end