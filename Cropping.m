function [mask,OUTimage]=Cropping(image,thresh)
    % Crop the retinal images so as to delete unuseful information of the
    % retinal images
    % thresh= threshold value for cropping retinal image
    % [mask,image]=Cropping(image)
    % mask=the mask of retinal image, OUTimage=the cropped image

    % create binary image
    mask=image(:,:,2)>thresh;
    % delete the noisy region on background and forground
    se=strel('disk',15);
    mask=imopen(mask,se);
    se=strel('disk',15);
    mask=imclose(mask,se);
    % find the boundingBox of the retina region
    [row,col]=find(mask);
    r1=min(row);
    r2=max(row);
    c1=min(col);
    c2=max(col);
    
    mask=mask(r1(1):r2(1),c1(1):c2(1));
    OUTimage=image(r1(1):r2(1),c1(1):c2(1),:);
 end