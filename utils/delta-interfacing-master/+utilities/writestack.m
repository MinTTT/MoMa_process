function writestack(fname,tiff_stack)

%write a Tiff file, appending each image as a new page
for ii = 1 : size(tiff_stack, 4)
    imwrite(tiff_stack(:,:,:,ii) , fname , 'WriteMode' , 'append') ;
end