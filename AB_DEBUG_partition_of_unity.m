%%%AB_DEBUG
%%% THIS IS THE LATEST
function[Recon] = AB_DEBUG_partition_of_unity(Image, w_size, imagepath, imagename, makePatches)
%Outpath - output path for the patches. 
%Recon     - Default set to create patches. if True then does
%reconstruction from patches in the path folder
% has a slight bug for odd-sized images , comes from the padding of the border slices
% wasnt part of the original code
 
%Rec = zeros(size(Matrix));

sx = w_size(1);
sy = w_size(2);
        
[d1,d2] = size(Image);

if d1<=2*sx || d2<=2*sy
   %error 
end


k1 = floor(d1/(2*sx)); % numer of windows in x
k2 = floor(d2/(2*sy)); % numer of windows in y

r1 = d1-k1*2*sx; % rest in x
r2 = d2-k2*2*sy; % rest in y


off_x = floor(r1/2);
off_y = floor(r2/2);

%% Generate Windows

tmp = sin(linspace(0,pi/2,sx)).^2;

wx0 = [ones(1,floor(r1/2)) 1-tmp zeros(1,sx-floor(r1/2))];
wx1 = [tmp 1-tmp];
wx2 = [zeros(1,sx-ceil(r1/2)) tmp ones(1,r1-floor(r1/2))];

tmp = sin(linspace(0,pi/2,sy)).^2;
% 
wy0 = [ones(1,floor(r2/2)) 1-tmp zeros(1,sy-floor(r2/2))];
wy1 = [tmp 1-tmp];
wy2 = [zeros(1,sy-ceil(r2/2)) tmp ones(1,r2-floor(r2/2))];

Recon = zeros(size(Image));

for u=1:2*k1+1  ,%number of windows
    for v=1:2*k2+1, %number of windows
        if u==1,
            indi_x = 1:2*sx;
            wx = wx0;
        elseif u==2*k1+1,   
            indi_x = (d1-2*sx+1):d1;
            wx = wx2;
        else
            indi_x = off_x+(u-2)*sx+(1:2*sx);
            wx = wx1;
        end
        if v==1,
            indi_y = 1:2*sy;
            wy = wy0;
        elseif v==2*k2+1, 
            indi_y = (d2-2*sy+1):d2;
            wy = wy2;  
        else
            indi_y = off_y+(v-2)*sy+(1:2*sy);
            wy = wy1;
        end
        
        
        Window{u,v} = wx'*wy;
       
        if  makePatches == true
           
            Local_image{u,v} = Image(indi_x,indi_y); % Alternative .*Window^p
             disp ( ' Making patches. HERE');
             disp(size(Local_image{u,v}));
             %disp(max(Local_image{u,v}));
            %imwrite ( cat(3,Local_image{u,v},Local_image{u,v},Local_image{u,v}), fullfile ( imagepath,'patches', ['u-',num2str(u), '-v-', num2str(v), '_image.png']));
            
             %imwrite ( cat(3,Local_image{u,v},Local_image{u,v},Local_image{u,v}),fullfile (imagepath,[char(imagename), '-u-',num2str(u), '-v-', num2str(v), '_image.png']));
             fid = fopen(fullfile (imagepath,[char(imagename), '-u-',num2str(u), '-v-', num2str(v), '_image.dat']), 'w');
             fwrite(fid, single(Local_image{u,v}),'float32');
              fclose(fid);
            
            %imwrite ( zeros (  length ( indi_x), length ( indi_y)), fullfile ( imagepath,'patches', ['u-',num2str(u), '-v-', num2str(v), '_label.png']));
        else 
            % do the Transformation   %
            disp( 'Reconstruction from patches  CURRENT');
            %Local_image_temp = ((imread ( fullfile(imagepath, [num2str(u),'-',num2str(v),'_image.png']))));
            %Local_image_temp = ((imread ( fullfile(imagepath, ['u-',num2str(u), '-v-', num2str(v), '_image.dat']))));
            disp( fullfile(imagepath, [char(imagename),'-u-',num2str(u), '-v-', num2str(v), '_image.dat']))
            fid=fopen( fullfile(imagepath, [char(imagename), '-u-',num2str(u), '-v-', num2str(v), '_image.dat']));
            out=fread ( fid,'float32');
            out1 = reshape(out, 2*sx, 2*sy);
            Local_image{u,v}=out1;
            fclose(fid);
            %Local_image = im2double(imresize(Local_image,0.5));
            %Recon(indi_x,indi_y) = Recon(indi_x,indi_y)+Window{u,v}.*Local_image{u,v}; % Alternative .*Window.^(1-p)
            
            Recon(indi_x,indi_y) = Recon(indi_x,indi_y)+Window{u,v}.*Local_image{u,v}; % Alternative .*Window.^(1-p)
            
            %imagesc(Rec);
            %subplot(1,2,2);imagesc(A);
            %pause(0.01)
        end       
    end
end
%{
if makePatches == true
    [status, msg, msgID] = mkdir(fullfile (imagepath,'patches'));
    disp ( status)
    for u=1:2*k1+1  ,%number of windows
        for v=1:2*k2+1,
            %GaN OUT with concatenation
            imwrite ( cat(3,Local_image{u,v},Local_image{u,v},Local_image{u,v}), fullfile ( imagepath,'patches', [num2str(u), '-', num2str(v), '_image.png']));
            %imwrite (cat(3,Local_image{u,v},Local_image{u,v},Local_image{u,v}), fullfile ( 'D:\Arindam\garb\images\', [num2str(u), '-', num2str(v), '_image.png']));
            %imwrite(uint8(zeros(128,128)), fullfile ( 'D:\Arindam\garb\images\', [num2str(u), '-', num2str(v), '_label.png']));
            imwrite ( uint8(zeros(128,128)), fullfile ( imagepath,'patches', [num2str(u), '-', num2str(v), '_label.png']));
        end
    end
end
%}

disp ( 'done');
end


