% *************************************************************************
% FILENAME:     webcam.m
%
% DESCRIPTION:	Example usage of webcam for exercises of Digital Image 
%               Processing.
%
% NOTES:        Type 'imaqhwinfo' to get informations about image 
%               acquisition adaptors available on the system.
%               The 'gentl' and 'winvideo' adaptors should be available.
%
% AUTHOR:       Mathias Hunold
%
% START DATE:	10 March 2017
%
% CHANGES:
%
% DATE      WHO                 DETAIL
% 10.03.17  Mathias Hunold      First published version.
% 18.09.17  Mathias Hunold      Rotate images by 180°
%
% *************************************************************************

if ~exist('vid_handler','var')                  % check if handler variable
                                                % already exists
	vid_handler = videoinput('gentl', 1);       % get a new video handler
else
    stop(vid_handler);                          % stop video capturing if
                                                % handler already exists
end

src = getselectedsource(vid_handler);           % get camera settings

triggerconfig(vid_handler, 'manual');           % configure 'manual'
start(vid_handler);                             % start capturing frames

e = [];
try                                             % catch the errors
	im = getsnapshot(vid_handler);              % get first image
    im = rot90(im,2);                           % rotate image by 180°

    % set up a new figure
	figure('NumberTitle','off','Name','Webcam','Color',[1 1 1])
	visualization_handle = imagesc(im,'Clipping','off');
	set(gca,'Position',[0 0 1 1],'Units','normalized')
	axis equal off;
	colormap(gray);

    % infinite loop to show video-frames
    while 1
        im = getsnapshot(vid_handler);          % get an image
        im = rot90(im,2);                       % rotate image by 180°

        %% ****** place your image processing code here :-) ******

        %% update visualization
        set(visualization_handle,'CData',im)    % set new image in figure

        drawnow                                 % show images immediately
    end

catch e                                         % error message is in 'e'
end

stop(vid_handler);                              % stop capturing frames

clear vid_handler

% check if there was an error and rethrow
% don't rethrow error if it is a Invalid-Handle-error: User closed figure
if ~isempty(e)
	if ~strcmp(e.identifier,'MATLAB:class:InvalidHandle')
		rethrow(e);
	end
end
