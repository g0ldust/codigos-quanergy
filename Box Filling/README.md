Python version used: 3.12.7

Along with the libraries mentioned in requirements.txt, you will need to configure sensor's tof library to be used in the program.

This program detects a open box in its field of view and measures its dimensions, angle of orientation as well as updates its fill rate in real time. 
To start measuring the fill rate and to get the dimensions of the box, press 'f' key. Then, you will see the box's dimensions displayed and as you go on filling the box, its fill percentage will update.
Note that, do not change the box position after the 'f' key is pressed.


Steps to run the program:
- Run the main script "box_fill.py"
- Enter the serial number of the sensor after the programs starts running.
- There will be 3 opencv windows, one of which is 'mask'. Make sure you see box edges in white color properly
without much points missing so that edge detection is smooth.
- Make sure the dot at the center of the streaming window falls on the box's base.
- Press f to finalize box position so that the box alignment is final and the program can start getting the box dimensions and fill rate.
- Press n to reset and start detecting a new box.
- Press q to quit the program.