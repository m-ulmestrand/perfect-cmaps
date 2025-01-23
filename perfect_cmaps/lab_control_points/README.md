In this directory you will find control points stored in LAB format.

Points will be stored with their lightness profile the colormap was created with, but can easily be reformatted to other profiles by changing the L-channel accordingly. This is done by changing the "lightness" argument in `get_colormap()`, or giving a different lightness profile in the main script with command line arguments. 

Note that it is not always possible to change lightness profile, or you may get suboptimal results, since colormaps are usually optimized for the lightness profile they are saved with. 
