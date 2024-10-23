# perfect-cmaps
This repository provides Python functionality for creating custom colormaps which are perfectly perceptually uniform.
The project came up as a result of there being to few easily accessible perceptually uniform colormaps, so I wanted to make a go at creating a good library where users can easily create their own. I wasn't quite happy with colormaps such as `viridis`, `plasma` or `magma` from `matplotlib`, so I started making some algorithmic colormaps. 

Here's an example of my colormap optimizer script, where you can draw your own colormap, and assign a lightness profile to it:


https://github.com/user-attachments/assets/2de077f7-776d-49b3-a8c0-564f42e4baed

These custom generated colormaps are theoretically perceptually uniform, but note that some screen settings may show colormap artifacts, making the colormaps appear less perceptually uniform. This may happen at lower lighting, for example.
