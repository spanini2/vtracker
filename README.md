# vtracker
A CV+DL Algorithm for detecting/highlighting volleyballs for the visually impaired 🏐

This project uses a HSV color picker to identify objects that are similar to the color of the ball. It then uses a binary classifier to detect the balls and non-balls. Furthermore, it uses a heuristic algorithm in order to choose the most viable option for a ball if there are multiple detected.

&nbsp;
<div align="center"> <i>A visual representation of the HSV pipeline, left is original frame, middle is only limited yellow colors, right is contours detected </i> </div>
&nbsp;
<img src="/hsv_filter_example_image.png" alt="">
