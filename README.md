# peoplecounter
Counting the number people entering and exiting a boundary

References :
https://www.youtube.com/watch?v=OS5qI9YBkfk
https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-and-count-vehicles-with-yolov8.ipynb?ref=blog.roboflow.com

1. Install Ultralytics
> pip install ultralytics

2. Install ByteTrack
git clone https://github.com/ifzhang/ByteTrack.git

cd /ByteTrack

pip3 install -q -r requirements.txt

python3 setup.py -q develop

pip install -q cython_bbox

pip install -q onemetric

3. Modify Code from "supervision" package - tools - line_counter.py
Centroid tracking - Under class LineCounter, edit "update" function
 - modify the boundary boxes of the object to just a smaller bounding box about the centroid
 - xmid = (x1 + x2)/2
 - ymid = (y1 + y2)/2
 - anchors =    Point(x= xmid+1, y= ymid+1),
                Point(x= xmid-1, y= ymid+1),
                Point(x= xmid+1, y=ymid-1),
                Point(x=xmid-1, y=ymid-1)
Calculate net flow - edit "annotate" function
 - modify one of the display boxes from line_counter.in_count or line_counter.out_count to line_counter.in_count - line_counter.out_count 
 - > in_text = f"net(in-out): {line_counter.in_count - line_counter.out_count}"

4. trackingstream.py
 - input stream path
 - resolution of images
 - Boundary Line (x: pixels from left, y: pixels from top


