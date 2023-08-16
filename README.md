# registration_icp_2d

A naive implementation if [iterative closest point](https://en.wikipedia.org/wiki/Iterative_closest_point). Unit test with two circles converges. Confirmed detected features from [feature_detector_fast](https://github.com/iwanders/feature_detector_fast) in two consecutive frames of a computer game converge to roughly equal the actual translation that occured between the two frames.

Most implementation time went into the [src/kdtree.rs](src/kdtree.rs) file, which is a neat single-container KD-tree with terminated leafs of up to n points. It finds exact nearest neighbour, it deduplicates points.


## License
License is `BSD-3-Clause`.
