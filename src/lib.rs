/*

Two sets of points:
    A and B
    B be the stable; Base.

So, in minimal form;

1. Pick n points from A.
    - For each point from A, pick closest point in B.
2. Determine transformation to map paired points from A to B.
3. Combine transformations, move point cloud A by B.
4. Repeat.

*/

pub mod nearest_neighbour;
