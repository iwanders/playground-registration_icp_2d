/*
    Ok, so we need something that will give us the (approximate) nearest neighbour.

    - Fast-ish.
    - No need for delete or insert.
*/

pub trait Scalar: std::cmp::PartialOrd + Default + std::marker::Copy + std::fmt::Debug {}
impl<T> Scalar for T where
    T: std::cmp::PartialOrd
        + Default
        + std::marker::Copy
        + std::fmt::Debug
        + std::ops::Sub
        + std::ops::Add
{
}

trait Necessary<T: Scalar> {
    fn div(&self, other: T) -> T;
    fn sub(&self, other: T) -> T;
    fn add(&self, other: T) -> T;
    fn mult(&self, other: T) -> T;
    fn min(&self, other: T) -> T;
    fn max(&self, other: T) -> T;
    // fn sqrt(&self) -> T;
    fn MAX() -> T;
    fn MIN() -> T;
    fn two() -> T;
}
impl Necessary<f32> for f32 {
    fn div(&self, other: f32) -> f32 {
        *self / other
    }
    fn sub(&self, other: f32) -> f32 {
        *self - other
    }
    fn add(&self, other: f32) -> f32 {
        *self + other
    }
    fn mult(&self, other: f32) -> f32 {
        *self * other
    }
    fn min(&self, other: f32) -> f32 {
        f32::min(*self, other)
    }
    fn max(&self, other: f32) -> f32 {
        f32::max(*self, other)
    }
    // fn sqrt(&self) -> f32 {
        // f32::sqrt(*self)
    // }
    fn two() -> f32 {
        2.0
    }
    fn MAX() -> f32 {
        f32::MAX
    }
    fn MIN() -> f32 {
        f32::MIN
    }
}

#[derive(Debug)]
pub enum Node<const D: usize, T: Scalar> {
    Split {
        /// Index to elements below of this pivot.
        left: usize,
        /// Pivot coordinate, an actual point.
        pivot: [T; D],
        /// Pivot dimension,
        dim: usize,
        /// Index to elements above or equal to this pivot.
        right: usize,
    },
    Points {
        points: Vec<[T; D]>, // could be a smallvec.
    },
    Placeholder,
}

fn distance<const D: usize, T: Scalar + Necessary<T>>(a: &[T; D], b: &[T; D]) -> T {
    let mut sum = T::default();
    for d in 0..D {
        let n = a[d].sub(b[d]);
        let nn = n.mult(n);
        sum = sum.add(nn);
    }
    sum
}

#[derive(Copy, Debug, Clone)]
struct BoundingBox<const D: usize, T: Scalar + Necessary<T>>{
    min: [T; D],
    max: [T; D],
}

impl<const D: usize, T: Scalar + Necessary<T>> BoundingBox<{ D }, T> {
    pub fn everything() -> Self {
        BoundingBox {
            min: [T::MIN(); D],
            max: [T::MAX(); D],
        }
    }

    /*
    pub fn corner(&self, index: usize) -> [T; D] {
        let mut corner = self.min;
        let mut index = index;

        let mut dim = 0;
        while index != 0 {
            if index & 1 == 0 {
                corner[dim] = self.min[dim];
            } else if index & 1 == 1 {
                corner[dim] = self.max[dim];
            }
            dim += 1;
            index = index >> 1;
        }
        corner
    }
    */

    pub fn min_norm(&self, p: &[T; D]) -> T {
        let mut best_point = self.min;
        for d in 0..D {
            best_point[d] = self.min[d].max(p[d].min(self.max[d]));
        }
        distance(&best_point, p)
    }

    pub fn split(&self, d: usize, value: T) -> (Self, Self) {
        let mut left = *self;
        let mut right = *self;
        left.max[d] = value;
        right.min[d] = value;
        (left, right)
    }
}




#[derive(Debug)]
struct KDTree<const D: usize, T: Scalar + Necessary<T>> {
    nodes: Vec<Node<D, T>>,
}

impl<const D: usize, T: Scalar + Necessary<T>> KDTree<{ D }, T> {
    pub fn from(limit: usize, points: &[[T; D]]) -> Self {
        assert!(D > 0);
        if points.is_empty() {
            return KDTree {
                nodes: vec![Node::Points{points: vec![]}]
            };
        }

        let partition = |indices: &[usize], dim: usize| -> ([T; D], Vec<usize>, Vec<usize>) {
            let mut indices = indices.to_vec();
            indices.sort_by(|a, b| points[*a][dim].partial_cmp(&points[*b][dim]).unwrap());

            // We don't want the median, because the median gives us headaches in our search as there may be points with an identical coordinate
            // on either side of the median. So we start with the median value, then determine the partition index based on that.
            let partition_point = points[indices[indices.len() / 2]];
            let up_to = indices.partition_point(|&i| points[i][dim] < partition_point[dim]);
            let final_partition_point = points[indices[up_to]];
            let below = indices[..up_to].to_vec();
            let above = indices[up_to..].to_vec();
            // println!();
            // println!("final_partition_point: {final_partition_point:?}, partition_point_start: {partition_point:?} indices: {indices:?}   {below:?}, {above:?}, dim : {dim}");
            // println!("below: {:?}", below.iter().map(|&i| points[i]).collect::<Vec::<_>>()); 
            // println!("above: {:?}", above.iter().map(|&i| points[i]).collect::<Vec::<_>>()); 
            (final_partition_point, below, above)
        };

        let mut nodes = vec![];

        #[derive(Debug)]
        struct ProcessNode {
            indices: Vec<usize>,
            precursor: usize,
            dim: usize,
        }

        // Make a list of indices, and deduplicate it.
        let mut sorted_indices = (0..points.len()).collect::<Vec<_>>();
        // Sort it, such that identical values are consecutive.
        sorted_indices.sort_by(|&a, &b| points[a].partial_cmp(&points[b]).unwrap());
        // Now, deduplicate it, this isn't the best, but we can't use dedup as we have an indirection.
        let mut indices = Vec::<usize>::with_capacity(points.len());
        let mut previous = points[sorted_indices[0]];
        indices.push(sorted_indices[0]);
        for index in sorted_indices {
            let point = points[index];
            if point  == previous {
                continue
            } else {
                previous = point;
                indices.push(index);
            }
        }

        // We use a deque, such that we can insert in the rear and pop from the front.
        // This ensures that we don't get a depth first tree.
        use std::collections::VecDeque;
        let mut to_process: VecDeque<ProcessNode> = VecDeque::new();
        to_process.push_back(ProcessNode {
            indices,
            precursor: 0,
            dim: 0,
        });
        nodes.push(Node::Placeholder);

        while !to_process.is_empty() {
            // Pop the new sequence of indices to work on.
            let v = to_process.pop_front().unwrap();
            let d = v.dim;
            let indices = v.indices;
            let precursor = v.precursor;

            if indices.len() <= limit {
                // No work to do, update the placeholder with the points.
                nodes[precursor] = Node::Points {
                    points: indices.into_iter().map(|i| points[i]).collect(),
                };
                continue;
            }

            // Now, we need to iterate over the indices, and this dimension is split by pivot.
            let (pivot, below, above) = partition(&indices, d);

            // Update this placeholder.
            let left = if below.len() <= limit {
                // Drop the points into a container.
                nodes.push(Node::Points {
                    points: below.into_iter().map(|i| points[i]).collect(),
                });
                nodes.len() - 1
            } else {
                let left = nodes.len();
                nodes.push(Node::Placeholder); // left node
                to_process.push_back(ProcessNode {
                    indices: below,
                    precursor: left,
                    dim: (d + 1) % D,
                });
                left
            };

            let right = if above.len() <= limit {
                // Drop the points into a container.
                nodes.push(Node::Points {
                    points: above.into_iter().map(|i| points[i]).collect(),
                });
                nodes.len() - 1
            } else {
                let right = nodes.len();
                nodes.push(Node::Placeholder); // left node
                to_process.push_back(ProcessNode {
                    indices: above,
                    precursor: right,
                    dim: (d + 1) % D,
                });
                right
            };

            nodes[precursor] = Node::Split {
                pivot,
                dim: d,
                left,
                right,
            };
        }

        KDTree::<{ D }, T> { nodes }
    }

    pub fn contains(&self, point: &[T; D]) -> bool {
        let mut index = 0;
        loop {
            match &self.nodes[index] {
                Node::<{ D }, T>::Placeholder => return false,
                Node::<{ D }, T>::Split {
                    left,
                    right,
                    pivot,
                    dim,
                } => {
                    if pivot == point {
                        return true;
                    }
                    if point[*dim] < pivot[*dim] {
                        index = *left;
                    } else {
                        index = *right;
                    }
                }
                Node::<{ D }, T>::Points { points } => {
                    return points.contains(point);
                }
            }
        }
        false
    }
    pub fn nearest(&self, search_point: &[T; D]) -> Option<[T; D]> {
        // Well... smarts :grimacing:


        let mut best_point: Option<[T; D]> = None;
        let mut best_distance: Option<T> = None;

        use std::collections::VecDeque;
        let mut indices = VecDeque::new();
        indices.push_back((0usize, BoundingBox::<{D}, T>::everything()));

        while let Some((index, bounding_box)) = indices.pop_front() {
            match &self.nodes[index] {
                Node::<{ D }, T>::Placeholder => {continue},
                Node::<{ D }, T>::Split {
                    left,
                    right,
                    pivot,
                    dim,
                } => {
                    let d = distance(search_point, pivot);
                    if let Some(current_best) = best_distance {
                        if d < current_best {
                            best_distance = Some(d);
                            best_point = Some(*pivot);
                        }
                    } else  {
                        best_distance = Some(d);
                        best_point = Some(*pivot);
                    }

                    // Need to explore this split.
                    if search_point[*dim] < pivot[*dim] {
                        // Explore left
                        

                    } else {
                        // Explore right.
                    }
                }
                Node::<{ D }, T>::Points { points } => {
                    // return points.contains(point);
                    for point in points {
                        let d = distance(search_point, point);
                        if let Some(current_best) = best_distance {
                            if d < current_best {
                                best_distance = Some(d);
                                best_point = Some(*point);
                            }
                        } else  {
                            best_distance = Some(d);
                            best_point = Some(*point);
                        }
                    }
                }
            }
        }
        best_point
    }
    /*
    */
}

#[cfg(test)]
mod test {
    use super::*;

    use rand::distributions::{Distribution, Standard, Uniform};
    use rand::Rng;
    use rand_xoshiro::rand_core::{RngCore, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_bounding_box() {
        {
            let b = BoundingBox::<1, f32>::everything();
            assert_eq!(b.min_norm(&[0.0]), 0.0);
            let (left, right) = b.split(0, 1.0);
            // left is [-infty, 1.0]
            // right is [1.0, infty]

            assert_eq!(left.min_norm(&[0.0]), 0.0);
            assert_eq!(right.min_norm(&[0.0]), 1.0);
            assert_eq!(right.min_norm(&[5.0]), 0.0);

            let (left, right) = right.split(0, 10.0);
            println!("Left: {left:?}:");
            // left is [1.0, 10.0]
            // right is [10.0, infty]

            assert_eq!(left.min_norm(&[0.0]), 1.0);
            assert_eq!(left.min_norm(&[5.0]), 0.0);
            assert_eq!(left.min_norm(&[10.0]), 0.0);
            assert_eq!(left.min_norm(&[11.0]), 1.0);

            assert_eq!(right.min_norm(&[5.0]), 25.0);
            assert_eq!(right.min_norm(&[10.0]), 0.0);
            assert_eq!(right.min_norm(&[15.0]), 0.0);
        }
        {
            let b = BoundingBox::<2, f32>::everything();
            assert_eq!(b.min_norm(&[0.0, 0.0]), 0.0);

        
            let (left_x, right_x) = b.split(0, 1.0);
            // left_x is [-infty, 1.0]
            // right_x is [1.0, infty]


            let (bottom_x, top_x) = left_x.split(1, 1.0);
            // bottom_x, x = [-infty, 1.0], y = [-infty, 1.0];
            // top_x, x = [-infty, 1.0], y = [1.0, infty];
            assert_eq!(bottom_x.min_norm(&[0.0, 0.0]), 0.0);
            assert_eq!(bottom_x.min_norm(&[1.0, 1.0]), 0.0);

            assert_eq!(bottom_x.min_norm(&[1.0, 2.0]), 1.0);
            assert_eq!(bottom_x.min_norm(&[2.0, 2.0]), 1.0 + 1.0);
            assert_eq!(bottom_x.min_norm(&[2.0, 1.0]), 1.0);

            let (top_x_left, top_x_right) = top_x.split(0, -1.0);
            // top_x_left, x = [-infty, -1.0], y = [1.0, infty];
            // top_x_right, x = [-1.0, 1.0], y = [1.0, infty];
            assert_eq!(top_x_left.min_norm(&[0.0, 0.0]), 1.0 + 1.0);
            assert_eq!(top_x_left.min_norm(&[0.0, 3.0]), 1.0);

            
            assert_eq!(top_x_right.min_norm(&[-2.0, 3.0]), 1.0);
            assert_eq!(top_x_right.min_norm(&[-1.0, 3.0]), 0.0);
            assert_eq!(top_x_right.min_norm(&[0.0, 3.0]), 0.0);
            assert_eq!(top_x_right.min_norm(&[1.0, 3.0]), 0.0);
            assert_eq!(top_x_right.min_norm(&[2.0, 3.0]), 1.0);

            // And the final split to give us a fully defined bounding box.
            let (top_x_right_bottom, top_x_right_top) = top_x_right.split(1, 10.0);
            // top_x_right_bottom, x = [-1.0, 1.0], y = [1.0, 10.0];
            // top_x_right_top, x = [-1.0, 1.0], y = [10.0, infty];
            assert_eq!(top_x_right_bottom.min_norm(&[0.0, 3.0]), 0.0);
            assert_eq!(top_x_right_bottom.min_norm(&[0.0, 9.0]), 0.0);
            assert_eq!(top_x_right_bottom.min_norm(&[0.5, 9.0]), 0.0);

            // Check corners.
            // Bottom left;
            assert_eq!(top_x_right_bottom.min_norm(&[-2.0, 1.0]), 1.0);
            assert_eq!(top_x_right_bottom.min_norm(&[-1.0, 0.0]), 1.0);
            assert_eq!(top_x_right_bottom.min_norm(&[-2.0, 0.0]), 1.0 + 1.0);
            // Bottom right.
            assert_eq!(top_x_right_bottom.min_norm(&[2.0, 1.0]), 1.0);
            assert_eq!(top_x_right_bottom.min_norm(&[1.0, 0.0]), 1.0);
            assert_eq!(top_x_right_bottom.min_norm(&[2.0, 0.0]), 1.0 + 1.0);
            // Top left
            assert_eq!(top_x_right_bottom.min_norm(&[-2.0, 10.0]), 1.0);
            assert_eq!(top_x_right_bottom.min_norm(&[-1.0, 11.0]), 1.0);
            assert_eq!(top_x_right_bottom.min_norm(&[-2.0, 11.0]), 1.0 + 1.0);
            // Top right
            assert_eq!(top_x_right_bottom.min_norm(&[2.0, 10.0]), 1.0);
            assert_eq!(top_x_right_bottom.min_norm(&[1.0, 11.0]), 1.0);
            assert_eq!(top_x_right_bottom.min_norm(&[2.0, 11.0]), 1.0 + 1.0);
        }
    }

    #[test]
    fn test_construct() {
        let points = [[0.5, 0.5], [0.25, 0.3], [0.1, 0.1], [0.1, 0.5], [0.1, 0.5]];
        let t = KDTree::<2, f32>::from(1, &points);
        println!("{t:#?}");
        for p in points.iter() {
            println!("Testing {p:?}");
            assert_eq!(t.contains(p), true);
        }
    }

    #[test]
    fn test_fixed_construct_2d_f32() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);

        let points = (0..10)
            .into_iter()
            .map(|_| [rng.gen::<f32>(), rng.gen::<f32>()])
            .collect::<Vec<_>>();

        let mut lower_bound = [f32::MAX, f32::MAX];
        let mut upper_bound = [f32::MIN, f32::MIN];
        for [x, y] in points.iter() {
            lower_bound[0] = lower_bound[0].min(*x);
            lower_bound[1] = lower_bound[1].min(*y);
            upper_bound[0] = upper_bound[0].max(*x);
            upper_bound[1] = upper_bound[1].max(*y);
        }

        let t = KDTree::from(3, &points);
        println!("{t:#?}");
        // Check if all points present.
        for p in points.iter() {
            println!("Testing {p:?}");
            assert!(t.contains(p));
        }
    }

    #[test]
    fn test_random_2d_f32() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let count = Uniform::from(1..1000usize);
        let limit = Uniform::from(1..1000usize);
        let duplicates = Uniform::from(1..10usize);
        let duplicate_count = Uniform::from(1..10usize);
        for _ in 0..1000 {
            let point_count = count.sample(&mut rng);
            let point_limit = limit.sample(&mut rng);
            let mut points = (0..point_count)
                .into_iter()
                .map(|_| [rng.gen::<f32>(), rng.gen::<f32>()])
                .collect::<Vec<_>>();

            let duplicate_index =  Uniform::from(0..points.len());
            // insert some duplicates
            for _ in 0..duplicates.sample(&mut rng) {
                // pick a point.
                let index = duplicate_index.sample(&mut rng);
                for _ in 0..duplicate_count.sample(&mut rng) {
                    points.push(points[index]);
                }
            }

            let mut lower_bound = [f32::MAX, f32::MAX];
            let mut upper_bound = [f32::MIN, f32::MIN];
            for [x, y] in points.iter() {
                lower_bound[0] = lower_bound[0].min(*x);
                lower_bound[1] = lower_bound[1].min(*y);
                upper_bound[0] = upper_bound[0].max(*x);
                upper_bound[1] = upper_bound[1].max(*y);
            }

            let t = KDTree::from(point_limit, &points);

            // Check if all points present.
            for p in points.iter() {
                assert!(t.contains(p));
            }

            // Check another 100 random points not in the tree.
            for _ in 0..100 {
                let p = [rng.gen::<f32>(), rng.gen::<f32>()];
                if points.contains(&p) {
                    // Could happen once in a blue moon...
                    continue;
                }
                assert!(!t.contains(&p));
            }
        }
    }
}
