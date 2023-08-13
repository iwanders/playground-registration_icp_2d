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

#[derive(Debug)]
struct KDTree<const D: usize, T: Scalar + Necessary<T>> {
    nodes: Vec<Node<D, T>>,
}

impl<const D: usize, T: Scalar + Necessary<T>> KDTree<{ D }, T> {
    pub fn from(limit: usize, points: &[[T; D]]) -> Self {
        assert!(D > 0);

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

        // We use a deque, such that we can insert in the rear and pop from the front.
        // This ensures that we don't get a depth first tree.
        use std::collections::VecDeque;
        let mut to_process: VecDeque<ProcessNode> = VecDeque::new();
        to_process.push_back(ProcessNode {
            indices: (0..points.len()).collect::<Vec<_>>(),
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
                    // println!("index: {index}, dim: {dim}, point {point:?}, pivot: {pivot:?}");
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
    /*
    pub fn nearest(&self, point: &[T; D]) -> Option<[T; D]> {
        // Well... smarts :grimacing:

        let mut lower_bound = [T::MAX(); D];
        let mut upper_bound = [T::MIN(); D];
        let mut best: Option<[T; D]> = None;

        let NN = |n: usize, min: [T; D], max: [T; D]| {

        };

        use std::collections::VecDeque;
        let mut indices = VecDeque::new();
        indices.push_back(0);
        let mut best = None;
        while let Some(index) = indices.pop_front() {
            match &self.nodes[index] {
                Node::<{ D }, T>::Placeholder => {continue},
                Node::<{ D }, T>::Split {
                    left,
                    right,
                    pivot,
                    dim,
                } => {
                    // Here, we need to determine if either index can yield a closer result.
                    // if point[*dim] < *pivot {
                        // index = *left;
                    // } else {
                        // index = *right;
                    // }
                    continue;
                }
                Node::<{ D }, T>::Points { points } => {
                    return points.contains(point);
                }
            }
        }
        best
    }
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
    fn test_construct() {
        let points = [[0.5, 0.5], [0.25, 0.3], [0.1, 0.1], [0.1, 0.5]];
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
        for _ in 0..1000 {
            let point_count = count.sample(&mut rng);
            let point_limit = limit.sample(&mut rng);
            let points = (0..point_count)
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
