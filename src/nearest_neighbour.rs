/*
    Ok, so we need something that will give us the ~~(approximate)~~ nearest neighbour.

    - Fast-ish.
    - No need for delete or insert.

    Approach;
    - Create a KD-tree
    - Terminate leafs after only 'n' points remain.
    - Put everything into a single vector such that search aren't random memory access but instead
      hop through memory that's close by.
*/

/// Super trait that defines what a scalar must support.
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

/// The operations we need on our scalar.
pub trait Necessary<T: Scalar> {
    fn div(&self, other: T) -> T;
    fn sub(&self, other: T) -> T;
    fn add(&self, other: T) -> T;
    fn mult(&self, other: T) -> T;
    fn min(&self, other: T) -> T;
    fn max(&self, other: T) -> T;
    // fn sqrt(&self) -> T;
    fn max_value() -> T;
    fn min_value() -> T;
    fn two() -> T;
}

/// Implement that for f32.
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
    fn max_value() -> f32 {
        f32::MAX
    }
    fn min_value() -> f32 {
        f32::MIN
    }
}

#[derive(Debug)]
/// A node in our kdtree.
enum Node<const D: usize, T: Scalar> {
    /// A split in the kdtree
    Split {
        /// Index to elements below of this pivot.
        left: usize,
        /// Pivot coordinate, the point this is is duplicated in the next level.
        pivot: [T; D],
        /// Pivot dimension.
        dim: usize,
        /// Index to elements above or equal to this pivot.
        right: usize,
    },
    /// Leaf in the kdtree holding actual points.
    Points {
        points: Vec<[T; D]>, // could be a smallvec.
    },
    /// A placeholder, used during construction.
    Placeholder,
}

/// A distance function.
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
/// Bounding box object necessary for the nearest neighbour search
struct BoundingBox<const D: usize, T: Scalar + Necessary<T>> {
    min: [T; D],
    max: [T; D],
}

impl<const D: usize, T: Scalar + Necessary<T>> BoundingBox<{ D }, T> {
    pub fn everything() -> Self {
        BoundingBox {
            min: [T::min_value(); D],
            max: [T::max_value(); D],
        }
    }

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
/// A KDTree representation of points, it deduplicates points.
pub struct KDTree<const D: usize, T: Scalar + Necessary<T>> {
    nodes: Vec<Node<D, T>>,
}

impl<const D: usize, T: Scalar + Necessary<T>> KDTree<{ D }, T> {
    /// Construct a tree from a slice of points, making leafs 'limit' size.
    pub fn from(limit: usize, points: &[[T; D]]) -> Self {
        assert!(D > 0);

        if points.is_empty() {
            return KDTree {
                nodes: vec![Node::Points { points: vec![] }],
            };
        }

        // Partition the current set of indices.
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
            (final_partition_point, below, above)
        };

        // Vector that will hold the KDtree nodes, the real tree is in here.
        let mut nodes = vec![];

        #[derive(Debug)]
        /// Helper struct for the nodes yet to be processed
        struct ProcessNode {
            indices: Vec<usize>,
            precursor: usize,
            dim: usize,
        }

        // Make a list of indices, and deduplicate it.
        let mut sorted_indices = (0..points.len()).collect::<Vec<_>>();
        // Sort it, such that identical values are consecutive.
        sorted_indices.sort_by(|&a, &b| points[a].partial_cmp(&points[b]).unwrap());
        // Now, deduplicate it, this isn't the best, but we can't use Vec::dedup as we have an indirection.
        let mut indices = Vec::<usize>::with_capacity(points.len());
        let mut previous = points[sorted_indices[0]];
        indices.push(sorted_indices[0]);
        for index in sorted_indices {
            let point = points[index];
            if point == previous {
                continue;
            } else {
                previous = point;
                indices.push(index);
            }
        }

        // We use a deque, such that we can insert in the rear and pop from the front.
        // This ensures that we don't get a depth first tree.
        nodes.push(Node::Placeholder); // push the first placeholder node
                                       // Push the list of ondices to work on.
        use std::collections::VecDeque;
        let mut to_process: VecDeque<ProcessNode> = VecDeque::new();
        to_process.push_back(ProcessNode {
            indices,
            precursor: 0,
            dim: 0,
        });

        while let Some(v) = to_process.pop_front() {
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

            // Split the remaining indices for this piece of work by the dimension.
            let (pivot, below, above) = partition(&indices, d);

            // Determine what to do with left.
            let left = if below.len() <= limit {
                // Drop the points into a container.
                nodes.push(Node::Points {
                    points: below.into_iter().map(|i| points[i]).collect(),
                });
                nodes.len() - 1
            } else {
                // Points still too large, push to the work queue.
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
                // Points still too large, push to the work queue.
                let right = nodes.len();
                nodes.push(Node::Placeholder); // right node
                to_process.push_back(ProcessNode {
                    indices: above,
                    precursor: right,
                    dim: (d + 1) % D,
                });
                right
            };

            // Finally, update the precursor, replacing the placeholder with a split pointing to
            // the correct nodes.
            nodes[precursor] = Node::Split {
                pivot,
                dim: d,
                left,
                right,
            };
        }

        KDTree { nodes }
    }

    /// Check if a point is contained in the tree.
    pub fn contains(&self, point: &[T; D]) -> bool {
        let mut index = 0;
        loop {
            match &self.nodes[index] {
                Node::<{ D }, T>::Placeholder => panic!("placeholder encountered during search"),
                Node::<{ D }, T>::Split {
                    left,
                    right,
                    pivot,
                    dim,
                } => {
                    if pivot == point {
                        // return true;
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
    }

    /// Retrieve the point nearest to a search point in the tree.
    pub fn nearest(&self, search_point: &[T; D]) -> Option<[T; D]> {
        // Well... smarts :grimacing:
        let mut best_value: Option<(T, [T; D])> = None;

        let update_best = |p: &[T; D], best_value: &mut Option<(T, [T; D])>| {
            let d = distance(search_point, p);
            if let Some((current_best, _current_point)) = best_value {
                if d < *current_best {
                    *best_value = Some((d, *p));
                }
            } else {
                *best_value = Some((d, *p));
            }
        };

        // Use a container to hold which index to explore and what bounding box is associated to
        // it.
        use std::collections::VecDeque;
        let mut indices = VecDeque::new();

        // Start with the first node, and the entire possible search space.
        indices.push_back((0usize, BoundingBox::<{ D }, T>::everything()));

        // Then, while there are searchable things.
        while let Some((index, bounding_box)) = indices.pop_front() {
            // Check if this is still relevant
            if let Some((best_distance, _best_point)) = best_value.as_ref() {
                if best_distance < &bounding_box.min_norm(search_point) {
                    // Current is better than this bounding box can ever be, no need to explore.
                    continue;
                }
            }

            match &self.nodes[index] {
                Node::<{ D }, T>::Placeholder => panic!("placeholder encountered during search"),
                Node::<{ D }, T>::Split {
                    left,
                    right,
                    pivot,
                    dim,
                } => {
                    // Determine the distance of this pivot to the current point, update best score.
                    update_best(pivot, &mut best_value);

                    // Next, split the current bounding box and explore the most likely region.
                    let (left_box, right_box) = bounding_box.split(*dim, pivot[*dim]);
                    if search_point[*dim] < pivot[*dim] {
                        // Explore left first
                        indices.push_back((*left, left_box));
                        indices.push_back((*right, right_box));
                    } else {
                        // Explore right first
                        indices.push_back((*right, right_box));
                        indices.push_back((*left, left_box));
                    }
                }
                Node::<{ D }, T>::Points { points } => {
                    // We found a leaf with actual points, linearly iterate through those.
                    for point in points {
                        update_best(point, &mut best_value);
                    }
                }
            }
        }
        best_value.map(|z| z.1)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use rand::distributions::{Distribution, Uniform};
    use rand::Rng;
    use rand_xoshiro::rand_core::SeedableRng;
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
            assert_eq!(left_x.max[0], 1.0);
            assert_eq!(right_x.min[0], 1.0);
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
            assert_eq!(top_x_right_top.min[1], 10.0);
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
            assert_eq!(t.nearest(p), Some(*p));
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

            let duplicate_index = Uniform::from(0..points.len());
            // insert some duplicates
            for _ in 0..duplicates.sample(&mut rng) {
                // pick a point.
                let index = duplicate_index.sample(&mut rng);
                for _ in 0..duplicate_count.sample(&mut rng) {
                    points.push(points[index]);
                }
            }

            let find_nearest = |search_point: &[f32; 2]| -> [f32; 2] {
                let mut best = points[0];
                let mut best_distance = f32::MAX;
                for p in points.iter() {
                    let d = distance(p, search_point);
                    if d < best_distance {
                        best = *p;
                        best_distance = d;
                    }
                }
                best
            };

            let t = KDTree::from(point_limit, &points);

            // Check if all points present.
            for p in points.iter() {
                assert!(t.contains(p));

                let v = t.nearest(p);
                assert_eq!(v, Some(*p));
            }

            // Check another 100 random points not in the tree.
            for _ in 0..100 {
                let p = [rng.gen::<f32>(), rng.gen::<f32>()];
                if points.contains(&p) {
                    // Could happen once in a blue moon...
                    continue;
                }
                assert!(!t.contains(&p));

                let v = t.nearest(&p);
                assert_eq!(v, Some(find_nearest(&p)));
            }
        }
    }
}
