/*
    Ok, so we need something that will give us the (approximate) nearest neighbour.

    - Fast-ish.
    - No need for delete or insert.
    - Can start with creating the tree like a quadtree.
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
}

#[derive(Debug)]
pub enum Node<const D: usize, T: Scalar> {
    Split {
        /// Index to elements below of this pivot.
        left: usize,
        /// Pivot coordinate, not an actual point.
        pivot: T,
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
    pub fn new_quadtree(
        limit: usize,
        points: &[[T; D]],
        min: [T; D],
        max: [T; D],
    ) -> KDTree<{ D }, T> {
        assert!(D > 0);

        let partition = |indices: &[usize], pivot: T, dim: usize| -> (Vec<usize>, Vec<usize>) {
            let mut below = vec![];
            let mut above = vec![];
            for i in indices.iter() {
                if points[*i][dim] < pivot {
                    below.push(*i);
                } else {
                    above.push(*i);
                }
            }
            (below, above)
        };

        let mut nodes = vec![];

        #[derive(Debug)]
        struct ProcessNode<const D: usize, T: Scalar + Necessary<T>> {
            indices: Vec<usize>,
            precursor: usize,
            dim: usize,
            min: [T; D],
            max: [T; D],
        }

        // We use a deque, such that we can insert in the rear and pop from the front.
        // This ensures that we don't get a depth first tree.
        use std::collections::VecDeque;
        let mut to_process: VecDeque<ProcessNode<{ D }, T>> = VecDeque::new();
        to_process.push_back(ProcessNode {
            indices: (0..points.len()).collect::<Vec<_>>(),
            precursor: 0,
            dim: 0,
            min,
            max,
        });
        nodes.push(Node::<{ D }, T>::Placeholder);

        while !to_process.is_empty() {
            // Pop the new sequence of indices to work on.
            let v = to_process.pop_front().unwrap();
            let d = v.dim;
            let indices = v.indices;
            let precursor = v.precursor;
            let min = v.min;
            let max = v.max;

            if indices.len() <= limit {
                // No work to do, update the placeholder with the points.
                nodes[precursor] = Node::<{ D }, T>::Points {
                    points: indices.into_iter().map(|i| points[i]).collect(),
                };
                continue;
            }

            // We have work to do, determine the pivot.
            let pivot = max[d].sub(min[d]).div(T::two()).add(min[d]);

            // Now, we need to iterate over the indices, and this dimension is split by pivot.
            let (below, above) = partition(&indices, pivot, d);
            let below_min = min;
            let mut below_max = max;
            below_max[d] = pivot;

            let mut above_min = min;
            let above_max = max;
            above_min[d] = pivot;

            // This is an optimisation here, it avoids placeholders in the list by reusing the
            // previous split if one group is empty, still bisecting the range though.
            if below.is_empty() || above.is_empty() {
                let (container, (min, max)) = if below.is_empty() {
                    (above, (above_min, above_max))
                } else {
                    (below, (below_min, below_max))
                };
                // Push back to the work queue to split in the other direction.
                to_process.push_back(ProcessNode {
                    indices: container,
                    precursor: precursor,
                    dim: (d + 1) % D,
                    min,
                    max,
                });
                continue;
            }

            // Update this placeholder.
            let left = if below.is_empty() {
                // don't do anything.
                // Insert placeholder.
                nodes.push(Node::<{ D }, T>::Placeholder); // left node
                nodes.len() - 1
            } else if below.len() < limit {
                // Drop the points into a container.
                nodes.push(Node::<{ D }, T>::Points {
                    points: below.into_iter().map(|i| points[i]).collect(),
                });
                nodes.len() - 1
            } else {
                let left = nodes.len();
                nodes.push(Node::<{ D }, T>::Placeholder); // left node
                to_process.push_back(ProcessNode {
                    indices: below,
                    precursor: left,
                    dim: (d + 1) % D,
                    min: below_min,
                    max: below_max,
                });
                left
            };

            let right = if above.is_empty() {
                // don't do anything.
                // Insert placeholder.
                nodes.push(Node::<{ D }, T>::Placeholder); // left node
                nodes.len() - 1
            } else if above.len() < limit {
                // Drop the points into a container.
                nodes.push(Node::<{ D }, T>::Points {
                    points: above.into_iter().map(|i| points[i]).collect(),
                });
                nodes.len() - 1
            } else {
                let right = nodes.len();
                nodes.push(Node::<{ D }, T>::Placeholder); // left node
                to_process.push_back(ProcessNode {
                    indices: above,
                    precursor: right,
                    dim: (d + 1) % D,
                    min: above_min,
                    max: above_max,
                });
                right
            };

            nodes[precursor] = Node::<{ D }, T>::Split {
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
                    if point[*dim] < *pivot {
                        index = *left;
                    } else {
                        index = *right;
                    }
                    continue;
                }
                Node::<{ D }, T>::Points { points } => {
                    return points.contains(point);
                }
            }
        }
        false
    }
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
        let points = [[0.5, 0.5], [0.25, 0.3], [0.1, 0.1]];
        let t = KDTree::<2, f32>::new_quadtree(2, &points, [0.0, 0.0], [1.0, 1.0]);
        for p in points.iter() {
            assert_eq!(t.contains(p), true);
        }
    }

    #[test]
    fn test_fixed_construct_2d_f32() {
        let points = [
            [0.38223928, 0.35961717],
            [0.011455476, 0.49527],
            [0.020565212, 0.85724735],
            [0.84550875, 0.29488564],
            [0.07423377, 0.31452203],
            [0.066070974, 0.104222596],
            [0.09381217, 0.06578922],
            [0.4003474, 0.62607],
            [0.11451006, 0.3271203],
            [0.40310985, 0.27699465],
            [0.13055545, 0.040342987],
            [0.2733367, 0.434321],
            [0.3780135, 0.13583922],
            [0.6477326, 0.9097044],
            [0.044103086, 0.2502429],
            [0.38766545, 0.3526219],
            [0.8611653, 0.8734964],
            [0.20379591, 0.51204795],
            [0.16359204, 0.71176416],
            [0.18152481, 0.56403023],
            [0.29694504, 0.3452716],
            [0.056833565, 0.03334576],
            [0.18193924, 0.21426392],
            [0.16607064, 0.24505132],
            [0.42822498, 0.58666754],
            [0.5212066, 0.7911238],
            [0.13704151, 0.02528733],
            [0.77074486, 0.7296772],
            [0.49131805, 0.30820596],
            [0.6398218, 0.6093775],
            [0.7529851, 0.013113797],
            [0.40955508, 0.92294186],
            [0.77238345, 0.41256005],
            [0.28462082, 0.7052928],
            [0.9053682, 0.81027925],
            [0.23959309, 0.81421494],
            [0.5918593, 0.76235086],
            [0.4959036, 0.6610549],
            [0.7556707, 0.6416542],
        ];
        let mut lower_bound = [f32::MAX, f32::MAX];
        let mut upper_bound = [f32::MIN, f32::MIN];
        for [x, y] in points.iter() {
            lower_bound[0] = lower_bound[0].min(*x);
            lower_bound[1] = lower_bound[1].min(*y);
            upper_bound[0] = upper_bound[0].max(*x);
            upper_bound[1] = upper_bound[1].max(*y);
        }

        let t = KDTree::new_quadtree(10, &points, lower_bound, upper_bound);
        println!("{t:?}");
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

            let t = KDTree::new_quadtree(point_limit, &points, lower_bound, upper_bound);

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
