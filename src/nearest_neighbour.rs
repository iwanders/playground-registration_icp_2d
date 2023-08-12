
/*
    Ok, so we need something that will give us the (approximate) nearest neighbour.

    - Fast-ish.
    - No need for delete or insert.
    - Can start with creating the tree like a quadtree.
*/

pub trait Scalar: std::cmp::PartialOrd + Default + std::marker::Copy + std::fmt::Debug {}
impl<T> Scalar for T
where
    T: std::cmp::PartialOrd + Default + std::marker::Copy + std::fmt::Debug  + std::ops::Sub , 
{}

trait Necessary<T: Scalar> {
    fn div(&self, other: T) -> T;
    fn sub(&self, other: T) -> T;
    fn two() -> T;
}
impl Necessary<f32> for f32 {
    fn div(&self, other: f32) -> f32 { *self / other }
    fn sub(&self, other: f32) -> f32 { *self - other }
    fn two() -> f32{ 2.0 }
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
    Placeholder
}


#[derive(Debug)]
struct KDTree<const D: usize, T: Scalar + Necessary<T>> {
    nodes: Vec<Node<D, T>>,
}


impl<const D: usize, T: Scalar+ Necessary<T>> KDTree<{D}, T> {

    pub fn new_quadtree(limit: usize, points: &[[T; D]], min: [T; D], max: [T; D]) -> KDTree<{D}, T> {
        assert!(D > 0);


        let partition = |indices: &[usize], pivot: T, dim: usize| -> (Vec<usize>, Vec<usize>)  {
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


        struct ProcessNode<const D: usize, T: Scalar+ Necessary<T>> {
            indices: Vec<usize>,
            precursor: usize,
            dim: usize,
            min: [T; D],
            max: [T; D],
        }

        // We use a deque, such that we can insert in the rear and pop from the front.
        // This ensures that we don't get a depth first tree.
        use std::collections::VecDeque;
        let mut to_process: VecDeque<ProcessNode<{D}, T>> = VecDeque::new();
        to_process.push_back(ProcessNode{
            indices: (0..points.len()).collect::<Vec<_>>(),
            precursor: 0,
            dim: 0,
            min,
            max
        });
        nodes.push(Node::<{D}, T>::Placeholder);


        while !to_process.is_empty() {

            // Pop the new sequence of indices to work on.
            let v = to_process.pop_front().unwrap();
            let d = v.dim;
            let indices = v.indices;
            let precursor = v.precursor;
            let min = v.min;
            let max = v.max;

            if indices.len() < limit {
                // No work to do, update the placeholder with the points.
                nodes[precursor] = Node::<{D}, T>::Points {
                    points: indices.into_iter().map(|i|{points[i]}).collect()
                };
                continue;
            }

            // We have work to do, determine the pivot.
            let pivot = max[d].sub(min[d]).div(T::two());

            // Now, we need to iterate over the indices, and this dimension is split by pivot.
            let (below, above) = partition(&indices, pivot, d);

            // Update this placeholder.
            let left = nodes.len();
            let right = nodes.len() + 1;
            nodes[precursor] = Node::<{D}, T>::Split{
                pivot,
                dim: d,
                left,
                right,
            };
            nodes.push(Node::<{D}, T>::Placeholder);  // left node
            nodes.push(Node::<{D}, T>::Placeholder);  // right node

            // Dump both sides into a bin.
            if !below.is_empty() {
                let min = min;
                let mut max = max;
                max[d] = pivot;
                to_process.push_back(ProcessNode{
                    indices: below,
                    precursor: left,
                    dim: (d + 1) % D,
                    min,
                    max
                });
            }

            if !above.is_empty() {
                let mut min = min;
                let max = max;
                min[d] = pivot;
                to_process.push_back(ProcessNode{
                    indices: above,
                    precursor: right,
                    dim: (d + 1) % D,
                    min,
                    max
                });
            }
        }

        KDTree::<{D}, T>{
            nodes,
        }
    }

    pub fn contains(&self, point: &[T; D]) -> bool {
        let mut index = 0;
        loop {
            match &self.nodes[index] {
                Node::<{D}, T>::Placeholder => return false,
                Node::<{D}, T>::Split {
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
                Node::<{D}, T>::Points{points} => {
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
    #[test]
    fn construct() {
        let points = [
            [0.5, 0.5],
            [0.25, 0.3],
            [0.1, 0.1],
        ];
        let t = KDTree::<2, f32>::new_quadtree(2, &points, [0.0, 0.0], [1.0, 1.0]);
        println!("{t:#?}");
        for p in points.iter() {
            assert_eq!(t.contains(p), true);
        }
    }
}


