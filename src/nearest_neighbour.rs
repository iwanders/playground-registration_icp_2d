
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
        pivot: [Option<T>; D],
        /// Index to elements above or equal to this pivot.
        right: usize,
    },
    Points {
        points: Vec<[T; D]>, // could be a smallvec.
    },
    Nothing
}


#[derive(Debug)]
struct KDTree<const D: usize, T: Scalar + Necessary<T>> {
    nodes: Vec<Node<D, T>>,
}

impl<const D: usize, T: Scalar+ Necessary<T>> KDTree<{D}, T> {
    pub fn new_quadtree(limit: usize, points: &[[T; D]], min: [T; D], max: [T; D]) -> KDTree<{D}, T> {
        assert!(D > 0);

        let mut indices = (0..points.len()).collect::<Vec<_>>();

        let mut d = 0;

        let partition = |indices: &[usize], pivot: T| -> (Vec<usize>, Vec<usize>)  {
            let mut below = vec![];
            let mut above = vec![];
            for i in indices.iter() {
                if points[*i][d] < pivot {
                    below.push(*i);
                } else {
                    above.push(*i);
                }
            }
            (below, above)
        };

        let mut nodes = vec![];

        loop {
            let mut both_satisfied = true;

            let pivot_value = max[d].sub(min[d]).div(T::two());
            // Now, we need to iterate over the indices, and this dimension is split by pivot.
            let (below, above) = partition(&indices, pivot_value);

            let mut pivot = [None; D];
            pivot[d] = Some(pivot_value);


            let node  = Node::<{D}, T>::Split {
                left: nodes.len() + 1,
                pivot,
                right: nodes.len() + 2,
            };
            nodes.push(node);

            let left_node = if below.len() < limit {
                Node::<{D}, T>::Points {
                    points: below.iter().map(|i|{points[*i]}).collect()
                }
            } else {
                both_satisfied = false;
                Node::<{D}, T>::Nothing
            };

            nodes.push(left_node);
            let right_node = if above.len() < limit {
                Node::<{D}, T>::Points {
                    points: below.iter().map(|i|{points[*i]}).collect()
                }
            } else {
                both_satisfied = false;
                Node::<{D}, T>::Nothing
            };
            nodes.push(right_node);


            if both_satisfied {
                break
            }
        }

        KDTree::<{D}, T>{
            nodes,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn construct() {
        let t = KDTree::<2, f32>::new_quadtree(5, &[
            [0.5, 0.5],
            [0.25, 0.3],
            [0.1, 0.1],
            ], [0.0, 0.0], [1.0, 1.0]);
        println!("{t:?}");
    }
}


