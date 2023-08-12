
/*
    Ok, so we need something that will give us the (approximate) nearest neighbour.

    - Fast-ish.
    - No need for delete or insert.
    - Can start with creating the tree like a quadtree.
*/

pub trait Scalar: std::cmp::PartialOrd + Default + std::marker::Copy + std::fmt::Debug {}
impl<T> Scalar for T
where
    T: std::cmp::PartialOrd + Default + std::marker::Copy + std::fmt::Debug , 
{}

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
    }
}


struct KDTree<const D: usize, T: Scalar> {
    nodes: Vec<Node<D, T>>,
}

impl<const D: usize, T: Scalar> KDTree<{D}, T> {
    pub fn new_quadtree(limit: usize, points: &[[T; D]], min: [T; D], max: [T; D]) -> KDTree<{D}, T> {


        let nodes= vec![];
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
        let t = KDTree::<2, f32>::new_quadtree(5, &[], [0.0, 0.0], [1.0, 1.0]);
    }
}


