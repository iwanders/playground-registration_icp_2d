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


pub struct IterativeClosestPoint2DTranslation {
    base: Vec<[f32; 2]>,
    moving: Vec<[f32; 2]>,
    transform: [f32; 2], // x, y
    iteration: usize,
    tree: nearest_neighbour::KDTree::<2, f32>,
}

impl IterativeClosestPoint2DTranslation {
    pub fn setup(base: &[[f32; 2]], other: &[[f32; 2]]) -> Self {
        IterativeClosestPoint2DTranslation {
            base: base.to_vec(),
            moving: other.to_vec(),
            transform: [0.0, 0.0],
            iteration: 0,
            tree: nearest_neighbour::KDTree::from(10, base),
        }
    }

    fn determine_translation(base: &[[f32; 2]], other: &[[f32; 2]]) -> [f32; 2] {
        let mut transform = [0.0, 0.0];
        assert!(base.len() == other.len());
        for (b, o) in base.iter().zip(other.iter()) {
            transform[0] += b[0] - o[0];
            transform[1] += b[1] - o[1];
        }
        [
            transform[0] / base.len() as f32,
            transform[1] / base.len() as f32,
        ]
    }

    fn apply_translation(points: &mut [[f32; 2]], transform: [f32; 2]) {
        for p in points.iter_mut() {
            p[0] = p[0] + transform[0];
            p[1] = p[1] + transform[1];
        }
    }

    pub fn iterate(&mut self, iterations: usize) {
        for _ in 0..iterations {
            let closest_points = self.moving.iter().map(|z| self.tree.nearest(z).unwrap()).collect::<Vec<_>>();
            let transform = Self::determine_translation(&self.base, &closest_points);
            println!("New closest t: {transform:?}");
            // Perform the transform.
            Self::apply_translation(&mut self.moving, transform);
            // Update the full transform.
            self.transform[0] += transform[0];
            self.transform[1] += transform[1];
        }
    }

    pub fn transform(&self) -> [f32; 2] {
        self.transform
    }
}


#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_icp_2d_circles() {
        // Create two circles that are offset, then fit the thing.
        let mut circle_base = vec![];
        let elements = 100;
        let radius = 1.0;

        let offset = (3.3, 8.8);
        for i in 0..elements {
            let p = (i as f32) / (elements - 1) as f32;
            circle_base.push([radius * ( p * 2.0 * std::f32::consts::PI).cos(), radius * ( p * 2.0 * std::f32::consts::PI).sin()]);
        }
        println!("circle_base: {circle_base:?}");

        let mut circle_two = circle_base.clone();
        for [a, b] in circle_two.iter_mut() {
            *a += offset.0;
            *b += offset.1;
        }
        println!("circle_two: {circle_two:?}");

        let mut icp = IterativeClosestPoint2DTranslation::setup(&circle_base, &circle_two);
        for i in 0..20 {
            icp.iterate(1);
            let t = icp.transform();
            println!("t; {t:?}");
        }
    }
}