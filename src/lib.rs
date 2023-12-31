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

pub mod kdtree;
pub mod util;

pub struct IterativeClosestPoint2DTranslation {
    base: Vec<[f32; 2]>,
    moving: Vec<[f32; 2]>,
    transform: [f32; 2], // x, y
    iteration: usize,
    tree: kdtree::KDTree<2, f32>,
}

impl IterativeClosestPoint2DTranslation {
    pub fn setup(base: &[[f32; 2]], other: &[[f32; 2]]) -> Self {
        IterativeClosestPoint2DTranslation {
            base: base.to_vec(),
            moving: other.to_vec(),
            transform: [0.0, 0.0],
            iteration: 0,
            tree: kdtree::KDTree::from(32, base),
        }
    }

    fn determine_translation(base: &[[f32; 2]], other: &[[f32; 2]]) -> [f32; 2] {
        let mut transform = [0.0, 0.0];
        assert!(base.len() == other.len());
        for (b, o) in base.iter().zip(other.iter()) {
            transform[0] += o[0] - b[0];
            transform[1] += o[1] - b[1];
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

    pub fn iterate(&mut self, iterations: usize, max_distance: f32) {
        let max_distance_square = max_distance * max_distance;
        for _ in 0..iterations {
            let closest_points = self
                .moving
                .iter()
                .map(|z| self.tree.nearest(z).unwrap())
                .collect::<Vec<_>>();

            let mut moving = vec![];
            let mut from_base = vec![];
            for p in self.moving.iter() {
                if let Some(nearest) = self.tree.nearest(p) {
                    if nearest.distance < max_distance_square {
                        moving.push(*p);
                        from_base.push(nearest.point);
                    }
                }
            }

            let transform = Self::determine_translation(&moving, &from_base);
            // Perform the transform.
            Self::apply_translation(&mut self.moving, transform);
            // Update the full transform.
            self.transform[0] += transform[0];
            self.transform[1] += transform[1];
            self.iteration += 1;
        }
    }

    pub fn transform(&self) -> [f32; 2] {
        self.transform
    }
    pub fn add_transform(&mut self, transform: [f32; 2]) {
        self.transform[0] += transform[0];
        self.transform[1] += transform[1];
        Self::apply_translation(&mut self.moving, transform);
    }
    pub fn moving(&self) -> &[[f32; 2]] {
        &self.moving
    }
    pub fn base(&self) -> &[[f32; 2]] {
        &self.base
    }
    pub fn iteration(&self) -> usize {
        self.iteration
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_icp_2d_circles() {
        // Create two circles that are offset, then fit the thing.
        let mut circle_base = vec![];
        let elements = 20000;
        let radius = 1.0;

        let offset = (3.3, 8.8);
        for i in 0..elements {
            let p = (i as f32) / (elements - 1) as f32;
            circle_base.push([
                radius * (p * 2.0 * std::f32::consts::PI).cos(),
                radius * (p * 2.0 * std::f32::consts::PI).sin(),
            ]);
        }
        // println!("circle_base: {circle_base:?}");

        let mut circle_two = circle_base.clone();
        for [a, b] in circle_two.iter_mut() {
            *a += offset.0;
            *b += offset.1;
        }
        // println!("circle_two: {circle_two:?}");

        let mut icp = IterativeClosestPoint2DTranslation::setup(&circle_base, &circle_two);
        let min = [-5.0, -5.0];
        let max = [15.0, 15.0];

        const DRAW_FRAMES: bool = false;
        for i in 0..20 {
            let t = icp.transform();
            println!("t; {t:?}");

            if DRAW_FRAMES {
                util::write_clouds(
                    &format!("/tmp/circles_{i:0>2}.svg"),
                    &min,
                    &max,
                    &circle_base,
                    icp.moving(),
                )
                .unwrap();
            }
            icp.iterate(1, f32::MAX);
        }
        let t = icp.transform();
        assert!((t[0] + offset.0).abs() < 0.0001);
        assert!((t[1] + offset.1).abs() < 0.0001);
    }
}
