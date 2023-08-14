#[derive(Copy, Debug, Clone)]
pub struct PointStyleId(usize);

// struct LineSegment {
// from: [f32; 2],
// to: [f32; 2],
// }

struct Point {
    p: [f32; 2],
    style: PointStyleId,
}

pub struct SVG {
    point_styles: Vec<String>,
    markers: Vec<Point>,
    // lines: Vec<LineSegment>,
    min: [f32; 2],
    max: [f32; 2],
    dimensions: [f32; 2],
    background: Option<String>,
}

impl SVG {
    pub fn new() -> Self {
        SVG {
            min: [0.0, 0.0],
            max: [1.0, 1.0],
            dimensions: [1000.0, 1000.0],
            markers: vec![],
            // lines: vec![],
            point_styles: vec![],
            background: None,
        }
    }

    pub fn set_bounds(&mut self, min: &[f32; 2], max: &[f32; 2]) {
        self.min = *min;
        self.max = *max;
    }

    pub fn set_dimensions(&mut self, dimensions: &[f32; 2]) {
        self.dimensions = *dimensions;
    }

    pub fn set_background(&mut self, color: &str) {
        self.background = Some(color.to_string());
    }

    pub fn add_markers(&mut self, v: &[[f32; 2]], style: PointStyleId) {
        self.markers.extend(v.iter().map(|&p| Point { p, style }));
    }

    pub fn add_point_style(&mut self, s: &str) -> PointStyleId {
        let key = self.point_styles.len();
        self.point_styles.push(format!("<g id=\"m{key}\">{s}</g>"));
        PointStyleId(key)
    }

    pub fn add_point_style_circle(&mut self, r: f32, fill: &str) -> PointStyleId {
        self.add_point_style(&format!(
            "<circle cx=\"0.0\" cy=\"0.0\" r=\"{r}\" fill=\"{fill}\"/>"
        ))
    }

    pub fn add_point_style_cross(&mut self, t: f32, stroke: &str) -> PointStyleId {
        self.add_point_style(&format!(r#"
            <g transform="scale({t}, {t})">
                <path d="M -1,-1 L 1,1 M -1,1 L 1,-1" fill="none" stroke="{stroke}" stroke-width="1"/>
                <path d="M -1,-1 L 1,1 M -1,1 L 1,-1" fill="none" stroke="{stroke}" stroke-width="1"/>
            </g>
        "#))
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        let [svg_width, svg_height] = self.dimensions;
        let [minx, miny] = self.min;
        let width = self.max[0] - self.min[0];
        let height = self.max[1] - self.min[1];
        s += "<svg id=\"svg_el\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" ";
        s += &format!("width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"{minx} {miny} {width} {height}\" version=\"1.1\">");
        if let Some(color) = self.background.as_ref() {
            s += &format!("<rect x=\"{minx}\" y=\"{miny}\" width=\"{width}\" height=\"{height}\" fill=\"{color}\" />");
        }
        s += "<defs>\n";
        for style in self.point_styles.iter() {
            s += style;
        }
        s += "</defs>\n";

        s += "";
        for point in self.markers.iter() {
            s += &format!(
                "<use xlink:href=\"#m{}\" x=\"{}\" y=\"{}\"/>",
                point.style.0, point.p[0], point.p[1]
            );
        }
        s += "";
        s += "</svg>";
        s
    }
}

pub fn draw_clouds(
    min: &[f32; 2],
    max: &[f32; 2],
    base: &[[f32; 2]],
    other: &[[f32; 2]],
) -> String {
    let mut s = SVG::new();
    s.set_bounds(min, max);
    s.set_background("white");
    let red_circle = s.add_point_style_circle(1.1, "red");
    let black_cross = s.add_point_style_cross(1.1, "green");

    s.add_markers(base, black_cross);
    s.add_markers(other, red_circle);

    s.render()
}

pub fn write_clouds(
    fname: &str,
    min: &[f32; 2],
    max: &[f32; 2],
    base: &[[f32; 2]],
    other: &[[f32; 2]],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::prelude::*;
    let mut file = std::fs::File::create(fname)?;
    let s = draw_clouds(min, max, base, other);
    file.write_all(s.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_svg_things() {
        use std::io::prelude::*;
        let mut s = SVG::new();
        s.set_bounds(&[0.0, 0.0], &[100.0, 100.0]);
        s.set_background("white");
        let red_circle = s.add_point_style_circle(0.5, "red");
        let black_cross = s.add_point_style_cross(0.5, "green");

        s.add_markers(&[[3.0, 3.0], [50.0, 30.0]], red_circle);
        s.add_markers(&[[3.0, 32.0], [70.0, 30.0]], black_cross);

        let mut file = std::fs::File::create("/tmp/test_svg.svg").unwrap();
        file.write_all(s.render().as_bytes()).unwrap();
    }
}
