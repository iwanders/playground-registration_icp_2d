use registration_icp_2d::{util, IterativeClosestPoint2DTranslation};

fn load_txt(fname: &str) -> Result<Vec<[f32; 2]>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::BufRead;
    use std::io::BufReader;
    let file = File::open(fname)?;
    let reader = BufReader::new(file);

    let mut res = vec![];
    for line in reader.lines() {
        let line = line?;
        let mut tokens = line.split(" ");
        let x = tokens.next().ok_or("not a float")?;
        let y = tokens.next().ok_or("not a float")?;
        let x = x.parse::<f32>()?;
        let y = y.parse::<f32>()?;
        res.push([x, y]);
    }
    Ok(res)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().len() == 1
        || (std::env::args().len() == 2 && std::env::args().nth(1) == Some("--help".to_string()))
    {
        println!("cargo r --release -- <input1> <input2>");
        return Ok(());
    }

    let input1 = std::env::args()
        .nth(1)
        .expect("expected .txt file for first points");
    let input2 = std::env::args()
        .nth(2)
        .expect("expected .txt file for second points");

    let cloud_base = load_txt(&input1)?;
    let cloud_moving = load_txt(&input2)?;

    let mut icp = IterativeClosestPoint2DTranslation::setup(&cloud_base, &cloud_moving);
    let min = [0.0, 0.0];
    let max = [1920.0, 1080.0];
    icp.add_transform([140.0, 280.0]);

    const DRAW_FRAMES: bool = true;
    for i in 0..20 {
        let t = icp.transform();
        println!("t; {t:?}");

        if DRAW_FRAMES {
            util::write_clouds(
                &format!("/tmp/icp_output_{i:0>2}.svg"),
                &min,
                &max,
                &cloud_base,
                icp.moving(),
            )
            .unwrap();
        }
        icp.iterate(1);
    }
    let t = icp.transform();

    Ok(())
}
