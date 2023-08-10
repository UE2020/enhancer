use image::{*, imageops::{blur}};
use rand::Rng;
use walkdir::WalkDir;

fn main() {
    let mut count = 0;

    for entry in WalkDir::new("./raw")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|f| f.file_type().is_file())
        .filter(|f| f.path().extension().is_some())
        .filter(|f| {
            matches!(
                f.path().extension().unwrap().to_str().unwrap(),
                "jpg" | "jpeg" | "png" | "gif"
            )
        })
    {
        let img = image::open(entry.path());
        match img {
            Ok(img) => {
                if img.width() < 224 * 2 || img.height() < 224 * 2 {
                    continue;
                }
                let (img, transformed_img) = transform(img);
                img.save(format!("./data/truth/{}.png", count)).unwrap();
                transformed_img
                    .save(format!("./data/transformed/{}.png", count))
                    .unwrap();
                println!(
                    "Saved {} and {}",
                    format!("./data/truth/{}.png", count),
                    format!("./data/transformed/{}.png", count)
                );
                count += 1;
            }
            Err(e) => println!("Bad image {}: {}", entry.path().display(), e),
        }
    }
}

fn transform(mut img: DynamicImage) -> (RgbaImage, RgbaImage) {
	let (w, h) = img.dimensions();
    let img = img.crop((w / 2) - (224/2), (h / 2) - (224/2), 224, 224);
    let og = img.clone().to_rgba8();
    let mut img = imageops::huerotate(&img, rand::thread_rng().gen_range(-50..50));
	let factor = rand::thread_rng().gen_range(0.5..2.0);
    img.enumerate_pixels_mut().for_each(|(_, _, pixel)| {
        pixel.0[0] = (pixel.0[0] as f32 * factor).clamp(0.0, 255.0) as u8;
        pixel.0[1] = (pixel.0[1] as f32 * factor).clamp(0.0, 255.0) as u8;
        pixel.0[2] = (pixel.0[2] as f32 * factor).clamp(0.0, 255.0) as u8;
    });
    for (_, _, pixel) in img.enumerate_pixels_mut() {
        let (r, g, b) = (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
        let factor = 0.5;
        let avg = (r + g + b) / 3.0;

        let new_r = avg + (r - avg) * factor;
        let new_g = avg + (g - avg) * factor;
        let new_b = avg + (b - avg) * factor;

        let new_r = new_r.clamp(0.0, 255.0) as u8;
        let new_g = new_g.clamp(0.0, 255.0) as u8;
        let new_b = new_b.clamp(0.0, 255.0) as u8;

        *pixel = Rgba([new_r, new_g, new_b, pixel[3]]);
    }

    for (_, _, pixel) in img.enumerate_pixels_mut() {
        let factor = 0.8;
        let new_pixel = Rgba([
            apply_contrast(pixel[0], factor),
            apply_contrast(pixel[1], factor),
            apply_contrast(pixel[2], factor),
            pixel[3],
        ]);

        *pixel = new_pixel;
    }

	let img = blur(&img, 1.0);

    (og, img)
}

fn apply_contrast(value: u8, factor: f32) -> u8 {
    let adjusted_value = ((value as f32 - 128.0) * factor) + 128.0;
    adjusted_value.clamp(0.0, 255.0) as u8
}
