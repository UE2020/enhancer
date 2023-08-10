use std::{
    path::{Path, PathBuf},
};

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Data, Tensor},
};

use image::{*, imageops::FilterType::Lanczos3};
use lab::Lab;

#[derive(Clone, Debug)]
pub struct ImageBatch<B: Backend> {
    pub truth: Tensor<B, 4>,
    pub transformed: Tensor<B, 4>,
}

pub type ImageItem = (RgbImage, RgbImage);

pub struct ImageBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ImageBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<ImageItem, ImageBatch<B>> for ImageBatcher<B> {
    fn batch(&self, items: Vec<ImageItem>) -> ImageBatch<B> {
        let truth = items
            .iter()
            .map(|(truth, _)| {
				let mut channels = [[[[0.0; 224]; 224]; 3]];
				for (x, y, pixel) in truth.enumerate_pixels() {
					let lab = Lab::from_rgb(&pixel.0);
					channels[0][0][x as usize][y as usize] = lab.l as f32 / 128.0;
					channels[0][1][x as usize][y as usize] = lab.a as f32 / 128.0;
					channels[0][2][x as usize][y as usize] = lab.b as f32 / 128.0;
				}
                Data::<f32, 4>::from(
                    channels,
                )
            })
            .map(|data| Tensor::<B, 4>::from_data(data.convert()))
            .collect();

        let transformed = items
            .iter()
            .map(|(_, transformed)| {
				let mut channels = [[[[0.0; 224]; 224]; 3]];
				for (x, y, pixel) in transformed.enumerate_pixels() {
					let lab = Lab::from_rgb(&pixel.0);
					channels[0][0][x as usize][y as usize] = lab.l as f32 / 128.0;
					channels[0][1][x as usize][y as usize] = lab.a as f32 / 128.0;
					channels[0][2][x as usize][y as usize] = lab.b as f32 / 128.0;
				}
                Data::<f32, 4>::from(
                    channels,
                )
            })
            .map(|data| Tensor::<B, 4>::from_data(data.convert()))
            .collect();

        let truth = Tensor::cat(truth, 0).to_device(&self.device);
        let transformed = Tensor::cat(transformed, 0).to_device(&self.device);

        ImageBatch { truth, transformed }
    }
}

pub struct ImageSet {
    path: PathBuf,
    count: usize,
}

impl ImageSet {
    pub fn new(path: impl AsRef<Path>, count: usize) -> Self {
        Self {
            count,
            path: path.as_ref().to_owned(),
        }
    }
}

impl Dataset<ImageItem> for ImageSet {
    fn len(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> Option<ImageItem> {
        let path = self.path.join(format!("truth/{}.png", index));
        let truth = image::open(path).expect("failed to load image");
		let truth = truth.resize_exact(224, 224, Lanczos3);
        let truth = truth.into_rgb8();

        let path = self.path.join(format!("transformed/{}.png", index));
        let transformed = image::open(path).expect("failed to load image");
		let transformed = transformed.resize_exact(224, 224, Lanczos3);
        let transformed = transformed.into_rgb8();

		Some((truth, transformed))
    }
}