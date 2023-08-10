use burn::module::Module;
use burn::optim::*;
use burn::record::{NoStdTrainingRecorder, Recorder};
use burn::tensor::{Data, Shape, Tensor};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    tensor::backend::ADBackend,
    train::{
        metric::{LossMetric},
        LearnerBuilder,
    },
};

mod model;
use image::imageops::FilterType::Lanczos3;
use model::*;

mod data;
use data::*;

mod lr;
use lr::*;

const ARTIFACT_DIR: &str = "./training-checkpoints";

#[derive(Config)]
pub struct AlphaZeroTrainerConfig {
    #[config(default = 255)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}

pub fn run<B: ADBackend>(device: B::Device) {
    // Config
    let config_optimizer = AdamConfig::new();
    let config = AlphaZeroTrainerConfig::new(config_optimizer);
    B::seed(config.seed);

    // Data
    let batcher_train = ImageBatcher::<B>::new(device.clone());
    let batcher_valid = ImageBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        //.num_workers(config.num_workers)
        .build(ImageSet::new("./data", 4400));
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        //.num_workers(config.num_workers)
        .build(ImageSet::new("./test-data", 100));

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .with_file_checkpointer(2, NoStdTrainingRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        //  .grads_accumulation(32)
        .build(Model::new(), config.optimizer.init(), StepLR::new(&[(0.001, 300), (0.0001, 400), (0.00001, usize::MAX)], 20, 1));

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    NoStdTrainingRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{ARTIFACT_DIR}/model").into(),
        )
        .expect("Failed to save trained model");
}

use burn_autodiff::ADBackendDecorator;
use burn_tch::{TchBackend, TchDevice};
use burn_wgpu::{AutoGraphicsApi, WgpuBackend, WgpuDevice};
use image::*;
use lab::Lab;
use std::env;

pub type InferenceBackend = burn_tch::TchBackend<f32>;

fn main() {
    #[cfg(not(target_os = "macos"))]
    let device = TchDevice::Cpu;
    #[cfg(target_os = "macos")]
    let device = TchDevice::Mps;

    run::<ADBackendDecorator<TchBackend<f32>>>(device);

	// let args: Vec<String> = env::args().collect();

	// let model: Model<InferenceBackend> = Model::new();
    // let record = burn::record::NoStdTrainingRecorder::default()
    //     .load("./training-checkpoints/best-model.bin".into())
    //     .expect("Failed to decode state");

	// let model = model.load_record(record);

	// let img = open(&args[1]).unwrap();
	// let (w, h) = img.dimensions();
	// let img = img.into_rgb8();
	// let mut channels = ndarray::ArrayBase::zeros((3, w as usize, h as usize));
	// for (x, y, pixel) in img.enumerate_pixels() {
	// 	let lab = Lab::from_rgb(&pixel.0);
	// 	channels[[0, x as usize, y as usize]] = lab.l as f32 / 128.0;
	// 	channels[[1, x as usize, y as usize]] = lab.a as f32 / 128.0;
	// 	channels[[2, x as usize, y as usize]] = lab.b as f32 / 128.0;
	// }
	// let data = Data::<f32, 4>::new(channels.clone().into_raw_vec(), Shape::new([1, 3, w as usize, h as usize]));
	// let tensor = Tensor::<InferenceBackend, 4>::from_data(data.convert());
	// let output = model.forward(tensor);
	// let mut img = RgbImage::new(w, h);
	// let matrix: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>> = ndarray::ArrayBase::from_shape_vec((2, w as usize, h as usize), output.into_data().value).unwrap();
	// for x in 0..w {
	// 	for y in 0..h {
	// 		let l = channels[[0, x as usize, y as usize]];
	// 		let a = matrix[[0, x as usize, y as usize]];
	// 		let b = matrix[[1, x as usize, y as usize]];
	// 		let lab = Lab {
	// 			l: l * 128.0,
	// 			a: a * 128.0,
	// 			b: b * 128.0
	// 		};
	// 		img.put_pixel(x as _, y as _, Rgb(lab.to_rgb()));
	// 		//img.get_pixel_mut(x, y)[i] = (plane[[x as usize, y as usize]].clamp(0.0, 255.0) * 255.0) as u8;
	// 	}
	// }
	// img.save("fixed.png").unwrap();
}