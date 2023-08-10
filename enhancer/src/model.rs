use std::sync::{Mutex, OnceLock};

use burn::{
    module::Module,
    nn::{
        self,
        conv::{Conv2d, Conv2dConfig},
        loss::MSELoss,
        PaddingConfig2d,
    },
    tensor::{
        backend::{ADBackend, Backend},
        Tensor,
    },
    train::{
        metric::{Adaptor, LossInput},
        TrainOutput, TrainStep, ValidStep,
    },
};

use tensorboard_rs as tensorboard;

use crate::*;

pub struct Writer(
    tensorboard::summary_writer::SummaryWriter,
    tensorboard::summary_writer::SummaryWriter,
    usize,
    usize,
);

unsafe impl Sync for Writer {}

fn global_data() -> &'static Mutex<Writer> {
    static INSTANCE: OnceLock<Mutex<Writer>> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        Mutex::new(Writer(
            tensorboard::summary_writer::SummaryWriter::new("logdir/train"),
            tensorboard::summary_writer::SummaryWriter::new("logdir/test"),
            0,
            0,
        ))
    })
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
	conv4: Conv2d<B>,

    activation: nn::ReLU,
}

impl<B: Backend> Model<B> {
    pub fn new() -> Self {
        let conv1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();
        let conv2 = Conv2dConfig::new([64, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();
        let conv3 = Conv2dConfig::new([64, 32], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();
		let conv4 = Conv2dConfig::new([32, 3], [3, 3])
			.with_padding(PaddingConfig2d::Explicit(1, 1))
			.init();

        Self {
            conv1,
            conv2,
            conv3,
			conv4,
            activation: nn::ReLU::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
		//eprintln!("Beginning inference");
        let x = self.conv1.forward(input);
        let x = self.activation.forward(x);
		//eprintln!("First convolution complete (1/4)");

        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
		//eprintln!("Second convolution complete (2/4)");

        let x = self.conv3.forward(x);
		let x = self.activation.forward(x);
		//eprintln!("Third convolution complete (3/4)");

		let x = self.conv4.forward(x);
		//eprintln!("Last convolution complete (4/4)");

        x.tanh()
    }

    pub fn forward_generation(&self, item: ImageBatch<B>) -> EnhancementOutput<B> {
        let targets = item.truth;
        let output = self.forward(item.transformed);
        let loss = MSELoss::new();
        let loss = loss.forward(output.clone(), targets.clone(), nn::loss::Reduction::Mean);

        EnhancementOutput {
            loss,
            target: targets,
        }
    }
}

pub struct EnhancementOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub target: Tensor<B, 4>,
}

impl<B: Backend> Adaptor<LossInput<B>> for EnhancementOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

use num_traits::cast::ToPrimitive;

impl<B: ADBackend> TrainStep<ImageBatch<B>, EnhancementOutput<B>> for Model<B> {
    fn step(&self, item: ImageBatch<B>) -> TrainOutput<EnhancementOutput<B>> {
        let item = self.forward_generation(item);
		let mut writer = global_data().lock().unwrap();
		let count = writer.2;
		writer.0.add_scalar("Loss", item.loss.clone().into_data().value[0].to_f32().unwrap(), count);
		writer.2 += 1;
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ImageBatch<B>, EnhancementOutput<B>> for Model<B> {
    fn step(&self, item: ImageBatch<B>) -> EnhancementOutput<B> {
        let item = self.forward_generation(item);
		let mut writer = global_data().lock().unwrap();
		let count = writer.3;
		writer.1.add_scalar("Loss", item.loss.clone().into_data().value[0].to_f32().unwrap(), count);
		writer.3 += 1;
		item
    }
}
