use burn::{
    backend::Wgpu,
    data::dataset::{Dataset, vision::MnistDataset},
};
use orangutan_rl::inference;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    let artifact_dir = "/tmp/guide";

    inference::infer::<MyBackend>(artifact_dir, device, MnistDataset::test().get(42).unwrap());
}
