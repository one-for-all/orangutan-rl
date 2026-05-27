pub mod double_integrator;
pub mod grid;
pub mod toy;

pub trait Observation {}

pub trait Env {
    fn reset(&mut self) -> impl Observation;
}
