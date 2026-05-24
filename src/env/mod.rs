pub mod dynamic;
pub mod grid;
pub mod toy;

pub trait Observation {}

pub trait Env {
    fn reset(&mut self) -> impl Observation;
}
