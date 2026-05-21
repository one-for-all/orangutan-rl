use burn::{
    Tensor,
    prelude::Backend,
    tensor::{BasicOps, TensorKind},
};
use rand::{RngExt, rng};
use ringbuffer::{ConstGenericRingBuffer, RingBuffer};

pub struct Memory<const CAP: usize> {
    pub state: ConstGenericRingBuffer<isize, CAP>,
    pub next_state: ConstGenericRingBuffer<isize, CAP>,
    pub action: ConstGenericRingBuffer<usize, CAP>,
    pub reward: ConstGenericRingBuffer<f32, CAP>,
    pub done: ConstGenericRingBuffer<bool, CAP>,
}

impl<const CAP: usize> Default for Memory<CAP> {
    fn default() -> Self {
        Self {
            state: ConstGenericRingBuffer::new(),
            next_state: ConstGenericRingBuffer::new(),
            action: ConstGenericRingBuffer::new(),
            reward: ConstGenericRingBuffer::new(),
            done: ConstGenericRingBuffer::new(),
        }
    }
}

impl<const CAP: usize> Memory<CAP> {
    pub fn push(
        &mut self,
        state: isize,
        next_state: isize,
        action: usize,
        reward: f32,
        done: bool,
    ) {
        self.state.push(state);
        self.next_state.push(next_state);
        self.action.push(action);
        self.reward.push(reward);
        self.done.push(done);
    }

    pub fn clear(&mut self) {
        self.state.clear();
        self.next_state.clear();
        self.action.clear();
        self.reward.clear();
        self.done.clear();
    }

    pub fn len(&self) -> usize {
        self.state.len()
    }
}

pub fn get_batch<B: Backend, const CAP: usize, T, K: TensorKind<B> + BasicOps<B>>(
    data: &ConstGenericRingBuffer<T, CAP>,
    indices: &Vec<usize>,
    converter: impl Fn(&T) -> Tensor<B, 1, K>,
) -> Tensor<B, 2, K> {
    Tensor::cat(
        indices
            .iter()
            .filter_map(|i| data.get(*i))
            .map(converter)
            .collect::<Vec<_>>(),
        0,
    )
    .reshape([indices.len() as i32, -1])
}

pub fn sample_indices(indices: Vec<usize>, size: usize) -> Vec<usize> {
    let mut rng = rng();
    let mut sample = Vec::<usize>::new();
    for _ in 0..size {
        unsafe {
            let index = rng.random_range(0..indices.len());
            sample.push(*indices.get_unchecked(index));
        }
    }

    sample
}
