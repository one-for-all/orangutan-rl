use burn::backend::{Autodiff, Wgpu};
use orangutan_rl::{env::toy::ToyEnv, run::run};

type Backend = Autodiff<Wgpu<f32, i32>>;

fn main() {
    let mut env = ToyEnv::new();
    let mut state = env.state();

    let mut agent = run::<Backend>();
    let mut done = false;
    while !done {
        let action = agent.react(&state);
        let (next_state, reward, next_done) = env.step(action);
        state = next_state;
        done = next_done;

        println!("{action}: {reward}, {state}, {done}");
    }
}
