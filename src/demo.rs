use orangutan_rl::{agent::Agent, env::Env};

fn main() {
    let mut env = Env::new();
    let mut state = env.state();
    let mut agent = Agent::new();
    let mut done = false;
    while !done {
        let action = agent.react(&state);
        let (next_state, reward, next_done) = env.step(action);
        state = next_state;
        done = next_done;

        println!("{action}: {reward}, {state}, {done}");
    }
}
