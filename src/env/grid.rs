/// 1-dimensional grid world environment
pub struct OneDimGridEnv {
    x: f32,

    i_step: usize, // max 5 steps
}

impl OneDimGridEnv {
    pub fn new() -> Self {
        Self { x: 0., i_step: 0 }
    }

    /// 1-dim continous observation
    pub fn obs_dim(&self) -> usize {
        1
    }

    /// 2 discrete actions
    pub fn n_acts(&self) -> usize {
        2
    }

    pub fn reset(&mut self) -> f32 {
        self.x = 0.;
        self.i_step = 0;
        self.x
    }

    pub fn step(&mut self, action: i32) -> (f32, f32, bool) {
        let mut reward = -1.;
        let mut done = false;
        match action {
            0 => {}
            1 => {
                self.x += 1.;
                if self.x == 3. {
                    reward = 10.;
                    done = true;
                }
            }
            _ => panic!("unknown action: {action}"),
        }

        let obs = self.x;
        self.i_step += 1;
        if self.i_step >= 5 {
            done = true;
        }
        return (obs, reward, done);
    }
}
