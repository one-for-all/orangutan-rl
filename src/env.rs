pub struct Env {
    pub pos: isize,
}

impl Env {
    pub fn new() -> Self {
        Self { pos: 0 }
    }

    /// Returns observation
    pub fn reset(&mut self) -> isize {
        self.pos = 0;
        0
    }

    /// Returns next obs, reward,
    pub fn step(&mut self, action: isize) -> (isize, f32) {
        self.pos += action;
        let reward = if self.pos == 3 { 1. } else { 0. };
        (self.pos, reward)
    }
}
