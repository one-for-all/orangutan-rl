pub struct ToyEnv {
    pub pos: isize,
}

impl ToyEnv {
    pub fn new() -> Self {
        Self { pos: 0 }
    }

    pub fn state(&self) -> isize {
        self.pos
    }

    /// Returns observation
    pub fn reset(&mut self) -> isize {
        self.pos = 0;
        0
    }

    /// Returns next obs, reward, done
    pub fn step(&mut self, action: usize) -> (isize, f32, bool) {
        match action {
            0 => self.pos -= 1,
            1 => self.pos += 1,
            _ => panic!(),
        }
        let mut reward = 0.;
        let mut done = false;
        if self.pos == 1 {
            reward = 1.;
            done = true;
        } else if self.pos == -1 {
            reward = -1.;
            done = true;
        }
        (self.pos, reward, done)
    }
}
