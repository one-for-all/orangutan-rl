pub struct Agent {
    pub q_table: [[f32; 2]; 5],
}

impl Agent {
    pub fn new() -> Self {
        Self {
            q_table: [[0.; 2]; 5], // 5 states, 2 actions
        }
    }

    pub fn get_action_and_value(&self, obs: isize) -> (isize, f32, f32) {
        (1, 0., 0.)
    }
}
