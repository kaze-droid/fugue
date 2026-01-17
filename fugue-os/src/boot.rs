use raylib::prelude::*;

pub struct BootSequence {
    logs: Vec<String>,
    visible_count: usize,
    timer: f32,
    pub is_finished: bool,
    scroll_offset: usize,
}

impl BootSequence {
    pub fn new() -> Self {
        // The "Hardcoded Aesthetic" logs
        let logs = vec![
            "Loading NEURO_KERNEL v0.9.2...".to_string(),
            "Mounting Virtual File System (VFS)...".to_string(),
            "Initializing Neural Weights...".to_string(),
            "Allocating Tensor Memory [256MB]...".to_string(),
            "Calibrating GAN Discriminator...".to_string(),
            "Connecting to Local LLM Bus...".to_string(),
            "Verifying GNN Node Integrity...".to_string(),
            "Starting Process Scheduler (RL-Agent)...".to_string(),
            "Loading User Space Drivers...".to_string(),
            "Mounting /dev/neural_net...".to_string(),
            "Checking VRAM Health...".to_string(),
            "Optimizing Hyperplane Boundaries...".to_string(),
            "Loading Desktop Environment...".to_string(),
            "Starting Window Compositor...".to_string(),
            "Establishing Uplink...".to_string(),
            "System Ready.".to_string(),
        ];

        Self {
            logs,
            visible_count: 0,
            timer: 0.0,
            is_finished: false,
            scroll_offset: 0,
        }
    }

    pub fn update(&mut self, dt: f32) {
        if self.is_finished { return; }

        self.timer += dt;

        // Speed of scrolling (0.1 = 100ms per line)
        // Make it faster towards the end for realism
        let speed = if self.visible_count < 5 { 0.3 } else { 0.15 };

        if self.timer > speed {
            self.timer = 0.0;
            if self.visible_count < self.logs.len() {
                self.visible_count += 1;
                
                // Auto-scroll logic: Keep the last 25 lines visible
                if self.visible_count > 25 {
                    self.scroll_offset = self.visible_count - 25;
                }
            } else {
                // Add a small delay at the end before finishing
                if self.timer > 1.0 { 
                    self.is_finished = true; 
                }
                // Hack: force finish after a moment
                self.is_finished = true;
            }
        }
    }

    pub fn draw(&self, d: &mut RaylibDrawHandle) {
        d.clear_background(Color::BLACK);

        let start_x = 20;
        let mut y = 20;
        let line_height = 20;

        // Draw visible lines based on scroll offset
        for i in self.scroll_offset..self.visible_count {
            if i >= self.logs.len() { break; }

            // Draw the "[  OK  ]" part
            d.draw_text("[", start_x, y, 20, Color::WHITE);
            d.draw_text(" OK ", start_x + 10, y, 20, Color::GREEN);
            d.draw_text("]", start_x + 58, y, 20, Color::WHITE);

            // Draw the message
            d.draw_text(&self.logs[i], start_x + 80, y, 20, Color::WHITE);

            y += line_height;
        }

        // Draw a blinking cursor at the bottom
        if (d.get_time() * 2.0) as i32 % 2 == 0 {
            d.draw_rectangle(start_x, y + 5, 10, 15, Color::WHITE);
        }
    }
}
