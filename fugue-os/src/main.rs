mod hardware;
mod desktop;
mod boot;

use hardware::VirtualHardware;
use desktop::{DesktopEnv, AppType};
use raylib::prelude::*;
use boot::BootSequence;
use std::io::Write;
use std::fs::OpenOptions;

enum AppState {
    Booting,
    Running
}

fn main() {
    let (mut rl, thread) = raylib::init()
        .size(1280, 800)
        .title("FUGUE OS")
        .resizable()
        .vsync()
        .build();

    let mut os_hw = VirtualHardware::new();
    // Initialize AI
    os_hw.load_brain();

    let mut desktop = DesktopEnv::new();
    let mut boot_seq = BootSequence::new();

    if let Ok(tex) = rl.load_texture(&thread, "wallpapers/Cyberpunk_Street.png") {
        desktop.wallpaper = Some(tex);
    } else {
        eprintln!("Failed to load wallpaper: wallpapers/Italian_Riveria.png");
    }


    // Start in booting mode
    let mut current_state = AppState::Booting;

    // Training Mode State
    let mut is_training = false;
    let mut training_timer = 0;

    while !rl.window_should_close() {
        match current_state {
            AppState::Booting => {
                boot_seq.update(rl.get_frame_time());
                if boot_seq.is_finished {
                    current_state = AppState::Running;
                    // Open the default app ONLY when boot finishes
                    desktop.open_window(&mut os_hw, AppType::SysMonitor);
                }
                let mut d = rl.begin_drawing(&thread);
                boot_seq.draw(&mut d);
            }

            AppState::Running => {
                while let Some(key) = rl.get_char_pressed() {
                    let k = key as u32;
                    // ASCII printable
                    if k >= 32 && k <= 125 {
                        os_hw.shell_buffer.push(key as char);
                    }
                }

                if rl.is_key_pressed(KeyboardKey::KEY_BACKSPACE) {
                    os_hw.shell_buffer.pop();
                }

                // Input Tracking
                let mouse_delta = rl.get_mouse_delta();
                let velocity = (mouse_delta.x.powi(2) + mouse_delta.y.powi(2)).sqrt();
                os_hw.system_state.mouse_velocity = velocity;

                if velocity > 0.1 || rl.get_key_pressed().is_some() || rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT) {
                    os_hw.system_state.time_since_last_input = 0.0;
                } else {
                    os_hw.system_state.time_since_last_input += rl.get_frame_time();
                }

                // Update
                os_hw.update_physics();
                desktop.update(&rl, &mut os_hw);

                // Shortcuts
                if rl.is_key_pressed(KeyboardKey::KEY_F1) { desktop.open_window(&mut os_hw, AppType::SysMonitor); }
                if rl.is_key_pressed(KeyboardKey::KEY_F2) { desktop.open_window(&mut os_hw, AppType::Terminal); }
                if rl.is_key_pressed(KeyboardKey::KEY_F3) { desktop.open_window(&mut os_hw, AppType::FileUniverse); } // NEW
                if rl.is_key_pressed(KeyboardKey::KEY_F4) { 
                    // Close all
                    while !desktop.windows.is_empty() { desktop.close_window(&mut os_hw, 0); }
                }
                if rl.is_key_pressed(KeyboardKey::KEY_F5) {
                    is_training = !is_training;
                    println!("TRAINING MODE: {}", is_training);
                }

                if is_training {
                    training_timer += 1;
                    if training_timer > 30 { // Log every 30 ticks (approx 0.5s)
                        training_timer = 0;
                        let state_vec = os_hw.system_state.to_state_vector();
                        let csv_line = state_vec.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(",");
                        
                        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("training_data.csv") {
                            let _ = writeln!(file, "{}", csv_line);
                        }
                    }
                }

                // Draw
                let mut d = rl.begin_drawing(&thread);
                desktop.draw(&mut d, &os_hw);
                
                if is_training {
                    d.draw_text("REC", 1200, 10, 20, Color::RED);
                }
            }
        }
    }
}
