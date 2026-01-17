mod hardware;
mod desktop;
mod boot;
mod theme_engine;

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
        .build();  // Removed vsync for manual FPS control

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

    // RL Scheduler Toggle
    let mut rl_scheduler_enabled = false;  // Default OFF

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
                // Check if terminal window is open
                let terminal_open = desktop.windows.iter().any(|w| w.app_type == AppType::Terminal && w.is_open);
                
                // Only capture input if terminal is open
                if terminal_open {
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
                }

                // Handle Enter key for theme commands (only if terminal is open)
                if rl.is_key_pressed(KeyboardKey::KEY_ENTER) && terminal_open {
                    let command = os_hw.shell_buffer.trim().to_string();
                    println!("[DEBUG] Enter pressed. Command: '{}'", command);
                    println!("[DEBUG] Theme engine available: {}", os_hw.theme_engine.is_some());
                    
                    // Check if it's a theme command
                    if command.starts_with("theme ") || command.starts_with("wallpaper ") {
                        println!("[DEBUG] Theme command detected");
                        let prompt = if command.starts_with("theme ") {
                            command.strip_prefix("theme ").unwrap_or("")
                        } else {
                            command.strip_prefix("wallpaper ").unwrap_or("")
                        };
                        
                        if !prompt.is_empty() {
                            println!("[DEBUG] Prompt: '{}'", prompt);
                            // Find matching wallpaper
                            if let Some(theme_engine) = &mut os_hw.theme_engine {
                                match theme_engine.find_wallpaper(prompt) {
                                    Ok((wallpaper_name, similarity)) => {
                                        println!("→ Loading '{}' (similarity: {:.2})", wallpaper_name, similarity);
                                        
                                        // Load the wallpaper
                                        let wallpaper_path = format!("wallpapers/{}.png", wallpaper_name);
                                        if let Ok(tex) = rl.load_texture(&thread, &wallpaper_path) {
                                            desktop.wallpaper = Some(tex);
                                        } else {
                                            eprintln!("Failed to load wallpaper: {}", wallpaper_path);
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("[ERROR] Theme engine error: {}", e);
                                    }
                                }
                            } else {
                                eprintln!("[ERROR] Theme engine not initialized!");
                            }
                        }
                    }
                    
                    // Clear shell buffer after command
                    os_hw.shell_buffer.clear();
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
                os_hw.rl_scheduler_enabled = rl_scheduler_enabled;
                os_hw.update_physics();
                desktop.update(&rl, &mut os_hw);

                // Manual FPS limiting based on CPU pressure
                rl.set_target_fps(os_hw.target_fps);

                // Shortcuts
                if rl.is_key_pressed(KeyboardKey::KEY_F1) { desktop.open_window(&mut os_hw, AppType::SysMonitor); }
                if rl.is_key_pressed(KeyboardKey::KEY_F2) { 
                    desktop.open_window(&mut os_hw, AppType::Terminal);
                    // Auto-spawn stress test if kernel is rescheduling
                    if os_hw.current_mindset == hardware::KernelMindset::Rescheduling {
                        os_hw.spawn_stress_test();
                        println!("STRESS TEST AUTO-SPAWNED (Kernel Rescheduling)");
                    }
                }
                if rl.is_key_pressed(KeyboardKey::KEY_F3) { desktop.open_window(&mut os_hw, AppType::FileUniverse); } // NEW
                if rl.is_key_pressed(KeyboardKey::KEY_F4) { 
                    // Close all
                    while !desktop.windows.is_empty() { desktop.close_window(&mut os_hw, 0); }
                }
                if rl.is_key_pressed(KeyboardKey::KEY_F5) {
                    is_training = !is_training;
                    println!("TRAINING MODE: {}", is_training);
                }
                if rl.is_key_pressed(KeyboardKey::KEY_F6) {
                    rl_scheduler_enabled = !rl_scheduler_enabled;
                    println!("RL SCHEDULER: {}", if rl_scheduler_enabled { "ENABLED" } else { "DISABLED" });
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
