mod hardware;
mod desktop;
mod boot;
mod graph_fs;

use hardware::VirtualHardware;
use desktop::{DesktopEnv, AppType, InputMode};
use raylib::prelude::*;
use boot::BootSequence;
use std::io::Write;
use std::fs::{OpenOptions, read_to_string};

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
    
    // Auto-save timer
    let mut autosave_timer = 0;

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
                // Handle input based on mode
                if desktop.input_mode == InputMode::Normal {
                    // Normal mode - shell buffer input
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
                } else if matches!(desktop.input_mode, InputMode::EditingFile(_)) {
                    // Editing mode - multiline text editor
                    while let Some(key) = rl.get_char_pressed() {
                        let k = key as u32;
                        // ASCII printable
                        if k >= 32 && k <= 125 {
                            if desktop.cursor_line < desktop.multiline_buffer.len() {
                                desktop.multiline_buffer[desktop.cursor_line].push(key as char);
                            }
                        }
                    }

                    if rl.is_key_pressed(KeyboardKey::KEY_BACKSPACE) {
                        if desktop.cursor_line < desktop.multiline_buffer.len() {
                            if !desktop.multiline_buffer[desktop.cursor_line].is_empty() {
                                desktop.multiline_buffer[desktop.cursor_line].pop();
                            } else if desktop.cursor_line > 0 {
                                // Delete empty line and move up
                                desktop.multiline_buffer.remove(desktop.cursor_line);
                                desktop.cursor_line -= 1;
                            }
                        }
                    }
                    
                    if rl.is_key_pressed(KeyboardKey::KEY_ENTER) {
                        // Add new line
                        desktop.cursor_line += 1;
                        desktop.multiline_buffer.insert(desktop.cursor_line, String::new());
                    }
                    
                    if rl.is_key_pressed(KeyboardKey::KEY_UP) && desktop.cursor_line > 0 {
                        desktop.cursor_line -= 1;
                    }
                    
                    if rl.is_key_pressed(KeyboardKey::KEY_DOWN) && desktop.cursor_line < desktop.multiline_buffer.len() - 1 {
                        desktop.cursor_line += 1;
                    }
                    
                    // Handle CTRL-S to save
                    if rl.is_key_down(KeyboardKey::KEY_LEFT_CONTROL) && rl.is_key_pressed(KeyboardKey::KEY_S) {
                        if let InputMode::EditingFile(file_id) = desktop.input_mode {
                            let content = desktop.multiline_buffer.join("\n");
                            if os_hw.file_system.write_file(file_id, content) {
                                println!("Saved file ID: {}", file_id);
                                // Also save to disk
                                if let Err(e) = os_hw.save_file_system() {
                                    eprintln!("Failed to save to disk: {}", e);
                                }
                            }
                        }
                    }
                    
                    // Handle CTRL-Q to cancel
                    if rl.is_key_down(KeyboardKey::KEY_LEFT_CONTROL) && rl.is_key_pressed(KeyboardKey::KEY_Q) {
                        desktop.input_mode = InputMode::Normal;
                        desktop.multiline_buffer = vec![String::new()];
                        desktop.cursor_line = 0;
                    }
                } else {
                    // Input mode - capture text for file creation/import
                    while let Some(key) = rl.get_char_pressed() {
                        let k = key as u32;
                        // ASCII printable
                        if k >= 32 && k <= 125 {
                            desktop.input_buffer.push(key as char);
                        }
                    }

                    if rl.is_key_pressed(KeyboardKey::KEY_BACKSPACE) {
                        desktop.input_buffer.pop();
                    }
                    
                    // Handle ENTER to confirm
                    if rl.is_key_pressed(KeyboardKey::KEY_ENTER) && !desktop.input_buffer.is_empty() {
                        match desktop.input_mode {
                            InputMode::CreatingFile => {
                                // Create a new file in the FileUniverse
                                let filename = desktop.input_buffer.clone();
                                let path = format!("/home/user/{}", filename);
                                let content = "New file created in FileUniverse".to_string();
                                let file_id = os_hw.file_system.create_file(filename, path, content);
                                println!("Created file with ID: {}", file_id);
                            }
                            InputMode::ImportingFile => {
                                // Import a file from the real filesystem
                                let file_path = desktop.input_buffer.clone();
                                match read_to_string(&file_path) {
                                    Ok(content) => {
                                        let filename = file_path.split(&['/', '\\']).last().unwrap_or(&file_path).to_string();
                                        let file_id = os_hw.file_system.create_file(filename.clone(), file_path.clone(), content);
                                        println!("Imported file '{}' with ID: {}", filename, file_id);
                                    }
                                    Err(e) => {
                                        eprintln!("Failed to import file '{}': {}", file_path, e);
                                    }
                                }
                            }
                            _ => {}
                        }
                        desktop.input_buffer.clear();
                        desktop.input_mode = InputMode::Normal;
                    }
                    
                    // Handle CTRL-Q to cancel
                    if rl.is_key_down(KeyboardKey::KEY_LEFT_CONTROL) && rl.is_key_pressed(KeyboardKey::KEY_Q) {
                        desktop.input_buffer.clear();
                        desktop.input_mode = InputMode::Normal;
                    }
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

                // Shortcuts (only in normal mode)
                if desktop.input_mode == InputMode::Normal {
                    if rl.is_key_pressed(KeyboardKey::KEY_F1) { desktop.open_window(&mut os_hw, AppType::SysMonitor); }
                    if rl.is_key_pressed(KeyboardKey::KEY_F2) { desktop.open_window(&mut os_hw, AppType::Terminal); }
                    if rl.is_key_pressed(KeyboardKey::KEY_F3) { desktop.open_window(&mut os_hw, AppType::FileUniverse); }
                    if rl.is_key_pressed(KeyboardKey::KEY_F4) { 
                        // Close all
                        while !desktop.windows.is_empty() { desktop.close_window(&mut os_hw, 0); }
                    }
                    if rl.is_key_pressed(KeyboardKey::KEY_F5) {
                        is_training = !is_training;
                        println!("TRAINING MODE: {}", is_training);
                    }
                    if rl.is_key_pressed(KeyboardKey::KEY_F6) {
                        // Create file mode
                        desktop.input_mode = InputMode::CreatingFile;
                        desktop.input_buffer.clear();
                        println!("CREATE FILE MODE - Enter filename");
                    }
                    if rl.is_key_pressed(KeyboardKey::KEY_F7) {
                        // Import file mode
                        desktop.input_mode = InputMode::ImportingFile;
                        desktop.input_buffer.clear();
                        println!("IMPORT FILE MODE - Enter file path");
                    }
                    if rl.is_key_pressed(KeyboardKey::KEY_F8) {
                        // Edit selected file
                        if let Some(file_id) = os_hw.selected_file_id {
                            if let Some(content) = os_hw.file_system.read_file(file_id) {
                                desktop.input_mode = InputMode::EditingFile(file_id);
                                desktop.multiline_buffer = content.lines().map(|s| s.to_string()).collect();
                                if desktop.multiline_buffer.is_empty() {
                                    desktop.multiline_buffer.push(String::new());
                                }
                                desktop.cursor_line = 0;
                                println!("EDITING FILE ID: {}", file_id);
                            }
                        } else {
                            println!("No file selected. Click on a file in FileUniverse first.");
                        }
                    }
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
                
                // Auto-save file system every 300 ticks (approx 5 seconds at 60fps)
                autosave_timer += 1;
                if autosave_timer > 300 {
                    autosave_timer = 0;
                    if let Err(e) = os_hw.save_file_system() {
                        eprintln!("Auto-save failed: {}", e);
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
    
    // Save file system on exit
    println!("Saving file system before exit...");
    if let Err(e) = os_hw.save_file_system() {
        eprintln!("Failed to save file system: {}", e);
    }
}
