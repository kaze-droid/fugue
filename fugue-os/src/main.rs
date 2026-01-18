mod hardware;
mod desktop;
mod boot;
mod theme_engine;
mod graph_fs;
mod slm;

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
        .build();  // Removed vsync for manual FPS control

    let mut os_hw = VirtualHardware::new();
    // Initialize AI
    os_hw.load_brain();

    let mut desktop = DesktopEnv::new();
    let mut boot_seq = BootSequence::new();

    if let Ok(tex) = rl.load_texture(&thread, "wallpapers/Cyberpunk_Street.png") {
        desktop.wallpaper = Some(tex);
    } else {
        eprintln!("Failed to load wallpaper: wallpapers/Cyberpunk_Street.png");
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
                // --- SLM STREAMING (Visual Only) ---
                while let Ok(token) = os_hw.stream_rx.try_recv() {
                    os_hw.partial_response.push_str(&token);
                }
                
                // --- INTENT EXECUTION ---
                while let Ok(intent) = os_hw.intent_rx.try_recv() {
                    os_hw.is_thinking = false;
                    os_hw.kernel_messages.push_front(format!("[KERNEL] {}", intent.response));
                    
                    match intent.action.as_str() {
                        "open" => {
                            if let Some(ref target) = intent.target {
                                let app = match target.to_lowercase().as_str() {
                                    "monitor" | "sysmonitor" => Some(AppType::SysMonitor),
                                    "files" | "fileuniverse" | "filesystem" => Some(AppType::FileUniverse),
                                    "terminal" | "shell" => Some(AppType::Terminal),
                                    _ => None
                                };
                                if let Some(app_type) = app {
                                    // Open N instances (default 1, max 20 for safety)
                                    let count = intent.count.unwrap_or(1).min(20);
                                    for _ in 0..count {
                                        desktop.open_window(&mut os_hw, app_type.clone());
                                    }
                                    if count > 1 {
                                        os_hw.kernel_messages.push_front(
                                            format!("[SYS] Spawned {} instances of {:?}", count, app_type)
                                        );
                                    }
                                }
                            }
                        }
                        "close" => {
                            if let Some(ref target) = intent.target {
                                let count = intent.count.unwrap_or(1) as usize;
                                match target.to_lowercase().as_str() {
                                    "all" => {
                                        while !desktop.windows.is_empty() {
                                            desktop.close_window(&mut os_hw, 0);
                                        }
                                    }
                                    "monitor" => {
                                        for _ in 0..count {
                                            if let Some(idx) = desktop.windows.iter().position(|w| w.app_type == AppType::SysMonitor) {
                                                desktop.close_window(&mut os_hw, idx);
                                            } else { break; }
                                        }
                                    }
                                    "terminal" => {
                                        for _ in 0..count {
                                            if let Some(idx) = desktop.windows.iter().position(|w| w.app_type == AppType::Terminal) {
                                                desktop.close_window(&mut os_hw, idx);
                                            } else { break; }
                                        }
                                    }
                                    "files" => {
                                        for _ in 0..count {
                                            if let Some(idx) = desktop.windows.iter().position(|w| w.app_type == AppType::FileUniverse) {
                                                desktop.close_window(&mut os_hw, idx);
                                            } else { break; }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        "theme" => {
                            // Change wallpaper using theme engine
                            if let Some(ref val) = intent.value {
                                if let Some(ref mut theme_engine) = os_hw.theme_engine {
                                    match theme_engine.find_wallpaper(val) {
                                        Ok((wallpaper_name, score)) => {
                                            let wallpaper_path = format!("wallpapers/{}.png", wallpaper_name);
                                            if let Ok(tex) = rl.load_texture(&thread, &wallpaper_path) {
                                                desktop.wallpaper = Some(tex);
                                                os_hw.kernel_messages.push_front(
                                                    format!("[THEME] Loaded '{}' (match: {:.0}%)", wallpaper_name, score * 100.0)
                                                );
                                            }
                                        }
                                        Err(e) => {
                                            os_hw.kernel_messages.push_front(format!("[THEME] Error: {}", e));
                                        }
                                    }
                                }
                            }
                            // Update accent/bg colors (strictly 6 hex chars)
                            if let (Some(acc), Some(bg)) = (&intent.accent_hex, &intent.bg_hex) {
                                desktop.update_theme(acc, bg);
                                os_hw.kernel_messages.push_front(
                                    format!("[THEME] Colors: accent=#{}, bg=#{}", 
                                        acc.trim_start_matches('#').chars().take(6).collect::<String>(),
                                        bg.trim_start_matches('#').chars().take(6).collect::<String>())
                                );
                            }
                        }
                        "toggle" => {
                            if let Some(ref target) = intent.target {
                                let enabled = intent.value.as_ref().map(|v| v.to_lowercase() == "on").unwrap_or(false);
                                let (name, toggled) = match target.to_lowercase().as_str() {
                                    "rl" | "scheduler" => {
                                        os_hw.rl_scheduler_enabled = enabled;
                                        ("RL Scheduler", true)
                                    }
                                    "vae" | "memory" | "defrag" => {
                                        os_hw.vae_enabled = enabled;
                                        ("VAE Neural Defrag", true)
                                    }
                                    "theme" => {
                                        os_hw.theme_enabled = enabled;
                                        ("Theme Engine", true)
                                    }
                                    "embeddings" | "semantic" => {
                                        os_hw.embeddings_enabled = enabled;
                                        ("File Embeddings", true)
                                    }
                                    _ => ("Unknown", false)
                                };
                                if toggled {
                                    os_hw.kernel_messages.push_front(
                                        format!("[SYS] {}: {}", name, if enabled { "ENABLED" } else { "DISABLED" })
                                    );
                                }
                            }
                        }
                        "create" => {
                            // Create a new file
                            if let Some(ref filename) = intent.value {
                                let path = format!("/home/user/{}", filename);
                                let content = String::new();
                                let file_id = os_hw.file_system.create_file(filename.clone(), path, content);
                                os_hw.kernel_messages.push_front(
                                    format!("[FS] Created file '{}' (ID: {:04X})", filename, file_id)
                                );
                                // Update edges if embeddings enabled
                                if os_hw.embeddings_enabled {
                                    os_hw.file_system.update_edges_by_similarity();
                                }
                            }
                        }
                        "edit" | "modify" => {
                            // Edit/modify a file
                            if let Some(ref target) = intent.target {
                                // Try to find file by name or ID
                                let file_id = if let Ok(id) = target.parse::<u32>() {
                                    Some(id)
                                } else {
                                    // Search by name
                                    os_hw.file_system.get_all_nodes().iter()
                                        .find(|n| n.name.to_lowercase().contains(&target.to_lowercase()))
                                        .map(|n| n.id)
                                };
                                
                                if let Some(id) = file_id {
                                    if let Some(ref new_content) = intent.value {
                                        if os_hw.file_system.write_file(id, new_content.clone()) {
                                            os_hw.kernel_messages.push_front(
                                                format!("[FS] Updated file ID {:04X}", id)
                                            );
                                        } else {
                                            os_hw.kernel_messages.push_front(
                                                format!("[FS] Failed to update file ID {:04X}", id)
                                            );
                                        }
                                    }
                                } else {
                                    os_hw.kernel_messages.push_front(
                                        format!("[FS] File '{}' not found", target)
                                    );
                                }
                            }
                        }
                        "delete" | "remove" => {
                            // Delete a file
                            if let Some(ref target) = intent.target {
                                // Try to find file by name or ID
                                let file_id = if let Ok(id) = target.parse::<u32>() {
                                    Some(id)
                                } else {
                                    // Search by name
                                    os_hw.file_system.get_all_nodes().iter()
                                        .find(|n| n.name.to_lowercase().contains(&target.to_lowercase()))
                                        .map(|n| n.id)
                                };
                                
                                if let Some(id) = file_id {
                                    if os_hw.file_system.remove_node(id) {
                                        os_hw.kernel_messages.push_front(
                                            format!("[FS] Deleted file ID {:04X}", id)
                                        );
                                    } else {
                                        os_hw.kernel_messages.push_front(
                                            format!("[FS] Failed to delete file ID {:04X}", id)
                                        );
                                    }
                                } else {
                                    os_hw.kernel_messages.push_front(
                                        format!("[FS] File '{}' not found", target)
                                    );
                                }
                            }
                        }
                        "chat" | _ => {
                            // Just show response (already added above)
                        }
                    }
                }
                
                // Check if terminal window is open (for SLM commands)
                let terminal_open = desktop.windows.iter().any(|w| w.app_type == AppType::Terminal && w.is_open);
                
                // Handle input based on mode
                if desktop.input_mode == InputMode::Normal {
                    // Normal mode - shell buffer input (only when terminal is open)
                    if terminal_open {
                        // Insert characters at cursor position
                        while let Some(key) = rl.get_char_pressed() {
                            let k = key as u32;
                            // ASCII printable
                            if k >= 32 && k <= 125 {
                                let cursor = os_hw.shell_cursor.min(os_hw.shell_buffer.len());
                                os_hw.shell_buffer.insert(cursor, key as char);
                                os_hw.shell_cursor = cursor + 1;
                            }
                        }

                        // --- HOLD-TO-DELETE (30 frames threshold) ---
                        if rl.is_key_down(KeyboardKey::KEY_BACKSPACE) {
                            os_hw.backspace_held_frames += 1;
                            
                            // Delete on initial press OR after holding for 30 frames (then every 3 frames)
                            let should_delete = rl.is_key_pressed(KeyboardKey::KEY_BACKSPACE) 
                                || (os_hw.backspace_held_frames > 30 && os_hw.backspace_held_frames % 3 == 0);
                            
                            if should_delete && os_hw.shell_cursor > 0 && !os_hw.shell_buffer.is_empty() {
                                let cursor = os_hw.shell_cursor.min(os_hw.shell_buffer.len());
                                if cursor > 0 {
                                    os_hw.shell_buffer.remove(cursor - 1);
                                    os_hw.shell_cursor = cursor - 1;
                                }
                            }
                        } else {
                            os_hw.backspace_held_frames = 0;
                        }
                        
                        // --- ARROW KEYS FOR CURSOR MOVEMENT ---
                        if rl.is_key_pressed(KeyboardKey::KEY_LEFT) {
                            if os_hw.shell_cursor > 0 {
                                os_hw.shell_cursor -= 1;
                            }
                        }
                        if rl.is_key_pressed(KeyboardKey::KEY_RIGHT) {
                            if os_hw.shell_cursor < os_hw.shell_buffer.len() {
                                os_hw.shell_cursor += 1;
                            }
                        }
                        // Home/End keys
                        if rl.is_key_pressed(KeyboardKey::KEY_HOME) {
                            os_hw.shell_cursor = 0;
                        }
                        if rl.is_key_pressed(KeyboardKey::KEY_END) {
                            os_hw.shell_cursor = os_hw.shell_buffer.len();
                        }
                        
                        // Handle Enter key - trigger SLM inference via Ollama
                        if rl.is_key_pressed(KeyboardKey::KEY_ENTER) && !os_hw.shell_buffer.is_empty() {
                            let prompt = os_hw.shell_buffer.clone();
                            os_hw.shell_buffer.clear();
                            os_hw.shell_cursor = 0;  // Reset cursor
                            os_hw.partial_response.clear();
                            os_hw.kernel_messages.push_front(format!("user@fugue-os:~$ {}", prompt));
                            os_hw.is_thinking = true;

                            // Run Ollama inference in background thread
                            if os_hw.ollama_available {
                                let s_tx = os_hw.stream_tx.clone();
                                let i_tx = os_hw.intent_tx.clone();
                                let prompt_clone = prompt.clone();
                                
                                std::thread::spawn(move || {
                                    if let Some(intent) = crate::slm::run_ollama_inference(prompt_clone, s_tx) {
                                        println!("[SLM] Parsed intent: {:?}", intent);
                                        let _ = i_tx.send(intent);
                                    }
                                });
                            } else {
                                os_hw.is_thinking = false;
                                os_hw.kernel_messages.push_front("[KERNEL] Ollama not available - start with: ollama serve".to_string());
                            }
                        }
                        
                        // --- CTRL+C: Terminate/Clear ---
                        if rl.is_key_down(KeyboardKey::KEY_LEFT_CONTROL) && rl.is_key_pressed(KeyboardKey::KEY_C) {
                            if os_hw.is_thinking {
                                // Terminate LLM call
                                os_hw.is_thinking = false;
                                os_hw.kernel_messages.push_front("^C [INTERRUPTED]".to_string());
                                if !os_hw.partial_response.is_empty() {
                                    os_hw.kernel_messages.push_front(format!("> {}", os_hw.partial_response));
                                }
                                os_hw.partial_response.clear();
                            } else if !os_hw.shell_buffer.is_empty() {
                                // Clear current input
                                os_hw.kernel_messages.push_front(format!("user@fugue-os:~$ {}^C", os_hw.shell_buffer));
                                os_hw.shell_buffer.clear();
                                os_hw.shell_cursor = 0;
                            }
                        }
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
                    // Input mode - capture text for file creation/import/search
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
                            InputMode::Searching => {
                                // Perform semantic search
                                let query = desktop.input_buffer.clone();
                                println!("Searching for: '{}'", query);
                                
                                let results = os_hw.file_system.search(&query, 10);
                                desktop.search_results = results.iter()
                                    .map(|r| (r.node_id, r.node.name.clone(), r.score))
                                    .collect();
                                
                                println!("Found {} results", desktop.search_results.len());
                                for (i, (_, name, score)) in desktop.search_results.iter().enumerate() {
                                    println!("  {}. {} ({:.1}%)", i + 1, name, score * 100.0);
                                }
                                
                                // Open FileUniverse if not already open
                                if !desktop.windows.iter().any(|w| w.app_type == AppType::FileUniverse && w.is_open) {
                                    desktop.open_window(&mut os_hw, AppType::FileUniverse);
                                }
                            }
                            InputMode::CreatingFile => {
                                // Create a new file in the FileUniverse
                                let filename = desktop.input_buffer.clone();
                                let path = format!("/home/user/{}", filename);
                                let content = "New file created in FileUniverse".to_string();
                                let file_id = os_hw.file_system.create_file(filename, path, content);
                                println!("Created file with ID: {}", file_id);
                                // Update edges after creating a new file
                                os_hw.file_system.update_edges_by_similarity();
                            }
                            InputMode::ImportingFile => {
                                // Import a file from the real filesystem
                                let file_path = desktop.input_buffer.clone();
                                match read_to_string(&file_path) {
                                    Ok(content) => {
                                        let filename = file_path.split(&['/', '\\']).last().unwrap_or(&file_path).to_string();
                                        let file_id = os_hw.file_system.create_file(filename.clone(), file_path.clone(), content);
                                        println!("Imported file '{}' with ID: {}", filename, file_id);
                                        // Update edges after importing a new file
                                        os_hw.file_system.update_edges_by_similarity();
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

                // Manual FPS limiting based on CPU pressure
                rl.set_target_fps(os_hw.target_fps);

                // Shortcuts (only in normal mode)
                if desktop.input_mode == InputMode::Normal {
                    if rl.is_key_pressed(KeyboardKey::KEY_F1) { desktop.open_window(&mut os_hw, AppType::SysMonitor); }
                    if rl.is_key_pressed(KeyboardKey::KEY_F2) { 
                        desktop.open_window(&mut os_hw, AppType::Terminal);
                        // Auto-spawn stress test if kernel is rescheduling
                        if os_hw.current_mindset == hardware::KernelMindset::Rescheduling {
                            os_hw.spawn_stress_test();
                            println!("STRESS TEST AUTO-SPAWNED (Kernel Rescheduling)");
                        }
                    }
                    if rl.is_key_pressed(KeyboardKey::KEY_F3) { desktop.open_window(&mut os_hw, AppType::FileUniverse); }
                    if rl.is_key_pressed(KeyboardKey::KEY_F4) { 
                        // Close all
                        while !desktop.windows.is_empty() { desktop.close_window(&mut os_hw, 0); }
                    }
                    if rl.is_key_pressed(KeyboardKey::KEY_F5) {
                        // Create file mode
                        desktop.input_mode = InputMode::CreatingFile;
                        desktop.input_buffer.clear();
                        println!("CREATE FILE MODE - Enter filename");
                    }
                    if rl.is_key_pressed(KeyboardKey::KEY_F6) {
                        // Import file mode
                        desktop.input_mode = InputMode::ImportingFile;
                        desktop.input_buffer.clear();
                        println!("IMPORT FILE MODE - Enter file path");
                    }
                    if rl.is_key_pressed(KeyboardKey::KEY_F7) {
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
                    if rl.is_key_pressed(KeyboardKey::KEY_F8) {
                        is_training = !is_training;
                        println!("TRAINING MODE: {}", is_training);
                    }
                    if rl.is_key_pressed(KeyboardKey::KEY_F9) {
                        // Search mode
                        desktop.input_mode = InputMode::Searching;
                        desktop.input_buffer.clear();
                        desktop.search_results.clear();
                        println!("SEARCH MODE - Enter keywords");
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
