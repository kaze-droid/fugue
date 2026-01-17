use raylib::prelude::*;
use crate::hardware::{VirtualHardware, BlockType, KernelMindset};
use std::collections::VecDeque;

pub struct Theme {
    pub win_bg: Color,
    pub border_active: Color,
    pub border_inactive: Color,
    pub header_active: Color,
    pub header_inactive: Color,
    pub text: Color,
    pub text_dim: Color,
    pub accent: Color,
    pub glitch: Color,
}

const CYBER_THEME: Theme = Theme {
    win_bg: Color::new(15, 15, 20, 245),
    border_active: Color::new(200, 200, 200, 255),
    border_inactive: Color::new(60, 60, 70, 255),
    header_active: Color::new(40, 40, 50, 255),
    header_inactive: Color::new(25, 25, 30, 255),
    text: Color::RAYWHITE,
    text_dim: Color::GRAY,
    accent: Color::new(0, 255, 200, 255),
    glitch: Color::MAGENTA,
};

#[derive(Clone, PartialEq)]
pub enum AppType {
    SysMonitor,
    Terminal,
    FileUniverse,
}

pub struct Window {
    pub title: String,
    pub rect: Rectangle,
    pub app_type: AppType,
    pub is_open: bool,
    pub process_id: Option<u32>,
    pub is_predicted_close: bool, 
    pub intent_score: f32
}

struct DragContext {
    is_dragging: bool,
    window_idx: usize,
    offset_x: f32,
    offset_y: f32,
}

struct MouseSnapshot {
    pos: Vector2,
    time: f64,
}

pub struct DesktopEnv {
    pub windows: Vec<Window>,
    drag_ctx: Option<DragContext>,
    mouse_history: VecDeque<MouseSnapshot>,
    pub wallpaper: Option<Texture2D>,
    pub input_mode: InputMode,
    pub input_buffer: String,
    pub multiline_buffer: Vec<String>, // For multiline editing
    pub cursor_line: usize,
}

#[derive(Clone, PartialEq)]
pub enum InputMode {
    Normal,
    CreatingFile,
    ImportingFile,
    EditingFile(u32), // Stores the file ID being edited
}

impl DesktopEnv {
    pub fn new() -> Self {
        Self {
            windows: Vec::new(),
            drag_ctx: None,
            mouse_history: VecDeque::new(),
            wallpaper: None,
            input_mode: InputMode::Normal,
            input_buffer: String::new(),
            multiline_buffer: vec![String::new()],
            cursor_line: 0,
        }
    }

    pub fn open_window(&mut self, hw: &mut VirtualHardware, app_type: AppType) {
        let (title, w, h, cpu_cost, ram_cost) = match app_type {
            AppType::SysMonitor => ("NEURAL MONITOR", 600.0, 400.0, 0.1, 5),
            AppType::Terminal => ("NEURO_SHELL", 500.0, 300.0, 0.05, 2),
            AppType::FileUniverse => ("GNN FILE SYSTEM", 500.0, 500.0, 0.2, 10),
        };

        let pid = hw.spawn_process(title, cpu_cost, ram_cost, true);
        let offset = (self.windows.len() as f32) * 30.0;
        
        self.windows.push(Window {
            title: title.to_string(),
            rect: Rectangle::new(50.0 + offset, 50.0 + offset, w, h),
            app_type,
            is_open: true,
            process_id: Some(pid),
            is_predicted_close: false,
            intent_score: 0.0
        });
    }

    pub fn close_window(&mut self, hw: &mut VirtualHardware, index: usize) {
        if let Some(pid) = self.windows[index].process_id {
            hw.kill_process(pid);
        }
        self.windows.remove(index);
    }

    pub fn update(&mut self, rl: &RaylibHandle, hw: &mut VirtualHardware) {
        let mouse_pos = rl.get_mouse_position();
        let is_dragging = self.drag_ctx.is_some() || rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT);
        let time = rl.get_time();
        let screen_w = rl.get_screen_width() as f32;
        let screen_h = rl.get_screen_height() as f32;
        let taskbar_h = 40.0;

        // 1. Update Ballistic History
        self.mouse_history.push_back(MouseSnapshot { pos: mouse_pos, time });
        if self.mouse_history.len() > 10 { self.mouse_history.pop_front(); }

        // 2. Predict Trajectory
        let predicted_pos = self.get_predicted_pos(250.0); // Predict 250ms ahead

        let mut window_to_close = None;

        for (i, win) in self.windows.iter_mut().enumerate() {
            if is_dragging {
                win.intent_score = 0.0;
                win.is_predicted_close = false;
                continue;
            }
            win.is_predicted_close = false;
            
            if let Some(pred) = predicted_pos {
                let close_btn = Rectangle::new(win.rect.x + win.rect.width - 30.0, win.rect.y, 30.0, 30.0);
                let btn_center = Vector2::new(close_btn.x + 15.0, close_btn.y + 15.0);

                if liang_barsky_intersect(mouse_pos, pred, close_btn) {
                    // 1. Calculate direction vectors
                    let to_center = Vector2::new(btn_center.x - mouse_pos.x, btn_center.y - mouse_pos.y);
                    let move_dir = Vector2::new(pred.x - mouse_pos.x, pred.y - mouse_pos.y);
                    
                    // 2. Normalize for dot product
                    let dist = (to_center.x.powi(2) + to_center.y.powi(2)).sqrt();
                    let move_mag = (move_dir.x.powi(2) + move_dir.y.powi(2)).sqrt();

                    if dist > 0.0 && move_mag > 0.0 {
                        let dot = (to_center.x / dist * move_dir.x / move_mag) + 
                                (to_center.y / dist * move_dir.y / move_mag);
                        
                        // 3. Weight intent by alignment
                        // If dot > 0.95, user is aiming within a ~18 degree cone of the center
                        if dist < 150.0 && dot > 0.93 {
                            win.is_predicted_close = true;
                            win.intent_score = (win.intent_score + 0.10).min(1.0); 
                        } else {
                            // Grazing: slow decay instead of increase
                            win.intent_score = (win.intent_score - 0.05).max(0.0);
                        }
                    }
                } else {
                    win.intent_score = (win.intent_score - 0.2).max(0.0);
                }
            }

            // AUTO-ACTUATION: Closing
            if win.intent_score >= 1.00 {
                window_to_close = Some(i);
            }
        }

        if let Some(idx) = window_to_close {
            self.close_window(hw, idx);
        }

        // --- FILE NODE AUTO-SELECTION ---
        if let Some(pred) = predicted_pos {
            for win in &self.windows {
                if win.app_type == AppType::FileUniverse {
                    let (inner_x, inner_y, inner_w, inner_h) = (win.rect.x + 10.0, win.rect.y + 40.0, win.rect.width - 20.0, win.rect.height - 50.0);
                    let offset = Vector2::new((inner_x + inner_w / 2.0) - 400.0, (inner_y + inner_h / 2.0) - 300.0);

                    for node in hw.file_system.get_all_nodes() {
                        let node_rect = Rectangle::new(node.x + offset.x - 10.0, node.y + offset.y - 10.0, 20.0, 20.0);
                        // If trajectory hits the node hitbox
                        if liang_barsky_intersect(mouse_pos, pred, node_rect) {
                            hw.selected_file_id = Some(node.id);
                        }
                    }
                }
            }
        }
        
        // 3. Update Dragging
        if let Some(ctx) = &self.drag_ctx {
            if rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT) {
                if ctx.window_idx < self.windows.len() {
                    let win = &mut self.windows[ctx.window_idx];
                    let mut new_x = mouse_pos.x - ctx.offset_x;
                    let mut new_y = mouse_pos.y - ctx.offset_y;
                    new_x = new_x.clamp(0.0, screen_w - win.rect.width);
                    new_y = new_y.clamp(0.0, screen_h - win.rect.height - taskbar_h);
                    win.rect.x = new_x;
                    win.rect.y = new_y;
                }
            } else {
                self.drag_ctx = None;
            }
        } else if rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT) {
            let mut action_close = None;
            let mut action_focus = None;
            let mut should_drag = false;

            for (i, win) in self.windows.iter().enumerate().rev() {
                if !win.is_open { continue; }
                if win.rect.check_collision_point_rec(mouse_pos) {
                    let close_hitbox = Rectangle::new(win.rect.x + win.rect.width - 30.0, win.rect.y, 30.0, 30.0);
                    let title_hitbox = Rectangle::new(win.rect.x, win.rect.y, win.rect.width - 30.0, 30.0);

                    if close_hitbox.check_collision_point_rec(mouse_pos) {
                        action_close = Some(i);
                    } else {
                        should_drag = title_hitbox.check_collision_point_rec(mouse_pos);
                        action_focus = Some(i);
                        if win.app_type == AppType::FileUniverse {
                            self.handle_file_click(win, hw, mouse_pos);
                        }
                    }
                    break;
                }
            }

            if let Some(index) = action_close {
                self.close_window(hw, index);
            } else if let Some(index) = action_focus {
                let window = self.windows.remove(index);
                self.windows.push(window);
                if should_drag {
                    let new_index = self.windows.len() - 1;
                    let win = &self.windows[new_index];
                    self.drag_ctx = Some(DragContext {
                        is_dragging: true,
                        window_idx: new_index,
                        offset_x: mouse_pos.x - win.rect.x,
                        offset_y: mouse_pos.y - win.rect.y,
                    });
                }
            }
        }

        // 4. Run Liang-Barsky on Close Buttons
        for win in &mut self.windows {
            win.is_predicted_close = false;
            if let Some(pred) = predicted_pos {
                let close_btn = Rectangle::new(win.rect.x + win.rect.width - 30.0, win.rect.y, 30.0, 30.0);
                if liang_barsky_intersect(mouse_pos, pred, close_btn) {
                    win.is_predicted_close = true;
                }
            }
        }
    }

    fn get_predicted_pos(&self, horizon_ms: f64) -> Option<Vector2> {
        if self.mouse_history.len() < 2 { return None; }
        let oldest = self.mouse_history.front()?;
        let newest = self.mouse_history.back()?;
        let dt = (newest.time - oldest.time) as f32;
        if dt <= 0.0 { return None; }

        let velocity = Vector2::new((newest.pos.x - oldest.pos.x) / dt, (newest.pos.y - oldest.pos.y) / dt);
        Some(Vector2::new(
            newest.pos.x + velocity.x * (horizon_ms / 1000.0) as f32,
            newest.pos.y + velocity.y * (horizon_ms / 1000.0) as f32
        ))
    }

    fn handle_file_click(&self, win: &Window, hw: &mut VirtualHardware, mouse_pos: Vector2) {
        let inner_x = win.rect.x + 10.0;
        let inner_y = win.rect.y + 40.0;
        let inner_w = win.rect.width - 20.0;
        let inner_h = win.rect.height - 50.0;
        let offset_x = (inner_x + inner_w / 2.0) - 400.0;
        let offset_y = (inner_y + inner_h / 2.0) - 300.0;

        let mut clicked_node = None;
        for node in hw.file_system.get_all_nodes() {
            let nx = node.x + offset_x;
            let ny = node.y + offset_y;
            let dx = mouse_pos.x - nx;
            let dy = mouse_pos.y - ny;
            if dx*dx + dy*dy < 36.0 {
                clicked_node = Some(node.id);
                break;
            }
        }
        hw.selected_file_id = clicked_node;
        if clicked_node.is_some() {
            hw.system_state.cpu_pressure = (hw.system_state.cpu_pressure + 0.05).min(1.0);
        }
    }

    pub fn draw(&mut self, d: &mut RaylibDrawHandle, hw: &VirtualHardware) {
        let sw = d.get_screen_width();
        let sh = d.get_screen_height();

        if let Some(tex) = &self.wallpaper {
            d.draw_texture_pro(tex, Rectangle::new(0.0, 0.0, tex.width as f32, tex.height as f32), 
                Rectangle::new(0.0, 0.0, sw as f32, sh as f32), Vector2::zero(), 0.0, Color::WHITE);
        } else {
            d.clear_background(Color::new(10, 10, 15, 255));
            for i in (0..sw).step_by(50) { d.draw_line(i, 0, i, sh, Color::new(255, 255, 255, 5)); }
            for i in (0..sh).step_by(50) { d.draw_line(0, i, sw, i, Color::new(255, 255, 255, 5)); }
        }

        // Draw Ballistic Projection Line
        if let Some(pred) = self.get_predicted_pos(250.0) {
            let start = self.mouse_history.back().unwrap().pos;
            d.draw_line_v(start, pred, Color::new(0, 255, 255, 80));
            d.draw_circle_v(pred, 3.0, Color::new(0, 255, 255, 120));
        }

        for (i, window) in self.windows.iter().enumerate() {
            if !window.is_open { continue; }
            let is_active = i == self.windows.len() - 1;
            self.draw_window_frame(d, window, is_active, |d, inner_rect| {
                match window.app_type {
                    AppType::SysMonitor => self.draw_sys_monitor(d, hw, inner_rect),
                    AppType::Terminal => self.draw_terminal(d, hw, inner_rect),
                    AppType::FileUniverse => self.draw_file_universe(d, hw, inner_rect),
                }
            });
        }
        self.draw_taskbar(d, hw, sw, sh);
        
        // Draw input dialog if in creation/import mode
        if self.input_mode != InputMode::Normal {
            self.draw_input_dialog(d);
        }
    }
    
    fn draw_input_dialog(&self, d: &mut RaylibDrawHandle) {
        let sw = d.get_screen_width() as f32;
        let sh = d.get_screen_height() as f32;
        
        match &self.input_mode {
            InputMode::EditingFile(_) => {
                // Full screen editor
                let editor_w = sw - 200.0;
                let editor_h = sh - 200.0;
                let editor_x = 100.0;
                let editor_y = 100.0;
                
                d.draw_rectangle(editor_x as i32, editor_y as i32, editor_w as i32, editor_h as i32, Color::new(15, 15, 20, 250));
                d.draw_rectangle_lines(editor_x as i32, editor_y as i32, editor_w as i32, editor_h as i32, CYBER_THEME.accent);
                
                d.draw_text("FILE EDITOR", editor_x as i32 + 10, editor_y as i32 + 10, 20, CYBER_THEME.accent);
                d.draw_text("CTRL-S to save | CTRL-Q to cancel", editor_x as i32 + 10, editor_y as i32 + 35, 12, CYBER_THEME.text_dim);
                
                // Draw content area
                let content_y = editor_y as i32 + 55;
                let content_h = editor_h as i32 - 65;
                d.draw_rectangle(editor_x as i32 + 10, content_y, editor_w as i32 - 20, content_h, Color::new(10, 10, 15, 255));
                
                // Draw lines of text
                let mut y_offset = content_y + 5;
                for (i, line) in self.multiline_buffer.iter().enumerate() {
                    let line_color = if i == self.cursor_line { Color::WHITE } else { CYBER_THEME.text_dim };
                    let display_line = if i == self.cursor_line {
                        format!("{}_", line)
                    } else {
                        line.clone()
                    };
                    d.draw_text(&display_line, editor_x as i32 + 15, y_offset, 14, line_color);
                    y_offset += 18;
                    if y_offset > content_y + content_h - 20 { break; }
                }
            }
            _ => {
                // Small dialog for file creation/import
                let dialog_w = 500.0;
                let dialog_h = 150.0;
                let dialog_x = (sw - dialog_w) / 2.0;
                let dialog_y = (sh - dialog_h) / 2.0;
                
                d.draw_rectangle(dialog_x as i32, dialog_y as i32, dialog_w as i32, dialog_h as i32, Color::new(20, 20, 30, 250));
                d.draw_rectangle_lines(dialog_x as i32, dialog_y as i32, dialog_w as i32, dialog_h as i32, CYBER_THEME.accent);
                
                let title = match self.input_mode {
                    InputMode::CreatingFile => "CREATE NEW FILE",
                    InputMode::ImportingFile => "IMPORT FILE PATH",
                    _ => "",
                };
                
                d.draw_text(title, dialog_x as i32 + 10, dialog_y as i32 + 10, 20, CYBER_THEME.accent);
                d.draw_text("Enter filename (or path to import):", dialog_x as i32 + 10, dialog_y as i32 + 45, 15, Color::WHITE);
                
                // Draw input box
                let input_box_y = dialog_y as i32 + 75;
                d.draw_rectangle(dialog_x as i32 + 10, input_box_y, dialog_w as i32 - 20, 30, Color::new(10, 10, 15, 255));
                d.draw_rectangle_lines(dialog_x as i32 + 10, input_box_y, dialog_w as i32 - 20, 30, CYBER_THEME.accent);
                d.draw_text(&format!("{}_", self.input_buffer), dialog_x as i32 + 15, input_box_y + 8, 15, Color::WHITE);
                
                d.draw_text("Press ENTER to confirm, CTRL-Q to cancel", dialog_x as i32 + 10, dialog_y as i32 + 120, 12, CYBER_THEME.text_dim);
            }
        }
    }

    fn draw_window_frame<F>(&self, d: &mut RaylibDrawHandle, win: &Window, is_active: bool, content: F)
    where F: FnOnce(&mut RaylibDrawHandle, Rectangle) {
        let x = win.rect.x as i32;
        let y = win.rect.y as i32;
        let w = win.rect.width as i32;
        let h = win.rect.height as i32;

        let border = if is_active { CYBER_THEME.border_active } else { CYBER_THEME.border_inactive };
        let mut header = if is_active { CYBER_THEME.header_active } else { CYBER_THEME.header_inactive };

        if win.is_predicted_close {
            let pulse = ((d.get_time() * 15.0).sin() * 40.0 + 40.0) as u8;
            header.r = header.r.saturating_add(pulse);
        }

        d.draw_rectangle(x + 8, y + 8, w, h, Color::new(0, 0, 0, 100));
        d.draw_rectangle(x, y, w, h, CYBER_THEME.win_bg);
        d.draw_rectangle_lines(x, y, w, h, border);
        d.draw_rectangle(x, y, w, 30, header);

        if win.intent_score > 0.0 {
            let bar_width = (win.intent_score * (w as f32 - 40.0)) as i32;
            d.draw_rectangle(x + 10, y + 25, bar_width, 2, CYBER_THEME.accent);
        }

        d.draw_text(&win.title, x + 10, y + 8, 10, CYBER_THEME.text);

        let close_color = if win.is_predicted_close { Color::WHITE } else { Color::RED };
        d.draw_circle(x + w - 15, y + 15, 5.0, close_color);
        d.draw_circle(x + w - 30, y + 15, 5.0, Color::YELLOW);

        content(d, Rectangle::new(win.rect.x + 10.0, win.rect.y + 40.0, win.rect.width - 20.0, win.rect.height - 50.0));
    }

    fn draw_sys_monitor(&self, d: &mut RaylibDrawHandle, hw: &VirtualHardware, r: Rectangle) {
        let (x, y, w, h) = (r.x as i32, r.y as i32, r.width as i32, r.height as i32);
        d.draw_text("NEURAL CORES", x + 10, y + 5, 10, CYBER_THEME.text_dim);
        for (i, load) in hw.cpu_load.iter().enumerate() {
            let fill = (*load * (w - 40) as f32) as i32;
            let by = y + 25 + (i as i32 * 25);
            d.draw_rectangle(x + 20, by, w - 40, 15, Color::new(10, 10, 15, 255));
            let color = if *load > 0.8 { Color::RED } else if *load > 0.5 { Color::ORANGE } else { Color::GREEN };
            d.draw_rectangle(x + 20, by, fill, 15, color);
        }
        d.draw_text("MEMORY MANIFOLD", x + 10, y + 140, 10, CYBER_THEME.text_dim);
        let (cell, cols) = (12, 24);
        for i in 0..crate::hardware::RAM_SIZE {
            let bx = x + 20 + (i % cols) as i32 * (cell + 2);
            let by = y + 160 + (i / cols) as i32 * (cell + 2);
            if by > y + h - 20 { break; }
            let color = match hw.ram[i].block_type {
                BlockType::Empty => Color::new(35, 35, 40, 255),
                BlockType::User => Color::BLUE,
                BlockType::System => Color::MAROON,
                BlockType::Glitch => CYBER_THEME.glitch,
            };
            d.draw_rectangle(bx, by, cell, cell, color);
            if hw.ram[i].heat > 0.1 {
                d.draw_rectangle(bx, by, cell, cell, Color::new(255, 255, 255, (hw.ram[i].heat * 200.0) as u8));
            }
        }
    }

    fn draw_terminal(&self, d: &mut RaylibDrawHandle, hw: &VirtualHardware, r: Rectangle) {
        let (x, y, w, h) = (r.x as i32, r.y as i32, r.width as i32, r.height as i32);
        d.draw_rectangle(x, y, w, h, Color::new(5, 5, 10, 200));
        d.draw_rectangle_lines(x, y, w, h, CYBER_THEME.text_dim);
        d.draw_text("user@fugue-os:~$ ./init_sequence", x + 10, y + 10, 10, CYBER_THEME.accent);
        d.draw_text("> Loading Fugue Kernel...", x + 10, y + 25, 10, CYBER_THEME.accent);
        d.draw_text(&format!("user@fugue-os:~$ {}_", hw.shell_buffer), x + 10, y + 45, 10, CYBER_THEME.text);
    }

    fn draw_file_universe(&self, d: &mut RaylibDrawHandle, hw: &VirtualHardware, r: Rectangle) {
        let (x, y, w, h) = (r.x as i32, r.y as i32, r.width as i32, r.height as i32);
        d.draw_rectangle(x, y, w, h, Color::BLACK);
        let offset = Vector2::new((r.x + r.width / 2.0) - 400.0, (r.y + r.height / 2.0) - 300.0);
        let edge_color = Color::new(100, 0, 200, ((d.get_time() * 3.0).sin() * 100.0 + 155.0) as u8);
        {
            let mut s = d.begin_scissor_mode(x + 2, y + 2, w - 4, h - 4);
            // Draw edges (connections)
            for node in hw.file_system.get_all_nodes() {
                let connections = hw.file_system.get_connections(node.id);
                for conn_id in connections {
                    if let Some(target) = hw.file_system.get_node(conn_id) {
                        s.draw_line_v(Vector2::new(node.x + offset.x, node.y + offset.y), 
                                     Vector2::new(target.x + offset.x, target.y + offset.y), edge_color);
                    }
                }
            }
            // Draw nodes
            for node in hw.file_system.get_all_nodes() {
                let pos = Vector2::new(node.x + offset.x, node.y + offset.y);
                let sel = Some(node.id) == hw.selected_file_id;
                s.draw_circle_v(pos, if sel { 6.0 } else { 4.0 }, if sel { CYBER_THEME.accent } else { Color::PURPLE });
                if sel { s.draw_circle_lines(pos.x as i32, pos.y as i32, 10.0, CYBER_THEME.accent); }
                s.draw_text(&node.name, pos.x as i32 + 8, pos.y as i32 - 5, 10, CYBER_THEME.text_dim);
            }
        }
        d.draw_rectangle_lines(x, y, w, h, CYBER_THEME.accent);
        // Draw node properties panel
        if let Some(sel) = hw.selected_file_id.and_then(|id| hw.file_system.get_node(id)) {
            let px = x + w - 160;
            let connections = hw.file_system.get_connections(sel.id);
            d.draw_rectangle(px, y + 10, 150, 120, Color::new(20, 20, 30, 240));
            d.draw_rectangle_lines(px, y + 10, 150, 120, CYBER_THEME.accent);
            d.draw_text("NODE PROPERTIES", px + 5, y + 15, 10, CYBER_THEME.accent);
            d.draw_text(&format!("ID: {:04X}", sel.id), px + 5, y + 35, 10, Color::WHITE);
            d.draw_text(&format!("LINKS: {}", connections.len()), px + 5, y + 50, 10, Color::WHITE);
            d.draw_text(&format!("SIZE: {} B", sel.size), px + 5, y + 65, 10, Color::WHITE);
            d.draw_text(&format!("TYPE: {}", sel.file_type), px + 5, y + 80, 10, Color::WHITE);
        }
    }

    fn draw_taskbar(&self, d: &mut RaylibDrawHandle, hw: &VirtualHardware, sw: i32, sh: i32) {
        d.draw_rectangle(0, sh - 40, sw, 40, Color::new(20, 20, 25, 230));
        d.draw_line(0, sh - 40, sw, sh - 40, CYBER_THEME.border_inactive);
        d.draw_text("START", 10, sh - 28, 20, CYBER_THEME.text);
        d.draw_text("|  [F1] Monitor  [F2] Terminal  [F3] Files  [F6] Create  [F7] Import  [F8] Edit", 90, sh - 28, 20, CYBER_THEME.text_dim);
        let (txt, clr) = match hw.current_mindset {
            KernelMindset::Idle => ("IDLE", CYBER_THEME.text_dim),
            KernelMindset::OptimizingRAM => ("OPTIMIZING RAM", Color::BLUE),
            KernelMindset::Rescheduling => ("RESCHEDULING", Color::RED),
            KernelMindset::Thinking => ("PREDICTING INPUT", CYBER_THEME.accent),
        };
        let disp = format!("KERNEL: {}", txt);
        d.draw_text(&disp, sw - d.measure_text(&disp, 20) - 15, sh - 30, 20, clr);
    }
}

fn liang_barsky_intersect(p1: Vector2, p2: Vector2, rect: Rectangle) -> bool {
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;

    let checks = [
        (-dx, p1.x - rect.x),
        (dx, rect.x + rect.width - p1.x),
        (-dy, p1.y - rect.y),
        (dy, rect.y + rect.height - p1.y),
    ];

    for (p, q) in checks {
        if p == 0.0 {
            if q < 0.0 { return false; }
        } else {
            let r = q / p;
            if p < 0.0 {
                if r > t1 { return false; }
                if r > t0 { t0 = r; }
            } else {
                if r < t0 { return false; }
                if r < t1 { t1 = r; }
            }
        }
    }
    t0 <= t1
}
