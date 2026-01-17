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
    win_bg: Color::new(15, 15, 20, 245), // Slightly transparent dark
    border_active: Color::new(200, 200, 200, 255),
    border_inactive: Color::new(60, 60, 70, 255),
    header_active: Color::new(40, 40, 50, 255),
    header_inactive: Color::new(25, 25, 30, 255),
    text: Color::RAYWHITE,
    text_dim: Color::GRAY,
    accent: Color::new(0, 255, 200, 255), // Cyan/Teal
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
}

struct DragContext {
    is_dragging: bool,
    window_idx: usize,
    offset_x: f32,
    offset_y: f32,
}

pub struct DesktopEnv {
    pub windows: Vec<Window>,
    drag_ctx: Option<DragContext>,
    mouse_trail: VecDeque<Vector2>,
    pub wallpaper: Option<Texture2D>,
}

impl DesktopEnv {
    pub fn new() -> Self {
        Self {
            windows: Vec::new(),
            drag_ctx: None,
            mouse_trail: VecDeque::new(),
            wallpaper: None,
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
        let screen_w = rl.get_screen_width() as f32;
        let screen_h = rl.get_screen_height() as f32;
        let taskbar_h = 40.0;

        // Ghost Trail
        self.mouse_trail.push_front(mouse_pos);
        if self.mouse_trail.len() > 20 { self.mouse_trail.pop_back(); }

        // Dragging
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

                        // --- 2. FILE INTERACTION: CLICK DETECTION ---
                        if win.app_type == AppType::FileUniverse {
                            // Replicate the offset logic from draw()
                            let inner_x = win.rect.x + 10.0;
                            let inner_y = win.rect.y + 40.0;
                            let inner_w = win.rect.width - 20.0;
                            let inner_h = win.rect.height - 50.0;
                            
                            let win_center_x = inner_x + (inner_w / 2.0);
                            let win_center_y = inner_y + (inner_h / 2.0);
                            let offset_x = win_center_x - 400.0;
                            let offset_y = win_center_y - 300.0;

                            // Check nodes
                            let mut clicked_node = None;
                            for node in &hw.files {
                                let nx = node.x + offset_x;
                                let ny = node.y + offset_y;
                                // Simple circle collision (radius 6.0 for easier clicking)
                                let dx = mouse_pos.x - nx;
                                let dy = mouse_pos.y - ny;
                                if dx*dx + dy*dy < 36.0 {
                                    clicked_node = Some(node.id);
                                    break;
                                }
                            }
                            
                            // Update Hardware State
                            hw.selected_file_id = clicked_node;
                            
                            // "Interaction Stress": Spike CPU slightly on click
                            if clicked_node.is_some() {
                                hw.system_state.cpu_pressure = (hw.system_state.cpu_pressure + 0.05).min(1.0);
                            }
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
    }

    pub fn draw(&mut self, d: &mut RaylibDrawHandle, hw: &VirtualHardware) {
        let screen_w = d.get_screen_width();
        let screen_h = d.get_screen_height();



        if let Some(tex) = &self.wallpaper {
            // Draw texture scaled to screen size
            let source_rec = Rectangle::new(0.0, 0.0, tex.width as f32, tex.height as f32);
            let dest_rec = Rectangle::new(0.0, 0.0, screen_w as f32, screen_h as f32);
            d.draw_texture_pro(tex, source_rec, dest_rec, Vector2::zero(), 0.0, Color::WHITE);
        } else {
            // Fallback to original grid code
            d.clear_background(Color::new(10, 10, 15, 255));

            for i in (0..screen_w).step_by(50) {
                d.draw_line(i, 0, i, screen_h, Color::new(255, 255, 255, 5));
            }

            for i in (0..screen_h).step_by(50) {
                d.draw_line(0, i, screen_w, i, Color::new(255, 255, 255, 5));
            }
        }

        // Ghost Trail
        for (i, pos) in self.mouse_trail.iter().enumerate() {
            let alpha = 100 - (i * 5);
            if alpha > 0 {
                d.draw_circle_v(*pos, 5.0, Color::new(0, 255, 255, alpha as u8));
            }
        }

        d.draw_fps(10,10);

        // Windows
        for (i, window) in self.windows.iter().enumerate() {
            if !window.is_open { continue; }
            let is_active = i == self.windows.len() - 1;
            
            // Use the helper to draw the frame
            self.draw_window_frame(d, window, is_active, |d, inner_rect| {
                match window.app_type {
                    AppType::SysMonitor => self.draw_sys_monitor(d, hw, inner_rect),
                    AppType::Terminal => self.draw_terminal(d, hw, inner_rect),
                    AppType::FileUniverse => self.draw_file_universe(d, hw, inner_rect),
                }
            });
        }

        self.draw_taskbar(d, hw, screen_w, screen_h);
    }

    // --- HELPER: UNIFIED WINDOW FRAME ---
    fn draw_window_frame<F>(&self, d: &mut RaylibDrawHandle, win: &Window, is_active: bool, content: F)
    where F: FnOnce(&mut RaylibDrawHandle, Rectangle) {
        let x = win.rect.x as i32;
        let y = win.rect.y as i32;
        let w = win.rect.width as i32;
        let h = win.rect.height as i32;

        // Shadow
        d.draw_rectangle(x + 8, y + 8, w, h, Color::new(0, 0, 0, 100));

        // Border & Header Colors
        let border = if is_active { CYBER_THEME.border_active } else { CYBER_THEME.border_inactive };
        let header = if is_active { CYBER_THEME.header_active } else { CYBER_THEME.header_inactive };

        // Main Body
        d.draw_rectangle(x, y, w, h, CYBER_THEME.win_bg);
        d.draw_rectangle_lines(x, y, w, h, border);

        // Header
        d.draw_rectangle(x, y, w, 30, header);
        d.draw_text(&win.title, x + 10, y + 8, 10, CYBER_THEME.text);

        // Controls
        d.draw_circle(x + w - 15, y + 15, 5.0, Color::RED);
        d.draw_circle(x + w - 30, y + 15, 5.0, Color::YELLOW);

        // Content Area Calculation
        let content_rect = Rectangle::new(
            win.rect.x + 10.0, 
            win.rect.y + 40.0, 
            win.rect.width - 20.0, 
            win.rect.height - 50.0
        );
        
        content(d, content_rect);
    }

    fn draw_sys_monitor(&self, d: &mut RaylibDrawHandle, hw: &VirtualHardware, r: Rectangle) {
        let x = r.x as i32;
        let y = r.y as i32;
        let w = r.width as i32;
        let h = r.height as i32;

        d.draw_text("NEURAL CORES", x + 10, y + 5, 10, CYBER_THEME.text_dim);
        for (i, load) in hw.cpu_load.iter().enumerate() {
            let bar_w = (w - 40) as f32;
            let fill = (*load * bar_w) as i32;
            let by = y + 25 + (i as i32 * 25);
            d.draw_rectangle(x + 20, by, bar_w as i32, 15, Color::new(10, 10, 15, 255));
            let color = if *load > 0.8 { Color::RED } else if *load > 0.5 { Color::ORANGE } else { Color::GREEN };
            d.draw_rectangle(x + 20, by, fill, 15, color);
        }

        d.draw_text("MEMORY MANIFOLD", x + 10, y + 140, 10, CYBER_THEME.text_dim);
        let cell_size = 12;
        let cols = 24;
        let start_x = x + 20;
        let start_y = y + 160;

        for (i, block) in hw.ram.iter().enumerate() {
            let bx = start_x + (i % cols) as i32 * (cell_size + 2);
            let by = start_y + (i / cols) as i32 * (cell_size + 2);
            if by > y + h - 20 { break; }
            if bx > x + w - 20 { continue; }

            let color = match block.block_type {
                BlockType::Empty => Color::new(35, 35, 40, 255),
                BlockType::User => Color::BLUE,
                BlockType::System => Color::MAROON,
                BlockType::Glitch => CYBER_THEME.glitch,
            };
            d.draw_rectangle(bx, by, cell_size, cell_size, color);
            if block.heat > 0.1 {
                let alpha = (block.heat * 200.0) as u8;
                d.draw_rectangle(bx, by, cell_size, cell_size, Color::new(255, 255, 255, alpha));
            }
        }
    }

    fn draw_terminal(&self, d: &mut RaylibDrawHandle, hw: &VirtualHardware, r: Rectangle) {
        let x = r.x as i32;
        let y = r.y as i32;
        let w = r.width as i32;
        let h = r.height as i32;

        d.draw_rectangle(x, y, w, h, Color::new(5, 5, 10, 200));
        d.draw_rectangle_lines(x, y, w, h, CYBER_THEME.text_dim);
        d.draw_text("user@fugue-os:~$ ./init_sequence", x + 10, y + 10, 10, CYBER_THEME.accent);
        d.draw_text("> Loading Fugue Kernel...", x + 10, y + 25, 10, CYBER_THEME.accent);
        d.draw_text("_", x + 10, y + 40, 10, CYBER_THEME.accent);

        let prompt = format!("user@fugue-os:~$ {}_", hw.shell_buffer);
        d.draw_text(&prompt, x + 10, y + 45, 10, CYBER_THEME.text);
    }

    fn draw_file_universe(&self, d: &mut RaylibDrawHandle, hw: &VirtualHardware, r: Rectangle) {
        let x = r.x as i32;
        let y = r.y as i32;
        let w = r.width as i32;
        let h = r.height as i32;

        d.draw_rectangle(x, y, w, h, Color::BLACK);

        let win_center_x = r.x + (r.width / 2.0);
        let win_center_y = r.y + (r.height / 2.0);
        let offset_x = win_center_x - 400.0;
        let offset_y = win_center_y - 300.0;

        let time = d.get_time();
        let pulse = (time * 3.0).sin(); // -1 to 1
        let alpha = ((pulse * 100.0) + 200.0) as u8; // 100 to 300
        let edge_color = Color::new(100, 0, 200, alpha);

        let to_window = |gx: f32, gy: f32| -> Vector2 {
            Vector2::new(gx + offset_x, gy + offset_y)
        };

        {
            let mut s = d.begin_scissor_mode(x + 2, y + 2, w - 4, h - 4);

            // Edges
            for node in &hw.files {
                for &conn_id in &node.connections {
                    if let Some(target) = hw.files.iter().find(|n| n.id == conn_id) {
                        let start = to_window(node.x, node.y);
                        let end = to_window(target.x, target.y);
                        s.draw_line_v(start, end, edge_color);
                    }
                }
            }

            // Nodes
            for node in &hw.files {
                let pos = to_window(node.x, node.y);
                let is_selected = Some(node.id) == hw.selected_file_id;
                
                let color = if is_selected { CYBER_THEME.accent } else { Color::PURPLE };
                let radius = if is_selected { 6.0 } else { 4.0 };

                s.draw_circle_v(pos, radius, color);
                
                // Draw ID if selected
                if is_selected {
                    s.draw_circle_lines(pos.x as i32, pos.y as i32, radius + 4.0, CYBER_THEME.accent);
                }
                
                s.draw_text(&node.name, pos.x as i32 + 8, pos.y as i32 - 5, 10, CYBER_THEME.text_dim);
            }
        } 

        d.draw_rectangle_lines(x, y, w, h, CYBER_THEME.accent);

        // --- 2. FILE INTERACTION: PROPERTIES PANEL ---
        if let Some(sel_id) = hw.selected_file_id {
            if let Some(node) = hw.files.iter().find(|n| n.id == sel_id) {
                let panel_x = x + w - 160;
                let panel_y = y + 10;
                d.draw_rectangle(panel_x, panel_y, 150, 100, Color::new(20, 20, 30, 240));
                d.draw_rectangle_lines(panel_x, panel_y, 150, 100, CYBER_THEME.accent);
                
                d.draw_text("NODE PROPERTIES", panel_x + 5, panel_y + 5, 10, CYBER_THEME.accent);
                d.draw_text(&format!("ID: {:04X}", node.id), panel_x + 5, panel_y + 25, 10, Color::WHITE);
                d.draw_text(&format!("LINKS: {}", node.connections.len()), panel_x + 5, panel_y + 40, 10, Color::WHITE);
                d.draw_text("STATUS: ACTIVE", panel_x + 5, panel_y + 55, 10, Color::GREEN);
            }
        }
    }

    fn draw_taskbar(&self, d: &mut RaylibDrawHandle, hw: &VirtualHardware, sw: i32, sh: i32) {
        let bar_h = 40;
        d.draw_rectangle(0, sh - bar_h, sw, bar_h, Color::new(20, 20, 25, 230));
        d.draw_line(0, sh - bar_h, sw, sh - bar_h, CYBER_THEME.border_inactive);
        d.draw_text("START", 10, sh - 28, 20, CYBER_THEME.text);
        d.draw_text("|  [F1] Monitor  [F2] Terminal  [F3] Files", 90, sh - 28, 20, CYBER_THEME.text_dim);

        let (status_text, status_color) = match hw.current_mindset {
            KernelMindset::Idle => ("IDLE", CYBER_THEME.text_dim),
            KernelMindset::OptimizingRAM => ("OPTIMIZING RAM", Color::BLUE),
            KernelMindset::Rescheduling => ("RESCHEDULING", Color::RED),
            KernelMindset::Thinking => ("PREDICTING INPUT", CYBER_THEME.accent),
        };
        let display_text = format!("KERNEL: {}", status_text);
        let font_size = 20;
        let text_width = d.measure_text(&display_text, font_size);
        
        // sw - text_width - margin (15px) ensures it stays on screen
        // sh - 30 centers it better within the 40px taskbar
        d.draw_text(&display_text, sw - text_width - 15, sh - 30, font_size, status_color);
    }
}
