use rand::Rng;
use ort::session::Session;
use ort::value::Value;
use ndarray::Array2;

pub const RAM_SIZE: usize = 256;
pub const CPU_CORES: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StateIndex {
    ActiveProcs,
    CpuPressure,
    WaitTime,
    RamUsage,
    FragScore,
    MouseVel,
    InputLatency,
    StateSize
}
pub const STATE_VECTOR_SIZE: usize = StateIndex::StateSize as usize; 

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockType {
    Empty,
    System,
    User,
    Glitch,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KernelMindset {
    Idle,
    OptimizingRAM,   // GAN
    Rescheduling,    // RL
    Thinking,        // LLM
}

#[derive(Clone, Copy, Debug)]
pub struct MemoryBlock {
    pub id: usize,
    pub block_type: BlockType,
    pub heat: f32,
    pub owner_pid: Option<u32>
}

#[derive(Clone, Debug)]
pub struct FileNode {
    pub id: u32,
    pub name: String,
    pub x: f32, 
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub connections: Vec<u32>
}

#[derive(Clone, Debug)]
pub struct Process {
    pub id: u32,
    pub name: String,
    pub cpu_impact: f32,
    pub ram_impact: usize,
    pub lifespan: u32,
    pub current_wait: f32 
}

#[derive(Debug, Clone, Copy)]
pub struct SystemState {
    pub active_processes: f32,
    pub cpu_pressure: f32,
    pub avg_wait_time: f32,
    pub ram_usage: f32,
    pub fragmentation_score: f32,
    pub mouse_velocity: f32,
    pub time_since_last_input: f32,
}

impl SystemState {
    pub fn new() -> Self {
        Self {
            active_processes: 0.0,
            cpu_pressure: 0.0,
            avg_wait_time: 0.0,
            ram_usage: 0.0,
            fragmentation_score: 0.0,
            mouse_velocity: 0.0,
            time_since_last_input: 0.0,
        }
    }

    pub fn to_state_vector(&self) -> Vec<f32> {
        let mut vec = vec![0.0; STATE_VECTOR_SIZE];
        vec[StateIndex::ActiveProcs as usize] = self.active_processes;
        vec[StateIndex::CpuPressure as usize] = self.cpu_pressure.clamp(0.0, 1.0);
        vec[StateIndex::WaitTime as usize] = self.avg_wait_time.min(1.0);
        vec[StateIndex::RamUsage as usize] = self.ram_usage.clamp(0.0, 1.0);
        vec[StateIndex::FragScore as usize] = self.fragmentation_score.clamp(0.0, 1.0);
        vec[StateIndex::MouseVel as usize] = (self.mouse_velocity / 100.0).min(1.0);
        vec[StateIndex::InputLatency as usize] = (self.time_since_last_input / 10.0).min(1.0);
        vec
    }
}

pub struct VirtualHardware {
    pub ram: [MemoryBlock; RAM_SIZE],
    pub cpu_load: [f32; CPU_CORES],
    pub processes: Vec<Process>,
    pub files: Vec<FileNode>,
    pub selected_file_id: Option<u32>,
    pub current_mindset: KernelMindset,
    pub system_state: SystemState,
    pub tick: u64,
    pub shell_buffer: String,
    pub model_session: Option<Session>,
    pub vae_session: Option<Session>,
    process_counter: u32,
}

impl VirtualHardware {
    pub fn load_brain(&mut self) {
        // Load Orchestrator
        self.model_session = Some(Session::builder()
            .unwrap()
            .commit_from_file("kernel_brain.onnx")
            .unwrap());

        self.vae_session = Some(Session::builder()
            .unwrap()
            .commit_from_file("memory_vae.onnx")
            .unwrap());
    }

    pub fn new() -> Self {
        let empty_block = MemoryBlock { id: 0, block_type: BlockType::Empty, heat: 0.0, owner_pid: None };
        
        let mut files = Vec::new();
        let mut rng = rand::rng();
        for i in 0..15 {
            files.push(FileNode {
                id: i,
                name: format!("node_{:02X}", i),
                x: rng.random_range(200.0..600.0),
                y: rng.random_range(200.0..400.0),
                vx: 0.0, vy: 0.0,
                connections: if i > 0 { vec![rng.random_range(0..i)] } else { vec![] },
            });
        }

        Self {
            ram: [empty_block; RAM_SIZE],
            cpu_load: [0.0; CPU_CORES],
            processes: Vec::new(),
            files,
            selected_file_id: None,
            current_mindset: KernelMindset::Idle,
            system_state: SystemState::new(),
            tick: 0,
            shell_buffer: String::new(),
            model_session: None,
            vae_session: None,
            process_counter: 0,
        }
    }

    pub fn update_mindset(&mut self) {
        if let Some(session) = &mut self.model_session {
            let state_vec = self.system_state.to_state_vector();
            let input_array = Array2::from_shape_vec((1, 7), state_vec).unwrap();

            // Explicitly create the tensor value
            let input_tensor = Value::from_array(input_array).unwrap();
            
            // Pass the tensor to the macro
            let outputs = session.run(ort::inputs!["float_input" => input_tensor]).unwrap();

            let output_tensor = outputs["output_label"].try_extract_tensor::<i64>().unwrap();
            let predicted_id = output_tensor.1[0];

            self.current_mindset = match predicted_id {
                0 => KernelMindset::Idle,
                1 => KernelMindset::OptimizingRAM,
                2 => KernelMindset::Thinking,
                3 => KernelMindset::Rescheduling,
                _ => KernelMindset::Idle,
            };
        }
    }

    pub fn neural_defrag(&mut self) {
        if let Some(session) = &mut self.vae_session {
            // 1. Flatten RAM into 0.0, 0.5, 1.0 for the AI
            let input_data: Vec<f32> = self.ram.iter().map(|b| {
                match b.block_type {
                    BlockType::Empty => 0.0,
                    BlockType::User => 0.5,
                    BlockType::System => 1.0,
                    BlockType::Glitch => 0.5,
                }
            }).collect();

            let input_array = Array2::from_shape_vec((1, 256), input_data).unwrap();
            let input_tensor = Value::from_array(input_array).unwrap();
            
            // 2. Run Inference
            let outputs = session.run(ort::inputs!["input" => input_tensor]).unwrap();
            let output_tensor = outputs["output"].try_extract_tensor::<f32>().unwrap();
            let view = output_tensor.1; // Get the array view

            // 3. Apply the "Neural Mask"
            // We use the AI's probability output to re-assign block types
            for i in 0..RAM_SIZE {
                let val = view[i];
                let old_type = self.ram[i].block_type;

                if val > 0.8 { self.ram[i].block_type = BlockType::System; }
                else if val > 0.3 { self.ram[i].block_type = BlockType::User; }
                else { self.ram[i].block_type = BlockType::Empty; }

                // If the AI changed the block, make it "glow" in the UI
                if old_type != self.ram[i].block_type {
                    self.ram[i].heat = 1.0; 
                }
            }
        }
    }

    pub fn update_sensors(&mut self) {
        let occupied_blocks = self.ram.iter().filter(|b| b.block_type != BlockType::Empty).count();
        self.system_state.ram_usage = occupied_blocks as f32 / RAM_SIZE as f32;

        let total_load: f32 = self.cpu_load.iter().sum();
        self.system_state.cpu_pressure = total_load / CPU_CORES as f32;
        self.system_state.active_processes = self.processes.len() as f32;

        if !self.processes.is_empty() {
            let total_wait: f32 = self.processes.iter().map(|p| p.current_wait).sum();
            self.system_state.avg_wait_time = total_wait / self.processes.len() as f32;
        } else {
            self.system_state.avg_wait_time = 0.0;
        }

        let mut switches = 0;
        for i in 0..RAM_SIZE-1 {
            if (self.ram[i].block_type == BlockType::Empty) != (self.ram[i+1].block_type == BlockType::Empty) {
                switches += 1;
            }
        }
        self.system_state.fragmentation_score = switches as f32 / RAM_SIZE as f32;
    }

    fn update_file_system(&mut self) {
        let len = self.files.len();
        for i in 0..len {
            for j in 0..len {
                if i == j { continue; }
                let dx = self.files[i].x - self.files[j].x;
                let dy = self.files[i].y - self.files[j].y;
                let dist_sq = dx*dx + dy*dy;
                if dist_sq > 0.1 && dist_sq < 10000.0 {
                    let force = 40.0 / dist_sq;
                    self.files[i].vx += dx * force;
                    self.files[i].vy += dy * force;
                }
            }
        }
        for node in self.files.iter_mut() {
            let dx = 400.0 - node.x; 
            let dy = 300.0 - node.y; 
            node.vx += dx * 0.005;
            node.vy += dy * 0.005;
            node.x += node.vx;
            node.y += node.vy;
            node.vx *= 0.90;
            node.vy *= 0.90;
        }
    }

    pub fn update_physics(&mut self) {
        let mut rng = rand::rng();
        self.tick += 1;
        self.update_mindset();

        // --- Trigger Neural Defrag ---
        if self.current_mindset == KernelMindset::OptimizingRAM && self.tick % 60 == 0 {
            self.neural_defrag();
        }

        self.update_file_system();
        self.processes.retain(|p| p.lifespan > 0);
        for p in self.processes.iter_mut() {
            if p.lifespan < u32::MAX { p.lifespan -= 1; }
        }
        let total_demand: f32 = self.processes.iter().map(|p| p.cpu_impact).sum();
        let total_capacity = CPU_CORES as f32 * 0.8;
        if total_demand > total_capacity {
            let starvation_factor = (total_demand - total_capacity) / self.processes.len() as f32;
            for p in self.processes.iter_mut() {
                p.current_wait += starvation_factor * rng.random_range(0.5..1.5);
            }
        } else {
            for p in self.processes.iter_mut() {
                p.current_wait = (p.current_wait - 0.1).max(0.0);
            }
        }
        let target_per_core = (total_demand / CPU_CORES as f32).clamp(0.0, 1.0);
        for i in 0..CPU_CORES {
            let variance = rng.random_range(-0.02..0.02);
            let target = (target_per_core + variance).clamp(0.0, 1.0);
            self.cpu_load[i] += (target - self.cpu_load[i]) * 0.1;
        }
        for block in self.ram.iter_mut() { block.heat *= 0.98; }
        self.update_sensors();
    }

    pub fn spawn_process(&mut self, name: &str, cpu: f32, ram: usize, persistent: bool) -> u32 {
        self.process_counter += 1;
        let pid = self.process_counter;
        let lifespan = if persistent { u32::MAX } else { 600 };
        let mut allocated = 0;
        let mut rng = rand::rng();
        for _ in 0..200 { 
            if allocated >= ram { break; }
            let idx = rng.random_range(0..RAM_SIZE);
            if self.ram[idx].block_type == BlockType::Empty {
                self.ram[idx].block_type = BlockType::User;
                self.ram[idx].heat = 1.0;
                self.ram[idx].owner_pid = Some(pid);
                allocated += 1;
            }
        }
        self.processes.push(Process {
            id: pid, name: name.to_string(), cpu_impact: cpu, ram_impact: ram, lifespan, current_wait: 0.0
        });
        pid
    }

    pub fn kill_process(&mut self, pid: u32) {
        self.processes.retain(|p| p.id != pid);
        for block in self.ram.iter_mut() {
            if block.owner_pid == Some(pid) {
                block.block_type = BlockType::Empty;
                block.owner_pid = None;
            }
        }
    }
}
