use rand::Rng;
use ort::session::Session;
use ort::value::Value;
use ndarray::Array2;
use crate::theme_engine::ThemeEngine;
use crate::graph_fs::GraphFileSystem;
use std::sync::mpsc::{self, Receiver, Sender};
use std::collections::VecDeque;

pub const RAM_SIZE: usize = 256;
pub const CPU_CORES: usize = 4;
pub const MAX_PROCESS_SLOTS: usize = 5;

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

// FileNode is now part of GraphFileSystem (graph_fs::graph::GraphNode)

#[derive(Clone, Debug)]
pub struct Process {
    pub id: u32,
    pub name: String,
    pub cpu_impact: f32,
    pub ram_impact: usize,
    pub lifespan: u32,
    pub current_wait: f32,
    // New fields for RL scheduler
    pub remaining_work: f32,  // Work units remaining
    pub is_ui: bool,          // Is this a UI process?
    pub priority: u8,         // Priority level (0-3)
    pub is_stress_test: bool, // Is this a stress test process?
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
    pub file_system: GraphFileSystem,
    pub selected_file_id: Option<u32>,
    pub current_mindset: KernelMindset,
    pub system_state: SystemState,
    pub tick: u64,
    pub shell_buffer: String,
    pub shell_cursor: usize,  // Cursor position in shell_buffer
    pub backspace_held_frames: u32,  // Track how long backspace is held
    pub model_session: Option<Session>,
    pub vae_session: Option<Session>,
    pub theme_engine: Option<ThemeEngine>,
    pub scheduler_session: Option<Session>,
    pub rl_scheduler_enabled: bool,
    pub vae_enabled: bool,           // VAE neural defrag
    pub theme_enabled: bool,         // Theme engine
    pub embeddings_enabled: bool,    // File embeddings
    pub last_scheduled_process: Option<usize>,  // Track which process was scheduled
    pub rl_decision_flash: f32,  // Visual flash when RL makes a decision
    pub target_fps: u32,  // Target FPS for frame limiting
    pub cpu_pressure: f32,  // Overall CPU pressure (0.0 to 1.0)
    // SLM Shell Integration (Ollama API)
    pub stream_tx: Sender<String>,
    pub stream_rx: Receiver<String>,
    pub intent_tx: Sender<crate::slm::ShellIntent>,
    pub intent_rx: Receiver<crate::slm::ShellIntent>,
    pub partial_response: String,
    pub is_thinking: bool,
    pub kernel_messages: VecDeque<String>,
    pub ollama_available: bool,  // Track if Ollama is running
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

        // Load RL Scheduler
        match Session::builder()
            .and_then(|builder| builder.commit_from_file("scheduler_policy.onnx"))
        {
            Ok(session) => {
                self.scheduler_session = Some(session);
                println!("[Hardware] RL Scheduler loaded successfully");
            }
            Err(e) => {
                eprintln!("[Hardware] Failed to load RL scheduler: {}", e);
                eprintln!("[Hardware] Continuing without RL scheduler - will use simple scheduling");
                self.scheduler_session = None;
            }
        }

        // Load Theme Engine
        match ThemeEngine::new() {
            Ok(engine) => {
                self.theme_engine = Some(engine);
                println!("[Hardware] Theme engine loaded successfully");
            }
            Err(e) => {
                eprintln!("[Hardware] Failed to load theme engine: {}", e);
                self.theme_engine = None;
            }
        }

        // Check if Ollama is available
        println!("[Hardware] Checking Ollama API availability...");
        match ureq::get("http://localhost:11434/api/tags").call() {
            Ok(_) => {
                self.ollama_available = true;
                println!("[Hardware] ✓ Ollama API is available");
            }
            Err(e) => {
                self.ollama_available = false;
                eprintln!("[Hardware] ✗ Ollama not available: {}", e);
                eprintln!("[Hardware] Make sure Ollama is running: ollama serve");
            }
        }
    }

    pub fn new() -> Self {
        let empty_block = MemoryBlock { id: 0, block_type: BlockType::Empty, heat: 0.0, owner_pid: None };
        
        // Try to load existing file system, or create new with sample files
        let mut file_system = match GraphFileSystem::load_from_file("fugue_filesystem.dat") {
            Ok(fs) => {
                println!("Loaded file system from disk ({} files)", fs.node_count());
                fs
            }
            Err(e) => {
                println!("Creating new file system: {}", e);
                let mut fs = GraphFileSystem::new();
                for i in 0..15 {
                    let name = format!("node_{:02X}.txt", i);
                    let path = format!("/home/user/{}", name);
                    let content = format!("Sample content for file {}", i);
                    fs.create_file(name, path, content);
                }
                fs
            }
        };
        
        // Initialize embedding service and generate embeddings at boot
        println!("[Hardware] Initializing file embeddings...");
        match file_system.init_embedding_service("text_embedding_model.onnx", "tokenizer.json") {
            Ok(_) => {
                println!("[Hardware] ✓ Embedding service initialized");
                // Generate embeddings for all files at boot
                match file_system.update_graph_with_embeddings() {
                    Ok(_) => println!("[Hardware] ✓ File embeddings generated and graph updated"),
                    Err(e) => eprintln!("[Hardware] ✗ Failed to generate embeddings: {}", e),
                }
            }
            Err(e) => {
                eprintln!("[Hardware] ✗ Failed to initialize embedding service: {}", e);
                eprintln!("[Hardware] File system will work without semantic embeddings");
            }
        }

        // Create channels for SLM communication
        let (stream_tx, stream_rx) = mpsc::channel();
        let (intent_tx, intent_rx) = mpsc::channel();

        Self {
            ram: [empty_block; RAM_SIZE],
            cpu_load: [0.0; CPU_CORES],
            processes: Vec::new(),
            file_system,
            selected_file_id: None,
            current_mindset: KernelMindset::Idle,
            system_state: SystemState::new(),
            tick: 0,
            shell_buffer: String::new(),
            shell_cursor: 0,
            backspace_held_frames: 0,
            model_session: None,
            vae_session: None,
            theme_engine: None,
            scheduler_session: None,
            rl_scheduler_enabled: false,  // Default OFF
            vae_enabled: true,            // VAE neural defrag ON by default
            theme_enabled: true,          // Theme engine ON by default
            embeddings_enabled: true,     // File embeddings ON by default
            last_scheduled_process: None,
            rl_decision_flash: 0.0,
            target_fps: 60,
            cpu_pressure: 0.0,
            stream_tx,
            stream_rx,
            intent_tx,
            intent_rx,
            partial_response: String::new(),
            is_thinking: false,
            kernel_messages: VecDeque::new(),
            ollama_available: false,
            process_counter: 0,
        }
    }
    
    /// Save the file system to disk
    pub fn save_file_system(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.file_system.save_to_file("fugue_filesystem.dat")?;
        println!("File system saved ({} files)", self.file_system.node_count());
        Ok(())
    }

    fn get_scheduler_state(&self) -> Vec<f32> {
        let mut state = vec![0.0; MAX_PROCESS_SLOTS * 4];
        
        for (i, process) in self.processes.iter().take(MAX_PROCESS_SLOTS).enumerate() {
            let base = i * 4;
            state[base] = process.current_wait;
            state[base + 1] = process.remaining_work;
            state[base + 2] = if process.is_ui { 1.0 } else { 0.0 };
            state[base + 3] = process.priority as f32;
        }
        
        state
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
        if !self.vae_enabled { return; }  // Skip if VAE disabled
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
        // Use GraphFileSystem's built-in physics update
        self.file_system.update_physics();
    }

    pub fn update_physics(&mut self) {
        let mut rng = rand::rng();
        self.tick += 1;
        
        // Update mindset naturally
        self.update_mindset();
        
        // Automatically enable RL scheduler when in Rescheduling or OptimizingRAM mode
        self.rl_scheduler_enabled = matches!(
            self.current_mindset, 
            KernelMindset::Rescheduling | KernelMindset::OptimizingRAM
        );

        // --- Trigger Neural Defrag ---
        if self.current_mindset == KernelMindset::OptimizingRAM && self.tick % 60 == 0 {
            self.neural_defrag();
        }

        self.update_file_system();
        self.processes.retain(|p| p.lifespan > 0);
        for p in self.processes.iter_mut() {
            if p.lifespan < u32::MAX { p.lifespan -= 1; }
        }
        
        // Calculate CPU pressure
        let total_demand: f32 = self.processes.iter().map(|p| p.cpu_impact).sum();
        self.cpu_pressure = (total_demand / (CPU_CORES as f32 * 0.8)).clamp(0.0, 1.0);
        
        // FPS always based on cpu_pressure - high pressure = laggy system
        if self.cpu_pressure > 0.7 {
            self.target_fps = 15;
        } else if self.cpu_pressure > 0.5 {
            self.target_fps = 25;
        } else if self.cpu_pressure > 0.3 {
            self.target_fps = 40;
        } else {
            self.target_fps = 60;
        }
        
        // RL-based process scheduling - reduces pressure over time
        if self.rl_scheduler_enabled 
            && !self.processes.is_empty() 
        {
            // Get state first before borrowing scheduler
            let state = self.get_scheduler_state();
            
            if let Some(scheduler) = &mut self.scheduler_session {
                let input_array = Array2::from_shape_vec((1, 20), state).unwrap();
                let input_tensor = Value::from_array(input_array).unwrap();
                
                let outputs = scheduler.run(ort::inputs!["state" => input_tensor]).unwrap();
                let action_probs = outputs["action_probs"].try_extract_tensor::<f32>().unwrap();
                
                // Select best action (process to schedule)
                let best_slot = action_probs.1.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                
                // Execute the scheduling decision
                if best_slot < self.processes.len() {
                    self.last_scheduled_process = Some(best_slot);
                    self.rl_decision_flash = 1.0;
                    
                    let process = &mut self.processes[best_slot];
                    
                    // RL throttles stress tests to maintain responsiveness
                    if process.is_stress_test {
                        process.remaining_work = (process.remaining_work - 0.3).max(0.0);  // Throttled
                    } else {
                        process.remaining_work = (process.remaining_work - 1.0).max(0.0);  // Normal
                    }
                    process.current_wait = 0.0;
                    
                    // Update wait times for other processes
                    for (i, p) in self.processes.iter_mut().enumerate() {
                        if i != best_slot && p.remaining_work > 0.0 {
                            p.current_wait += 1.0;
                        }
                    }
                    
                    // RL actively reduces pressure by smart scheduling
                    self.cpu_pressure = (self.cpu_pressure - 0.03).max(0.0);
                }
            }
            
            // Visual "cheat": Increase heat on stress test RAM blocks when under pressure
            if self.cpu_pressure > 0.7 {
                for block in self.ram.iter_mut() {
                    if let Some(pid) = block.owner_pid {
                        if self.processes.iter().any(|p| p.id == pid && p.is_stress_test) {
                            block.heat = (block.heat + 0.3).min(2.0);  // Make it look hot!
                        }
                    }
                }
            }
        } else {
            // Clear RL scheduling indicators
            self.last_scheduled_process = None;
            
            // Fallback to simple scheduling when RL not active
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
        }
        
        // CPU load distribution - different behavior based on RL mode
        if self.rl_scheduler_enabled && self.current_mindset == KernelMindset::Rescheduling {
            // RL mode: Efficient scheduling - CPU cores go down quickly
            let target_per_core = (self.cpu_pressure * 0.6).clamp(0.0, 1.0);  // RL keeps it lower
            for i in 0..CPU_CORES {
                let variance = rng.random_range(-0.03..0.03);
                let target = (target_per_core + variance).clamp(0.0, 1.0);
                // Fast decay when RL is optimizing
                if self.cpu_load[i] > target {
                    self.cpu_load[i] += (target - self.cpu_load[i]) * 0.4; // Quick drop
                } else {
                    self.cpu_load[i] += (target - self.cpu_load[i]) * 0.15; // Normal rise
                }
            }
        } else {
            // Simple mode: Standard target_per_core logic - less efficient
            let target_per_core = (self.cpu_pressure).clamp(0.0, 1.0);  // Uses full pressure
            for i in 0..CPU_CORES {
                let variance = rng.random_range(-0.02..0.02);
                let target = (target_per_core + variance).clamp(0.0, 1.0);
                self.cpu_load[i] += (target - self.cpu_load[i]) * 0.08; // Slower, uniform
            }
        }
        for block in self.ram.iter_mut() { block.heat *= 0.98; }
        self.update_sensors();
    }

    pub fn spawn_process(&mut self, name: &str, cpu: f32, ram: usize, persistent: bool, is_ui: bool) -> u32 {
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
        
        let remaining_work = if is_ui { 
            rng.random_range(3.0..10.0)  // UI tasks are shorter
        } else {
            rng.random_range(5.0..20.0)
        };
        
        self.processes.push(Process {
            id: pid,
            name: name.to_string(),
            cpu_impact: cpu,
            ram_impact: ram,
            lifespan,
            current_wait: 0.0,
            remaining_work,
            is_ui,
            priority: rng.random_range(0..4),
            is_stress_test: false,
        });
        pid
    }
    
    pub fn spawn_stress_test(&mut self) -> u32 {
        self.process_counter += 1;
        let pid = self.process_counter;
        let mut allocated = 0;
        let mut rng = rand::rng();
        for _ in 0..200 {
            if allocated >= 50 { break; }
            let idx = rng.random_range(0..RAM_SIZE);
            if self.ram[idx].block_type == BlockType::Empty {
                self.ram[idx].block_type = BlockType::User;
                self.ram[idx].heat = 1.0;
                self.ram[idx].owner_pid = Some(pid);
                allocated += 1;
            }
        }
        self.processes.push(Process {
            id: pid,
            name: "STRESS_TEST".to_string(),
            cpu_impact: 2.5,
            ram_impact: 50,
            lifespan: 600,
            current_wait: 0.0,
            remaining_work: rng.random_range(100.0..200.0),
            is_ui: false,
            priority: 0,
            is_stress_test: true,
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
