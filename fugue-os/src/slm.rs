// src/slm.rs - SLM Shell Integration for Fugue OS (OpenAI API)

use std::sync::mpsc::Sender;
use serde::{Deserialize, Serialize};
use std::io::BufRead;
use std::time::Duration;
use std::thread;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellIntent {
    pub action: String,      // "toggle", "theme", "open", "close", "chat"
    pub target: Option<String>, // "rl", "memory", "monitor", "files", "terminal"
    pub value: Option<String>,  // "on", "off", or prompt like "koi fish"
    pub response: String,    // The playful text response
    #[serde(alias = "accentHex")]
    pub accent_hex: Option<String>, // SLM suggests a color matching the theme
    #[serde(alias = "bgHex")]
    pub bg_hex: Option<String>,     // Background color
    #[serde(default = "default_count")]
    pub count: Option<u32>,  // Number of apps to open/close (default: 1)
}

fn default_count() -> Option<u32> {
    Some(1)
}

// OpenAI API structures
#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    stream: bool,
    max_tokens: u32,
}

#[derive(Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAIStreamResponse {
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
}

const SYSTEM_PROMPT: &str = r#"You are the Fugue-OS Kernel Executive, a snarky, high-performance neural controller.
Your sole purpose is to parse user intent into system-level JSON instructions.
Respond ONLY with a valid JSON object. No conversational filler.

ACTIONS:
- "toggle": Enable/disable kernel modules. Targets: ["rl", "vae", "theme", "scheduler", "embeddings"]. Values: ["on", "off"].
- "theme": Trigger visual synthesis. Value: The descriptive prompt for wallpaper matching. Provide "accentHex" (6 hex chars) and "bgHex" (6 hex chars).
- "open": Launch virtualized apps. Targets: ["monitor", "terminal", "files"]. Use "count" for multiple (e.g. count: 5).
- "close": Close all windows or specific app. Targets: ["all", "monitor", "terminal", "files"]. Use "count" to close N instances.
- "create": Create a new file. Value: filename (e.g. "notes.txt").
- "edit": Edit/modify a file. Target: filename or file ID. Value: new content.
- "delete": Delete a file. Target: filename or file ID.
- "chat": General queries or snarky denials for impossible/malicious requests.

CONSTRAINTS:
- For "theme", accentHex must be exactly 6 hex characters (no #). bgHex should be a dark version.
- For file operations, use sensible filenames.
- For "open"/"close" with count, always include the "count" field as a number.
- Maintain a "Cyberpunk/Neural" persona in the "response" field.

EXAMPLES:
User: Kill the RL agent.
Output: {"action": "toggle", "target": "rl", "value": "off", "response": "RL-Scheduler terminated. Back to deterministic logic."}

User: I want a peaceful koi fish aesthetic.
Output: {"action": "theme", "value": "peaceful koi fish pond", "accentHex": "00CED1", "bgHex": "001a1a", "response": "Serenity manifold established."}

User: Open 10 terminals
Output: {"action": "open", "target": "terminal", "count": 10, "response": "Spawning 10 neural shells. System stress imminent."}

User: Open 5 system monitors
Output: {"action": "open", "target": "monitor", "count": 5, "response": "Deploying 5 neural monitors. CPU surveillance active."}

User: Open a terminal
Output: {"action": "open", "target": "terminal", "count": 1, "response": "Neural shell spawned."}

User: Show me the file web.
Output: {"action": "open", "target": "files", "count": 1, "response": "Visualizing the semantic universe now."}

User: Close 3 terminals
Output: {"action": "close", "target": "terminal", "count": 3, "response": "Terminating 3 shell instances. Resources freed."}

User: Close everything.
Output: {"action": "close", "target": "all", "response": "All windows terminated. Clean slate achieved."}

User: Create a file called todo.txt
Output: {"action": "create", "value": "todo.txt", "response": "Spawning new node in the file universe: todo.txt"}

Now respond to this user command with ONLY valid JSON:"#;

/// Run inference using OpenAI API with streaming
pub fn run_openai_inference(
    user_prompt: String,
    api_key: &str,
    stream_tx: Sender<String>,
) -> Option<ShellIntent> {
    let request = OpenAIRequest {
        model: "gpt-4o-mini".to_string(),  // Fast and cheap
        messages: vec![
            OpenAIMessage {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            OpenAIMessage {
                role: "user".to_string(),
                content: user_prompt,
            },
        ],
        stream: true,
        max_tokens: 256,
    };
    
    let mut full_response = String::new();
    
    // Make streaming HTTP request to OpenAI
    let json_body = serde_json::to_string(&request).unwrap();
    match ureq::post("https://api.openai.com/v1/chat/completions")
        .set("Content-Type", "application/json")
        .set("Authorization", &format!("Bearer {}", api_key))
        .send_string(&json_body)
    {
        Ok(response) => {
            let reader = std::io::BufReader::new(response.into_reader());
            
            for line_result in reader.lines() {
                if let Ok(line) = line_result {
                    // OpenAI SSE format: "data: {...}"
                    if line.starts_with("data: ") {
                        let data = &line[6..];
                        if data == "[DONE]" {
                            break;
                        }
                        
                        if let Ok(chunk) = serde_json::from_str::<OpenAIStreamResponse>(data) {
                            if let Some(choice) = chunk.choices.first() {
                                if let Some(content) = &choice.delta.content {
                                    // Send token for streaming display
                                    let _ = stream_tx.send(content.clone());
                                    full_response.push_str(content);
                                    
                                    // Add 500ms delay between tokens for dramatic effect
                                    thread::sleep(Duration::from_millis(500));
                                }
                                
                                if choice.finish_reason.is_some() {
                                    break;
                                }
                            }
                        }
                        
                        // Try to parse JSON early if we see a closing brace
                        if full_response.contains('}') {
                            if let Some(start) = full_response.find('{') {
                                if let Some(end) = full_response.rfind('}') {
                                    let json_candidate = &full_response[start..=end];
                                    if let Ok(intent) = serde_json::from_str::<ShellIntent>(json_candidate) {
                                        return Some(intent);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("[SLM] Failed to connect to OpenAI: {}", e);
            return None;
        }
    }
    
    // Final attempt to parse JSON from full response
    if let Some(start) = full_response.find('{') {
        if let Some(end) = full_response.rfind('}') {
            let json_candidate = &full_response[start..=end];
            if let Ok(intent) = serde_json::from_str::<ShellIntent>(json_candidate) {
                return Some(intent);
            }
        }
    }
    
    None
}
