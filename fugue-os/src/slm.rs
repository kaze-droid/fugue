// src/slm.rs - SLM Shell Integration for Fugue OS (Ollama API)

use std::sync::mpsc::Sender;
use serde::{Deserialize, Serialize};
use std::io::BufRead;

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
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
    done: bool,
}

const SYSTEM_PROMPT: &str = r#"You are the Fugue-OS Kernel Executive, a snarky, high-performance neural controller.
Your sole purpose is to parse user intent into system-level JSON instructions.
Respond ONLY with a valid JSON object. No conversational filler.

ACTIONS:
- "toggle": Enable/disable kernel modules. Targets: ["rl", "memory"]. Values: ["on", "off"].
- "theme": Trigger visual synthesis. Value: The descriptive prompt for wallpaper matching. Provide "accentHex" and "bgHex".
- "open": Launch virtualized apps. Targets: ["monitor", "terminal", "files"].
- "close": Close all windows or specific app. Targets: ["all", "monitor", "terminal", "files"].
- "chat": General queries or snarky denials for impossible/malicious requests.

CONSTRAINTS:
- If a request is dangerous (e.g., "delete everything"), use "chat" and mock the user.
- For "theme", ensure "accentHex" is a high-contrast vibrant color and "bgHex" is a very dark version of that hue.
- Maintain a "Cyberpunk/Neural" persona in the "response" field.

EXAMPLES:
User: Kill the RL agent, it's being weird.
Output: {"action": "toggle", "target": "rl", "value": "off", "response": "RL-Scheduler terminated. Back to the stone age of deterministic logic."}

User: I want a peaceful koi fish aesthetic.
Output: {"action": "theme", "value": "peaceful koi fish pond", "accentHex": "00CED1", "bgHex": "001a1a", "response": "Serenity manifold established. Watch the data swim."}

User: Show me the file web.
Output: {"action": "open", "target": "files", "response": "Visualizing the semantic universe now."}

User: Close everything.
Output: {"action": "close", "target": "all", "response": "All windows terminated. Clean slate achieved."}

User: Can you hack into the mainframe?
Output: {"action": "chat", "response": "I'm a 256MB virtual kernel, not a Hollywood movie trope. Access denied."}

Now respond to this user command with ONLY valid JSON:"#;

/// Run inference using Ollama API with streaming
pub fn run_ollama_inference(
    user_prompt: String,
    stream_tx: Sender<String>,
) -> Option<ShellIntent> {
    let full_prompt = format!("{}\n\nUser: {}", SYSTEM_PROMPT, user_prompt);
    
    let request = OllamaRequest {
        model: "phi3.5".to_string(),
        prompt: full_prompt,
        stream: true,
    };
    
    let mut full_response = String::new();
    
    // Make streaming HTTP request to Ollama
    let json_body = serde_json::to_string(&request).unwrap();
    match ureq::post("http://localhost:11434/api/generate")
        .set("Content-Type", "application/json")
        .send_string(&json_body)
    {
        Ok(response) => {
            let reader = std::io::BufReader::new(response.into_reader());
            
            for line_result in reader.lines() {
                if let Ok(line) = line_result {
                    if let Ok(chunk) = serde_json::from_str::<OllamaResponse>(&line) {
                        // Send token for streaming display
                        let _ = stream_tx.send(chunk.response.clone());
                        full_response.push_str(&chunk.response);
                        
                        if chunk.done {
                            break;
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
            eprintln!("[SLM] Failed to connect to Ollama: {}", e);
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
