import pandas as pd
import numpy as np

# Load your real data
try:
    df = pd.read_csv('training_data.csv', header=None)
    # Give columns names for easier logic
    df.columns = ['procs', 'cpu', 'wait', 'ram', 'frag', 'mouse', 'delay']
    real_data = df.values
    print(f"Seeding with {len(real_data)} real samples...")
except FileNotFoundError:
    print("Error: training_data.csv not found.")
    exit()

def augment_os_data(original_row, factor=10):
    augmented = []
    
    # Unpack the original row
    p, c, w, r, f, m, d = original_row

    for _ in range(factor):
        # 1. Identify the "State" of the seed row (using your kernel logic)
        is_meltdown = c > 0.8 and r > 0.8
        is_idle = c < 0.1 and p == 0
        is_thinking = m > 0.4 and d < 0.2
        
        # 2. Set "Mutation Strengths" (Jitter)
        # In Meltdown, CPU and RAM stay very close to the seed (low jitter)
        # In Idle, CPU stays low, but Mouse/Delay can wander more.
        
        if is_meltdown:
            # CPU/RAM are "Locked" at high values
            new_c = c + np.random.uniform(-0.01, 0.01)
            new_r = r + np.random.uniform(-0.01, 0.01)
            new_d = d + np.random.uniform(-0.05, 0.05) # Delay doesn't matter much
            new_m = m + np.random.uniform(-0.1, 0.1)
        elif is_idle:
            # Everything stays low, but Delay can fluctuate a lot
            new_c = c + np.random.uniform(-0.005, 0.005)
            new_r = 0.0 # Force zero in idle
            new_d = d + np.random.uniform(-0.1, 0.1)
            new_m = 0.0
        elif is_thinking:
            # Mouse velocity is high, Delay must stay near zero
            new_c = c + np.random.uniform(-0.05, 0.05)
            new_r = r + np.random.uniform(-0.05, 0.05)
            new_d = 0.0 
            new_m = m + np.random.uniform(-0.1, 0.1)
        else:
            # General "Middle Ground" jitter
            new_c = c + np.random.uniform(-0.03, 0.03)
            new_r = r + np.random.uniform(-0.03, 0.03)
            new_d = d + np.random.uniform(-0.05, 0.05)
            new_m = m + np.random.uniform(-0.05, 0.05)

        # Apply common mutations
        new_p = p + np.random.randint(-1, 2) # Change proc count by +/- 1
        new_w = w + np.random.uniform(-0.02, 0.02)
        new_f = f + np.random.uniform(-0.02, 0.02)

        # Final Step: Clamp values between 0.0 and 1.0 (Crucial!)
        row = [
            max(0, new_p),
            np.clip(new_c, 0.0, 1.0),
            np.clip(new_w, 0.0, 1.0),
            np.clip(new_r, 0.0, 1.0),
            np.clip(new_f, 0.0, 1.0),
            np.clip(new_m, 0.0, 1.0),
            np.clip(new_d, 0.0, 1.0)
        ]
        augmented.append(row)
    
    return augmented

# Generate new data
all_data = []
for row in real_data:
    all_data.append(row) # Keep the original
    mutations = augment_os_data(row, factor=3) # Create 5 variations of every row
    all_data.extend(mutations)

df_augmented = pd.DataFrame(all_data)
df_augmented.to_csv('training_data_augmented.csv', index=False, header=False)
print(f"Saved {len(df_augmented)} samples to training_data_augmented.csv")
