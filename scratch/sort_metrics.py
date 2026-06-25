import os
import re
import pandas as pd

def extract_config_num(config_str):
    # Match the leading number in "X. Model Name"
    match = re.match(r'^(\d+)\.', str(config_str))
    if match:
        return int(match.group(1))
    return 999  # Fallback

def get_balancing_order(bal_str):
    order = {
        "None": 0,
        "Pairwise": 1,
        "Cluster": 2,
        "RST-Neighborhood": 3,
        "DSNB": 4
    }
    return order.get(str(bal_str), 99)

def main():
    csv_path = "metrics.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print(f"Loading metrics from {csv_path}...")
    df = pd.read_csv(csv_path, keep_default_na=False)
    df["Balancing"] = df["Balancing"].replace({"nan": "None", "": "None"})
    
    # Create temporary sorting keys
    df["_config_num"] = df["Model Configuration"].apply(extract_config_num)
    df["_balancing_order"] = df["Balancing"].apply(get_balancing_order)
    
    # Sort by config number first, then by balancing method order
    df_sorted = df.sort_values(by=["_config_num", "_balancing_order"]).copy()
    
    # Drop temporary columns
    df_sorted = df_sorted.drop(columns=["_config_num", "_balancing_order"])
    
    # Save back to CSV
    df_sorted.to_csv("metrics.csv", index=False)
    print("Saved sorted metrics.csv in workspace.")
    
    # Save back to Markdown in workspace
    with open("metrics.md", "w", encoding="utf-8") as f:
        f.write("# SQuAD Sentence Salience Comparative Results\n\n")
        f.write(df_sorted.to_markdown(index=False))
    print("Saved sorted metrics.md in workspace.")
    
    # Save copies to brain artifacts directory
    brain_dir = r"C:\Users\maddu\.gemini\antigravity\brain\64294085-ec70-478d-8d81-2ec235975552"
    if os.path.exists(brain_dir):
        df_sorted.to_csv(os.path.join(brain_dir, "metrics.csv"), index=False)
        with open(os.path.join(brain_dir, "metrics.md"), "w", encoding="utf-8") as f:
            f.write("# SQuAD Sentence Salience Comparative Results\n\n")
            f.write(df_sorted.to_markdown(index=False))
        print("Saved copies to brain artifacts directory.")
        
        # Also update the metrics summary to reflect the same order or mention it
        print("Updating walkthrough.md to include the sorted table...")
        update_walkthrough(df_sorted, brain_dir)

def update_walkthrough(df_sorted, brain_dir):
    walkthrough_path = os.path.join(brain_dir, "walkthrough.md")
    if not os.path.exists(walkthrough_path):
        return
        
    with open(walkthrough_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # We want to replace the table in walkthrough.md.
    # The table starts with "| Model Configuration" and ends before "---" or "## 3."
    table_pattern = r"\| Model Configuration.*?\n\n"
    # Find the table in walkthrough.md
    markdown_table = df_sorted.to_markdown(index=False)
    
    # Find start and end indices of the table
    lines = content.split("\n")
    start_idx = -1
    end_idx = -1
    for idx, line in enumerate(lines):
        if line.startswith("| Model Configuration"):
            start_idx = idx
            # Find the first blank line after this
            for j in range(idx + 1, len(lines)):
                if lines[j].strip() == "":
                    end_idx = j
                    break
            break
            
    if start_idx != -1 and end_idx != -1:
        new_lines = lines[:start_idx] + [markdown_table] + lines[end_idx:]
        new_content = "\n".join(new_lines)
        with open(walkthrough_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Updated walkthrough.md in brain directory with sorted table.")
        
        # Also copy it to workspace if walkthrough.md exists there?
        # Actually walkthrough.md is only in the brain directory according to list_dir.

if __name__ == "__main__":
    main()
