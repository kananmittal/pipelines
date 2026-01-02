import os

src = "/Users/kananmittal/Desktop/untitled folder"
dst = "/Users/kananmittal/Desktop/llm pipeline"

print(f"Renaming '{src}' to '{dst}'...")

try:
    if os.path.exists(src):
        os.rename(src, dst)
        print("Success! Folder renamed.")
        print("IMPORTANT: The current agent session may lose access. Please restart or update workspace.")
    else:
        print(f"Error: Source folder '{src}' not found.")
except Exception as e:
    print(f"Failed to rename: {e}")
