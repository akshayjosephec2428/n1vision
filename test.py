import os

# Your exact folder paths
img_folder = r"C:\Users\aksha\Downloads\nutri\train\labels"
lbl_folder = r"C:\Users\aksha\Downloads\nutri\train\images"

# Get all file names without the .jpg or .txt at the end
images = set([os.path.splitext(f)[0] for f in os.listdir(img_folder) if f != "classes.txt"])
labels = set([os.path.splitext(f)[0] for f in os.listdir(lbl_folder) if f != "classes.txt"])

# Find the ones that don't match
unmatched_images = images - labels
unmatched_labels = labels - images

print("--- SCAN COMPLETE ---")
print(f"Total Images Found: {len(images)}")
print(f"Total Labels Found: {len(labels)}\n")

print(f"Images without a matching label: {len(unmatched_images)}")
if unmatched_images:
    print("Here are a few examples:", list(unmatched_images)[:5])

print(f"\nLabels without a matching image: {len(unmatched_labels)}")
if unmatched_labels:
    print("Here are a few examples:", list(unmatched_labels)[:5])