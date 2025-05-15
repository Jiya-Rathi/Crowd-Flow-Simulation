import subprocess

# List of sequence IDs
seq_ids = [
    "00048", "00049", "00050", "00051", "00052", "00053", "00054", "00055",
    "00056", "00057", "00058", "00081", "00083", "00084", "00085", "00086",
    "00087", "00089", "00090", "00091", "00092", "00093", "00094", "00093",
    "00096", "00097", "00098", "00099", "00106", "00107", "00109", "00110",
    "00059", "00060", "00064", "00066", "00067", "00068", "00071", "00072",
    "00073", "00076", "00077", "00078", "00079", "00080", "00100", "00102",
    "00104"
]

for seq_id in seq_ids:
    cmd = [
        "python3", "visualize_coco.py",
        "--seq_id", seq_id,
        "--annotation_file", f"./annotations_with_ids/{seq_id}_with_ids.txt",
        "--image_dir", "./sequences",
        "--output_dir", f"./viz_output/{seq_id}"
    ]
    print(f"Running visualization for {seq_id}...")
    subprocess.run(cmd)
