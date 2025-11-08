#!/bin/bash
TRACE_NAME=boonchu_trace_3185279
REPORT="osrt_sum"
OUT_DIR="summary"
mkdir -p $OUT_DIR

python3 <<EOF
import json
import pandas as pd
import os

trace_name = "${TRACE_NAME}"
report = "${REPORT}"
out_dir = "${OUT_DIR}"   # <-- ส่งค่า Bash variable เข้า Python
fname = f"{trace_name}_{report}_{report}.json"

if not os.path.isfile(fname) or os.path.getsize(fname) == 0:
    print(f"ERROR: {fname} not found or empty.")
else:
    with open(fname, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"ERROR: {fname} is not valid JSON.")
            data = []

    if data:
        # ใส่ชื่อ report
        for row in data:
            row['report'] = report

        # สร้าง DataFrame
        df = pd.DataFrame(data)

        # Export CSV
        csv_file = os.path.join(out_dir, f"{trace_name}_summary.csv")
        df.to_csv(csv_file, index=False)

        # Export Markdown
        md_file = os.path.join(out_dir, f"{trace_name}_summary.md")
        with open(md_file, "w") as f:
            f.write(df.to_markdown(index=False))

        print(f"Summary exported to: {csv_file}, {md_file}")
    else:
        print("No data found in JSON file.")
EOF
