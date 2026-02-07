import pdfplumber
import os

pdf_files = [
    "/Users/liuzhen/Work/Liu Zhen/模型开发/小室模型/文献/德士古煤气化炉模型研究.pdf",
    "/Users/liuzhen/Work/Liu Zhen/模型开发/小室模型/文献/李政-texaco气化炉建模-cfb.pdf"
]

output_file = "docs/literature_extraction_raw.txt"

with open(output_file, 'w', encoding='utf-8') as f_out:
    for pdf_path in pdf_files:
        f_out.write(f"\n\n{'='*50}\nEXTRACTING: {os.path.basename(pdf_path)}\n{'='*50}\n\n")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    f_out.write(f"\n--- Page {i+1} ---\n")
                    text = page.extract_text()
                    if text:
                        f_out.write(text)
                    else:
                        f_out.write("[No text extracted]")
        except Exception as e:
            f_out.write(f"\n[ERROR processing file]: {str(e)}\n")

print(f"Extraction complete. Saved to {output_file}")
