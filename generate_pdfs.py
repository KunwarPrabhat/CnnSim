import os
import subprocess

with open('MetalNet_Report.tex', 'r') as f:
    lines = f.readlines()

# We only need up to line 437
base_lines = lines[:437]

# Let's replace the last few lines for each variation
def generate_pdf(filename, names_text):
    # Find the flushright block
    out_lines = base_lines.copy()
    for i in range(len(out_lines)-1, -1, -1):
        if r'\begin{flushright}' in out_lines[i]:
            # Replace lines after this
            out_lines[i+1] = names_text + '\n'
            out_lines[i+2] = r'\end{flushright}' + '\n'
            # Delete any extra lines before \end{flushright}
            del out_lines[i+3:]
            break
            
    out_lines.append(r'\end{document}' + '\n')
    
    tex_filename = f"{filename}.tex"
    with open(tex_filename, 'w') as f:
        f.writelines(out_lines)
        
    print(f"Compiling {tex_filename}...")
    subprocess.run(["./tectonic", tex_filename])
    
    os.makedirs("Final_pdf", exist_ok=True)
    subprocess.run(["mv", f"{filename}.pdf", f"Final_pdf/{filename}.pdf"])
    
generate_pdf("pdf1", r"\textbf{Raj Kunwar Prabhat}\\" + "\n" + r"\textbf{22050440078}")
generate_pdf("pdf2", r"\textbf{Mayan Kumar}\\" + "\n" + r"\textbf{22050440058}")
generate_pdf("pdf3", r"\textbf{Sujeet Kumar}\\" + "\n" + r"\textbf{22050440097}")

all_three = r"\textbf{Raj Kunwar Prabhat \hspace{0.2cm} 22050440078}\\" + "\n" + \
            r"\textbf{Mayan Kumar \hspace{1cm} 22050440058}\\" + "\n" + \
            r"\textbf{Sujeet Kumar \hspace{0.8cm} 22050440097}"
# Alternatively, use tabular in flushright:
all_three_tabular = r"""\begin{tabular}{@{}lr@{}}
\textbf{Raj Kunwar Prabhat} & \textbf{22050440078} \\
\textbf{Mayan Kumar} & \textbf{22050440058} \\
\textbf{Sujeet Kumar} & \textbf{22050440097} \\
\end{tabular}"""

generate_pdf("pdf4", all_three_tabular)
print("Done!")
