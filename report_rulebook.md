# MetalNet Dissertation - Report Rulebook

## Overview
This document outlines all formatting rules, guidelines, and specifications implemented in the MetalNet project dissertation, following Cambridge Institute of Technology standards and the provided format guidelines.

---

## Section 1: Document Structure Rules

### 1.1 Front Matter Organization
- ✅ **Title Page** (Page 1)
  - Centered bold title in large font
  - Submission statement
  - Degree and department information
  - Three student names with university roll numbers
  - Institution name and year

- ✅ **Declaration Certificate** (Page 2)
  - Bold title centered
  - Authentication statement of original work
  - Space for guide signature and date
  - HOD signature section at bottom
  - Professional formatting with proper spacing

- ✅ **Certificate of Approval** (Page 3)
  - Institution header
  - Project approval statement
  - Space for internal and external examiner signatures
  - HOD department confirmation

- ✅ **Acknowledgement** (Page 4)
  - Personal gratitude section
  - Individual student signatures with roll numbers
  - 1.5 line spacing maintained

- ✅ **Abstract** (Page 5)
  - 250-300 word summary of project
  - Includes problem statement, methodology, key findings
  - 5+ keywords listed at bottom
  - Complete overview without external references

### 1.2 Indexing Pages
- ✅ **Table of Contents**
  - Centered title: "TABLE OF CONTENTS"
  - Chapter entries prefixed with "CHAPTER X --"
  - Roman numerals (i, ii, iii, ...) for front matter pages
  - Arabic numerals (1, 2, 3, ...) for main content pages
  - Proper indentation for sections and subsections

- ✅ **List of Figures**
  - Numbered figures with page references
  - Descriptive captions

- ✅ **List of Tables**
  - Numbered tables with page references
  - Descriptive captions

- ✅ **List of Abbreviations**
  - Alphabetically organized abbreviation table
  - Technical terms with full expansions
  - 25+ commonly used abbreviations

---

## Section 2: Typography and Formatting Rules

### 2.1 Font Specifications
| Element | Font | Size | Style |
|---------|------|------|-------|
| Main body text | Times New Roman | 12pt | Regular |
| Chapter titles | Arial/Sans-serif | 16pt | Bold |
| Section headings | Times New Roman | 14pt | Bold |
| Subsection headings | Times New Roman | 12pt | Bold |
| Captions (Figures/Tables) | Times New Roman | 10pt | Regular |
| Code listings | Courier New | 10pt | Regular (monospace) |

### 2.2 Line Spacing
- ✅ **Body text**: 1.5 line spacing throughout document
- ✅ **Paragraph spacing**: 10pt before, 4pt after (CSS-like margins)
- ✅ **Table cells**: Enhanced spacing with `\setlength{\cellspacetoplimit}{8pt}` and `\arraystretch{1.4}`

### 2.3 Page Margins
| Margin | Value |
|--------|-------|
| Top | 1.0 inch |
| Bottom | 1.0 inch |
| Left | 1.5 inches |
| Right | 1.0 inch |
| Paper Size | A4 |

### 2.4 Color and Links
- ✅ Black text throughout (no colored text except for code highlighting)
- ✅ Hyperlinks in blue for external URLs
- ✅ Cross-references in black (chapter/figure references)
- ✅ Code syntax highlighting:
  - Keywords: Blue
  - Comments: Gray
  - Strings: Red
  - Numbers: Default

---

## Section 3: Chapter Structure Rules

### 3.1 Chapter Format
- ✅ **Chapter Title Format**: `\chapter{CHAPTER NAME}`
  - Automatically numbered with Roman numerals
  - Full-page heading (not inline)
  - 16pt bold Arial font

- ✅ **Section Format**: `\section{Section Name}`
  - 14pt bold Times New Roman
  - Numbered as 1.1, 1.2, etc.
  - Contributes to table of contents

- ✅ **Subsection Format**: `\subsection{Subsection Name}`
  - 12pt bold Times New Roman
  - Numbered as 1.1.1, 1.1.2, etc.
  - Nested indentation in TOC

### 3.2 Chapter Organization (Mandatory)
All chapters follow this structure:

```
Chapter X: CHAPTER TITLE
├── Section X.1: First Major Topic
│   ├── Subsection X.1.1: Detailed Topic
│   ├── Subsection X.1.2: Another Topic
│   └── Text content with references
├── Section X.2: Second Major Topic
│   ├── List items
│   └── Enumerated points
└── References to figures and tables
```

### 3.3 Required Chapters
1. **Chapter 1: Introduction**
   - 1.1 Background and Motivation (400-500 words)
   - 1.2 Problem Statement (150-200 words)
   - 1.3 Objectives of the Project (4-6 bullet points)
   - 1.4 Scope of the Project (150-200 words)
   - 1.5 Organization of the Report (overview of chapters)

2. **Chapter 2: Literature Review**
   - 2.1 Related Works (10-15 papers, 1200-1500 words)
   - 2.2 Comparative Analysis (table comparing approaches)
   - 2.3 Gap Identification (identifying research gaps)

3. **Chapter 3: System Analysis and Design**
   - 3.1 System Requirements (hardware & software tables)
   - 3.2 System Architecture (with architecture diagram)
   - 3.3 Data Flow Diagrams (Level 0 and Level 1)
   - 3.4 Design Patterns and Implementation Details

4. **Chapter 4: Implementation**
   - 4.1 Technologies and Tools Used (with justifications)
   - 4.2 Module Description (with screenshots/diagrams)
   - 4.3 Algorithm/Methodology (with pseudo-code)
   - 4.4 Code Implementation (with code samples)

5. **Chapter 5: Testing and Results**
   - 5.1 Testing Strategy
   - 5.2 Test Cases and Results (with tables)
   - 5.3 Performance Analysis (with benchmarks)

6. **Chapter 6: Conclusion and Future Work**
   - 6.1 Conclusion (200-300 words)
   - 6.2 Future Enhancements (short-term, medium-term, long-term)

---

## Section 4: Content Rules

### 4.1 Text Content
- ✅ **Word Count Requirement**: Minimum 50-60 pages total
- ✅ **Paragraph Structure**: 
  - Opening sentence introduces topic
  - 3-5 supporting sentences
  - Concluding sentence transitions to next topic
  - Maximum 10-12 lines per paragraph

- ✅ **Technical Terminology**:
  - First use includes full term with abbreviation in parentheses
  - Example: "Convolutional Neural Networks (CNNs)"
  - Subsequent uses employ abbreviation only

- ✅ **Citation Format**: IEEE format for references
  - Books: `[#] Author, Title, Edition, Publisher, Year`
  - Journals: `[#] Author, "Title," Journal, vol., no., pp., Month Year`
  - Web: `[#] Author, "Title," Website. [Online]. Available: URL`

### 4.2 Enumeration Rules
- ✅ **Bullet Points**: Use for unordered lists
  - Each item starts with keyword or bold topic
  - Items should be roughly equal length
  - Maximum 8 items per list

- ✅ **Numbered Lists**: Use for ordered processes
  - Steps in algorithms, procedures
  - Ranked items with clear priority
  - Each item on separate line

- ✅ **Nested Lists**: Maximum 2 levels of nesting
  - First level uses •, second uses -
  - Proper indentation maintained

---

## Section 5: Figures and Tables

### 5.1 Figure Rules
- ✅ **Naming Convention**: `\label{fig:descriptive_name}`
- ✅ **Caption Format**: `\caption{Figure X.Y -- Descriptive Title}`
  - Caption placed BELOW figure
  - Number format: Chapter.Sequential
  - Text size: 10pt Times New Roman

- ✅ **Figure Placement**:
  - Use `[H]` to place figure at specific location
  - Surrounded by white space
  - Referenced in text as "Figure X.Y"

- ✅ **Figure Requirements**:
  - Minimum 600px width for readability
  - Clear labels on axes and elements
  - Self-explanatory with caption
  - High contrast, professional appearance

### 5.2 Table Rules
- ✅ **Table Formatting**:
  - Centered horizontally on page
  - Clear borders with `|` separators
  - Header row in bold
  - Cell spacing with `\cellspacetoplimit` = 8pt

- ✅ **Header Row Specification**:
  - Bold text with `\textbf{}`
  - Column alignment: Left (l), Center (c), Right (r)
  - Use `Sc` for smart-cell formatting with built-in spacing

- ✅ **Caption Format**: `\caption{Table X.Y -- Descriptive Title}`
  - Caption placed ABOVE table
  - 10pt font size
  - Number format: Chapter.Sequential

- ✅ **Table Example**:
```latex
\begin{table}[H]
\centering
\caption{Table Title}
\begin{tabular}{|Sc|Sc|Sc|}
\hline
\textbf{Column 1} & \textbf{Column 2} & \textbf{Column 3} \\
\hline
Data 1 & Data 2 & Data 3 \\
\hline
\end{tabular}
\end{table}
```

### 5.3 Code Listing Rules
- ✅ **Format**: `\begin{lstlisting}[language=C++]...\end{lstlisting}`
- ✅ **Line Numbers**: Enabled on left side
- ✅ **Syntax Highlighting**:
  - Keywords: Blue
  - Comments: Gray  
  - Strings: Red
  - Background: White

- ✅ **Code Snippets**:
  - Maximum 30 lines per snippet
  - Include brief explanation before
  - Only show relevant code sections
  - No full files unless in appendix

---

## Section 6: Reference and Citation Rules

### 6.1 Citation Format (IEEE Standard)
```
[1] Author Initials. Surname, "Title of article," Title of Journal, vol. X, no. Y, pp. xxx--xxx, Abbrev. Month Year.
[2] A. Surname and B. Surname, Title of Book, Xth ed. City: Publisher, Year.
[3] Author, "Title of webpage," Website Name. [Online]. Available: URL. [Accessed: Day Month Year].
```

### 6.2 Reference Placement
- ✅ References listed at end of document
- ✅ Numbered [1], [2], [3], ... in order of appearance
- ✅ Chapter heading: "References" (no number)
- ✅ Added to table of contents with `\addcontentsline{toc}{chapter}{References}`

### 6.3 Internal References
- ✅ **Chapter references**: "see Chapter \ref{ch:intro}"
- ✅ **Section references**: "Section \ref{sec:architecture}"
- ✅ **Figure references**: "Figure \ref{fig:system_arch}"
- ✅ **Table references**: "Table \ref{tab:performance}"
- ✅ **Equation references**: "Equation \ref{eq:conv}"

---

## Section 7: Appendices

### 7.1 Appendix Organization
- ✅ **Appendix A**: Source Code Highlights
  - Key algorithms and implementations
  - 100-200 lines per section
  - Well-commented code samples

- ✅ **Appendix B**: Screenshots and Sample Outputs
  - Profiler GUI screenshots
  - Benchmark output logs
  - Training session examples
  - Performance graphs

### 7.2 Appendix Numbering
- ✅ Chapters use letters: Appendix A, Appendix B
- ✅ Sections within appendices numbered: A.1, A.2, B.1, etc.
- ✅ Each appendix starts on new page

---

## Section 8: Document Metadata and Settings

### 8.1 LaTeX Package Requirements
```latex
\documentclass[12pt,oneside,a4paper]{report}
\usepackage[utf-8]{inputenc}
\usepackage[a4paper,top=1in,bottom=1in,left=1.5in,right=1in]{geometry}
\usepackage{setspace}      % Line spacing
\usepackage{times}         % Times font
\usepackage{hyperref}      % Clickable references
\usepackage{listings}      % Code listings
\usepackage{xcolor}        % Syntax highlighting
\usepackage{booktabs}      % Professional tables
\usepackage{caption}       % Figure/table captions
\usepackage{tocloft}       % TOC customization
\usepackage{cellspace}     % Cell spacing in tables
```

### 8.2 Document Class Settings
- ✅ **Document class**: `report` (not article or book)
- ✅ **Default font size**: 12pt
- ✅ **Page style**: oneside
- ✅ **Paper size**: a4paper (297 × 210 mm)

### 8.3 Compilation Settings
- ✅ **Compiler**: pdflatex (or xelatex)
- ✅ **Encoding**: UTF-8
- ✅ **Output**: PDF format
- ✅ **Interaction mode**: nonstopmode (for automated compilation)

---

## Section 9: Numbering Conventions

### 9.1 Chapter and Section Numbering
- ✅ **Chapters**: Roman numerals in TOC, Auto-numbered in document
- ✅ **Sections**: 1.1, 1.2, 1.3 (Chapter.Section)
- ✅ **Subsections**: 1.1.1, 1.1.2 (Chapter.Section.Subsection)
- ✅ **Sub-subsections**: 1.1.1.1 (limited use, max 3 levels)

### 9.2 Figure Numbering
- ✅ **Format**: Figure X.Y (Chapter.Sequential)
- ✅ **Example**: Figure 3.1, Figure 3.2, Figure 4.1
- ✅ **Continuous across chapters**
- ✅ **Referenced by label**: `\ref{fig:name}`

### 9.3 Table Numbering
- ✅ **Format**: Table X.Y (Chapter.Sequential)
- ✅ **Example**: Table 2.1, Table 3.2
- ✅ **Continuous across chapters**
- ✅ **Referenced by label**: `\ref{tab:name}`

### 9.4 Equation Numbering
- ✅ **Format**: (X.Y) displayed to right of equation
- ✅ **Example**: (1.1), (1.2), (2.1)
- ✅ **Reference**: "Equation \ref{eq:name}"

---

## Section 10: Page Numbering and Headers/Footers

### 10.1 Page Numbering Scheme
- ✅ **Front matter (Title through Abstract)**:
  - Roman numerals: i, ii, iii, iv, v
  - Centered in footer
  - No header text

- ✅ **Main content (Chapter 1 onwards)**:
  - Arabic numerals: 1, 2, 3, ...
  - Centered in footer
  - Reset to page 1 at Chapter 1

### 10.2 Header/Footer Formatting
- ✅ **Headers**: Removed (no running headers)
- ✅ **Footers**: Centered page number only
- ✅ **Page 1 of chapters**: Footers removed (plain style)
- ✅ **Odd/even pages**: Same formatting on both sides

---

## Section 11: Special Formatting Rules

### 11.1 Mathematical Equations
- ✅ **Inline math**: `$formula$` (small size)
- ✅ **Display math**: `$$formula$$` or `\[formula\]` (full line)
- ✅ **Equation environment** for numbered equations:
```latex
\begin{equation}
E = mc^2
\label{eq:einstein}
\end{equation}
```

### 11.2 Algorithms and Pseudo-code
- ✅ **Presented as numbered blocks**
- ✅ **Clear variable naming**
- ✅ **Indentation shows control flow**
- ✅ **Comments with `//`**
- ✅ **Function declarations with `function` keyword**

### 11.3 Emphasis and Special Text
- ✅ **Bold text**: `\textbf{text}` for emphasis
- ✅ **Italic text**: `\textit{text}` for book titles, variables
- ✅ **Code text**: `\texttt{code}` for inline code
- ✅ **Small caps**: Not used (readability concern)

### 11.4 Lists Within Text
- ✅ **Inline lists**: Use parentheses, separated by semicolons
  - Example: "(a) first item; (b) second item; (c) third item"
- ✅ **Bullet lists**: Use `\begin{itemize}`
- ✅ **Numbered lists**: Use `\begin{enumerate}`

---

## Section 12: Project-Specific Rules (MetalNet)

### 12.1 Content Requirements
- ✅ **Technical depth**: Include implementation details, algorithms, code samples
- ✅ **Benchmark results**: Quantitative performance comparisons with PyTorch
- ✅ **Optimization focus**: Emphasize Implicit GEMM, SIMD, cache tiling, fusion
- ✅ **Complete autodiff**: DAG-based automatic differentiation explained
- ✅ **Memory analysis**: Detailed memory footprint comparisons

### 12.2 Student Information
- ✅ **Names on title page**:
  - Raj Kunwar Prabhat (22050440078)
  - Mayan Kumar (22050440058)
  - Sujeet Kumar (22050440097)
- ✅ **Names on acknowledgement page** with individual roll numbers
- ✅ **Contribution notes in Chapter 6** (Conclusions)

### 12.3 Institution Information
- ✅ **Institution**: Cambridge Institute of Technology
- ✅ **Location**: Tatisilwai, Ranchi -- 835103
- ✅ **Department**: Computer Science & Engineering
- ✅ **HOD**: Prof. DEEPAK KUMAR VERMA
- ✅ **Degree**: Bachelor of Technology (B.Tech)

---

## Section 13: Quality Assurance Checklist

### 13.1 Content Checklist
- ✅ Abstract includes problem, methodology, findings, keywords
- ✅ Each chapter has introduction and conclusion
- ✅ All referenced figures and tables are present
- ✅ No broken cross-references (`??` symbols)
- ✅ All equations are numbered (if referenced)
- ✅ Bibliography entries are complete and accurate

### 13.2 Formatting Checklist
- ✅ Consistent font sizes throughout
- ✅ Consistent line spacing (1.5)
- ✅ Proper indentation of all lists
- ✅ Margins are correct (1", 1", 1.5", 1")
- ✅ Page numbers visible on all pages
- ✅ Chapter titles are centered and bold

### 13.3 Completeness Checklist
- ✅ Minimum 60 pages of content
- ✅ All 6 chapters present
- ✅ References section with 15+ entries
- ✅ Table of Contents complete and accurate
- ✅ List of Figures with all figures referenced
- ✅ List of Tables with all tables referenced
- ✅ Abbreviations list with 25+ terms

---

## Section 14: Compilation and Distribution

### 14.1 LaTeX Compilation Command
```bash
pdflatex -interaction=nonstopmode MetalNet_Dissertation.tex
# First pass: generates TOC structure
pdflatex -interaction=nonstopmode MetalNet_Dissertation.tex
# Second pass: resolves all cross-references
```

### 14.2 Output Files
- ✅ **Primary**: `MetalNet_Dissertation.pdf` (for submission)
- ✅ **Source**: `MetalNet_Dissertation.tex` (for backup)
- ✅ **Aux files**: `.aux`, `.toc`, `.lof`, `.lot` (intermediate, can delete)

### 14.3 Hardbound Preparation
- ✅ **Cover color**: Light sky-blue (RGB: 173, 216, 230)
- ✅ **Text color**: Black
- ✅ **Copies**: 6 hardbound copies required
- ✅ **Distribution**:
  - 1 for project guide
  - 1 for department
  - 1 for central library
  - 3 for student team members

---

## Section 15: Final Validation Rules

### 15.1 Pre-Submission Checks
```
□ Document compiles without errors
□ All page numbers are correct
□ TOC page numbers match actual sections
□ All figures are referenced in text
□ All tables are referenced in text
□ No "??" symbols (unresolved references)
□ Margins are correct on all pages
□ Font sizes match specifications
□ Line spacing is 1.5 throughout
□ No spelling or grammatical errors
□ Bibliography entries are in IEEE format
□ All student names and roll numbers present
□ Institutional information correct
□ Total page count is 60+ pages
```

### 15.2 PDF Quality Validation
- ✅ PDF is searchable (text-based, not image)
- ✅ Hyperlinks are functional
- ✅ Page dimensions are A4
- ✅ All fonts are embedded
- ✅ Color mode is RGB (for digital) or CMYK (for printing)

---

## Appendix: Common LaTeX Commands Used

### Document Structure
```latex
\chapter{Chapter Name}           % New chapter
\section{Section Name}           % Major section
\subsection{Subsection Name}     % Subsection
\newpage                         % New page break
\pagebreak                       % Page break with no fill
\clearpage                       % Clear all floats
```

### Text Formatting
```latex
\textbf{bold}                    % Bold text
\textit{italic}                  % Italic text
\texttt{monospace}               % Monospace/code
\underline{underlined}           % Underlined text
\small \normalsize \large        % Font size changes
```

### Lists and Enumerations
```latex
\begin{itemize}
  \item First item
  \item Second item
\end{itemize}

\begin{enumerate}
  \item First point
  \item Second point
\end{enumerate}
```

### Figures and Tables
```latex
\begin{figure}[H]
  \centering
  \includegraphics{image.pdf}
  \caption{Figure caption}
  \label{fig:name}
\end{figure}

\begin{table}[H]
  \centering
  \caption{Table caption}
  \begin{tabular}{|c|c|}
    ...
  \end{tabular}
  \label{tab:name}
\end{table}
```

### References and Labels
```latex
\label{fig:name}                 % Create label
\ref{fig:name}                   % Reference label
\cite{key}                       % Citation
\begin{thebibliography}{99}      % Bibliography environment
  \bibitem{key} Reference text
\end{thebibliography}
```

---

## Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | June 9, 2026 | Initial dissertation creation with all required sections |
| 1.1 | June 9, 2026 | Fixed Declaration Certificate formatting; Added roman numerals to TOC; Improved table UI with enhanced spacing |
| 1.2 | June 9, 2026 | Added profiling methodology section; Expanded performance analysis with component breakdown and framework comparison; Added design trade-offs analysis; Expanded conclusion with reproducibility and broader impact sections |

---

**End of Report Rulebook**

*This rulebook serves as a comprehensive reference for all formatting, structural, and content requirements implemented in the MetalNet dissertation. It ensures consistency, clarity, and adherence to Cambridge Institute of Technology standards.*
