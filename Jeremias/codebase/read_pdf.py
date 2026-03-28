import PyPDF2

def extract_text(pdf_path, out_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        with open(out_path, 'w', encoding='utf-8') as out_file:
            for page in reader.pages:
                out_file.write(page.extract_text() + '\n')

if __name__ == '__main__':
    extract_text('lab1.pdf', 'pdf_content.txt')
