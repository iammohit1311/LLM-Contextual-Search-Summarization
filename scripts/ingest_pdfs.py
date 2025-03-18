import os
from src.pdf_parser import extract_text_from_pdf
from src.indexer import create_faiss_index, create_faiss_index_cosine
from config import SIMILARITY_MODE

DATA_FOLDER = "../data/"

pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

if not pdf_files:
    print("❌ No PDF files found in the 'data/' folder.")
    exit(1)

all_text_chunks = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(DATA_FOLDER, pdf_file)
    print(f"📄 Extracting text from: {pdf_path}")

    pdf_text = extract_text_from_pdf(pdf_path)

    if not pdf_text:
        print(f"⚠️ No text extracted from {pdf_file}. Skipping...")
        continue

    print(f"✅ Extracted {len(pdf_text)} characters from {pdf_file}.")

    text_chunks = pdf_text.split("\n\n")
    if not text_chunks:
        print(f"⚠️ No valid text chunks found in {pdf_file}. Skipping...")
        continue

    all_text_chunks.extend(text_chunks)

print(f"🔹 Total text chunks collected: {len(all_text_chunks)}")

if all_text_chunks:
    if SIMILARITY_MODE == "euclidean":
        create_faiss_index(all_text_chunks)
    elif SIMILARITY_MODE == "cosine":
        create_faiss_index_cosine(all_text_chunks)
    print("✅ FAISS index successfully created for all PDFs.")
else:
    print("❌ No valid text found in any PDFs. FAISS indexing skipped.")
