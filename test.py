from pypdf import PdfReader

reader = PdfReader(r"C:\Users\vansh\OneDrive\Desktop\rag data pieline\rag\data\OOPS lecture notes Complete.pdf")

print("Total Pages:", len(reader.pages))

for i, page in enumerate(reader.pages[:5]):
    print(f"\n Page {i+1}\n")
    text = page.extract_text()
    if text:
        print(text[:1000])  
        print("No text found on this page.")