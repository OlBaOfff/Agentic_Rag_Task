from pathlib import Path

#Szövegként betölti a sample dokumentumot

def load_sample():
     file_path = Path(__file__).parent.parent.parent / "data" / "sample" / "transformer_intro.txt"
     with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    

if __name__ == "__main__":
    text = load_sample()
    print("Document content:\n")
    print(text)