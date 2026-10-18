from book_data import load_books
from book_profiles import build_book_profiles


books = load_books()

print(f"Building profiles for {len(books)} books...")

build_book_profiles(books)

print("Book profiles saved successfully.")