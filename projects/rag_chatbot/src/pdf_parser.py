import pdfplumber


def text_parser(path):
    parsed_data = []
    pages = pdfplumber.open(path).pages
    for i, page in enumerate(pages):
        text = page.extract_text()
        if text is None:
            continue
        text = {"page": i + 1, "text": text}
        parsed_data.append(text)
    return parsed_data