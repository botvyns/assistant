import re
from typing import List

def receive_main_text(documents: List[Document]) -> List[Document]:
    """
    Extracts and cleans the main body content from a list of document objects.

    :param documents: A list of document objects. Each document object should have a 'page_content' attribute containing the raw text content of the document.
    :type documents: List[Document]

    :return: A list of document objects with the main body content extracted and normalized. 
    :rtype: List[Document]

    Note: This function modifies input documents in place.
    """
    doc_main_body = []
    for doc in documents:
        header_start = doc.page_content.split('Підтримати')[-1].split('Neformat.com.ua ©')[0]
        up_to_site_mention = re.sub(r'\xa0|&a|quot;|lt;|amp;', '\n', header_start).strip()
        up_to_site_mention = up_to_site_mention.replace("\n ", "").replace(" \n", "")
        up_to_site_mention = re.sub(r'[\t\r\f]+', ' ', up_to_site_mention)
        normalised = re.sub(r'\n{2,}', '\n\n', up_to_site_mention)
        doc.page_content = normalised
        doc_main_body.append(doc)
    return doc_main_body
