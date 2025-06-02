import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import base64


def load_pdf(pdf_path):
    try:
        return fitz.open(pdf_path)
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None


def find_title_position(doc, title):
    for page_num, page in enumerate(doc):
        instances = page.search_for(title)
        if instances:
            return page_num, instances

    return 0, None


def is_inside_or_equal(rect1, rect2):
    """Check if rect2 is completely inside rect1 or exactly equal to it."""
    return (
        rect1.x0 <= rect2.x0 and
        rect1.y0 <= rect2.y0 and
        rect1.x1 >= rect2.x1 and
        rect1.y1 >= rect2.y1
    )


def highlight_paragraph_words(doc, title_page, title_rect, paragraph):
    highlighted_rects = {}  # {page_num: [rects]}
    highlights_by_page = {}
    min_y0_tracker = {}  # {page_num: current_min_y0}

    words = paragraph.split()

    last_page = title_page


    for i, word in enumerate(words):
        word = word.strip()

        for page_num in range(last_page, min(last_page + 2, len(doc))):
            page = doc[page_num]
            rects = page.search_for(word)
            rects.sort(key=lambda x: x.y0)

            # On the title page, ignore rects above the title
            if page_num == title_page:
                rects = [r for r in rects if r.y0 >= title_rect.y1]

            page_highlights = highlighted_rects.get(page_num, [])

            # Filter out already-highlighted or upward rects
            filtered_rects = []

            for r in rects:
                already_highlighted = any(
                    is_inside_or_equal(existing, r)
                    for existing in page_highlights
                )

                # Enforce downward-only rule
                min_y0 = min_y0_tracker.get(page_num)
                below_min_y = r.y0 >= min_y0 if min_y0 else True

                if not already_highlighted and below_min_y:
                    filtered_rects.append(r)

            if filtered_rects:
                first_rect = filtered_rects[0]

                if page_num not in highlights_by_page:
                    highlights_by_page[page_num] = []
                if page_num not in highlighted_rects:
                    last_page = max(last_page, page_num)
                    highlighted_rects[page_num] = []

                highlights_by_page[page_num].append(first_rect)
                highlighted_rects[page_num].append(first_rect)

                # Update minimum y0 for that page
                min_y0_tracker[page_num] = max(
                    min_y0_tracker.get(page_num, first_rect.y0),
                    first_rect.y0
                )

                break  # Move to next word after first match

    return highlights_by_page


def render_page_with_highlights(page, highlights):
    for rect in highlights:
        highlight = page.add_highlight_annot(rect)
        highlight.set_colors(stroke=[1, 1, 0])
        highlight.update()

    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    return img


def display_pdf_with_highlights(pdf_path, context_list):
    doc = load_pdf(pdf_path)
    if doc is None:
        return

    all_highlights = {}

    for context in context_list:
        lines = context.split('\n', 1)
        title = lines[0]
        paragraph = lines[1].strip()

        if not title:
            continue

        title_page, title_rect = find_title_position(doc, title)
        if not title_rect:
            continue

        all_highlights[title_page] = title_rect

        title_rect = title_rect[0]

        highlights_by_page = highlight_paragraph_words(doc, title_page, title_rect, paragraph)

        for page_num, highlight in highlights_by_page.items():
            if page_num not in all_highlights:
                all_highlights[page_num] = []
            all_highlights[page_num].append(highlight)

    if not all_highlights:
        st.warning("No highlights could be generated.")
        doc.close()
        return

    sorted_pages = sorted(all_highlights.keys())

    st.write(f"Viser {len(sorted_pages)} markerte sider")

    html_content = """
    <div style="height: 600px; overflow-y: auto; border: 2px solid #ddd;
                border-radius: 10px; padding: 15px; background-color: #f9f9f9;
                margin: 10px 0; width: fit-content; max-width: 100%;">
    """

    for i, page_num in enumerate(sorted_pages):
        page = doc[page_num]
        highlights = all_highlights[page_num]

        try:
            img = render_page_with_highlights(page, highlights)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            html_content += f"""
            <div style="margin-bottom: 25px;">
                <h4 style="color: #333; margin-bottom: 5px;">Side {page_num + 1}</h4>
                <img src="data:image/png;base64,{img_base64}"
                     style="width: 100%; max-width: 800px; height: auto;
                            border: 1px solid #ccc; border-radius: 5px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);" />
            </div>
            """
            if i < len(sorted_pages) - 1:
                html_content += '<hr style="margin: 20px 0; border-color: #ddd;">'
        except Exception as e:
            html_content += f'<p style="color: red;">Error rendering page {page_num + 1}: {e}</p>'

    html_content += "</div>"
    st.components.v1.html(html_content, height=650, scrolling=False)
    doc.close()


def render_pdf_viewer(pdf_path, context_list):
    if not context_list:
        st.info("No context available. Ask a question to see document highlights.")
        return

    display_pdf_with_highlights(pdf_path, context_list)
