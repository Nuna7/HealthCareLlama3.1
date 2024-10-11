import re

def clean_response(text):
    """
    Retrieve only the necessay part of model output
    """
    text = re.split(r'Answer:', text, 1)[-1]
    text = re.split(r'Note:', text, 1)[0]
    
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    text = parse_response(text)
    text = text[-1]['content']
    
    return text.strip()

def parse_response(response):
    """
    Format the response.
    """
    sections = response.split('\n\n')
    parsed_sections = []
    for section in sections:
        section_lines = section.split('\n')
        if len(section_lines) > 1:
            parsed_sections.append({
                'title': section_lines[0],
                'content': '\n'.join(section_lines[1:])
            })
        else:
            parsed_sections.append({
                'content': section
            })
    
    return parsed_sections