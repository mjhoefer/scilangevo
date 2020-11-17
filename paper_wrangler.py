#
# The purpose of this file is to extract the lemmatized words from the paper corpus
# It essentially walks the directory, loads each json file, and writes all the text into a single file
# We're using .words as an extension to distinguish
#


from pathlib import Path
import json
import string


def extract_lemma_text(json_file):
    data = json.loads(json_file.read_bytes())
    contents = ""

    # now we need to get the paper contents into something that can be parsed using TWEC.
    # This will largely be concatinating the tokens found in the json structure.
    data['full_text'] = ''
    data['lemmas'] = []
    for section in data['sections']:
        for subsection in section['subsections']:
            for sentence in subsection['sentences']:
                for token in sentence['tokens']:
                    if token['lemma'] not in string.punctuation:
                        contents += token['lemma'] + ' '

    return contents


def write_out_contents(contents, path_to_write):
    with open(path_to_write, 'w', encoding='utf8') as outfile:
        outfile.write(contents)


def extract_text_and_write(json_file):
    content = extract_lemma_text(json_file)
    to_write = json_file.parent / (json_file.stem + '.words')
    write_out_contents(content, to_write)


p = Path('.')
p = p / '..' / 'project_code' / 'papers'
p = p / 'acl-arc-json' / 'json'
root_directory = p
files_to_parse = []
ids_already_complete = []

# walk dir and get files we need to parse
for path_object in root_directory.glob('**/*'):
    if path_object.is_file():
        # gather all json files
        if path_object.suffix == '.json':
            files_to_parse.append(path_object)
        elif path_object.suffix == '.words':
            # just save off the ID
            ids_already_complete.append(path_object.stem)

num_remaining = len(files_to_parse) - len(ids_already_complete)
print("Need to parse ", num_remaining)

num_complete = 0

# now go through and actually parse any files we haven't parsed yet
for file in files_to_parse:
    if file.stem not in ids_already_complete:
        if num_complete % 100 == 0:
            print("Completed ", num_complete, " out of ", num_remaining, ", ", num_complete / num_remaining,
                  "% complete")
        extract_text_and_write(file)
        num_complete += 1

exit()

#### OLD STUFF BELOW

# # We can do a quick parse on the index of the paper to determine its path
# # J11-1005
#
# paper_id = 'J11-1005'
#
# # need first char and the last two nums to get it all.
# first_char = paper_id[0]
# two_nums = paper_id[1:3]
#
# paper_dir = p / first_char / (first_char + two_nums)
#
# # handle to particular file
# paper = paper_dir / (paper_id + '.json')
#
# data = json.loads(paper.read_bytes())
#
# # now we need to get the paper contents into something that can be parsed using TWEC.
# # This will largely be concatinating the tokens found in the json structure.
# data['full_text'] = ''
# data['lemmas'] = []
# for section in data['sections']:
#     for
# subsection in section['subsections']:
# for sentence in subsection['sentences']:
#     for
# token in sentence['tokens']:
# if token['lemma'] not in string.punctuation:
#     data['full_text'] += token['lemma'] + ' '
# data['lemmas'].append(token['lemma'])
#
# to_write = paper_dir / (paper_id + '_lemma_text.txt')
#
# # now write the words out to a file
# with open(to_write, 'w', encoding='utf8') as outfile:
#     outfile.write(data['full_text'])
