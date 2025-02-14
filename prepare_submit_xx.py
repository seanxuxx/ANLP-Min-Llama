import os
import sys
import zipfile

required_files = {'run_llama.py',
                  'llama.py',
                  'optimizer.py',
                  'classifier.py',
                  'rope.py',
                  'generated-sentence-temp-0.txt',
                  'generated-sentence-temp-1.txt',
                  'sst-dev-prompting-output.txt',
                  'sst-test-prompting-output.txt',
                  'sst-dev-finetuning-output.txt',
                  'sst-test-finetuning-output.txt',
                  'cfimdb-dev-prompting-output.txt',
                  'cfimdb-test-prompting-output.txt',
                  'cfimdb-dev-finetuning-output.txt',
                  'cfimdb-test-finetuning-output.txt'}

optional_files = {'sst-dev-advanced-output.txt',
                  'sst-test-advanced-output.txt',
                  'cfimdb-dev-advanced-output.txt',
                  'cfimdb-test-advanced-output.txt',
                  'feedback.txt'}


my_files = [
    'run_llama.py',
    'base_llama.py',
    'llama.py',
    'rope.py',
    'classifier.py',
    'config.py',
    'optimizer.py',
    'sanity_check.py',
    'tokenizer.py',
    'utils.py',
    'README.md',
    'structure.md',
    'sanity_check.data',
    'generated-sentence-temp-0.txt',
    'generated-sentence-temp-1.txt',
    'sst-dev-prompting-output.txt',
    'sst-test-prompting-output.txt',
    'sst-dev-finetuning-output.txt',
    'sst-test-finetuning-output.txt',
    'cfimdb-dev-prompting-output.txt',
    'cfimdb-test-prompting-output.txt',
    'cfimdb-dev-finetuning-output.txt',
    'cfimdb-test-finetuning-output.txt',
    'setup.sh',
]


def main(path: str, aid: str):
    aid = aid.strip()
    if os.path.isdir(path):
        with zipfile.ZipFile(f"{aid}.zip", 'w') as zz:
            for root, dirs, files in os.walk(path):
                if '.git' in root or '__pycache__' in root:
                    continue  # ignore some parts
                for file in files:
                    if file not in my_files:
                        continue
                    ff = os.path.join(root, file)
                    rpath = os.path.relpath(ff, path)
                    zz.write(ff, os.path.join(".", aid, rpath))
                    if rpath in required_files:
                        required_files.remove(rpath)
        assert len(required_files) == 0, breakpoint()


if __name__ == '__main__':
    main(path=os.getcwd(), aid='xiaoxu')
