import os
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

if __name__ == '__main__':
    aid = 'xiaoxu'
    with zipfile.ZipFile(f"{aid}.zip", 'w') as zz:
        for file in my_files:
            if file.endswith(".zip"):
                continue
            ff = os.path.join(os.getcwd(), file)
            rpath = os.path.relpath(ff, os.getcwd())
            zz.write(ff, os.path.join(".", aid, rpath))
            if rpath in required_files:
                required_files.remove(rpath)
    assert len(required_files) == 0, breakpoint()
