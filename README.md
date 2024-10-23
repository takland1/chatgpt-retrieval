# chatgpt-retrieval

Simple script to use ChatGPT on your own files.

## Installation

Install [Langchain](https://github.com/hwchase17/langchain) and other required packages.
```
pip install -r requirements.txt
```
Besides, it needs other programme like `libmagic`, try
```commandline
brew install libmagic
```
if it is run on a mac.

Modify `constants.py.default` to use your own [OpenAI API key](https://platform.openai.com/account/api-keys), and rename it to `constants.py`.

Place your own data into `Docs/data.txt`.

## Example usage
Test reading `Docs/data.txt` file.
```
> python chatgpt.py "what is my dog's name"
Your dog's name is Sunny.
```

Test reading `Docs/cat.pdf` file.
```
> python chatgpt.py "what is my cat's name"
Your cat's name is Muffy.
```

To Run the html frontend
```
python server.py
```