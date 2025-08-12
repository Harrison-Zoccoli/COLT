COLT - ignore ts pls



# The following s:
Just hit these jwns:
uv python pin 3.12
uv venv --python 3.12
source .venv/bin/activate


# .env add



Then make a webhook that goes to 8765 and put that jawn in your voice api for telnyx


# Extra Yap
Get started with PipeCat
More or less read and follow:
https://docs.pipecat.ai/getting-started
https://github.com/pipecat-ai/pipecat
Much of this is included above (with improvements like using uv).
But we want to do inbound calling with Telnyx, so follow below.

Telnyx
To use a Telnyx phone number, follow:
https://github.com/pipecat-ai/pipecat/tree/main/examples/telnyx-chatbot
Specifically, the section "Configure Telnyx TeXML application".
Put the streams.xml file into a subdir 'templates'.

Basic testing, style checking and linting
For basic syntax check, use pyflakes.
For style checking, use pycodestyle and/or flake8.
For linting, use pylint and ruff.
For static type checking, use mypy.
For a basic package-level check, do a uv build.

Thus, for main.py:

pyflakes main.py    # expect no output (no errors)
flake8 main.py      # expect no output (no suggestions)
pylint main.py      # expect score of 10.0 / 10.0
mypy main.py        # expect 'Success: no issues found...'
ruff check          # expect 'All checks passed!'
The final ruff check actually checks all included .py files.

Also do a uv build as a basic integration test:

uv build            # expect 'Successfully built dist/...'
