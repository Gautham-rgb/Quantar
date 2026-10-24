# Quantar Language Support

VS Code support for Quantar's `.qtr` and `.quantar` files.

## Features

- Python syntax highlighting with Quantar's symbolic operators highlighted in
  a consistent cyan accent.
- Parser diagnostics from Quantar's language server.
- Semantic highlighting for operators registered by the project or user code.

## Setup

Install the extension, then install Quantar with its LSP support in the Python
interpreter selected by the VS Code Python extension:

```powershell
python -m pip install "quantar[lsp]"
```

For development from a Quantar checkout, run this at the project root instead:

```powershell
python -m pip install -e ".[lsp]"
cd extension
npm install
```

Open the project root in VS Code and press F5 to launch the extension host.

## Custom operators

Set `QUANTAR_OPERATOR_MODULES` to a comma-separated list of Python modules
that register operators. The language server imports these modules at startup.
