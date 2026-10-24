const path = require("path");
const fs = require("fs");
const vscode = require("vscode");
const { LanguageClient } = require("vscode-languageclient/node");

let client;

/** @param {vscode.ExtensionContext} context */
async function activate(context) {
    const projectRoot = path.join(context.extensionPath, "..");
    const sourceRoot = path.join(projectRoot, "src");
    const isSourceCheckout = fs.existsSync(path.join(sourceRoot, "lang_systems", "lsp"));
    const pythonExtension = vscode.extensions.getExtension("ms-python.python");
    let pythonCommand = "python";
    let pythonArgs = [];

    if (pythonExtension) {
        const pythonApi = await pythonExtension.activate();
        const workspaceUri = vscode.workspace.workspaceFolders?.[0]?.uri;
        const execution = await pythonApi.settings?.getExecutionDetails?.(workspaceUri);
        if (Array.isArray(execution?.execCommand)) {
            if (execution.execCommand.length > 0) {
                [pythonCommand, ...pythonArgs] = execution.execCommand;
            }
        } else if (execution?.execCommand) {
            pythonCommand = execution.execCommand;
            pythonArgs = execution.execArgs || [];
        }
    }

    const serverOptions = {
        command: pythonCommand,
        args: [...pythonArgs, "-u", "-m", "lang_systems.lsp"],
        options: {
            cwd: isSourceCheckout
                ? projectRoot
                : vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || context.extensionPath,
            env: {
                ...process.env,
                PYTHONPATH: [
                    ...(isSourceCheckout ? [sourceRoot] : []),
                    process.env.PYTHONPATH,
                ]
                    .filter(Boolean)
                    .join(path.delimiter),
            },
        },
    };

    const outputChannel = vscode.window.createOutputChannel("Quantar");
    client = new LanguageClient(
        "quantarLanguageServer",
        "Quantar Language Server",
        serverOptions,
        {
            documentSelector: [{ scheme: "file", language: "quantar" }],
            outputChannel,
        },
    );
    context.subscriptions.push(client, outputChannel);
    client.start().catch((error) => {
        outputChannel.appendLine(`Failed to start the language server: ${error.message}`);
        outputChannel.show(true);
        const installCommand = isSourceCheckout
            ? 'python -m pip install -e ".[lsp]" (from the project root)'
            : 'python -m pip install "quantar[lsp]"';
        void vscode.window.showErrorMessage(
            `Quantar could not start its language server. Install Quantar with LSP support in the selected Python environment: ${installCommand}. Then reload VS Code.`,
        );
    });
}

function deactivate() {
    return client?.stop();
}

module.exports = { activate, deactivate };
