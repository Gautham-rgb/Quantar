import { spawn } from "child_process";
import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;
let output: vscode.LogOutputChannel;

function config<T>(key: string, fallback: T): T {
  return vscode.workspace.getConfiguration("quantar").get<T>(key, fallback);
}

function workspaceRoot(): string | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  output = vscode.window.createOutputChannel("Quantar", { log: true });
  context.subscriptions.push(output);

  const pythonPath = config("pythonPath", "python");
  const serverModule = config("serverModule", "lang_systems.lsp");
  const operatorModules = config<string[]>("operatorModules", []);
  const root = workspaceRoot();

  const serverOptions: ServerOptions = {
    command: pythonPath,
    args: ["-m", serverModule],
    options: {
      cwd: root,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        QUANTAR_OPERATOR_MODULES: operatorModules.join(","),
      },
    },
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [
      { scheme: "file", language: "quantar" },
    ],
    outputChannel: output,
  };

  client = new LanguageClient(
    "quantar",
    "Quantar Language Server",
    serverOptions,
    clientOptions,
  );

  context.subscriptions.push(client);
  context.subscriptions.push(
    vscode.commands.registerCommand("quantar.runFile", () => runFile(false)),
    vscode.commands.registerCommand("quantar.compileFile", () => runFile(true)),
  );

  try {
    await client.start();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    output.appendLine(`Failed to start language server: ${message}`);
    void vscode.window.showErrorMessage(
      `Quantar language server failed to start: ${message}`,
    );
  }
}

async function runFile(compile: boolean): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.languageId !== "quantar") {
    void vscode.window.showWarningMessage("Open a Quantar (.qtr) file first.");
    return;
  }

  await editor.document.save();
  const filename = editor.document.uri.fsPath;
  const pythonPath = config("pythonPath", "python");
  const args = ["-m", "lang_systems.runner", filename];

  let outputPath: string | undefined;
  if (compile) {
    outputPath = filename.replace(/\.qtr$/i, ".py");
    args.push("--compile", outputPath);
  }

  output.show(true);
  output.appendLine(`> ${pythonPath} ${args.join(" ")}`);

  const child = spawn(pythonPath, args, {
    cwd: workspaceRoot(),
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  child.stdout?.on("data", (data) => output.append(data.toString()));
  child.stderr?.on("data", (data) => output.append(data.toString()));
  child.on("error", (error) => {
    output.appendLine(`Failed to start ${pythonPath}: ${error.message}`);
  });
  child.on("close", (code) => {
    output.appendLine(`[exit ${code ?? "unknown"}]`);

    if (compile && code === 0 && outputPath) {
      void vscode.workspace
        .openTextDocument(vscode.Uri.file(outputPath))
        .then((document) =>
          vscode.window.showTextDocument(document, { preview: true }),
        );
    }
  });
}

export async function deactivate(): Promise<void> {
  await client?.stop();
}
