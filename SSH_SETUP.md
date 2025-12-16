# VS Code Remote SSH Setup for RunPod

This guide explains how to connect your local VS Code directly to your RunPod GPU instance. This allows you to edit files and run code on the powerful GPU server as if it were your local machine.

## 1. Prerequisites

1.  **OpenSSH Client**: This is usually installed by default on Windows 10/11. You can check by running `ssh` in your PowerShell terminal. If it says "command not found", you need to enable it in Windows "Optional Features".
2.  **VS Code Extension**: You need the **Remote - SSH** extension by Microsoft.
    *   Open VS Code Extensions (Ctrl+Shift+X).
    *   Search for `ms-vscode-remote.remote-ssh`.
    *   Click **Install**.

## 2. Get Connection Details from RunPod

1.  Go to your RunPod Dashboard -> **My Pods**.
2.  Click the **Connect** button on your active Pod.
3.  Look for the **SSH over exposed port** command. It looks like this:
    ```bash
    ssh root@123.45.67.89 -p 12345 -i ~/.ssh/id_ed25519
    ```
    *Note: RunPod often provides a command with a key file (`-i`). If you haven't set up SSH keys, you might just use the password method. If using password, the command is just `ssh root@IP -p PORT`.*

## 3. Connect via VS Code

1.  **Open Remote Menu**: Click the green icon **><** in the bottom-left corner of VS Code (or press `F1` and type `Remote-SSH: Connect to Host...`).
2.  **Select "Connect to Host..."**.
3.  **Enter Connection String**: Paste the SSH command you got from RunPod.
    *   *Example:* `root@194.23.1.2 -p 30456` (Remove the `ssh` part if pasting into the box, or paste the whole command if prompted).
4.  **Select OS**: Choose **Linux** when asked for the platform of the remote host.
5.  **Authenticate**:
    *   If you set up an SSH key, it should connect automatically.
    *   If not, it will ask for a **password**. You can find the password in the RunPod "Connect" menu (under "Credentials" or environment variables).

## 4. Setup the Remote Environment

Once connected, a new VS Code window will open. This window is running **on the server**.

1.  **Open Terminal**: Press `Ctrl + ` ` (backtick). This is now the RunPod terminal.
2.  **Clone Your Repo**:
    ```bash
    git clone https://github.com/Jabir281/3D_CNN.git
    cd 3D_CNN
    ```
3.  **Install Python Extension**: You need to install the Python extension **on the remote server**.
    *   Go to Extensions (Ctrl+Shift+X).
    *   You will see a section "SSH: ...". Install the Python extension there.
4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Download Data**: Follow the steps in `RUNPOD_GUIDE.md` (Step 4) to download the dataset using `wget`.

## 5. Start Coding

Now you can open files in the file explorer on the left. When you save, it saves to the RunPod server. When you run code, it runs on the A100 GPU.
