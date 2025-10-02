#!/usr/bin/env python3
"""
Google Colab GitHub Actions Self-Hosted Runner Setup
Fixes the sudo/root user issue in Colab environment
"""

import subprocess
import os
import sys

def run_command(cmd, check=True):
    """Run shell command and return result"""
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    print(f"✅ Success: {result.stdout}")
    return True

def setup_non_root_user():
    """Create non-root user for GitHub Actions runner"""
    print("👤 Setting up non-root user for GitHub Actions...")
    
    # Create a non-root user
    run_command("useradd -m -s /bin/bash runner", check=False)
    run_command("usermod -aG sudo runner", check=False)
    
    # Set password for runner user
    run_command("echo 'runner:runner123' | chpasswd", check=False)
    
    # Give runner sudo without password for setup
    run_command("echo 'runner ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers", check=False)
    
    print("✅ Non-root user 'runner' created successfully!")

def setup_github_runner():
    """Download and setup GitHub Actions runner"""
    print("📥 Downloading GitHub Actions runner...")
    
    # Create runner directory
    run_command("mkdir -p /home/runner/actions-runner")
    run_command("chown -R runner:runner /home/runner/actions-runner")
    
    # Download runner as runner user
    commands = [
        "cd /home/runner/actions-runner",
        "curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz",
        "tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz",
        "rm actions-runner-linux-x64-2.311.0.tar.gz"
    ]
    
    for cmd in commands:
        run_command(f"su - runner -c '{cmd}'")
    
    print("✅ GitHub Actions runner downloaded successfully!")

def configure_runner(token, repo_url):
    """Configure the GitHub Actions runner"""
    print("⚙️ Configuring GitHub Actions runner...")
    
    config_cmd = f"""
    cd /home/runner/actions-runner && 
    ./config.sh --url {repo_url} --token {token} --name colab-gpu-runner --labels gpu,colab,t4 --unattended
    """
    
    result = run_command(f"su - runner -c '{config_cmd}'")
    if result:
        print("✅ Runner configured successfully!")
        return True
    else:
        print("❌ Runner configuration failed!")
        return False

def start_runner():
    """Start the GitHub Actions runner"""
    print("🚀 Starting GitHub Actions runner...")
    
    start_cmd = """
    cd /home/runner/actions-runner && 
    nohup ./run.sh > runner.log 2>&1 &
    """
    
    run_command(f"su - runner -c '{start_cmd}'")
    print("✅ Runner started in background!")
    print("📋 Check status with: su - runner -c 'tail -f /home/runner/actions-runner/runner.log'")

def main():
    """Main setup function"""
    print("🧠 LamAI Colab GitHub Runner Setup")
    print("=" * 50)
    
    # Get configuration from user
    repo_url = "https://github.com/NaveenSingh9999/LamAI-ChatBot-Model-ML-AI-Trainable"
    
    print(f"📍 Repository: {repo_url}")
    print("\n🔑 You need to get your registration token from:")
    print("   GitHub Repo → Settings → Actions → Runners → New self-hosted runner")
    
    token = input("\n🎟️ Enter your registration token: ").strip()
    
    if not token:
        print("❌ Registration token is required!")
        sys.exit(1)
    
    # Setup process
    setup_non_root_user()
    setup_github_runner() 
    
    if configure_runner(token, repo_url):
        start_runner()
        print("\n🎉 Setup completed successfully!")
        print("🔥 Your Colab is now a GitHub Actions runner!")
        print("💡 Go to GitHub Actions tab and run your workflow!")
    else:
        print("\n❌ Setup failed!")

if __name__ == "__main__":
    main()