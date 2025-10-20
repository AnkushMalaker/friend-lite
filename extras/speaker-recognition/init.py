#!/usr/bin/env python3
"""
Friend-Lite Speaker Recognition Setup Script
Interactive configuration for speaker recognition service
"""

import argparse
import getpass
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import set_key
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text


class SpeakerRecognitionSetup:
    def __init__(self, args=None):
        self.console = Console()
        self.config: Dict[str, Any] = {}
        self.args = args or argparse.Namespace()

    def print_header(self, title: str):
        """Print a colorful header"""
        self.console.print()
        panel = Panel(
            Text(title, style="cyan bold"),
            style="cyan",
            expand=False
        )
        self.console.print(panel)
        self.console.print()

    def print_section(self, title: str):
        """Print a section header"""
        self.console.print()
        self.console.print(f"[magenta]► {title}[/magenta]")
        self.console.print("[magenta]" + "─" * len(f"► {title}") + "[/magenta]")

    def prompt_value(self, prompt: str, default: str = "") -> str:
        """Prompt for a value with optional default"""
        try:
            return Prompt.ask(prompt, default=default)
        except EOFError:
            self.console.print(f"Using default: {default}")
            return default

    def prompt_password(self, prompt: str) -> str:
        """Prompt for password (hidden input)"""
        while True:
            try:
                password = getpass.getpass(f"{prompt}: ")
                if password:  # Just need non-empty
                    return password
                self.console.print("[yellow][WARNING][/yellow] Token is required")
            except (EOFError, KeyboardInterrupt):
                self.console.print("[red][ERROR][/red] Token is required for speaker recognition")
                sys.exit(1)

    def prompt_choice(self, prompt: str, choices: Dict[str, str], default: str = "1") -> str:
        """Prompt for a choice from options"""
        self.console.print(prompt)
        for key, desc in choices.items():
            self.console.print(f"  {key}) {desc}")
        self.console.print()

        while True:
            try:
                choice = Prompt.ask("Enter choice", default=default)
                if choice in choices:
                    return choice
                self.console.print(f"[red]Invalid choice. Please select from {list(choices.keys())}[/red]")
            except EOFError:
                self.console.print(f"Using default choice: {default}")
                return default

    def backup_existing_env(self):
        """Backup existing .env file"""
        env_path = Path(".env")
        if env_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f".env.backup.{timestamp}"
            shutil.copy2(env_path, backup_path)
            self.console.print(f"[blue][INFO][/blue] Backed up existing .env file to {backup_path}")

    def setup_hf_token(self):
        """Configure Hugging Face token"""
        self.print_section("Hugging Face Token Setup")
        self.console.print("Required for pyannote speaker recognition models")
        self.console.print("Get yours from: https://huggingface.co/settings/tokens")
        self.console.print()

        # Check if provided via command line
        if hasattr(self.args, 'hf_token') and self.args.hf_token:
            self.config["HF_TOKEN"] = self.args.hf_token
            self.console.print("[green][SUCCESS][/green] HF Token configured from command line")
        else:
            hf_token = self.prompt_password("HF Token")
            self.config["HF_TOKEN"] = hf_token
            self.console.print("[green][SUCCESS][/green] HF Token configured")

    def setup_compute_mode(self):
        """Configure compute mode (CPU/GPU)"""
        self.print_section("Compute Mode Configuration")

        # Check if provided via command line
        if hasattr(self.args, 'compute_mode') and self.args.compute_mode:
            compute_mode = self.args.compute_mode
            self.console.print(f"[green][SUCCESS][/green] Compute mode configured from command line: {compute_mode}")
        else:
            choices = {
                "1": "CPU-only (works everywhere)",
                "2": "GPU acceleration (requires NVIDIA+CUDA)"
            }
            choice = self.prompt_choice("Choose compute mode:", choices, "1")
            compute_mode = "gpu" if choice == "2" else "cpu"

        self.config["COMPUTE_MODE"] = compute_mode

        # Set CUDA_VERSION for Docker build
        if compute_mode == "cpu":
            self.config["CUDA_VERSION"] = "cpu"
        else:
            # Default to cu121 for GPU mode (can be overridden in .env)
            self.config["CUDA_VERSION"] = "cu121"

        self.console.print(f"[blue][INFO][/blue] Using {compute_mode.upper()} mode")

    def setup_deepgram(self):
        """Configure Deepgram API key if provided"""
        # Only set if provided via command line
        if hasattr(self.args, 'deepgram_api_key') and self.args.deepgram_api_key:
            self.config["DEEPGRAM_API_KEY"] = self.args.deepgram_api_key
            self.console.print("[green][SUCCESS][/green] Deepgram API key configured from command line")

    def setup_https(self):
        """Configure HTTPS settings"""
        # Check if HTTPS configuration provided via command line
        if hasattr(self.args, 'enable_https') and self.args.enable_https:
            enable_https = True
            server_ip = getattr(self.args, 'server_ip', 'localhost')
            self.console.print(f"[green][SUCCESS][/green] HTTPS configured via command line: {server_ip}")
        else:
            # Interactive configuration
            self.print_section("HTTPS Configuration (Optional)")
            self.console.print("HTTPS is required for microphone access in browsers")
            self.console.print()

            choices = {
                "1": "HTTP mode (development, localhost only)",
                "2": "HTTPS mode with SSL (production, remote access, microphone access)"
            }
            choice = self.prompt_choice("Choose connection mode:", choices, "1")
            enable_https = (choice == "2")

            if enable_https:
                self.console.print()
                self.console.print("[blue][INFO][/blue] For distributed deployments, use your Tailscale IP")
                self.console.print("[blue][INFO][/blue] For local-only access, use 'localhost'")
                self.console.print("Examples: localhost, 100.64.1.2, your-domain.com")
                server_ip = self.prompt_value("Server IP/Domain for SSL certificate", "localhost")

        if enable_https:
            self.config["REACT_UI_HTTPS"] = "true"
            self.config["REACT_UI_PORT"] = "5175"

            # Generate SSL certificates
            self.console.print("[blue][INFO][/blue] Generating SSL certificates...")
            ssl_script = Path("ssl/generate-ssl.sh")
            if ssl_script.exists():
                try:
                    subprocess.run([str(ssl_script), server_ip], check=True, timeout=180)
                    self.console.print("[green][SUCCESS][/green] SSL certificates generated")
                except subprocess.TimeoutExpired:
                    self.console.print("[yellow][WARNING][/yellow] SSL certificate generation timed out")
                except subprocess.CalledProcessError:
                    self.console.print("[yellow][WARNING][/yellow] SSL certificate generation failed")
            else:
                self.console.print(f"[yellow][WARNING][/yellow] SSL script not found at {ssl_script}")

            # Generate nginx.conf from template
            self.console.print("[blue][INFO][/blue] Creating nginx configuration...")
            nginx_template = Path("nginx.conf.template")
            if nginx_template.exists():
                try:
                    with open(nginx_template, 'r') as f:
                        nginx_content = f.read()

                    # Replace TAILSCALE_IP with server_ip
                    nginx_content = nginx_content.replace('TAILSCALE_IP', server_ip)

                    with open('nginx.conf', 'w') as f:
                        f.write(nginx_content)

                    self.console.print(f"[green][SUCCESS][/green] nginx.conf created for: {server_ip}")
                except Exception as e:
                    self.console.print(f"[yellow][WARNING][/yellow] nginx.conf generation failed: {e}")
            else:
                self.console.print("[yellow][WARNING][/yellow] nginx.conf.template not found")

            self.console.print()
            self.console.print("📋 [bold]HTTPS Mode URLs:[/bold]")
            self.console.print(f"   🌐 HTTPS Access: https://localhost:8444/")
            self.console.print(f"   🌐 HTTP Redirect: http://localhost:8081/ → HTTPS")
            self.console.print(f"   📱 Service API: https://localhost:8444/api/")
            self.console.print(f"   💡 Accept SSL certificate in browser")
        else:
            self.config["REACT_UI_HTTPS"] = "false"
            self.config["REACT_UI_PORT"] = "5174"

            self.console.print()
            self.console.print("📋 [bold]HTTP Mode URLs:[/bold]")
            self.console.print("   📱 Service API: http://localhost:8085")
            self.console.print("   📱 Web Interface: http://localhost:5174")
            self.console.print("   ⚠️  Note: Microphone access may not work over HTTP on remote connections")

    def generate_env_file(self):
        """Generate .env file from template and update with configuration"""
        env_path = Path(".env")
        env_template = Path(".env.template")

        # Backup existing .env if it exists
        if env_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = env_path.with_suffix(f'.backup.{timestamp}')
            shutil.copy2(env_path, backup_path)
            self.console.print(f"[blue][INFO][/blue] Backed up existing .env to {backup_path.name}")

        # Copy template to .env
        if env_template.exists():
            shutil.copy2(env_template, env_path)
            self.console.print("[blue][INFO][/blue] Copied .env.template to .env")
        else:
            self.console.print("[yellow][WARNING][/yellow] .env.template not found, creating new .env")
            env_path.touch(mode=0o600)

        # Update configured values using set_key
        env_path_str = str(env_path)
        for key, value in self.config.items():
            if value:  # Only set non-empty values
                set_key(env_path_str, key, value)

        # Ensure secure permissions
        os.chmod(env_path, 0o600)

        self.console.print("[green][SUCCESS][/green] .env file configured successfully with secure permissions")

    def show_summary(self):
        """Show configuration summary"""
        self.print_section("Configuration Summary")
        self.console.print()

        self.console.print(f"✅ HF Token: {'Configured' if self.config.get('HF_TOKEN') else 'Not configured'}")
        self.console.print(f"✅ Compute Mode: {self.config.get('COMPUTE_MODE', 'Not configured')}")
        self.console.print(f"✅ HTTPS Enabled: {self.config.get('REACT_UI_HTTPS', 'false')}")
        if self.config.get('DEEPGRAM_API_KEY'):
            self.console.print(f"✅ Deepgram API Key: Configured")

    def show_model_agreement_links(self):
        """Show required Hugging Face model agreement links"""
        self.print_section("⚠️  Required: Accept Model Agreements")
        self.console.print()

        self.console.print("Before using speaker recognition, you must accept agreements for these gated models:")
        self.console.print()

        model_links = [
            ("Speaker Diarization", "https://huggingface.co/pyannote/speaker-diarization-community-1"),
            ("Segmentation Model", "https://huggingface.co/pyannote/segmentation-3.0"),
            ("Segmentation Model2", "https://huggingface.co/pyannote/segmentation-3.1"),
            ("Embedding Model", "https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM")
        ]

        for idx, (name, url) in enumerate(model_links, 1):
            self.console.print(f"   {idx}. [cyan]{name}[/cyan]")
            self.console.print(f"      {url}")
            self.console.print()

        self.console.print("[yellow]→[/yellow] Open each link above and click 'Agree and access repository'")
        self.console.print("[yellow]→[/yellow] You must be logged into Hugging Face with the same account used for HF_TOKEN")
        self.console.print()

    def show_next_steps(self):
        """Show next steps"""
        self.print_section("Next Steps")
        self.console.print()

        self.console.print("1. Start the speaker recognition service:")
        if self.config.get('REACT_UI_HTTPS') == 'true':
            self.console.print("   [cyan]docker compose up --build -d[/cyan]")
        else:
            self.console.print("   [cyan]docker compose up --build -d speaker-service web-ui[/cyan]")

    def run(self):
        """Run the complete setup process"""
        self.print_header("🗣️ Speaker Recognition Setup")
        self.console.print("Configure speaker identification and enrollment service")
        self.console.print()

        try:
            # Run setup steps
            self.setup_hf_token()
            self.setup_compute_mode()
            self.setup_deepgram()
            self.setup_https()

            # Generate files
            self.print_header("Configuration Complete!")
            self.generate_env_file()

            # Show results
            self.show_summary()
            self.show_model_agreement_links()
            self.show_next_steps()

            self.console.print()
            self.console.print("[green][SUCCESS][/green] Speaker Recognition setup complete! 🎉")

        except KeyboardInterrupt:
            self.console.print()
            self.console.print("[yellow]Setup cancelled by user[/yellow]")
            sys.exit(0)
        except Exception as e:
            self.console.print(f"[red][ERROR][/red] Setup failed: {e}")
            sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Speaker Recognition Service Setup")
    parser.add_argument("--hf-token",
                       help="Hugging Face token (default: prompt user)")
    parser.add_argument("--compute-mode",
                       choices=["cpu", "gpu"],
                       help="Compute mode: cpu or gpu (default: prompt user)")
    parser.add_argument("--deepgram-api-key",
                       help="Deepgram API key (optional)")
    parser.add_argument("--enable-https", action="store_true",
                       help="Enable HTTPS configuration (default: prompt user)")
    parser.add_argument("--server-ip",
                       help="Server IP/domain for SSL certificate (default: prompt user)")

    args = parser.parse_args()

    setup = SpeakerRecognitionSetup(args)
    setup.run()


if __name__ == "__main__":
    main()
