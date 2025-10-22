#!/usr/bin/env python3
"""
Friend-Lite Advanced Backend Interactive Setup Script
Interactive configuration for all services and API keys
"""

import argparse
import getpass
import os
import secrets
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import get_key, set_key
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text


class FriendLiteSetup:
    def __init__(self, args=None):
        self.console = Console()
        self.config: Dict[str, Any] = {}
        self.args = args or argparse.Namespace()
        
        # Check if we're in the right directory
        if not Path("pyproject.toml").exists() or not Path("src").exists():
            self.console.print("[red][ERROR][/red] Please run this script from the backends/advanced directory")
            sys.exit(1)

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
            # Always provide a default to avoid EOF issues
            return Prompt.ask(prompt, default=default)
        except EOFError:
            self.console.print(f"Using default: {default}")
            return default

    def prompt_password(self, prompt: str) -> str:
        """Prompt for password (hidden input)"""
        while True:
            try:
                password = getpass.getpass(f"{prompt}: ")
                if len(password) >= 8:
                    return password
                self.console.print("[yellow][WARNING][/yellow] Password must be at least 8 characters")
            except (EOFError, KeyboardInterrupt):
                # For non-interactive environments, generate a secure password
                self.console.print("[yellow][WARNING][/yellow] Non-interactive environment detected")
                password = f"admin-{secrets.token_hex(8)}"
                self.console.print(f"Generated secure password: {password}")
                return password

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

    def read_existing_env_value(self, key: str) -> str:
        """Read a value from existing .env file"""
        env_path = Path(".env")
        if not env_path.exists():
            return None

        value = get_key(str(env_path), key)
        # get_key returns None if key doesn't exist or value is empty
        return value if value else None

    def mask_api_key(self, key: str, show_chars: int = 5) -> str:
        """Mask API key showing only first and last few characters"""
        if not key or len(key) <= show_chars * 2:
            return key

        # Remove quotes if present
        key_clean = key.strip("'\"")

        return f"{key_clean[:show_chars]}{'*' * min(15, len(key_clean) - show_chars * 2)}{key_clean[-show_chars:]}"

    def setup_authentication(self):
        """Configure authentication settings"""
        self.print_section("Authentication Setup")
        self.console.print("Configure admin account for the dashboard")
        self.console.print()

        self.config["ADMIN_EMAIL"] = self.prompt_value("Admin email", "admin@example.com")
        self.config["ADMIN_PASSWORD"] = self.prompt_password("Admin password (min 8 chars)")
        self.config["AUTH_SECRET_KEY"] = secrets.token_hex(32)

        self.console.print("[green][SUCCESS][/green] Admin account configured")

    def setup_transcription(self):
        """Configure transcription provider"""
        self.print_section("Speech-to-Text Configuration")
        
        choices = {
            "1": "Deepgram (recommended - high quality, requires API key)",
            "2": "Mistral (Voxtral models - requires API key)", 
            "3": "Offline (Parakeet ASR - requires GPU, runs locally)",
            "4": "None (skip transcription setup)"
        }
        
        choice = self.prompt_choice("Choose your transcription provider:", choices, "1")

        if choice == "1":
            self.console.print("[blue][INFO][/blue] Deepgram selected")
            self.console.print("Get your API key from: https://console.deepgram.com/")

            # Check for existing API key
            existing_key = self.read_existing_env_value("DEEPGRAM_API_KEY")
            if existing_key and existing_key not in ['your_deepgram_api_key_here', 'your-deepgram-key-here']:
                masked_key = self.mask_api_key(existing_key)
                prompt_text = f"Deepgram API key ({masked_key}) [press Enter to reuse, or enter new]"
                api_key_input = self.prompt_value(prompt_text, "")
                api_key = api_key_input if api_key_input else existing_key
            else:
                api_key = self.prompt_value("Deepgram API key (leave empty to skip)", "")

            if api_key:
                self.config["TRANSCRIPTION_PROVIDER"] = "deepgram"
                self.config["DEEPGRAM_API_KEY"] = api_key
                self.console.print("[green][SUCCESS][/green] Deepgram configured")
            else:
                self.console.print("[yellow][WARNING][/yellow] No API key provided - transcription will not work")

        elif choice == "2":
            self.config["TRANSCRIPTION_PROVIDER"] = "mistral"
            self.console.print("[blue][INFO][/blue] Mistral selected")
            self.console.print("Get your API key from: https://console.mistral.ai/")

            # Check for existing API key
            existing_key = self.read_existing_env_value("MISTRAL_API_KEY")
            if existing_key and existing_key not in ['your_mistral_api_key_here', 'your-mistral-key-here']:
                masked_key = self.mask_api_key(existing_key)
                prompt_text = f"Mistral API key ({masked_key}) [press Enter to reuse, or enter new]"
                api_key_input = self.prompt_value(prompt_text, "")
                api_key = api_key_input if api_key_input else existing_key
            else:
                api_key = self.prompt_value("Mistral API key (leave empty to skip)", "")

            model = self.prompt_value("Mistral model", "voxtral-mini-2507")

            if api_key:
                self.config["MISTRAL_API_KEY"] = api_key
                self.config["MISTRAL_MODEL"] = model
                self.console.print("[green][SUCCESS][/green] Mistral configured")
            else:
                self.console.print("[yellow][WARNING][/yellow] No API key provided - transcription will not work")

        elif choice == "3":
            self.config["TRANSCRIPTION_PROVIDER"] = "offline"
            self.console.print("[blue][INFO][/blue] Offline Parakeet ASR selected")
            parakeet_url = self.prompt_value("Parakeet ASR URL", "http://host.docker.internal:8767")
            self.config["PARAKEET_ASR_URL"] = parakeet_url
            self.console.print("[yellow][WARNING][/yellow] Remember to start Parakeet service: cd ../../extras/asr-services && docker compose up parakeet")

        elif choice == "4":
            self.console.print("[blue][INFO][/blue] Skipping transcription setup")

    def setup_llm(self):
        """Configure LLM provider"""
        self.print_section("LLM Provider Configuration")
        
        choices = {
            "1": "OpenAI (GPT-4, GPT-3.5 - requires API key)",
            "2": "Ollama (local models - requires Ollama server)",
            "3": "Skip (no memory extraction)"
        }
        
        choice = self.prompt_choice("Choose your LLM provider for memory extraction:", choices, "1")

        if choice == "1":
            self.config["LLM_PROVIDER"] = "openai"
            self.console.print("[blue][INFO][/blue] OpenAI selected")
            self.console.print("Get your API key from: https://platform.openai.com/api-keys")

            # Check for existing API key
            existing_key = self.read_existing_env_value("OPENAI_API_KEY")
            if existing_key and existing_key not in ['your_openai_api_key_here', 'your-openai-key-here']:
                masked_key = self.mask_api_key(existing_key)
                prompt_text = f"OpenAI API key ({masked_key}) [press Enter to reuse, or enter new]"
                api_key_input = self.prompt_value(prompt_text, "")
                api_key = api_key_input if api_key_input else existing_key
            else:
                api_key = self.prompt_value("OpenAI API key (leave empty to skip)", "")

            model = self.prompt_value("OpenAI model", "gpt-4o-mini")
            base_url = self.prompt_value("OpenAI base URL (for proxies/compatible APIs)", "https://api.openai.com/v1")

            if api_key:
                self.config["OPENAI_API_KEY"] = api_key
                self.config["OPENAI_MODEL"] = model
                self.config["OPENAI_BASE_URL"] = base_url
                self.console.print("[green][SUCCESS][/green] OpenAI configured")
            else:
                self.console.print("[yellow][WARNING][/yellow] No API key provided - memory extraction will not work")

        elif choice == "2":
            self.config["LLM_PROVIDER"] = "ollama"
            self.console.print("[blue][INFO][/blue] Ollama selected")
            
            base_url = self.prompt_value("Ollama server URL", "http://host.docker.internal:11434")
            if not base_url.endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
                self.console.print(f"[blue][INFO][/blue] Automatically appending /v1 to Ollama URL: {base_url}")

            model = self.prompt_value("Ollama model", "llama3.2")
            
            embedder_model = self.prompt_value("Ollama embedder model", "nomic-embed-text:latest")
            
            self.config["OLLAMA_BASE_URL"] = base_url
            self.config["OLLAMA_MODEL"] = model
            self.config["OLLAMA_EMBEDDER_MODEL"] = embedder_model
            self.console.print("[green][SUCCESS][/green] Ollama configured")
            self.console.print("[yellow][WARNING][/yellow] Make sure Ollama is running and all required models (LLM and embedder) are pulled")

        elif choice == "3":
            self.console.print("[blue][INFO][/blue] Skipping LLM setup - memory extraction disabled")

    def setup_memory(self):
        """Configure memory provider"""
        self.print_section("Memory Storage Configuration")
        
        choices = {
            "1": "Friend-Lite Native (Qdrant + custom extraction)",
            "2": "OpenMemory MCP (cross-client compatible, external server)"
        }
        
        choice = self.prompt_choice("Choose your memory storage backend:", choices, "1")

        if choice == "1":
            self.config["MEMORY_PROVIDER"] = "friend_lite"
            self.console.print("[blue][INFO][/blue] Friend-Lite Native memory provider selected")
            
            qdrant_url = self.prompt_value("Qdrant URL", "qdrant")
            self.config["QDRANT_BASE_URL"] = qdrant_url
            self.console.print("[green][SUCCESS][/green] Friend-Lite memory provider configured")

        elif choice == "2":
            self.config["MEMORY_PROVIDER"] = "openmemory_mcp"
            self.console.print("[blue][INFO][/blue] OpenMemory MCP selected")
            
            mcp_url = self.prompt_value("OpenMemory MCP server URL", "http://host.docker.internal:8765")
            client_name = self.prompt_value("OpenMemory client name", "friend_lite")
            user_id = self.prompt_value("OpenMemory user ID", "openmemory")
            
            self.config["OPENMEMORY_MCP_URL"] = mcp_url
            self.config["OPENMEMORY_CLIENT_NAME"] = client_name
            self.config["OPENMEMORY_USER_ID"] = user_id
            self.console.print("[green][SUCCESS][/green] OpenMemory MCP configured")
            self.console.print("[yellow][WARNING][/yellow] Remember to start OpenMemory: cd ../../extras/openmemory-mcp && docker compose up -d")

    def setup_optional_services(self):
        """Configure optional services"""
        self.print_section("Optional Services")

        # Check if speaker service URL provided via args
        if hasattr(self.args, 'speaker_service_url') and self.args.speaker_service_url:
            self.config["SPEAKER_SERVICE_URL"] = self.args.speaker_service_url
            self.console.print(f"[green][SUCCESS][/green] Speaker Recognition configured via args: {self.args.speaker_service_url}")
        else:
            try:
                enable_speaker = Confirm.ask("Enable Speaker Recognition?", default=False)
            except EOFError:
                self.console.print("Using default: No")
                enable_speaker = False
                
            if enable_speaker:
                speaker_url = self.prompt_value("Speaker Recognition service URL", "http://host.docker.internal:8001")
                self.config["SPEAKER_SERVICE_URL"] = speaker_url
                self.console.print("[green][SUCCESS][/green] Speaker Recognition configured")
                self.console.print("[blue][INFO][/blue] Start with: cd ../../extras/speaker-recognition && docker compose up -d")
        
        # Check if ASR service URL provided via args  
        if hasattr(self.args, 'parakeet_asr_url') and self.args.parakeet_asr_url:
            self.config["PARAKEET_ASR_URL"] = self.args.parakeet_asr_url
            self.console.print(f"[green][SUCCESS][/green] Parakeet ASR configured via args: {self.args.parakeet_asr_url}")

    def setup_network(self):
        """Configure network settings"""
        self.print_section("Network Configuration")

        self.config["BACKEND_PUBLIC_PORT"] = self.prompt_value("Backend port", "8000")
        self.config["WEBUI_PORT"] = self.prompt_value("Web UI port", "5173")

    def setup_https(self):
        """Configure HTTPS settings for microphone access"""
        # Check if HTTPS configuration provided via command line
        if hasattr(self.args, 'enable_https') and self.args.enable_https:
            enable_https = True
            server_ip = getattr(self.args, 'server_ip', 'localhost')
            self.console.print(f"[green][SUCCESS][/green] HTTPS configured via command line: {server_ip}")
        else:
            # Interactive configuration
            self.print_section("HTTPS Configuration (Optional)")
            
            try:
                enable_https = Confirm.ask("Enable HTTPS for microphone access?", default=False)
            except EOFError:
                self.console.print("Using default: No")
                enable_https = False
            
            if enable_https:
                self.console.print("[blue][INFO][/blue] HTTPS enables microphone access in browsers")
                self.console.print("[blue][INFO][/blue] For distributed deployments, use your Tailscale IP (e.g., 100.64.1.2)")
                self.console.print("[blue][INFO][/blue] For local-only access, use 'localhost'")
                server_ip = self.prompt_value("Server IP/Domain for SSL certificate (Tailscale IP or localhost)", "localhost")
        
        if enable_https:
            
            # Generate SSL certificates
            self.console.print("[blue][INFO][/blue] Generating SSL certificates...")
            # Use path relative to this script's directory
            script_dir = Path(__file__).parent
            ssl_script = script_dir / "ssl" / "generate-ssl.sh"
            if ssl_script.exists():
                try:
                    # Run from the backend directory so paths work correctly
                    subprocess.run([str(ssl_script), server_ip], check=True, cwd=str(script_dir), timeout=180)
                    self.console.print("[green][SUCCESS][/green] SSL certificates generated")
                except subprocess.TimeoutExpired:
                    self.console.print("[yellow][WARNING][/yellow] SSL certificate generation timed out after 3 minutes")
                except subprocess.CalledProcessError:
                    self.console.print("[yellow][WARNING][/yellow] SSL certificate generation failed")
            else:
                self.console.print(f"[yellow][WARNING][/yellow] SSL script not found at {ssl_script}")
            
            # Generate nginx.conf from template
            self.console.print("[blue][INFO][/blue] Creating nginx configuration...")
            nginx_template = script_dir / "nginx.conf.template"
            if nginx_template.exists():
                try:
                    with open(nginx_template, 'r') as f:
                        nginx_content = f.read()
                    
                    # Replace TAILSCALE_IP with server_ip
                    nginx_content = nginx_content.replace('TAILSCALE_IP', server_ip)
                    
                    with open('nginx.conf', 'w') as f:
                        f.write(nginx_content)
                    
                    self.console.print(f"[green][SUCCESS][/green] nginx.conf created for: {server_ip}")
                    self.config["HTTPS_ENABLED"] = "true"
                    self.config["SERVER_IP"] = server_ip
                    
                except Exception as e:
                    self.console.print(f"[yellow][WARNING][/yellow] nginx.conf generation failed: {e}")
            else:
                self.console.print("[yellow][WARNING][/yellow] nginx.conf.template not found")

            # Generate Caddyfile from template
            self.console.print("[blue][INFO][/blue] Creating Caddyfile configuration...")
            caddyfile_template = script_dir / "Caddyfile.template"
            if caddyfile_template.exists():
                try:
                    with open(caddyfile_template, 'r') as f:
                        caddyfile_content = f.read()

                    # Replace TAILSCALE_IP with server_ip
                    caddyfile_content = caddyfile_content.replace('TAILSCALE_IP', server_ip)

                    with open('Caddyfile', 'w') as f:
                        f.write(caddyfile_content)

                    self.console.print(f"[green][SUCCESS][/green] Caddyfile created for: {server_ip}")

                except Exception as e:
                    self.console.print(f"[yellow][WARNING][/yellow] Caddyfile generation failed: {e}")
            else:
                self.console.print("[yellow][WARNING][/yellow] Caddyfile.template not found")
        else:
            self.config["HTTPS_ENABLED"] = "false"

    def generate_env_file(self):
        """Generate .env file from template and update with configuration"""
        env_path = Path(".env")
        env_template = Path(".env.template")

        # Backup existing .env if it exists
        self.backup_existing_env()

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

    def copy_config_templates(self):
        """Copy other configuration files"""
        if not Path("memory_config.yaml").exists() and Path("memory_config.yaml.template").exists():
            shutil.copy2("memory_config.yaml.template", "memory_config.yaml")
            self.console.print("[green][SUCCESS][/green] memory_config.yaml created")

        if not Path("diarization_config.json").exists() and Path("diarization_config.json.template").exists():
            shutil.copy2("diarization_config.json.template", "diarization_config.json")
            self.console.print("[green][SUCCESS][/green] diarization_config.json created")

    def show_summary(self):
        """Show configuration summary"""
        self.print_section("Configuration Summary")
        self.console.print()
        
        self.console.print(f"✅ Admin Account: {self.config.get('ADMIN_EMAIL', 'Not configured')}")
        self.console.print(f"✅ Transcription: {self.config.get('TRANSCRIPTION_PROVIDER', 'Not configured')}")
        self.console.print(f"✅ LLM Provider: {self.config.get('LLM_PROVIDER', 'Not configured')}")
        self.console.print(f"✅ Memory Provider: {self.config.get('MEMORY_PROVIDER', 'friend_lite')}")
        # Auto-determine URLs based on HTTPS configuration
        if self.config.get('HTTPS_ENABLED') == 'true':
            server_ip = self.config.get('SERVER_IP', 'localhost')
            self.console.print(f"✅ Backend URL: https://{server_ip}/")
            self.console.print(f"✅ Dashboard URL: https://{server_ip}/")
        else:
            backend_port = self.config.get('BACKEND_PUBLIC_PORT', '8000')
            webui_port = self.config.get('WEBUI_PORT', '5173')
            self.console.print(f"✅ Backend URL: http://localhost:{backend_port}")
            self.console.print(f"✅ Dashboard URL: http://localhost:{webui_port}")

    def show_next_steps(self):
        """Show next steps"""
        self.print_section("Next Steps")
        self.console.print()
        
        self.console.print("1. Start the main services:")
        self.console.print("   [cyan]docker compose up --build -d[/cyan]")
        self.console.print()
        
        # Auto-determine URLs for next steps
        if self.config.get('HTTPS_ENABLED') == 'true':
            server_ip = self.config.get('SERVER_IP', 'localhost')
            self.console.print("2. Access the dashboard:")
            self.console.print(f"   [cyan]https://{server_ip}/[/cyan]")
            self.console.print()
            self.console.print("3. Check service health:")
            self.console.print(f"   [cyan]curl -k https://{server_ip}/health[/cyan]")
        else:
            webui_port = self.config.get('WEBUI_PORT', '5173')
            backend_port = self.config.get('BACKEND_PUBLIC_PORT', '8000')
            self.console.print("2. Access the dashboard:")
            self.console.print(f"   [cyan]http://localhost:{webui_port}[/cyan]")
            self.console.print()
            self.console.print("3. Check service health:")
            self.console.print(f"   [cyan]curl http://localhost:{backend_port}/health[/cyan]")

        if self.config.get("MEMORY_PROVIDER") == "openmemory_mcp":
            self.console.print()
            self.console.print("4. Start OpenMemory MCP:")
            self.console.print("   [cyan]cd ../../extras/openmemory-mcp && docker compose up -d[/cyan]")

        if self.config.get("TRANSCRIPTION_PROVIDER") == "offline":
            self.console.print()
            self.console.print("5. Start Parakeet ASR:")
            self.console.print("   [cyan]cd ../../extras/asr-services && docker compose up parakeet -d[/cyan]")

    def run(self):
        """Run the complete setup process"""
        self.print_header("🚀 Friend-Lite Interactive Setup")
        self.console.print("This wizard will help you configure Friend-Lite with all necessary services.")
        self.console.print("We'll ask for your API keys and preferences step by step.")
        self.console.print()

        try:
            # Backup existing config
            self.backup_existing_env()

            # Run setup steps
            self.setup_authentication()
            self.setup_transcription()
            self.setup_llm()
            self.setup_memory()
            self.setup_optional_services()
            self.setup_network()
            self.setup_https()

            # Generate files
            self.print_header("Configuration Complete!")
            self.generate_env_file()
            self.copy_config_templates()

            # Show results
            self.show_summary()
            self.show_next_steps()

            self.console.print()
            self.console.print("[green][SUCCESS][/green] Setup complete! 🎉")
            self.console.print()
            self.console.print("For detailed documentation, see:")
            self.console.print("  • Docs/quickstart.md")
            self.console.print("  • MEMORY_PROVIDERS.md")
            self.console.print("  • CLAUDE.md")

        except KeyboardInterrupt:
            self.console.print()
            self.console.print("[yellow]Setup cancelled by user[/yellow]")
            sys.exit(0)
        except Exception as e:
            self.console.print(f"[red][ERROR][/red] Setup failed: {e}")
            sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Friend-Lite Advanced Backend Setup")
    parser.add_argument("--speaker-service-url", 
                       help="Speaker Recognition service URL (default: prompt user)")
    parser.add_argument("--parakeet-asr-url", 
                       help="Parakeet ASR service URL (default: prompt user)")
    parser.add_argument("--enable-https", action="store_true",
                       help="Enable HTTPS configuration (default: prompt user)")
    parser.add_argument("--server-ip", 
                       help="Server IP/domain for SSL certificate (default: prompt user)")
    
    args = parser.parse_args()
    
    setup = FriendLiteSetup(args)
    setup.run()


if __name__ == "__main__":
    main()