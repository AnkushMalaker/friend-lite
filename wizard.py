#!/usr/bin/env python3
"""
Friend-Lite Root Setup Orchestrator
Handles service selection and delegation only - no configuration duplication
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import get_key
from rich import print as rprint
from rich.console import Console
from rich.prompt import Confirm

console = Console()

def read_env_value(env_file_path, key):
    """Read a value from an .env file using python-dotenv"""
    env_path = Path(env_file_path)
    if not env_path.exists():
        return None

    value = get_key(str(env_path), key)
    # get_key returns None if key doesn't exist or value is empty
    return value if value else None

def is_placeholder(value, *placeholder_variants):
    """
    Check if a value is a placeholder by normalizing both the value and placeholders.
    Treats 'your-key-here' and 'your_key_here' as equivalent.

    Args:
        value: The value to check
        placeholder_variants: One or more placeholder strings to check against

    Returns:
        True if value matches any placeholder variant (after normalization)
    """
    if not value:
        return True

    # Normalize by replacing hyphens with underscores
    normalized_value = value.replace('-', '_').lower()

    for placeholder in placeholder_variants:
        normalized_placeholder = placeholder.replace('-', '_').lower()
        if normalized_value == normalized_placeholder:
            return True

    return False

SERVICES = {
    'backend': {
        'advanced': {
            'path': 'backends/advanced',
            'cmd': ['uv', 'run', '--with-requirements', 'setup-requirements.txt', 'python', 'init.py'],
            'description': 'Advanced AI backend with full feature set',
            'required': True
        }
    },
    'extras': {
        'speaker-recognition': {
            'path': 'extras/speaker-recognition',
            'cmd': ['uv', 'run', '--with-requirements', 'setup-requirements.txt', 'python', 'init.py'],
            'description': 'Speaker identification and enrollment'
        },
        'asr-services': {
            'path': 'extras/asr-services',
            'cmd': ['./setup.sh'],
            'description': 'Offline speech-to-text (Parakeet)'
        },
        'openmemory-mcp': {
            'path': 'extras/openmemory-mcp',
            'cmd': ['./setup.sh'],
            'description': 'OpenMemory MCP server'
        }
    }
}

def check_service_exists(service_name, service_config):
    """Check if service directory and script exist"""
    service_path = Path(service_config['path'])
    if not service_path.exists():
        return False, f"Directory {service_path} does not exist"

    # For services with Python init scripts, check if init.py exists
    if service_name in ['advanced', 'speaker-recognition']:
        script_path = service_path / 'init.py'
        if not script_path.exists():
            return False, f"Script {script_path} does not exist"
    else:
        # For other extras, check if setup.sh exists
        script_path = service_path / 'setup.sh'
        if not script_path.exists():
            return False, f"Script {script_path} does not exist (will be created in Phase 2)"

    return True, "OK"

def select_services():
    """Let user select which services to setup"""
    console.print("🚀 [bold cyan]Friend-Lite Service Setup[/bold cyan]")
    console.print("Select which services to configure:\n")
    
    selected = []
    
    # Backend is required
    console.print("📱 [bold]Backend (Required):[/bold]")
    console.print("  ✅ Advanced Backend - Full AI features")
    selected.append('advanced')
    
    # Optional extras
    console.print("\n🔧 [bold]Optional Services:[/bold]")
    for service_name, service_config in SERVICES['extras'].items():
        # Check if service exists
        exists, msg = check_service_exists(service_name, service_config)
        if not exists:
            console.print(f"  ⏸️  {service_config['description']} - [dim]{msg}[/dim]")
            continue
        
        try:
            enable_service = Confirm.ask(f"  Setup {service_config['description']}?", default=False)
        except EOFError:
            console.print("Using default: No")
            enable_service = False
            
        if enable_service:
            selected.append(service_name)
    
    return selected

def cleanup_unselected_services(selected_services):
    """Backup and remove .env files from services that weren't selected"""
    
    all_services = list(SERVICES['backend'].keys()) + list(SERVICES['extras'].keys())
    
    for service_name in all_services:
        if service_name not in selected_services:
            if service_name == 'advanced':
                service_path = Path(SERVICES['backend'][service_name]['path'])
            else:
                service_path = Path(SERVICES['extras'][service_name]['path'])
            
            env_file = service_path / '.env'
            if env_file.exists():
                # Create backup with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = service_path / f'.env.backup.{timestamp}.unselected'
                env_file.rename(backup_file)
                console.print(f"🧹 [dim]Backed up {service_name} configuration to {backup_file.name} (service not selected)[/dim]")

def run_service_setup(service_name, selected_services, https_enabled=False, server_ip=None):
    """Execute individual service setup script"""
    if service_name == 'advanced':
        service = SERVICES['backend'][service_name]
        
        # For advanced backend, pass URLs of other selected services and HTTPS config
        cmd = service['cmd'].copy()
        if 'speaker-recognition' in selected_services:
            cmd.extend(['--speaker-service-url', 'http://host.docker.internal:8085'])
        if 'asr-services' in selected_services:
            cmd.extend(['--parakeet-asr-url', 'http://host.docker.internal:8767'])
        
        # Add HTTPS configuration
        if https_enabled and server_ip:
            cmd.extend(['--enable-https', '--server-ip', server_ip])
            
    else:
        service = SERVICES['extras'][service_name]
        cmd = service['cmd'].copy()
        
        # Add HTTPS configuration for services that support it
        if service_name == 'speaker-recognition' and https_enabled and server_ip:
            cmd.extend(['--enable-https', '--server-ip', server_ip])
        
        # For speaker-recognition, try to pass API keys and config if available
        if service_name == 'speaker-recognition':
            # Pass Deepgram API key from backend if available
            backend_env_path = 'backends/advanced/.env'
            deepgram_key = read_env_value(backend_env_path, 'DEEPGRAM_API_KEY')
            if deepgram_key and not is_placeholder(deepgram_key, 'your_deepgram_api_key_here', 'your-deepgram-api-key-here'):
                cmd.extend(['--deepgram-api-key', deepgram_key])
                console.print("[blue][INFO][/blue] Found existing DEEPGRAM_API_KEY from backend config, reusing")

            # Pass HF Token from existing speaker recognition .env if available
            speaker_env_path = 'extras/speaker-recognition/.env'
            hf_token = read_env_value(speaker_env_path, 'HF_TOKEN')
            if hf_token and not is_placeholder(hf_token, 'your_huggingface_token_here', 'your-huggingface-token-here'):
                cmd.extend(['--hf-token', hf_token])
                console.print("[blue][INFO][/blue] Found existing HF_TOKEN, reusing")

            # Pass compute mode from existing .env if available
            compute_mode = read_env_value(speaker_env_path, 'COMPUTE_MODE')
            if compute_mode in ['cpu', 'gpu']:
                cmd.extend(['--compute-mode', compute_mode])
                console.print(f"[blue][INFO][/blue] Found existing COMPUTE_MODE ({compute_mode}), reusing")
        
        # For openmemory-mcp, try to pass OpenAI API key from backend if available
        if service_name == 'openmemory-mcp':
            backend_env_path = 'backends/advanced/.env'
            openai_key = read_env_value(backend_env_path, 'OPENAI_API_KEY')
            if openai_key and not is_placeholder(openai_key, 'your_openai_api_key_here', 'your-openai-api-key-here', 'your_openai_key_here', 'your-openai-key-here'):
                cmd.extend(['--openai-api-key', openai_key])
                console.print("[blue][INFO][/blue] Found existing OPENAI_API_KEY from backend config, reusing")
    
    console.print(f"\n🔧 [bold]Setting up {service_name}...[/bold]")
    
    # Check if service exists before running
    exists, msg = check_service_exists(service_name, service)
    if not exists:
        console.print(f"❌ {service_name} setup failed: {msg}")
        return False
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=service['path'],
            check=True,
            timeout=300  # 5 minute timeout for service setup
        )
        
        console.print(f"✅ {service_name} setup completed")
        return True
            
    except FileNotFoundError as e:
        console.print(f"❌ {service_name} setup failed: {e}")
        return False
    except subprocess.TimeoutExpired as e:
        console.print(f"❌ {service_name} setup timed out after {e.timeout} seconds")
        return False
    except subprocess.CalledProcessError as e:
        console.print(f"❌ {service_name} setup failed with exit code {e.returncode}")
        return False
    except Exception as e:
        console.print(f"❌ {service_name} setup failed: {e}")
        return False

def show_service_status():
    """Show which services are available"""
    console.print("\n📋 [bold]Service Status:[/bold]")
    
    # Check backend
    exists, msg = check_service_exists('advanced', SERVICES['backend']['advanced'])
    status = "✅" if exists else "❌"
    console.print(f"  {status} Advanced Backend - {msg}")
    
    # Check extras
    for service_name, service_config in SERVICES['extras'].items():
        exists, msg = check_service_exists(service_name, service_config)
        status = "✅" if exists else "⏸️"
        console.print(f"  {status} {service_config['description']} - {msg}")

def main():
    """Main orchestration logic"""
    console.print("🎉 [bold green]Welcome to Friend-Lite![/bold green]\n")
    
    # Show what's available
    show_service_status()
    
    # Service Selection
    selected_services = select_services()
    
    if not selected_services:
        console.print("\n[yellow]No services selected. Exiting.[/yellow]")
        return
    
    # HTTPS Configuration (for services that need it)
    https_enabled = False
    server_ip = None
    
    # Check if we have services that benefit from HTTPS
    https_services = {'advanced', 'speaker-recognition'} # advanced will always need https then
    needs_https = bool(https_services.intersection(selected_services))
    
    if needs_https:
        console.print("\n🔒 [bold cyan]HTTPS Configuration[/bold cyan]")
        console.print("HTTPS enables microphone access in browsers and secure connections")
        
        try:
            https_enabled = Confirm.ask("Enable HTTPS for selected services?", default=False)
        except EOFError:
            console.print("Using default: No")
            https_enabled = False
        
        if https_enabled:
            console.print("\n[blue][INFO][/blue] For distributed deployments, use your Tailscale IP")
            console.print("[blue][INFO][/blue] For local-only access, use 'localhost'")
            console.print("Examples: localhost, 100.64.1.2, your-domain.com")
            
            while True:
                try:
                    server_ip = console.input("Server IP/Domain for SSL certificates [localhost]: ").strip()
                    if not server_ip:
                        server_ip = "localhost"
                    break
                except EOFError:
                    server_ip = "localhost"
                    break
            
            console.print(f"[green]✅[/green] HTTPS configured for: {server_ip}")
    
    # Pure Delegation - Run Each Service Setup
    console.print(f"\n📋 [bold]Setting up {len(selected_services)} services...[/bold]")
    
    # Clean up .env files from unselected services (creates backups)
    cleanup_unselected_services(selected_services)
    
    success_count = 0
    failed_services = []
    
    for service in selected_services:
        if run_service_setup(service, selected_services, https_enabled, server_ip):
            success_count += 1
        else:
            failed_services.append(service)
    
    # Final Summary
    console.print(f"\n🎊 [bold green]Setup Complete![/bold green]")
    console.print(f"✅ {success_count}/{len(selected_services)} services configured successfully")
    
    if failed_services:
        console.print(f"❌ Failed services: {', '.join(failed_services)}")
    
    # Next Steps
    console.print("\n📖 [bold]Next Steps:[/bold]")
    
    # Service Management Commands
    console.print("1. Start all configured services:")
    console.print("   [cyan]uv run --with-requirements setup-requirements.txt python services.py start --all --build[/cyan]")
    console.print("")
    console.print("2. Or start individual services:")
    
    configured_services = []
    if 'advanced' in selected_services and 'advanced' not in failed_services:
        configured_services.append("backend")
    if 'speaker-recognition' in selected_services and 'speaker-recognition' not in failed_services:
        configured_services.append("speaker-recognition") 
    if 'asr-services' in selected_services and 'asr-services' not in failed_services:
        configured_services.append("asr-services")
    if 'openmemory-mcp' in selected_services and 'openmemory-mcp' not in failed_services:
        configured_services.append("openmemory-mcp")
        
    if configured_services:
        service_list = " ".join(configured_services)
        console.print(f"   [cyan]uv run --with-requirements setup-requirements.txt python services.py start {service_list}[/cyan]")
    
    console.print("")
    console.print("3. Check service status:")
    console.print("   [cyan]uv run --with-requirements setup-requirements.txt python services.py status[/cyan]")
    
    console.print("")
    console.print("4. Stop services when done:")
    console.print("   [cyan]uv run --with-requirements setup-requirements.txt python services.py stop --all[/cyan]")
    
    console.print(f"\n🚀 [bold]Enjoy Friend-Lite![/bold]")
    
    # Show individual service usage
    console.print(f"\n💡 [dim]Tip: You can also setup services individually:[/dim]")
    console.print(f"[dim]   cd backends/advanced && uv run --with-requirements setup-requirements.txt python init.py[/dim]")
    console.print(f"[dim]   cd extras/speaker-recognition && uv run --with-requirements setup-requirements.txt python init.py[/dim]")

if __name__ == "__main__":
    main()