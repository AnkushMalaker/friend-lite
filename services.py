#!/usr/bin/env python3
"""
Friend-Lite Service Management
Start, stop, and manage configured services
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os

from rich import print as rprint
from rich.console import Console
from rich.table import Table
from dotenv import dotenv_values

console = Console()

SERVICES = {
    'backend': {
        'path': 'backends/advanced',
        'compose_file': 'docker-compose.yml',
        'description': 'Advanced Backend + WebUI',
        'ports': ['8000', '5173']
    },
    'speaker-recognition': {
        'path': 'extras/speaker-recognition', 
        'compose_file': 'docker-compose.yml',
        'description': 'Speaker Recognition Service',
        'ports': ['8085', '5174/8444']
    },
    'asr-services': {
        'path': 'extras/asr-services',
        'compose_file': 'docker-compose.yml', 
        'description': 'Parakeet ASR Service',
        'ports': ['8767']
    },
    'openmemory-mcp': {
        'path': 'extras/openmemory-mcp',
        'compose_file': 'docker-compose.yml',
        'description': 'OpenMemory MCP Server', 
        'ports': ['8765']
    }
}

def check_service_configured(service_name):
    """Check if service is configured (has .env file)"""
    service = SERVICES[service_name]
    service_path = Path(service['path'])
    
    # Backend uses advanced init, others use .env
    if service_name == 'backend':
        return (service_path / '.env').exists()
    else:
        return (service_path / '.env').exists()

def run_compose_command(service_name, command, build=False):
    """Run docker compose command for a service"""
    service = SERVICES[service_name]
    service_path = Path(service['path'])
    
    if not service_path.exists():
        console.print(f"[red]❌ Service directory not found: {service_path}[/red]")
        return False
        
    compose_file = service_path / service['compose_file']
    if not compose_file.exists():
        console.print(f"[red]❌ Docker compose file not found: {compose_file}[/red]")
        return False
    
    cmd = ['docker', 'compose']
    
    # Handle speaker-recognition service specially
    if service_name == 'speaker-recognition' and command == 'up':
        # Read configuration to determine how to start
        env_file = service_path / '.env'
        if env_file.exists():
            env_values = dotenv_values(env_file)
            compute_mode = env_values.get('COMPUTE_MODE', 'cpu')
            https_enabled = env_values.get('REACT_UI_HTTPS', 'false')
            
            if https_enabled.lower() == 'true':
                # HTTPS mode: start with profile for all services (includes nginx)
                if compute_mode == 'gpu':
                    cmd.extend(['--profile', 'gpu'])
                else:
                    cmd.extend(['--profile', 'cpu'])
                cmd.extend(['up', '-d'])
            else:
                # HTTP mode: start specific services with profile (no nginx)
                if compute_mode == 'gpu':
                    cmd.extend(['--profile', 'gpu'])
                else:
                    cmd.extend(['--profile', 'cpu'])
                cmd.extend(['up', '-d', 'speaker-service-gpu' if compute_mode == 'gpu' else 'speaker-service-cpu', 'web-ui'])
        else:
            # Fallback: just start base service
            cmd.extend(['up', '-d'])
    else:
        # Standard compose commands for other services
        if command == 'up':
            cmd.extend(['up', '-d'])
        elif command == 'down':
            cmd.extend(['down'])
        elif command == 'restart':
            cmd.extend(['restart'])
        elif command == 'status':
            cmd.extend(['ps'])
    
    if command == 'up' and build:
        cmd.append('--build')
    
    try:
        # For commands that need real-time output (build), stream to console
        if build and command == 'up':
            console.print(f"[dim]Building {service_name} containers...[/dim]")
            process = subprocess.Popen(
                cmd,
                cwd=service_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Simply stream all output with coloring
            all_output = []
            
            if process.stdout is None:
                raise RuntimeError("Process stdout is None - unable to read command output")
            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                
                # Store for error context
                all_output.append(line)
                
                # Print with appropriate coloring
                if 'error' in line.lower() or 'failed' in line.lower():
                    console.print(f"  [red]{line}[/red]")
                elif 'Successfully' in line or 'Started' in line or 'Created' in line:
                    console.print(f"  [green]{line}[/green]")
                elif 'Building' in line or 'Creating' in line:
                    console.print(f"  [cyan]{line}[/cyan]")
                elif 'warning' in line.lower():
                    console.print(f"  [yellow]{line}[/yellow]")
                else:
                    console.print(f"  [dim]{line}[/dim]")
            
            # Wait for process to complete
            process.wait()
            
            # If build failed, show error summary
            if process.returncode != 0:
                console.print(f"\n[red]❌ Build failed for {service_name}[/red]")
                return False
            
            return True
        else:
            # For non-build commands, run silently unless there's an error
            result = subprocess.run(
                cmd,
                cwd=service_path,
                capture_output=True,
                text=True,
                check=False,
                timeout=120  # 2 minute timeout for service status checks
            )
            
            if result.returncode == 0:
                return True
            else:
                console.print(f"[red]❌ Command failed[/red]")
                if result.stderr:
                    console.print("[red]Error output:[/red]")
                    # Show all error output
                    for line in result.stderr.splitlines():
                        console.print(f"  [dim]{line}[/dim]")
                return False
            
    except subprocess.TimeoutExpired:
        console.print(f"[red]❌ Command timed out after 2 minutes for {service_name}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]❌ Error running command: {e}[/red]")
        return False

def start_services(services, build=False):
    """Start specified services"""
    console.print(f"🚀 [bold]Starting {len(services)} services...[/bold]")
    
    success_count = 0
    for service_name in services:
        if service_name not in SERVICES:
            console.print(f"[red]❌ Unknown service: {service_name}[/red]")
            continue
            
        if not check_service_configured(service_name):
            console.print(f"[yellow]⚠️  {service_name} not configured, skipping[/yellow]")
            continue
            
        console.print(f"\n🔧 Starting {service_name}...")
        if run_compose_command(service_name, 'up', build):
            console.print(f"[green]✅ {service_name} started[/green]")
            success_count += 1
        else:
            console.print(f"[red]❌ Failed to start {service_name}[/red]")
    
    console.print(f"\n[green]🎉 {success_count}/{len(services)} services started successfully[/green]")

def stop_services(services):
    """Stop specified services"""
    console.print(f"🛑 [bold]Stopping {len(services)} services...[/bold]")
    
    success_count = 0
    for service_name in services:
        if service_name not in SERVICES:
            console.print(f"[red]❌ Unknown service: {service_name}[/red]")
            continue
            
        console.print(f"\n🔧 Stopping {service_name}...")
        if run_compose_command(service_name, 'down'):
            console.print(f"[green]✅ {service_name} stopped[/green]")
            success_count += 1
        else:
            console.print(f"[red]❌ Failed to stop {service_name}[/red]")
    
    console.print(f"\n[green]🎉 {success_count}/{len(services)} services stopped successfully[/green]")

def show_status():
    """Show status of all services"""
    console.print("📊 [bold]Service Status:[/bold]\n")
    
    table = Table()
    table.add_column("Service", style="cyan")
    table.add_column("Configured", justify="center")
    table.add_column("Description", style="dim")
    table.add_column("Ports", style="green")
    
    for service_name, service_info in SERVICES.items():
        configured = "✅" if check_service_configured(service_name) else "❌"
        ports = ", ".join(service_info['ports'])
        table.add_row(
            service_name,
            configured, 
            service_info['description'],
            ports
        )
    
    console.print(table)
    
    console.print("\n💡 [dim]Use 'python services.py start --all' to start all configured services[/dim]")

def main():
    parser = argparse.ArgumentParser(description="Friend-Lite Service Management")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start services')
    start_parser.add_argument('services', nargs='*', 
                            help='Services to start: backend, speaker-recognition, asr-services, openmemory-mcp (or use --all)')
    start_parser.add_argument('--all', action='store_true', help='Start all configured services')
    start_parser.add_argument('--build', action='store_true', help='Build images before starting')
    
    # Stop command  
    stop_parser = subparsers.add_parser('stop', help='Stop services')
    stop_parser.add_argument('services', nargs='*',
                           help='Services to stop: backend, speaker-recognition, asr-services, openmemory-mcp (or use --all)')
    stop_parser.add_argument('--all', action='store_true', help='Stop all services')
    
    # Status command
    subparsers.add_parser('status', help='Show service status')
    
    args = parser.parse_args()
    
    if not args.command:
        show_status()
        return
    
    if args.command == 'status':
        show_status()
        
    elif args.command == 'start':
        if args.all:
            services = [s for s in SERVICES.keys() if check_service_configured(s)]
        elif args.services:
            # Validate service names
            invalid_services = [s for s in args.services if s not in SERVICES]
            if invalid_services:
                console.print(f"[red]❌ Invalid service names: {', '.join(invalid_services)}[/red]")
                console.print(f"Available services: {', '.join(SERVICES.keys())}")
                return
            services = args.services
        else:
            console.print("[red]❌ No services specified. Use --all or specify service names.[/red]")
            return
            
        start_services(services, args.build)
        
    elif args.command == 'stop':
        if args.all:
            services = list(SERVICES.keys())
        elif args.services:
            # Validate service names
            invalid_services = [s for s in args.services if s not in SERVICES]
            if invalid_services:
                console.print(f"[red]❌ Invalid service names: {', '.join(invalid_services)}[/red]")
                console.print(f"Available services: {', '.join(SERVICES.keys())}")
                return
            services = args.services
        else:
            console.print("[red]❌ No services specified. Use --all or specify service names.[/red]")
            return
            
        stop_services(services)

if __name__ == "__main__":
    main()