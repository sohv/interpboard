#!/usr/bin/env python3
"""
Command-line interface for InterpBoard.
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="InterpBoard CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  interpboard analyze --model gpt2 --text "The capital of France is Paris."
  interpboard demo --model gpt2-medium
  interpboard serve --port 8501
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run interpretability analysis')
    analyze_parser.add_argument('--model', required=True, help='Model name or path')
    analyze_parser.add_argument('--text', required=True, help='Text to analyze')
    analyze_parser.add_argument('--methods', nargs='+', 
                               default=['integrated_gradients'],
                               help='Attribution methods to use')
    analyze_parser.add_argument('--output', help='Output directory for results')
    analyze_parser.add_argument('--device', default='auto', help='Device to use')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    demo_parser.add_argument('--model', default='gpt2', help='Model name')
    demo_parser.add_argument('--interface', choices=['streamlit', 'gradio'], 
                           default='streamlit', help='Interface type')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start web server')
    serve_parser.add_argument('--port', type=int, default=8501, help='Port number')
    serve_parser.add_argument('--host', default='localhost', help='Host address')
    
    # Version command
    subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'analyze':
        run_analysis(args)
    elif args.command == 'demo':
        run_demo(args)
    elif args.command == 'serve':
        start_server(args)
    elif args.command == 'version':
        show_version()

def run_analysis(args):
    """Run interpretability analysis."""
    try:
        from interpboard.dashboards import create_unified_dashboard
        
        print(f"Loading model: {args.model}")
        attribution_dashboard, ablation_dashboard = create_unified_dashboard(
            args.model, device=args.device
        )
        
        print(f"Analyzing text: '{args.text}'")
        result = attribution_dashboard.analyze(
            args.text,
            methods=args.methods,
            visualize=True,
            save_html=True
        )
        
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            print(f"Results saved to: {output_dir}")
        
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

def run_demo(args):
    """Run interactive demo."""
    try:
        if args.interface == 'streamlit':
            import subprocess
            subprocess.run([
                'streamlit', 'run', 
                str(Path(__file__).parent / 'demo' / 'streamlit_app.py'),
                '--', '--model', args.model
            ])
        elif args.interface == 'gradio':
            from interpboard.demo.gradio_app import create_interface
            interface = create_interface(args.model)
            interface.launch()
    except ImportError as e:
        print(f"Demo interface not available: {e}")
        print("Install with: pip install interpboard[demo]")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting demo: {e}")
        sys.exit(1)

def start_server(args):
    """Start web server."""
    try:
        import subprocess
        subprocess.run([
            'streamlit', 'run',
            str(Path(__file__).parent / 'web' / 'app.py'),
            '--server.port', str(args.port),
            '--server.address', args.host
        ])
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

def show_version():
    """Show version information."""
    try:
        from interpboard import __version__
        print(f"InterpBoard v{__version__}")
    except ImportError:
        print("Version information not available")

if __name__ == '__main__':
    main()