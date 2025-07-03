#!/usr/bin/env python
"""
Test script to verify the Real-ESRGAN API endpoints
"""
import requests
import os
import time
import argparse
from rich.console import Console
from rich.panel import Panel
from rich import print
from rich.progress import Progress

console = Console()

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    with console.status("[bold green]Testing health endpoint..."):
        try:
            response = requests.get(f"{base_url}/health")
            response.raise_for_status()
            data = response.json()
            
            console.print(Panel.fit(
                f"Health Status: {data['status']}\n"
                f"Redis: {data['redis']}\n"
                f"CPU Usage: {data['system'].get('cpu_usage', 'N/A')}\n"
                f"Memory Usage: {data['system'].get('memory_usage', 'N/A')}\n"
                f"Disk Usage: {data['system'].get('disk_usage', 'N/A')}\n"
                f"Load Average: {data['system'].get('load_average', 'N/A')}",
                title="Health Endpoint", 
                border_style="green"
            ))
            return True
        except Exception as e:
            console.print(f"[bold red]Health endpoint test failed: {e}")
            return False

def test_upscale_endpoint(base_url, image_path, direct_process=False):
    """Test the upscale endpoint"""
    if not os.path.exists(image_path):
        console.print(f"[bold red]Image not found: {image_path}")
        return False
    
    status_text = "direct processing" if direct_process else "background processing"
    with console.status(f"[bold green]Testing upscale endpoint with {status_text}..."):
        try:
            # Prepare form data
            files = {'file': open(image_path, 'rb')}
            params = {
                'scale': 2,
                'face_enhance': False,
                'direct_process': direct_process
            }
            
            # Make the request
            start_time = time.time()
            response = requests.post(f"{base_url}/upscale", params=params, files=files)
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - start_time
            
            console.print(Panel.fit(
                f"Job ID: {data['job_id']}\n"
                f"Status: {data['status']}\n"
                f"Message: {data['message']}\n"
                f"Time taken: {elapsed:.2f} seconds\n" +
                (f"Result URL: {data['result_url']}" if 'result_url' in data else ""),
                title="Upscale Endpoint",
                border_style="green"
            ))
            
            # If background processing, check status
            if not direct_process and data['status'] == 'queued':
                job_id = data['job_id']
                console.print("[yellow]Job queued for processing. Checking status...")
                
                max_checks = 10
                with Progress() as progress:
                    task = progress.add_task("[green]Waiting for processing...", total=max_checks)
                    
                    for i in range(max_checks):
                        progress.update(task, advance=1)
                        time.sleep(5)  # Wait 5 seconds between checks
                        
                        status_response = requests.get(f"{base_url}/status/{job_id}")
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            progress.update(task, description=f"[green]Status: {status_data.get('status', 'unknown')}")
                            
                            if status_data.get('status') in ['completed', 'failed']:
                                console.print(Panel.fit(
                                    f"Final Status: {status_data.get('status')}\n" +
                                    (f"Result URL: {status_data.get('result_url', 'N/A')}" if 'result_url' in status_data else ""),
                                    title="Final Job Status",
                                    border_style="green" if status_data.get('status') == 'completed' else "red"
                                ))
                                break
                        else:
                            console.print(f"[red]Failed to get status: {status_response.status_code}")
            
            return True
        except Exception as e:
            console.print(f"[bold red]Upscale endpoint test failed: {e}")
            if 'response' in locals():
                console.print(f"[bold red]Response: {response.text}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Test Real-ESRGAN API endpoints")
    parser.add_argument("--url", default="https://imag-upscaler-server-production.up.railway.app", help="Base URL for the API")
    parser.add_argument("--image", required=True, help="Path to the test image file")
    parser.add_argument("--direct", action="store_true", help="Use direct processing mode")
    
    args = parser.parse_args()
    
    console.print("[bold]Real-ESRGAN API Test[/bold]")
    console.print(f"Base URL: {args.url}")
    console.print(f"Test Image: {args.image}")
    console.print(f"Direct Processing: {args.direct}")
    console.print("---")
    
    # Test endpoints
    if test_health_endpoint(args.url):
        test_upscale_endpoint(args.url, args.image, args.direct)

if __name__ == "__main__":
    main()
