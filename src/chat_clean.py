#!/usr/bin/env python3
"""
Clean chat interface - shows only conversation, no debug logs.
"""

from agent_system import main_system_streaming, initialize_system, set_debug_mode
import time

def main():
    """Clean chat with minimal output."""
    # Disable all debug logging
    set_debug_mode(False)
    
    print("🌙 Luna (Clean Mode)")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    # Initialize system quietly
    try:
        initialize_system()
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif not user_input:
                continue
            
            # Start latency tracking
            start_time = time.time()
            
            # Clean Luna response
            print("🌙 Luna: ", end="", flush=True)
            first_chunk_time = None
            for chunk in main_system_streaming(user_input):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                print(chunk, end="", flush=True)
            print()  # New line after response
            
            # Calculate and display latency metrics
            end_time = time.time()
            total_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            if first_chunk_time:
                first_chunk_latency = (first_chunk_time - start_time) * 1000
                print(f"⏱️  Latency - First chunk: {first_chunk_latency:.1f}ms, Total: {total_time:.1f}ms")
            else:
                print(f"⏱️  Latency - Total: {total_time:.1f}ms")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()