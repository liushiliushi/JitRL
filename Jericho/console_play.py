from jericho import *
import argparse
from src.env import JerichoEnv

def main():
    parser = argparse.ArgumentParser(description='Play Jericho games through console')
    parser.add_argument('--rom_path', default='jericho-games/zork3.z5', type=str,
                      help='Path to the game ROM file')
    args = parser.parse_args()

    # Initialize the environment
    env = JerichoEnv(args.rom_path, seed=0, get_valid=True)
    obs, info = env.reset()
    
    print("\n=== Welcome to Jericho Console Player ===")
    print("Type 'quit' to exit, 'look' to see the current room, 'inventory' to see your items")
    print("Enter your commands as you would in a text adventure game\n")
    
    print(obs)  # Print initial observation
    
    while True:
        # Get user input
        command = input("\n> ").strip().lower()
        
        if command == 'quit':
            print("Thanks for playing!")
            break
            
        # Execute the command
        obs, reward, done, info = env.step(command)
        
        # Print the result and reward
        print("\n" + obs + "\n")
        print(f"Reward: {reward}")
        print(f"Options: {info['valid']}")
        if done:
            print("\nGame Over!")
            print(f"Final Score: {info.get('score', 0)}")
            break

if __name__ == "__main__":
    main()