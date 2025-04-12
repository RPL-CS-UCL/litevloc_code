#! /usr/bin/env python

import argparse
import os
import numpy as np
import re

def natural_sort_key(s):
    """
    Key function for natural sorting of strings containing numbers
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def main():
    parser = argparse.ArgumentParser(description='Generate random scene orders')
    parser.add_argument('--dir', help='Directory containing scene files')
    parser.add_argument('--output', '-o', default='scene_orders.npy',
                       help='Output file name (default: scene_orders.npy)')
    parser.add_argument('--num_orders', '-n', type=int, default=5,
                       help='Number of scene orders to generate (including original, default: 5)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()

    # Validate directory exists
    if not os.path.isdir(args.dir):
        parser.error(f"Directory '{args.dir}' does not exist")

    # Load and sort scene names in ascending order
    scenes = []
    for folder in sorted(os.listdir(args.dir), key=natural_sort_key):
        scenes.append(folder)
    if not scenes:
        parser.error(f"Directory '{args.dir}' is empty")

    # Set random seed
    np.random.seed(args.seed)

    # Generate scene orders
    original_order = np.array(scenes)
    orders = [original_order]  # Start with original order
    
    # Generate random permutations
    for _ in range(args.num_orders - 1):
        orders.append(np.random.permutation(original_order))

    # Save as numpy array
    print(orders)
    with open(args.output, 'w') as f:
        for order in orders:
            f.write(' '.join(order) + '\n')
    print(f"Saved {args.num_orders} scene orders to {args.output}")

if __name__ == '__main__':
    main()
