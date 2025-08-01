import json
import argparse
import os

def replace_descriptions(input_path, output_path):
    """
    Reads a JSON file containing a list of objects, replaces the value
    of the 'description' key in each object with "This is a photo.",
    and writes the result to a new JSON file.

    Args:
        input_path (str): The path to the input JSON file.
        output_path (str): The path where the modified JSON file will be saved.
    """
    try:
        # Read the original JSON data from the input file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ensure the data is a list of dictionaries as expected
        if not isinstance(data, list):
            print("Error: The JSON file does not contain a list.")
            return

        # Iterate through each item and modify the 'description'
        modified_count = 0
        for item in data:
            if isinstance(item, dict) and 'description' in item:
                item['description'] = "This is a photo."
                modified_count += 1
            else:
                print(f"Warning: Found an item that is not a dictionary or lacks a 'description' key: {item}")


        # Write the modified data to the new output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        print(f"Success! Replaced descriptions in {modified_count} items.")
        print(f"Input file:  '{os.path.abspath(input_path)}'")
        print(f"Output file: '{os.path.abspath(output_path)}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{input_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Replace the 'description' field in a JSON file with 'This is a photo.'."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the source JSON file (e.g., generated_rgb_descriptions.json)."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the new, modified JSON file (e.g., generated_non-informative_text.json)."
    )

    args = parser.parse_args()
    
    # Run the main function with the provided file paths
    replace_descriptions(args.input_file, args.output_file)