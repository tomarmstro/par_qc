import par_corrections
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a file.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the file to process."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with the provided file path
    par_corrections.main(args.input_file)


# %%
if __name__ == "__main__":
    main()