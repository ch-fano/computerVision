import os


def delete_duplicates(src_dir, dpl_dir):

    if not os.path.exists(src_dir) or not os.path.isdir(src_dir):
        raise Exception("Source directory does not exist")

    if not os.path.exists(dpl_dir) or not os.path.isdir(dpl_dir):
        raise Exception("Duplicates directory does not exist")

    for dpl_file in os.listdir(dpl_dir):
        # Check if the file also exists in the source directory
        src_file_path = os.path.join(src_dir, dpl_file)
        if os.path.exists(src_file_path) and os.path.isfile(src_file_path):
            os.remove(src_file_path)
            print(f"Deleted {src_file_path}")

if __name__ == "__main__":
    src_directories = [

    ]
    duplicates_directories = [

    ]

    for i in range(len(src_directories)):
        delete_duplicates(src_directories[i], duplicates_directories[i])