import os

def create_ramdisk(directory, size):
        # Check if the directory exists already
        if os.path.isdir(directory):
            self.logger.debug(f"Directory {directory} already exists.")
            return

        os.system(f"mkdir {directory}")

        # Create the ramdisk
        # os.system(f"sudo mount -t tmpfs -o size={size}M tmpfs {directory}")
        print(f"Created ramdisk at {directory} with size {size}MB.")

def clean_ramdisk(directory):
    # Check if the directory exists already
    if not os.path.isdir(directory):
        print(f"Directory {directory} does not exists.")
        return

    # Remove the ramdisk
    import shutil

    shutil.rmtree(directory)
    print(f"Ramdisk content at {directory} cleared")


rd="./rd"

#create_ramdisk(rd,100)

clean_ramdisk(rd)
