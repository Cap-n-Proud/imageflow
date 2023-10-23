import sys
sys.path.append("..")  # Add the parent directory to the Python path

# Now you can import modules from the parent directory
from ProcessMedia import *



fname = "mnt/Photos/005-PhotoBook/2023/09/_DSC4552.jpg"

# f=  ProcessMedia.set_rank(fname,fname)

import asyncio

async def main():
    process_media = ProcessMedia()
    await process_media.set_rank(fname)

if __name__ == "__main__":
    asyncio.run(main())