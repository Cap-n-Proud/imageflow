import logging
import parser

# parse the arguments

FORMAT = "[%(asctime)s][%(name)s][%(levelname)s][%(message)s]"

logger = logging.getLogger()
logger.setLevel(eval("logging." + args.logLevel))
formatter = logging.Formatter(FORMAT)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(eval("logging." + args.logLevel))
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(args.logFilePath + args.logName)
file_handler.setLevel(eval("logging." + args.logLevel))
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
