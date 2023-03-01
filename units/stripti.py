import re


def stripIt(s):
    """
    Removes all html tags from a string, leaving just the
    content behind.
    """
    a = re.sub('<.*?>', '', s).replace('"', '').replace("\n", "")

    return re.sub('\s{2,}', ' ', a)


data = """
<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN"  "http://www.apple.com/DTDs/PropertyList-1.0.dtd"><plist version="1.0"><dict><key>studio</key><string>studio</string><key>cast</key><array><dict><key>name</key><string></string></dict></array><key>directors</key><array><dict><key>name</key><string></string></dict></array><key>producers</key><array><dict><key>name</key><string></string></dict></array><key>codirectors</key><array><dict><key>name</key><string>codirector</string></dict></array><key>screenwriters</key><array><dict><key>name</key><string></string></dict></array></dict></plist>
"""

print(stripIt(data))
