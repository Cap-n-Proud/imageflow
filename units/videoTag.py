
fps = 30
frame_count = 30000
sampling_strategy = ["0.1", "0.5", "0.9"]

for sample in sampling_strategy:
    print(f"{frame_count*float(sample):n}")
a = " "
dct = {'a': 1, 'b': 2}
newline = "\n"  # \escapes are not allowed inside f-strings
# print(f'{a.join(f"{key}: {value}" for key, value in dct.items())}')


print(a.join(str(int(frame_count * float(sample)))
      for sample in sampling_strategy))

result = [int(frame_count * float(sample)) for sample in sampling_strategy]
sampling_frames = ','.join(map("{:,}".format, result))

print(
    f"frames per second: {fps}, frame count: {frame_count}, sampling frames: {a.join(sampling_frames.split(','))}")
