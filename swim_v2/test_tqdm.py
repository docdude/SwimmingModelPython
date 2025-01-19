from tqdm import tqdm
import time

# Simulate static positions
positions = [1, 2, 3]

bars = {}
for pos in positions:
    bars[pos] = tqdm(total=100, desc=f"Bar {pos}", position=pos)

for i in range(100):
    time.sleep(0.1)
    for bar in bars.values():
        bar.update(1)
