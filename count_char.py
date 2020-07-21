import os

maxi = 0
for link in os.listdir("/home/han/Documents/gen_text/train"):
	if link.endswith(".txt"):
		with open("/home/han/Documents/gen_text/train/" + link, 'r') as f:
			for line in f:
				if len(line) > maxi:
					print(maxi)
					maxi = len(line)
		f.close()

print(maxi)

