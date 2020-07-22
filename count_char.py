import os

maxi = 0
for link in os.listdir("/home/han/Documents/gen_text/train"):
	if link.endswith(".txt"):
		with open("/home/han/Documents/gen_text/train/" + link, 'r') as f:
			for line in f:
				if len(line) > maxi:
					if len(line) > 46:
						print(line)
						image_name = link[:-3] + 'jpg'
						os.remove("/home/han/Documents/gen_text/train/" + link)
						os.remove("/home/han/Documents/gen_text/train/" + image_name)
					maxi = len(line)
					print(maxi)
					
		f.close()



