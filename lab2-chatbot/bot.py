import sys
import aiml

kernel = aiml.Kernel()
kernel.learn(sys.argv[1])

while True:
	print(kernel.respond(input("> ")))
