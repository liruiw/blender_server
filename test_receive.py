import zmq
import time
import sys

port = "15562"
if len(sys.argv) > 1:
    port =  sys.argv[1]
    int(port)
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)
# socket.bind("tcp://*:%s" % port)

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: ", message)
    time.sleep (1)
    socket.send_string("World from %s" % port)