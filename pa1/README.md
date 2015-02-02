## CS149 Programming Assignment 1 -- Chat Server
John Doherty
doherty1

Explanation:
For this assignment I created a queue of incoming connections (sockets) that need to be handled and a pool of threads that take connections from the queue and handle their requests. The threads are started right when the server is launched and run in an endless loop until the program ends. Each thread checks if there are any connections in the queue that need to be handled. If there is nothing, they wait() until the main server loop receives an incoming connection, adds it to the queue, and calls notifyAll() to wake up the threads. From there one thread will obtain the lock on the connection queue remove the connection and process it. All of this prevents spin-waiting.

The most time intensive request that we need to handle is the request for the most recent posts. For this request we want to check for new messages or wait for them to arrive if none are found. To implement this without spin-waiting, the thread will wait(15) if there are no new messages. If a thread adds a new message to that room it calls notifyAll() and all threads waiting on a new messages will be notified.

The only data structures that need to be shared between threads are the queue of connections, the ChatState for each room, and the hash map that maps room names to ChatStates. To properly protect the queue I made sure that a lock on the queue is obtained before checking whether there are items in the queue, removing an item from the queue, or adding an item to the queue. To protect the hash map from room names to ChatStates I made sure that a lock was obtained while we checked if an item was in the hash map and proceeded to add it to the map if it was not there. Finally, to protect the ChatState I made sure that methods used to write to the state or read multiple values from the state are synchronized. This prevents weird interleaving of reads and writes.



